# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Student-side full-vocabulary KL distillation loss (FSDP/VeOmni engines).

Same data contract as the Megatron variant (see
``verl.trainer.distillation.megatron.full_vocab_kl``): teacher hidden states
are fetched from TransferQueue via per-sample artifacts and full-vocab teacher
logits are rebuilt on the fly with the frozen teacher lm_head. The difference
is that FSDP/VeOmni student logits are full-vocab (not TP-sharded), so the
teacher lm_head is loaded whole and the softmax needs no cross-TP collectives;
Ulysses SP only shards the sequence dimension, which is aligned by slicing the
teacher rows exactly like the top-K path does.

Memory: teacher logits are recomputed per chunk of ``full_vocab_chunk_tokens``
tokens and again in backward from the saved hidden states. Note the chunk
buffers are full-vocab wide here (no TP division), so long-vocab runs may want
a smaller ``full_vocab_chunk_tokens`` than the Megatron path.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab_common import (
    artifacts_of,
    fetch_teacher_hidden,
    get_full_teacher_lm_head,
)
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor
from verl.workers.config import DistillationConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class _FullVocabKL(torch.autograd.Function):
    """Per-token KL between teacher (rebuilt from hidden states) and student logits.

    Non-vocab-parallel variant of the Megatron implementation: full-vocab
    logits, plain softmax, no TP collectives. Teacher logits are recomputed in
    backward from the saved hidden states instead of being kept alive.

    Columns where the teacher log-prob is not finite (padded vocab tail) are
    excluded from the loss and receive zero / student-only gradient.
    """

    @staticmethod
    def forward(
        ctx,
        student_logits: torch.Tensor,  # [tokens, V], requires grad
        teacher_hidden: torch.Tensor,  # [tokens, H]
        teacher_lm_head: torch.Tensor,  # [V, H]
        reverse: bool,
    ):
        teacher_logits = teacher_hidden @ teacher_lm_head.t()
        if teacher_logits.shape[-1] < student_logits.shape[-1]:
            teacher_logits = F.pad(
                teacher_logits, (0, student_logits.shape[-1] - teacher_logits.shape[-1]), value=float("-inf")
            )
        elif teacher_logits.shape[-1] > student_logits.shape[-1]:
            teacher_logits = teacher_logits[..., : student_logits.shape[-1]]

        student_logps = F.log_softmax(student_logits.float(), dim=-1)
        teacher_logps = F.log_softmax(teacher_logits.float(), dim=-1)
        student_probs = student_logps.exp()

        finite_teacher = torch.isfinite(teacher_logps)
        if reverse:
            # KL(student || teacher) = sum_v s * (log s - log t)
            diff = torch.where(finite_teacher, student_logps - teacher_logps, torch.zeros_like(student_logps))
            per_token_kl = (student_probs * diff).sum(dim=-1)
        else:
            # KL(teacher || student) = sum_v t * (log t - log s)
            teacher_probs = teacher_logps.exp()
            diff = torch.where(finite_teacher, teacher_logps - student_logps, torch.zeros_like(teacher_logps))
            per_token_kl = (teacher_probs * diff).sum(dim=-1)

        ctx.save_for_backward(student_probs, teacher_hidden, teacher_lm_head, per_token_kl)
        ctx.reverse = reverse
        ctx.student_dtype = student_logits.dtype
        return per_token_kl

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        student_probs, teacher_hidden, teacher_lm_head, per_token_kl = ctx.saved_tensors
        teacher_logits = teacher_hidden @ teacher_lm_head.t()
        if teacher_logits.shape[-1] < student_probs.shape[-1]:
            teacher_logits = F.pad(
                teacher_logits, (0, student_probs.shape[-1] - teacher_logits.shape[-1]), value=float("-inf")
            )
        elif teacher_logits.shape[-1] > student_probs.shape[-1]:
            teacher_logits = teacher_logits[..., : student_probs.shape[-1]]
        teacher_logps = F.log_softmax(teacher_logits.float(), dim=-1)

        if ctx.reverse:
            # d/dz_j KL(s||t) = s_j * ((log s_j - log t_j) - KL)
            student_logps = student_probs.log()
            diff = student_logps - teacher_logps
            valid = torch.isfinite(diff) & (student_probs > 0)
            grad = torch.where(valid, student_probs * (diff - per_token_kl.unsqueeze(-1)), torch.zeros_like(diff))
        else:
            # d/dz_j KL(t||s) = s_j - t_j
            teacher_probs = teacher_logps.exp()
            grad = student_probs - teacher_probs
        grad = grad * grad_output.unsqueeze(-1)
        return grad.to(ctx.student_dtype), None, None, None


def _teacher_lm_heads(distillation_config: DistillationConfig, device: torch.device) -> dict[str, torch.Tensor]:
    """teacher_key -> full [V, H] frozen teacher lm_head (cached per worker)."""
    loss_config = distillation_config.distillation_loss
    return {
        key: get_full_teacher_lm_head(
            checkpoint_path=loss_config.full_vocab_lm_head_checkpoint or teacher_config.model_path,
            layer=loss_config.full_vocab_lm_head_layer,
            device=device,
        )
        for key, teacher_config in distillation_config.teacher_models.items()
    }


def _compute_full_vocab_kl(
    *,
    student_logits: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    reverse: bool,
) -> dict[str, torch.Tensor]:
    device = student_logits.device
    artifacts = artifacts_of(data)
    loss_config = config.distillation_loss
    teacher_lm_heads = _teacher_lm_heads(config, device)

    # 1. Fetch teacher hidden states, zero each sample's last row (structural
    #    marker: no teacher target past the sequence end), and pack to
    #    [1, total_nnz, H] exactly like the top-K teacher tensors.
    teacher_keys = sorted(teacher_lm_heads.keys())
    teacher_index = {key: i for i, key in enumerate(teacher_keys)}
    hidden_rows, index_rows = [], []
    for i, artifact in enumerate(artifacts):
        if artifact["teacher_name"] not in teacher_index:
            raise KeyError(
                f"full-vocab distillation: no teacher lm_head for teacher {artifact['teacher_name']!r} "
                f"(sample {i}); loaded: {teacher_keys}."
            )
        hidden = fetch_teacher_hidden(artifact).to(
            device=device, dtype=teacher_lm_heads[artifact["teacher_name"]].dtype
        )
        hidden[-1] = 0
        hidden_rows.append(hidden)
        index_rows.append(
            torch.full((hidden.shape[0],), teacher_index[artifact["teacher_name"]], dtype=torch.long, device=device)
        )
    teacher_hidden = torch.nested.nested_tensor(hidden_rows, layout=torch.jagged).values().unsqueeze(0)
    teacher_tidx = torch.nested.nested_tensor(index_rows, layout=torch.jagged).values().unsqueeze(0)

    # 2. Diagnostics: per-token teacher hidden norm (0 on padded/structural rows).
    teacher_hidden_norm = teacher_hidden.float().norm(dim=-1).detach()

    # 3. Slice across the Ulysses SP group like the top-K path (dim=1 is the
    #    packed sequence; trailing dims are preserved).
    if get_ulysses_sequence_parallel_world_size() > 1:
        teacher_hidden = slice_input_tensor(teacher_hidden, dim=1)
        teacher_tidx = slice_input_tensor(teacher_tidx, dim=1)
        teacher_hidden_norm = slice_input_tensor(teacher_hidden_norm, dim=1)
    assert teacher_hidden.shape[:2] == student_logits.shape[:2], (
        f"teacher hidden shape {teacher_hidden.shape} does not match student logits "
        f"shape {student_logits.shape} after SP slicing."
    )

    # 4. Chunked full-vocab KL. Rows are grouped by teacher inside each chunk
    #    (multi-teacher micro batches select the matching lm_head per row).
    student_flat = student_logits.reshape(-1, student_logits.shape[-1])
    hidden_flat = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
    index_flat = teacher_tidx.reshape(-1)
    num_tokens = student_flat.shape[0]
    chunk_tokens = loss_config.full_vocab_chunk_tokens

    loss_chunks = []
    for start in range(0, num_tokens, chunk_tokens):
        end = min(start + chunk_tokens, num_tokens)
        index_chunk = index_flat[start:end]
        order = torch.argsort(index_chunk)
        student_chunk = student_flat[start:end][order]
        hidden_chunk = hidden_flat[start:end][order]
        index_sorted = index_chunk[order]

        per_teacher = []
        boundaries = torch.searchsorted(index_sorted, torch.arange(len(teacher_keys) + 1, device=device))
        for t, key in enumerate(teacher_keys):
            lo, hi = boundaries[t].item(), boundaries[t + 1].item()
            if hi == lo:
                continue
            per_teacher.append(
                _FullVocabKL.apply(
                    student_chunk[lo:hi],
                    hidden_chunk[lo:hi],
                    teacher_lm_heads[key],
                    reverse,
                )
            )
        loss_sorted = torch.cat(per_teacher, dim=0)
        # Undo the per-teacher sort with an autograd-friendly gather.
        inverse_order = torch.argsort(order)
        loss_chunks.append(loss_sorted[inverse_order])

    distillation_losses = torch.cat(loss_chunks, dim=0).reshape(student_logits.shape[:-1])
    return {
        "distillation_losses": distillation_losses,
        "teacher_hidden_norm": teacher_hidden_norm.reshape(student_logits.shape[:-1]),
    }


def compute_forward_kl_full_vocab(
    *,
    student_logits: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    data_format: str = "thd",
    teacher_lm_head_shards: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """Full-vocabulary forward KL (teacher || student) per-token losses.

    Args:
        student_logits: (bsz, seqlen/sp_size, vocab_size), requires grad.
        data: micro batch carrying ``teacher_full_vocab_artifact`` per sample.
        config: DistillationConfig.
        data_format: unused on the FSDP path (SP slicing is format-agnostic).
        teacher_lm_head_shards: unused on the FSDP path; the full lm_head is
            loaded and cached per worker (see ``get_full_teacher_lm_head``).

    Returns:
        - distillation_losses: (bsz, seqlen/sp_size)
        - teacher_hidden_norm: (bsz, seqlen/sp_size)
    """
    return _compute_full_vocab_kl(student_logits=student_logits, data=data, config=config, reverse=False)


def compute_reverse_kl_full_vocab(
    *,
    student_logits: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    data_format: str = "thd",
    teacher_lm_head_shards: Optional[dict[str, torch.Tensor]] = None,
) -> dict[str, torch.Tensor]:
    """Full-vocabulary reverse KL (student || teacher) per-token losses.

    Same arguments and returns as :func:`compute_forward_kl_full_vocab`.
    """
    return _compute_full_vocab_kl(student_logits=student_logits, data=data, config=config, reverse=True)
