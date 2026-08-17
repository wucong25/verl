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

"""Student-side full-vocabulary KL distillation loss (Megatron engine).

The teacher exports pre-lm_head hidden states to TransferQueue (see
``verl.workers.rollout.vllm_rollout.full_vocab_hidden_export``); the training
batch only carries per-sample artifact metadata dicts
(``data["teacher_full_vocab_artifact"]``). This module fetches the hidden
states from TransferQueue, rebuilds full-vocab teacher logits on the fly with
the frozen teacher lm_head shard of this TP rank, and computes the per-token
forward/reverse KL against the student's vocab-parallel logits.

Memory: teacher logits are recomputed per chunk of ``full_vocab_chunk_tokens``
tokens and again in backward from the saved hidden states, so the autograd
graph never pins a full ``[tokens, vocab]`` teacher-logits tensor.
"""

import json
import logging
import os
from typing import Any

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine
from verl.trainer.distillation.megatron.losses import vocab_parallel_log_softmax
from verl.utils.fs import copy_to_local
from verl.workers.config import DistillationConfig

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_TQ_INITIALIZED = False


def _ensure_tq_initialized() -> None:
    global _TQ_INITIALIZED
    if _TQ_INITIALIZED:
        return
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise RuntimeError(
            "full-vocab distillation requires the transfer_queue package "
            "(pip install TransferQueue) on the Megatron actor workers."
        ) from exc
    tq.init()
    _TQ_INITIALIZED = True


def load_teacher_lm_head_shard(
    *,
    checkpoint_path: str,
    layer: str,
    vocab_size: int,
    tp_rank: int,
    tp_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Load this TP rank's vocab shard of the frozen teacher lm_head.

    Args:
        checkpoint_path: HF checkpoint directory (local/HDFS/model-id) containing
            safetensors weights, or a direct path to a safetensors file.
        layer: lm_head weight name; ``"auto"`` prefers ``*lm_head.weight`` and falls
            back to ``*embed_tokens.weight`` (tied word embeddings).
        vocab_size: student vocab size; the teacher must share the tokenizer.
        tp_rank / tp_size: this rank's vocab shard is rows
            ``[tp_rank * ceil(vocab_size / tp_size), ...)``.
        device: device to place the shard on.

    Returns:
        ``[ceil(vocab_size / tp_size), hidden_size]`` tensor (zero-padded at the
        vocab tail when ``vocab_size`` is not divisible by ``tp_size``).
    """
    from safetensors import safe_open

    local_path = copy_to_local(checkpoint_path)
    if os.path.isdir(local_path):
        index_file = os.path.join(local_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            with open(index_file) as f:
                weight_map = json.load(f)["weight_map"]
        else:
            single_file = os.path.join(local_path, "model.safetensors")
            if not os.path.exists(single_file):
                raise FileNotFoundError(
                    f"full-vocab distillation: no safetensors weights found in {local_path!r} "
                    "(expected model.safetensors or model.safetensors.index.json)."
                )
            with safe_open(single_file, framework="pt") as f:
                weight_map = {key: single_file for key in f.keys()}
    elif os.path.isfile(local_path):
        with safe_open(local_path, framework="pt") as f:
            weight_map = {key: local_path for key in f.keys()}
    else:
        raise FileNotFoundError(f"full-vocab distillation: teacher checkpoint {local_path!r} does not exist.")

    if layer == "auto":
        candidates = [k for k in weight_map if k.endswith("lm_head.weight")]
        if not candidates:
            candidates = [k for k in weight_map if k.endswith("embed_tokens.weight")]
        if not candidates:
            raise KeyError(
                f"full-vocab distillation: could not find a lm_head/embed_tokens weight in {local_path!r}; "
                "set distillation_loss.full_vocab_lm_head_layer explicitly."
            )
        layer = sorted(candidates)[0]
    if layer not in weight_map:
        raise KeyError(
            f"full-vocab distillation: layer {layer!r} not found in {local_path!r}. "
            f"Available keys include: {sorted(weight_map)[:10]} ..."
        )

    per_partition = (vocab_size + tp_size - 1) // tp_size
    start = tp_rank * per_partition
    end = min(start + per_partition, vocab_size)
    with safe_open(weight_map[layer], framework="pt") as f:
        shard = f.get_slice(layer)[start:end]
    if shard.shape[0] < per_partition:
        # Zero rows produce logit 0 for the padded vocab tail; the KL computation
        # masks non-finite/missing teacher columns, so only the softmax denominator
        # is affected by exp(0)=1 per padded row. Keep the shard rectangular.
        logger.warning(
            f"full-vocab distillation: teacher lm_head vocab ({vocab_size}) is not divisible by "
            f"tp_size ({tp_size}); zero-padding the last shard from {shard.shape[0]} to {per_partition} rows."
        )
        shard = F.pad(shard, (0, 0, 0, per_partition - shard.shape[0]))
    return shard.to(device=device)


def _fetch_teacher_hidden(artifact: dict[str, Any]) -> torch.Tensor:
    """Fetch one sample's teacher hidden states from TransferQueue (CPU tensor)."""
    _ensure_tq_initialized()
    import transfer_queue as tq

    batch = tq.kv_batch_get(keys=[artifact["key"]], partition_id=artifact["partition_id"])
    hidden = batch["hidden"][0]
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(f"full-vocab distillation: TQ entry {artifact['key']!r} is not a tensor: {type(hidden)}.")
    if hidden.shape[0] != artifact["seq_len"]:
        raise ValueError(
            f"full-vocab distillation: TQ entry {artifact['key']!r} has {hidden.shape[0]} rows, "
            f"but the artifact records seq_len={artifact['seq_len']}."
        )
    return hidden


class _VocabParallelFullVocabKL(torch.autograd.Function):
    """Per-token KL between teacher (rebuilt from hidden states) and student logits.

    Both distributions are vocab-parallel: the softmax normalizers are computed
    with all-reduces over the TP group, and the per-token KL partial sums are
    all-reduced as well. Teacher logits are recomputed in backward from the
    saved hidden states instead of being kept alive.

    Columns where the teacher log-prob is not finite (padded vocab tail) are
    excluded from the loss and receive zero / student-only gradient.
    """

    @staticmethod
    def forward(
        ctx,
        student_logits: torch.Tensor,  # [tokens, V/tp], requires grad
        teacher_hidden: torch.Tensor,  # [tokens, H]
        teacher_lm_head: torch.Tensor,  # [V/tp, H]
        reverse: bool,
    ):
        from megatron.core.parallel_state import get_tensor_model_parallel_group

        tp_group = get_tensor_model_parallel_group()
        teacher_logits = teacher_hidden @ teacher_lm_head.t()
        if teacher_logits.shape[-1] < student_logits.shape[-1]:
            teacher_logits = F.pad(
                teacher_logits, (0, student_logits.shape[-1] - teacher_logits.shape[-1]), value=float("-inf")
            )
        elif teacher_logits.shape[-1] > student_logits.shape[-1]:
            teacher_logits = teacher_logits[..., : student_logits.shape[-1]]

        student_logps = vocab_parallel_log_softmax(student_logits)
        teacher_logps = vocab_parallel_log_softmax(teacher_logits)
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
        torch.distributed.all_reduce(per_token_kl, op=torch.distributed.ReduceOp.SUM, group=tp_group)

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
        teacher_logps = vocab_parallel_log_softmax(teacher_logits)

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


def _artifacts_of(data: TensorDict) -> list[dict[str, Any]]:
    artifacts = data.get("teacher_full_vocab_artifact", None)
    if artifacts is None:
        raise KeyError(
            "full-vocab distillation: the micro batch has no 'teacher_full_vocab_artifact' field; "
            "the agent loop must attach the teacher hidden-state artifact to every sample."
        )
    if hasattr(artifacts, "tolist"):
        artifacts = artifacts.tolist()
    artifacts = list(artifacts)
    if any(a is None for a in artifacts):
        raise ValueError("full-vocab distillation: some samples carry no teacher hidden-state artifact.")
    return artifacts


def _compute_full_vocab_kl(
    *,
    student_logits: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    data_format: str,
    teacher_lm_head_shards: dict[str, torch.Tensor],
    reverse: bool,
) -> dict[str, torch.Tensor]:
    device = student_logits.device
    artifacts = _artifacts_of(data)
    loss_config = config.distillation_loss

    # 1. Fetch teacher hidden states and build jagged [bsz, seqlen, H] tensors.
    #    The last row of each sample is zeroed: it would predict a token past the
    #    sequence end (no teacher target), and the norm-0 row doubles as the
    #    structural marker consumed by the teacher_hidden_coverage metric.
    teacher_keys = sorted(teacher_lm_head_shards.keys())
    teacher_index = {key: i for i, key in enumerate(teacher_keys)}
    hidden_rows, index_rows = [], []
    for i, artifact in enumerate(artifacts):
        if artifact["teacher_name"] not in teacher_index:
            raise KeyError(
                f"full-vocab distillation: no teacher lm_head shard for teacher {artifact['teacher_name']!r} "
                f"(sample {i}); loaded shards: {teacher_keys}."
            )
        shard_dtype = teacher_lm_head_shards[artifact["teacher_name"]].dtype
        hidden = _fetch_teacher_hidden(artifact).to(device=device, dtype=shard_dtype)
        hidden[-1] = 0
        hidden_rows.append(hidden)
        index_rows.append(
            torch.full((hidden.shape[0],), teacher_index[artifact["teacher_name"]], dtype=torch.long, device=device)
        )
    teacher_hidden_nested = torch.nested.nested_tensor(hidden_rows, layout=torch.jagged)
    teacher_index_nested = torch.nested.nested_tensor(index_rows, layout=torch.jagged)

    # 2. Split across CP groups exactly like the top-k distillation path.
    if data_format == "thd":
        teacher_hidden_cp, *_ = preprocess_thd_engine(teacher_hidden_nested, pre_process=True)
        teacher_index_cp, *_ = preprocess_thd_engine(teacher_index_nested, pre_process=True)
    else:
        teacher_hidden_cp, *_ = preprocess_bshd_engine(teacher_hidden_nested, pre_process=True)
        teacher_index_cp, *_ = preprocess_bshd_engine(teacher_index_nested, pre_process=True)
    assert teacher_hidden_cp.shape[:-1] == student_logits.shape[:-1], (
        f"teacher hidden shape {teacher_hidden_cp.shape} does not match student logits "
        f"shape {student_logits.shape} after CP split."
    )

    # 3. Diagnostics: per-token teacher hidden norm (0 on padded/structural rows).
    teacher_hidden_norm = teacher_hidden_cp.float().norm(dim=-1).detach()
    teacher_hidden_norm = teacher_hidden_norm.reshape(student_logits.shape[:-1])

    # 4. Chunked vocab-parallel KL. Rows are grouped by teacher inside each chunk
    #    (multi-teacher micro batches select the matching lm_head shard per row).
    student_flat = student_logits.reshape(-1, student_logits.shape[-1])
    hidden_flat = teacher_hidden_cp.reshape(-1, teacher_hidden_cp.shape[-1])
    index_flat = teacher_index_cp.reshape(-1)
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
                _VocabParallelFullVocabKL.apply(
                    student_chunk[lo:hi],
                    hidden_chunk[lo:hi],
                    teacher_lm_head_shards[key],
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
        "teacher_hidden_norm": teacher_hidden_norm,
    }


def compute_forward_kl_full_vocab(
    *,
    student_logits: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    data_format: str,
    teacher_lm_head_shards: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Full-vocabulary forward KL (teacher || student) per-token losses.

    Args:
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size), requires grad.
        data: micro batch carrying ``teacher_full_vocab_artifact`` per sample.
        config: DistillationConfig.
        data_format: "thd" or "bshd".
        teacher_lm_head_shards: teacher_key -> [V/tp, H] lm_head shard of this TP rank.

    Returns:
        - distillation_losses: (bsz, seqlen/cp_size)
        - teacher_hidden_norm: (bsz, seqlen/cp_size)
    """
    return _compute_full_vocab_kl(
        student_logits=student_logits,
        data=data,
        config=config,
        data_format=data_format,
        teacher_lm_head_shards=teacher_lm_head_shards,
        reverse=False,
    )


def compute_reverse_kl_full_vocab(
    *,
    student_logits: torch.Tensor,
    data: TensorDict,
    config: DistillationConfig,
    data_format: str,
    teacher_lm_head_shards: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Full-vocabulary reverse KL (student || teacher) per-token losses.

    Same arguments and returns as :func:`compute_forward_kl_full_vocab`.
    """
    return _compute_full_vocab_kl(
        student_logits=student_logits,
        data=data,
        config=config,
        data_format=data_format,
        teacher_lm_head_shards=teacher_lm_head_shards,
        reverse=True,
    )
