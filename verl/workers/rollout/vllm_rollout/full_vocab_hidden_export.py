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

"""Teacher-side hidden-state export for full-vocabulary KL distillation.

In full-vocab OPD the teacher does not return top-k logprobs. Instead the
teacher server performs a prefill-only forward (``max_tokens=1``,
``prompt_logprobs=0``) and the pre-lm_head hidden states of every input
position are captured, written to TransferQueue, and referenced by a small
artifact metadata dict that rides with the training batch. The student
(Megatron) rebuilds full-vocab teacher logits on the fly from the exported
hidden states and the frozen teacher lm_head
(see ``verl.trainer.distillation.megatron.full_vocab_kl``).

TransferQueue layout:

- partition: ``full_vocab_hidden_{prefix}_{teacher_name}_step_{step}``
- key: ``{teacher_name}/step={step}/sample={uid}``
- field: ``hidden`` — a ``[seq_len, hidden_size]`` tensor, row ``t`` is the
  hidden state after input token ``t`` (i.e. it predicts token ``t + 1``).
"""

import logging
import os
from types import MethodType
from typing import Any, Optional

import torch

from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

FULL_VOCAB_TQ_FIELD = "hidden"

_TQ_INITIALIZED = False


def full_vocab_partition_name(prefix: str, teacher_name: str, step: int) -> str:
    """TransferQueue partition holding one step's hidden states of one teacher."""
    return f"full_vocab_hidden_{prefix}_{teacher_name}_step_{step}"


def full_vocab_sample_key(teacher_name: str, step: int, uid: str) -> str:
    """TransferQueue key of one sample's hidden-state tensor."""
    return f"{teacher_name}/step={step}/sample={uid}"


def _ensure_tq_initialized() -> None:
    global _TQ_INITIALIZED
    if _TQ_INITIALIZED:
        return
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise RuntimeError(
            "full-vocab distillation requires the transfer_queue package "
            "(pip install TransferQueue) on the teacher server."
        ) from exc
    tq.init()
    _TQ_INITIALIZED = True


class FullVocabHiddenWorkerExtension(vLLMColocateWorkerExtension):
    """vLLM worker extension adding hidden-state capture RPCs for teacher servers.

    ``start_hidden_capture`` wraps the model's ``compute_logits`` so the next
    forward's pre-lm_head hidden states are stashed on the worker;
    ``fetch_captured_hidden`` returns the stashed tensor (on CPU) and restores
    the original ``compute_logits``. The teacher server serializes the whole
    capture -> forward -> fetch sequence with a per-server lock, so a single
    shared buffer per worker is sufficient.

    Hidden states are replicated across TP ranks (only the vocab dimension is
    sharded in ``compute_logits``), and only pipeline stages owning the lm_head
    run ``compute_logits`` at all — hence :func:`unwrap_captured_hidden` simply
    takes the first non-None worker result.
    """

    def start_hidden_capture(self):
        """Arm the capture: the next ``compute_logits`` call stashes its input hidden states."""
        model = self.model_runner.model
        if not hasattr(model, "compute_logits"):
            raise RuntimeError(
                f"full-vocab export: model of type {type(model).__name__} has no 'compute_logits' "
                "method; hidden-state capture is only implemented for standard causal-LM vLLM models."
            )
        # Restore a leftover wrapper first in case a previous capture cycle was
        # interrupted between start and fetch.
        self._restore_compute_logits()

        original_compute_logits = model.compute_logits
        extension = self

        def compute_logits(model_, hidden_states, *args, **kwargs):
            if extension._fv_capture_armed and extension._fv_captured_hidden is None:
                # Detach from the inference graph; the prefill call comes first,
                # the one-token decode call afterwards finds the buffer filled.
                extension._fv_captured_hidden = hidden_states.detach()
            return original_compute_logits(hidden_states, *args, **kwargs)

        model.compute_logits = MethodType(compute_logits, model)
        self._fv_original_compute_logits = original_compute_logits
        self._fv_captured_hidden = None
        self._fv_capture_armed = True

    def fetch_captured_hidden(self) -> Optional[torch.Tensor]:
        """Return the captured ``[seq_len, hidden_size]`` hidden states (CPU) and disarm."""
        hidden = getattr(self, "_fv_captured_hidden", None)
        self._fv_capture_armed = False
        self._fv_captured_hidden = None
        self._restore_compute_logits()
        if hidden is None:
            return None
        return hidden.to("cpu", copy=True)

    def _restore_compute_logits(self):
        original = getattr(self, "_fv_original_compute_logits", None)
        if original is not None:
            self.model_runner.model.compute_logits = original
            self._fv_original_compute_logits = None


def unwrap_captured_hidden(rpc_result: Any) -> Optional[torch.Tensor]:
    """Extract the captured hidden-state tensor from a ``collective_rpc`` result.

    ``collective_rpc`` returns one result per worker; TP workers hold identical
    replicas of the hidden states and workers without an lm_head (non-final
    pipeline stages) return None, so the first tensor result is the capture.
    Returns None when no worker captured anything.
    """
    if isinstance(rpc_result, torch.Tensor):
        return rpc_result
    if isinstance(rpc_result, (list, tuple)):
        for item in rpc_result:
            if isinstance(item, torch.Tensor):
                return item
    return None


def export_hidden_to_tq(
    *,
    hidden: torch.Tensor,
    seq_len: int,
    teacher_name: str,
    step: int,
    uid: str,
    prefix: str,
) -> dict[str, Any]:
    """Write one sample's teacher hidden states to TransferQueue and return the artifact.

    Args:
        hidden: ``[L, hidden_size]`` pre-lm_head hidden states, row ``t`` predicts
            token ``t + 1``. ``L`` must be at least ``seq_len``; extra trailing rows
            (e.g. the one-token decode step) are trimmed.
        seq_len: number of input tokens the teacher prefilled.
        teacher_name: teacher key, selects the partition and the student's lm_head shard.
        step: trainer global step, selects the partition.
        uid: per-sample unique id, selects the key inside the partition.
        prefix: run-unique partition prefix (isolates runs sharing a TQ cluster).

    Returns:
        Artifact metadata dict; the student fetches the tensor via
        ``partition_id``/``key`` and validates ``seq_len``/``hidden_size``/``dtype``.

    Fail-loud: any shape or TransferQueue problem raises — a silent skip would
    silently disable distillation for the sample.
    """
    from tensordict import TensorDict

    if hidden.dim() != 2:
        raise ValueError(f"full-vocab export: expected a 2D hidden-state tensor, got shape {tuple(hidden.shape)}.")
    if hidden.shape[0] < seq_len:
        raise ValueError(
            f"full-vocab export: captured hidden has {hidden.shape[0]} rows but the request prefilled "
            f"{seq_len} tokens; the capture is truncated or belongs to a different request."
        )
    if hidden.shape[0] > seq_len:
        hidden = hidden[:seq_len]

    hidden = hidden.contiguous().cpu()
    partition_id = full_vocab_partition_name(prefix, teacher_name, step)
    key = full_vocab_sample_key(teacher_name, step, uid)

    _ensure_tq_initialized()
    import transfer_queue as tq

    fields = TensorDict({FULL_VOCAB_TQ_FIELD: hidden.unsqueeze(0)}, batch_size=(1,))
    tq.kv_put(
        key=key,
        partition_id=partition_id,
        fields=fields,
        tag={"global_steps": int(step), "teacher": teacher_name},
    )
    return {
        "teacher_name": teacher_name,
        "step": int(step),
        "uid": uid,
        "partition_id": partition_id,
        "key": key,
        "seq_len": int(seq_len),
        "hidden_size": int(hidden.shape[1]),
        "dtype": str(hidden.dtype).replace("torch.", ""),
    }
