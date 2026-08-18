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

"""TransferQueue export helpers for full-vocabulary KL distillation.

In full-vocab OPD the teacher's pre-lm_head hidden states are computed by a
forward-only training engine (Megatron/FSDP, see
``verl.experimental.teacher_loop.teacher_engine``) and written to
TransferQueue; only a small artifact metadata dict rides with the training
batch. The student (Megatron) rebuilds full-vocab teacher logits on the fly
from the exported hidden states and the frozen teacher lm_head
(see ``verl.trainer.distillation.megatron.full_vocab_kl``).

TransferQueue layout:

- partition: ``full_vocab_hidden_{prefix}_{teacher_name}_step_{step}``
- key: ``{teacher_name}/step={step}/sample={uid}``
- field: ``hidden`` — a ``[seq_len, hidden_size]`` tensor, row ``t`` is the
  hidden state after input token ``t`` (i.e. it predicts token ``t + 1``).
"""

import logging
import os
from typing import Any, Optional

import torch

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


def resolve_partition_prefix(experiment_name: Optional[str]) -> str:
    """Sanitize the experiment name for use in full-vocab TQ partition names.

    None/empty falls back to the ``VERL_FULL_VOCAB_EXPERIMENT_NAME`` env var,
    then ``"default_exp"``.
    """
    import re

    if not experiment_name:
        experiment_name = os.getenv("VERL_FULL_VOCAB_EXPERIMENT_NAME", "default_exp")
    return re.sub(r"[^0-9A-Za-z_-]+", "_", str(experiment_name)).strip("_")


def _ensure_tq_initialized() -> None:
    global _TQ_INITIALIZED
    if _TQ_INITIALIZED:
        return
    try:
        import transfer_queue as tq
    except ImportError as exc:
        raise RuntimeError(
            "full-vocab distillation requires the transfer_queue package "
            "(pip install TransferQueue) on the teacher/actor workers."
        ) from exc
    tq.init()
    _TQ_INITIALIZED = True


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
            are trimmed.
        seq_len: number of input tokens of the sample.
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
            f"full-vocab export: hidden has {hidden.shape[0]} rows but the sample has "
            f"{seq_len} tokens; the teacher forward output is truncated or misaligned."
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
