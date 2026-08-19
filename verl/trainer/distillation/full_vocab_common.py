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

"""Engine-agnostic helpers for full-vocabulary KL distillation.

Shared by the Megatron (vocab-parallel) and FSDP/VeOmni (full-vocab) student
loss implementations: TransferQueue artifact reads, teacher lm_head loading,
and micro-batch artifact extraction. Keep this module free of any
megatron/FSDP imports so both backends can use it.
"""

import json
import logging
import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.distillation.full_vocab_export import _ensure_tq_initialized
from verl.utils.fs import copy_to_local

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def load_teacher_lm_head_shard(
    *,
    checkpoint_path: str,
    layer: str,
    vocab_size: Optional[int],
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
            ``None`` loads the full lm_head (FSDP/VeOmni path).
        tp_rank / tp_size: this rank's vocab shard is rows
            ``[tp_rank * ceil(vocab_size / tp_size), ...)``. Ignored when
            ``vocab_size`` is None.
        device: device to place the shard on.

    Returns:
        ``[ceil(vocab_size / tp_size), hidden_size]`` tensor (zero-padded at the
        vocab tail when ``vocab_size`` is not divisible by ``tp_size``), or the
        full ``[V, hidden_size]`` lm_head when ``vocab_size`` is None.
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

    if vocab_size is None:
        with safe_open(weight_map[layer], framework="pt") as f:
            return f.get_tensor(layer).to(device=device)

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


_FULL_LM_HEAD_CACHE: dict[tuple, torch.Tensor] = {}


def get_full_teacher_lm_head(*, checkpoint_path: str, layer: str, device: torch.device) -> torch.Tensor:
    """The full ``[V, H]`` frozen teacher lm_head, cached per worker process.

    Used by the FSDP/VeOmni full-vocab path, where student logits are not
    vocab-sharded (or already gathered to full vocab).
    """
    cache_key = (checkpoint_path, layer, str(device))
    if cache_key not in _FULL_LM_HEAD_CACHE:
        _FULL_LM_HEAD_CACHE[cache_key] = load_teacher_lm_head_shard(
            checkpoint_path=checkpoint_path,
            layer=layer,
            vocab_size=None,
            tp_rank=0,
            tp_size=1,
            device=device,
        )
    return _FULL_LM_HEAD_CACHE[cache_key]


def fetch_teacher_hidden(artifact: dict[str, Any]) -> torch.Tensor:
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


def artifacts_of(data: TensorDict) -> list[dict[str, Any]]:
    """Per-sample teacher hidden-state artifact dicts of the micro batch."""
    artifacts = data.get("teacher_full_vocab_artifact", None)
    if artifacts is None:
        raise KeyError(
            "full-vocab distillation: the micro batch has no 'teacher_full_vocab_artifact' field; "
            "the trainer must run the teacher engine forward "
            "(TeacherEngineManager.compute_teacher_hidden, v1 trainer only) before update_actor."
        )
    # The column is a NonTensorStack of per-sample dicts (items may be NonTensorData).
    result = []
    for i in range(len(artifacts)):
        item = artifacts[i]
        if hasattr(item, "data") and not isinstance(item, dict):  # NonTensorData
            item = item.data
        result.append(item)
    if any(a is None for a in result):
        raise ValueError("full-vocab distillation: some samples carry no teacher hidden-state artifact.")
    return result
