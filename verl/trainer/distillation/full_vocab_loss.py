# Copyright 2026 verl contributors
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

"""Align teacher distributions with the student engine's token/vocabulary shards."""

import torch
import torch.nn.functional as F

from verl.trainer.distillation.full_vocab import full_vocab_forward_kl


def compute_full_vocab_loss(
    config,
    distillation_config,
    data,
    student_logits,
    data_format,
    *,
    quant_align_size=None,
    min_local_rows=None,
    pad_to_length_bucket=None,
    cp_layout="zigzag",
    local_cp_size=None,
):
    teacher = data["teacher_logprobs"]
    if not teacher.is_nested:
        raise ValueError("Full-vocabulary teacher probabilities must be jagged.")
    vocabulary_size = teacher.shape[-1]
    group = None
    if config.strategy in {"fsdp", "fsdp2", "fsdp_turbo", "veomni"}:
        from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, slice_input_tensor

        teacher = teacher.values().unsqueeze(0)
        if get_ulysses_sequence_parallel_world_size() > 1:
            teacher = slice_input_tensor(teacher, dim=1)
        if student_logits.shape[-1] < vocabulary_size:
            raise ValueError("Student vocabulary is smaller than the aligned teacher vocabulary.")
        student_logits = student_logits[..., :vocabulary_size]
    elif config.strategy == "megatron":
        from megatron.core import parallel_state as mpu

        from verl.models.mcore.util import preprocess_bshd_engine, preprocess_thd_engine

        if data_format == "thd":
            teacher, *_ = preprocess_thd_engine(
                teacher,
                pre_process=True,
                quant_align_size=quant_align_size,
                min_local_rows=min_local_rows,
                pad_to_length_bucket=pad_to_length_bucket,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
            )
        else:
            teacher, *_ = preprocess_bshd_engine(
                teacher,
                pre_process=True,
                use_fp8_padding=quant_align_size is not None,
            )
        group = mpu.get_tensor_model_parallel_group()
        width = student_logits.shape[-1]
        world_size = mpu.get_tensor_model_parallel_world_size()
        rank = mpu.get_tensor_model_parallel_rank()
        if width * world_size < vocabulary_size:
            raise ValueError("Student vocabulary is smaller than the aligned teacher vocabulary.")
        teacher = F.pad(teacher, (0, width * world_size - vocabulary_size), value=-torch.inf)
        teacher = teacher[..., rank * width : (rank + 1) * width]
        global_ids = torch.arange(width, device=student_logits.device) + rank * width
        student_logits = student_logits.masked_fill(global_ids >= vocabulary_size, -torch.inf)
    else:
        raise ValueError(f"Unsupported full-vocabulary student backend: {config.strategy}")
    teacher = teacher.to(device=student_logits.device)
    losses = full_vocab_forward_kl(
        student_logits,
        teacher,
        vocab_parallel_group=group,
        chunk_size=distillation_config.distillation_loss.full_vocab_chunk_size,
    )
    return {"distillation_losses": losses}
