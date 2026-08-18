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

"""FSDP teacher engine for full-vocabulary KL distillation.

A forward-only variant of the LM-head engine that returns the pre-lm_head
hidden states instead of logits: the HF backbone is called directly (falling
back to ``output_hidden_states=True``), yielding the post-final-norm hidden
states without materializing logits. Sequence-parallel shards are gathered and
unpadded with the same utilities as per-token outputs, and the result is
packed as nested ``[bsz, j1, hidden]`` exactly like ``log_probs``.
"""

import logging
import os
from contextlib import nullcontext
from typing import ContextManager

import torch
from tensordict import TensorDict
from torch.distributed.tensor import DTensor

from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_id, get_device_name
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="distill_teacher", backend=["fsdp", "fsdp2"], device=["cuda", "npu"])
class FSDPDistillTeacherEngine(FSDPEngineWithLMHead):
    """Forward-only FSDP engine exporting pre-lm_head hidden states."""

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        assert forward_only, "FSDPDistillTeacherEngine only supports forward_only=True"
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        model_inputs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)
        input_ids = micro_batch["input_ids"]

        autocast_dtype = getattr(self, "_autocast_dtype", torch.bfloat16)
        autocast_ctx: ContextManager = (
            nullcontext()
            if autocast_dtype == torch.float32
            else torch.autocast(device_type=device_name, dtype=autocast_dtype)
        )
        with autocast_ctx:
            # Prefer calling the backbone directly: output_hidden_states=True on the
            # full CausalLM materializes every layer's hidden states, which is
            # needlessly expensive. The backbone's last_hidden_state is the
            # post-final-norm, pre-lm_head hidden states.
            backbone = getattr(self.module, "model", None)
            if backbone is not None:
                backbone_output = backbone(**model_inputs, use_cache=False)
                hidden = backbone_output.last_hidden_state
            else:
                raw_output = self.module(
                    **model_inputs,
                    use_cache=False,
                    output_hidden_states=True,
                )
                hidden = raw_output.hidden_states[-1]

        # With TP, hidden states may be DTensors sharded on the hidden dim.
        if isinstance(hidden, DTensor):
            hidden = hidden.full_tensor()

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        if use_remove_padding:
            hidden = hidden.squeeze(0)  # ((total_nnz / sp) + pad, H)
            hidden = self._gather_and_unpad_packed(hidden, output_args["pad_size"])  # (total_nnz, H)
            hidden_nested = torch.nested.nested_tensor_from_jagged(hidden, input_ids.offsets())
        else:
            seq_lens = input_ids.offsets().diff().tolist()
            hidden_nested = torch.nested.nested_tensor(
                [h[:length] for h, length in zip(hidden, seq_lens, strict=True)], layout=torch.jagged
            )

        model_output = {"hidden_states": hidden_nested}
        loss = torch.tensor(1.0, device=device_name)
        output = {
            "model_output": model_output,
            "loss": loss.detach().item(),
            "metrics": {},
        }
        return loss, output
