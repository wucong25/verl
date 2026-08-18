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

"""Megatron teacher engine for full-vocabulary KL distillation.

A forward-only variant of the LM-head engine that returns the pre-lm_head
hidden states instead of logits: ``post_process`` is temporarily disabled
around the model call, so ``GPTModel._postprocess`` returns the (post
final-norm) hidden states and the lm_head matmul is skipped entirely. The
packed hidden states are unpacked to nested ``[bsz, j1, hidden]`` with the
same CP gather/restore path as per-token outputs, so downstream collection
works exactly like ``log_probs``.
"""

import logging
import os
from functools import partial
from typing import Iterator

from tensordict import TensorDict

from verl.models.mcore.util import postprocess_thd_engine, preprocess_thd_engine
from verl.utils.device import get_device_id
from verl.utils.megatron_utils import unwrap_model
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@EngineRegistry.register(model_type="distill_teacher", backend="megatron")
class MegatronDistillTeacherEngine(MegatronEngineWithLMHead):
    """Forward-only Megatron engine exporting pre-lm_head hidden states."""

    def forward_step(
        self, batch_iter: Iterator[TensorDict], model, logits_processor_func, postprocess_micro_batch_func
    ):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())

        if not self.engine_config.use_remove_padding:
            raise NotImplementedError(
                "MegatronDistillTeacherEngine requires use_remove_padding=True (thd format); "
                "the bshd format is not supported for full-vocab teacher forwards."
            )
        if self.engine_config.dynamic_context_parallel:
            raise NotImplementedError("MegatronDistillTeacherEngine does not support dynamic_context_parallel.")

        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]

        unwrapped_model = unwrap_model(model)
        cp_layout = self._get_context_parallel_layout(unwrapped_model)
        fp8 = unwrapped_model.config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]

        input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
            input_ids,
            pre_process=unwrapped_model.pre_process,
            use_fp8_padding=use_fp8_padding,
            cp_layout=cp_layout,
        )

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        # Temporarily disable post_process so GPTModel._postprocess returns the
        # pre-lm_head hidden states (the final norm lives in the decoder and is
        # still applied). The flag is restored right after the forward.
        is_last_stage = unwrapped_model.post_process
        unwrapped_model.post_process = False
        try:
            hidden = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=None,
                packed_seq_params=packed_seq_params,
                **model_kwargs,
            )
        finally:
            unwrapped_model.post_process = is_last_stage

        if not is_last_stage:
            # Non-last pipeline stage: pass the decoder output along (p2p).
            return hidden, partial(postprocess_micro_batch_func, data=batch)

        # Last pipeline stage: `hidden` is [s_local, 1, H]. With sequence
        # parallelism it is sharded along s; gather the full local sequence first.
        if self.tf_config.sequence_parallel:
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

            hidden = gather_from_sequence_parallel_region(hidden)
        hidden = hidden.transpose(0, 1)  # [1, s_local, H]
        hidden_nested = postprocess_thd_engine(
            hidden,
            packed_seq_params,
            input_ids,
            batch_size=input_ids.shape[0],
            post_process=True,
            cp_layout=cp_layout,
        )
        output = {"hidden_states": hidden_nested}
        return output, partial(postprocess_micro_batch_func, data=batch)
