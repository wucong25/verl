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
hidden states are unpacked to nested ``[bsz, j1, hidden]`` with the same
CP/SP gather and restore path as per-token outputs, so downstream collection
works exactly like ``log_probs``. Both the thd (remove-padding) and bshd
(padded) data formats are supported.
"""

import logging
import os
from functools import partial
from typing import Iterator

from tensordict import TensorDict

from verl.models.mcore.util import (
    build_vlm_attn_mask_bshd,
    build_vlm_attn_mask_thd,
    postprocess_bshd_engine,
    postprocess_thd_engine,
    preprocess_bshd_engine,
    preprocess_thd_engine,
)
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

        if self.engine_config.dynamic_context_parallel:
            raise NotImplementedError("MegatronDistillTeacherEngine does not support dynamic_context_parallel.")

        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        batch_size = input_ids.shape[0]

        unwrapped_model = unwrap_model(model)
        cp_layout = self._get_context_parallel_layout(unwrapped_model)
        fp8 = unwrapped_model.config.fp8
        use_fp8_padding = fp8 in ["e4m3", "hybrid"]
        data_format = "thd" if self.engine_config.use_remove_padding else "bshd"
        vision_model = hasattr(self.model_config.hf_config, "vision_config")

        model_kwargs = {}
        if "pixel_values" in multi_modal_inputs:
            model_kwargs["pixel_values"] = multi_modal_inputs["pixel_values"].to(input_ids.device)
        if "image_grid_thw" in multi_modal_inputs:
            model_kwargs["image_grid_thw"] = multi_modal_inputs["image_grid_thw"].to(input_ids.device)
        if "pixel_values_videos" in multi_modal_inputs:
            model_kwargs["pixel_values_videos"] = multi_modal_inputs["pixel_values_videos"].to(input_ids.device)
        if "video_grid_thw" in multi_modal_inputs:
            model_kwargs["video_grid_thw"] = multi_modal_inputs["video_grid_thw"].to(input_ids.device)

        if data_format == "thd":
            input_ids_rmpad, packed_seq_params, _ = preprocess_thd_engine(
                input_ids,
                pre_process=unwrapped_model.pre_process,
                use_fp8_padding=use_fp8_padding,
                cp_layout=cp_layout,
            )
            attention_mask = None
            if vision_model and model_kwargs.get("pixel_values") is not None:
                # For VLM models, pass bshd-format `input_ids` and `attention_mask`.
                input_ids_rmpad, attention_mask = build_vlm_attn_mask_thd(
                    input_ids, self.model_config.tokenizer.pad_token_id
                )
            forward_kwargs = {
                "input_ids": input_ids_rmpad,
                "attention_mask": attention_mask,
                "position_ids": None,
                "packed_seq_params": packed_seq_params,
            }
        else:
            input_ids_bshd, attention_mask_bshd, position_ids_bshd = preprocess_bshd_engine(
                input_ids,
                pre_process=unwrapped_model.pre_process,
                use_fp8_padding=use_fp8_padding,
            )
            if vision_model:
                # For VLM models, pass bshd-format `input_ids` and `attention_mask`.
                model_input_ids, model_attention_mask = build_vlm_attn_mask_bshd(
                    input_ids, batch_size, pad_token_id=self.model_config.tokenizer.pad_token_id
                )
                model_position_ids = None
            else:
                model_input_ids, model_attention_mask, model_position_ids = (
                    input_ids_bshd,
                    attention_mask_bshd,
                    position_ids_bshd,
                )
            forward_kwargs = {
                "input_ids": model_input_ids,
                "attention_mask": model_attention_mask,
                "position_ids": model_position_ids,
            }

        # Temporarily disable post_process so GPTModel._postprocess returns the
        # pre-lm_head hidden states (the final norm lives in the decoder and is
        # still applied). The flag is restored right after the forward.
        is_last_stage = unwrapped_model.post_process
        unwrapped_model.post_process = False
        try:
            hidden = model(**forward_kwargs, **model_kwargs)
        finally:
            unwrapped_model.post_process = is_last_stage

        if not is_last_stage:
            # Non-last pipeline stage: pass the decoder output along (p2p).
            return hidden, partial(postprocess_micro_batch_func, data=batch)

        # Last pipeline stage: `hidden` is [s_local, b, H] (b == 1 for thd). With
        # sequence parallelism it is sharded along s; gather the full local
        # sequence first, then go batch-first for the unpack helpers.
        if self.tf_config.sequence_parallel:
            from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

            hidden = gather_from_sequence_parallel_region(hidden)
        hidden = hidden.transpose(0, 1).contiguous()  # [s, b, H] -> [b, s, H]

        if data_format == "thd":
            hidden_nested = postprocess_thd_engine(
                hidden,
                packed_seq_params,
                input_ids,
                batch_size=batch_size,
                post_process=True,
                cp_layout=cp_layout,
            )
        else:
            hidden_nested = postprocess_bshd_engine(hidden, attention_mask_bshd, post_process=True)

        output = {"hidden_states": hidden_nested}
        return output, partial(postprocess_micro_batch_func, data=batch)
