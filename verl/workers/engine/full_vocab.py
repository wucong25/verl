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

"""Full-vocabulary teacher outputs from the training engines.

Each row predicts the next token after its corresponding input token. Consumers
must align response targets with that causal offset, without crossing samples.
"""

import torch
from torch.distributed.tensor import DTensor

from verl.utils import tensordict_utils as tu


def validate_full_vocab_inference(data, *, forward_only, use_fused_kernels):
    requested = tu.get_non_tensor_data(data, key="return_full_logits", default=False)
    if requested and (not forward_only or use_fused_kernels):
        raise ValueError("return_full_logits requires forward-only inference with fused kernels disabled.")
    return requested


def fsdp_full_vocab_output(engine, raw_logits, output_args, data):
    """Restore packed token order for FSDP and its FSDP Turbo subclass."""
    if isinstance(raw_logits, DTensor):
        raw_logits = raw_logits.full_tensor()
    input_ids = data["input_ids"]
    if not input_ids.is_nested:
        raise ValueError("Full-vocabulary inference requires jagged input_ids.")
    if tu.get_non_tensor_data(data, key="use_remove_padding", default=True):
        values = engine._gather_and_unpad_packed(raw_logits.squeeze(0), output_args["pad_size"])
    else:
        lengths = input_ids.offsets().diff().tolist()
        values = torch.cat([row[:length] for row, length in zip(raw_logits, lengths, strict=True)])
    config = engine.model_config.hf_config
    config = getattr(config, "text_config", config)
    values = values[..., : config.vocab_size]
    if values.shape[0] != input_ids.values().shape[0]:
        raise ValueError("Teacher logits do not match the unpadded input token count.")
    return {"full_logits": torch.nested.nested_tensor_from_jagged(values.detach(), input_ids.offsets())}


def megatron_full_vocab_output(logits, hf_config):
    """Gather TP vocabulary shards; the engine restores CP token order next."""
    from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

    config = getattr(hf_config, "text_config", hf_config)
    logits = gather_from_tensor_model_parallel_region(logits)
    return {"full_logits": logits[..., : config.vocab_size].detach()}
