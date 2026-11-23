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

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.workers.engine.fsdp.fsdp_turbo_impl import FSDPTurboEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine.full_vocab import validate_full_vocab_inference


@pytest.mark.parametrize("engine_cls", [FSDPEngineWithLMHead, FSDPTurboEngineWithLMHead])
@pytest.mark.parametrize("packed", [False, True])
def test_full_logits_preserve_token_order(engine_cls, packed):
    engine = object.__new__(engine_cls)
    engine.use_ulysses_sp = False
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=7)))
    ids = torch.nested.nested_tensor_from_jagged(torch.arange(5), torch.tensor([0, 3, 5]))
    data = TensorDict({"input_ids": ids}, batch_size=[2])
    tu.assign_non_tensor(data, return_full_logits=True, use_remove_padding=packed)
    expected = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    if packed:
        logits = torch.cat([expected, torch.full((1, 8), -999.0)]).unsqueeze(0)
    else:
        logits = torch.stack([expected[:3], torch.cat([expected[3:], torch.full((1, 8), -999.0)])])
    output = engine.prepare_model_outputs(SimpleNamespace(logits=logits), {"pad_size": 1}, data, None)
    actual = output["full_logits"]
    torch.testing.assert_close(actual.values(), expected[:, :7])
    assert actual.offsets().tolist() == [0, 3, 5]
    assert not actual.requires_grad


@pytest.mark.parametrize("forward_only,fused", [(False, False), (True, True)])
def test_full_logits_reject_unsupported_modes(forward_only, fused):
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, return_full_logits=True)
    with pytest.raises(ValueError, match="forward-only"):
        validate_full_vocab_inference(data, forward_only=forward_only, use_fused_kernels=fused)


def test_megatron_export_keeps_temperature_one_and_removes_vocab_padding(monkeypatch):
    from megatron.core.tensor_parallel import mappings

    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    engine = object.__new__(MegatronEngineWithLMHead)
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(vocab_size=7))
    logits = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    original = logits.clone()
    other_shard = logits + 100
    monkeypatch.setattr(
        mappings,
        "gather_from_tensor_model_parallel_region",
        lambda values: torch.cat([values, other_shard], dim=-1),
    )
    data = TensorDict({}, batch_size=[])
    tu.assign_non_tensor(data, return_full_logits=True)
    output = engine._lm_head_logits_processor(
        logits,
        torch.zeros(1, 3, dtype=torch.long),
        torch.full((1, 3), 0.25),
        calculate_sum_pi_squared=False,
        calculate_entropy=False,
        distillation_use_topk=False,
        distillation_only=False,
        score_centering=False,
        logits_processor_func=None,
        batch=data,
        data_format="thd",
    )
    assert set(output) == {"full_logits"}
    torch.testing.assert_close(output["full_logits"], torch.cat([original, other_shard], -1)[..., :7])
    torch.testing.assert_close(logits, original)
