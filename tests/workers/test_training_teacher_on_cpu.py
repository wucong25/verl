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

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.experimental.teacher_loop.training_teacher import TeacherTrainingWorker
from verl.utils import tensordict_utils as tu


@pytest.mark.parametrize("rope_axes", [None, 4])
def test_teacher_request_preserves_positions_and_returns_single_payload(rope_axes):
    worker = object.__new__(TeacherTrainingWorker)
    worker.engine = SimpleNamespace(is_mp_src_rank_with_outputs=lambda: True, get_data_parallel_rank=lambda: 0)
    positions = torch.arange(5) if rope_axes is None else torch.arange(5).repeat(rope_axes, 1)
    captured = {}

    def infer(data):
        captured.update(data=data)
        logits = torch.randn(5, 11)
        return TensorDict(
            {"full_logits": torch.nested.nested_tensor_from_jagged(logits, torch.tensor([0, 5]))}, batch_size=[1]
        )

    worker.infer_batch = infer
    result = worker.infer_teacher_sequence([1, 2, 3, 4, 5], positions, {"pixel_values": torch.zeros(1, 3, 2, 2)})
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    assert result.shape == (5, 11)
    data = captured["data"]
    torch.testing.assert_close(data["position_ids"].values(), positions)
    assert tu.get_non_tensor_data(data, "return_full_logits", False)
    assert not tu.get_non_tensor_data(data, "compute_loss", True)
    assert data["position_ids"].offsets().tolist() == [0, 5]
    assert len(data["multi_modal_inputs"]) == 1


@pytest.mark.parametrize("precomputed", [False, True])
def test_training_teacher_multimodal_request_in_both_agent_paths(precomputed):
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput, AgentLoopWorker

    captured = {}
    positions = torch.arange(5).repeat(4, 1)
    mm_inputs = {"pixel_values": torch.randn(1, 3, 2, 2)}

    async def compute(**kwargs):
        captured.update(kwargs)
        return torch.zeros(5, 1, dtype=torch.int32), torch.randn(5, 11)

    def multimodal(output, ids):
        assert ids.tolist() == [[1, 2, 3, 4, 5]]
        return mm_inputs

    def position_ids(ids, mask, mm, kwargs):
        assert mask.tolist() == [[1, 1, 1, 1, 1]]
        assert mm is mm_inputs
        return positions.unsqueeze(0)

    worker = SimpleNamespace(
        distillation_enabled=True,
        teacher_key="data_source",
        teacher_server_manager=SimpleNamespace(
            _resolve_teacher_key=lambda key: "teacher",
            teacher_model_configs={"teacher": SimpleNamespace(inference=SimpleNamespace(name="training_engine"))},
            compute_teacher_logprobs_single=compute,
        ),
        _compute_multi_modal_inputs=multimodal,
        _compute_position_ids=position_ids,
    )
    output = AgentLoopOutput(
        prompt_ids=[1, 2, 3],
        response_ids=[4, 5],
        response_mask=[1, 1],
        metrics=AgentLoopMetrics(),
    )
    asyncio.run(
        AgentLoopWorker._compute_teacher_logprobs(
            worker,
            output,
            [1, 2, 3],
            [4, 5],
            False,
            teacher_position_ids=positions if precomputed else None,
            teacher_multi_modal_inputs=mm_inputs if precomputed else None,
        )
    )
    torch.testing.assert_close(captured["position_ids"], positions)
    assert captured["multi_modal_inputs"] is mm_inputs
    assert captured["sequence_ids"] == [1, 2, 3, 4, 5]
    assert output.extra_fields["teacher_logprobs"].shape == (5, 11)


def test_teacher_client_transports_arrays_in_both_directions():
    from verl.experimental.teacher_loop.training_teacher import TrainingTeacherClient

    async def remote(tokens, full_vocab, topk, positions, multimodal):
        assert isinstance(positions, np.ndarray)
        assert isinstance(multimodal["pixel_values"], np.ndarray)
        assert isinstance(multimodal["nested"][0], np.ndarray)
        ids = np.zeros((len(tokens), 1), dtype=np.int32)
        probabilities = np.full((len(tokens), 11), -np.log(11), dtype=np.float32)
        ids.setflags(write=False)
        probabilities.setflags(write=False)
        return ids, probabilities

    replica = SimpleNamespace(compute=SimpleNamespace(remote=remote))
    client = TrainingTeacherClient([replica], full_vocab=True, topk=3)
    ids, logprobs = asyncio.run(
        client.compute(
            [1, 2],
            torch.arange(2),
            {"pixel_values": torch.ones(1, 3, 2, 2), "nested": [torch.ones(1, dtype=torch.bfloat16)]},
        )
    )
    assert isinstance(ids, torch.Tensor)
    assert isinstance(logprobs, torch.Tensor)
    torch.testing.assert_close(logprobs.exp().sum(-1), torch.ones(2))
    logprobs.add_(1)  # The returned tensor must own writable storage.
