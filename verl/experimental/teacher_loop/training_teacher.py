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

"""Serialized teacher requests backed by verl training engines.

A replica executes one sequence collectively. FSDP ranks share the sequence,
while Megatron uses its TP/PP partitioning. Only one rank returns the payload.
This favors bounded request memory and correctness over teacher batching.
"""

import numpy as np
import ray
import torch
from tensordict import NonTensorData, NonTensorStack, TensorDict

from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import FSDPEngineConfig, HFModelConfig, McoreEngineConfig, TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker


def _to_numpy(value):
    """Keep engine-patched torch serializers out of CPU-only Ray actors."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_to_numpy(item) for item in value)
    return value


def _to_torch(value):
    if isinstance(value, np.ndarray):
        # Ray's object-store views can be read-only. Give downstream model
        # preprocessing writable storage instead of relying on implicit copies.
        return torch.from_numpy(value.copy() if not value.flags.writeable else value)
    if isinstance(value, dict):
        return {key: _to_torch(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_to_torch(item) for item in value)
    return value


class TeacherTrainingWorker(TrainingWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def infer_teacher_sequence(self, token_ids, position_ids=None, multi_modal_inputs=None):
        position_ids = _to_torch(position_ids)
        multi_modal_inputs = _to_torch(multi_modal_inputs)
        ids = torch.tensor(token_ids, dtype=torch.long)
        offsets = torch.tensor([0, len(token_ids)], dtype=torch.int64)
        positions = torch.arange(len(token_ids)) if position_ids is None else position_ids
        data = TensorDict(
            {
                "input_ids": torch.nested.nested_tensor_from_jagged(ids, offsets),
                "position_ids": torch.nested.nested_tensor_from_jagged(
                    positions,
                    offsets,
                    jagged_dim=positions.ndim,
                ),
                "loss_mask": torch.ones(1, len(token_ids), dtype=torch.bool),
            },
            batch_size=[1],
        )
        if multi_modal_inputs:
            data["multi_modal_inputs"] = NonTensorStack(NonTensorData(multi_modal_inputs))
        tu.assign_non_tensor(
            data,
            global_token_num=None,
            return_full_logits=True,
            compute_loss=False,
            return_model_output=True,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            temperature=1.0,
        )
        result = self.infer_batch(data)
        if self.engine.is_mp_src_rank_with_outputs() and self.engine.get_data_parallel_rank() == 0:
            # A plain array prevents engine-specific tensor metadata and device
            # storage reducers from crossing the Ray process boundary.
            return result["full_logits"].values().detach().float().cpu().numpy()
        return None


@ray.remote(num_cpus=1)
class TrainingTeacherReplica:
    def __init__(self, teacher_config, resource_pool, device_name):
        kwargs = dict(teacher_config.training_engine)
        strategy = kwargs.pop("strategy", "fsdp2")
        engine_cls = McoreEngineConfig if strategy == "megatron" else FSDPEngineConfig
        kwargs.update(
            strategy=strategy,
            forward_only=True,
            use_fused_kernels=False,
            use_dynamic_bsz=False,
            infer_micro_batch_size_per_gpu=1,
        )
        engine = engine_cls(**kwargs)
        model_kwargs = dict(teacher_config.training_model)
        model_kwargs.update(
            path=teacher_config.model_path,
            use_remove_padding=engine.use_remove_padding,
            use_fused_kernels=False,
            enable_gradient_checkpointing=False,
        )
        model = HFModelConfig(**model_kwargs)
        config = TrainingWorkerConfig(
            model_type="language_model",
            model_config=model,
            engine_config=engine,
            optimizer_config=None,
            checkpoint_config=CheckpointConfig(),
        )
        self.workers = RayWorkerGroup(
            resource_pool=resource_pool,
            ray_cls_with_init=RayClassWithInitArgs(ray.remote(TeacherTrainingWorker), config=config),
            device_name=device_name,
        )
        self.workers.reset()

    def ready(self):
        return True

    def compute(self, token_ids, full_vocab, topk, position_ids=None, multi_modal_inputs=None):
        results = self.workers.infer_teacher_sequence(token_ids, position_ids, multi_modal_inputs)
        tensors = [value for value in results if value is not None]
        if len(tensors) != 1:
            raise RuntimeError(f"Expected one teacher output, received {len(tensors)}")
        logits = torch.from_numpy(tensors[0])
        if full_vocab:
            logprobs = torch.empty_like(logits, dtype=torch.float32)
            for start in range(0, logits.shape[0], 128):
                logprobs[start : start + 128] = logits[start : start + 128].float().log_softmax(-1)
            # Dense mode has implicit vocabulary IDs; avoid allocating [S,V] integer IDs.
            ids = torch.zeros(logits.shape[0], 1, dtype=torch.int32)
        else:
            if topk is None or not 0 < topk <= logits.shape[-1]:
                raise ValueError("Teacher topk must be between one and vocabulary size.")
            logprobs, ids = logits.float().log_softmax(-1).topk(topk, dim=-1)
            ids = ids.to(torch.int32)
        return ids.numpy(), logprobs.numpy()


class TrainingTeacherClient:
    def __init__(self, replicas, full_vocab, topk):
        self.replicas = replicas
        self.full_vocab = full_vocab
        self.topk = topk
        self.next_replica = 0

    async def compute(self, token_ids, position_ids=None, multi_modal_inputs=None):
        replica = self.replicas[self.next_replica % len(self.replicas)]
        self.next_replica += 1
        ids, logprobs = await replica.compute.remote(
            token_ids,
            self.full_vocab,
            self.topk,
            _to_numpy(position_ids),
            _to_numpy(multi_modal_inputs),
        )

        return _to_torch(ids), _to_torch(logprobs)
