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

"""Engine-based teacher forward for full-vocabulary KL distillation.

Replaces the inference-server teacher path for full-vocab OPD: instead of
capturing hidden states from a vLLM prefill, a forward-only training engine
(Megatron or FSDP, see ``model_type="distill_teacher"`` engine
implementations) runs the teacher forward over the training batch inside the
trainer loop. Each worker writes its samples' hidden states directly to
TransferQueue and only the small per-sample artifact metadata dicts
(``teacher_full_vocab_artifact``) are appended back as a TQ column, so the
student-side consumption (``verl.trainer.distillation.megatron.full_vocab_kl``)
is unchanged.
"""

import logging
import os
from typing import Optional
from uuid import uuid4

import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorStack
from transfer_queue import KVBatchMeta

from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import split_resource_pool
from verl.trainer.distillation.full_vocab_export import export_hidden_to_tq, resolve_partition_prefix
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import CheckpointConfig, DistillationConfig, HFModelConfig, OptimizerConfig
from verl.workers.config.engine import TrainingWorkerConfig
from verl.workers.engine_workers import TrainingWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DistillTeacherWorker(TrainingWorker):
    """TrainingWorker running a frozen teacher model, exporting hidden states to TQ."""

    def __init__(self, config: TrainingWorkerConfig):
        # Keep the teacher model path unresolved until the worker is placed, so
        # HDFS models are copied into that node's local cache instead of the
        # controller's local /tmp (same trick as the rollout server path).
        if not isinstance(config.model_config, HFModelConfig):
            config.model_config = omega_conf_to_dataclass(config.model_config)
        # Import the distill-teacher engine implementation for the chosen backend
        # so it registers itself with EngineRegistry before TrainingWorker builds it.
        strategy = config.engine_config.strategy
        if strategy == "megatron":
            import verl.workers.engine.megatron.teacher_engine_impl  # noqa: F401
        elif strategy in ("fsdp", "fsdp2"):
            import verl.workers.engine.fsdp.teacher_engine_impl  # noqa: F401
        else:
            raise ValueError(f"Unsupported distill teacher engine strategy: {strategy!r}")
        super().__init__(config)
        self.teacher_key = self.config.extra_context["teacher_key"]
        self.tq_prefix = self.config.extra_context["tq_prefix"]

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="train"), blocking=False)
    def compute_teacher_hidden(self, data: TensorDict) -> Optional[TensorDict]:
        """Run the teacher forward and export per-sample hidden states to TransferQueue.

        Returns a one-column TensorDict with the per-sample artifact metadata; the
        tqbridge wrapper appends it to the samples' TQ records. Only the mp-src rank
        of each DP shard produces output (and writes TQ entries), so every sample is
        exported exactly once.
        """
        global_steps = tu.get_non_tensor_data(data, key="global_steps", default=None)
        if global_steps is None:
            raise ValueError(
                "compute_teacher_hidden requires 'global_steps' in the batch non-tensor fields; "
                "the trainer sets it via batch.extra_info before the call."
            )
        global_steps = int(global_steps.item() if hasattr(global_steps, "item") else global_steps)

        default_keys = dict(
            use_remove_padding=self.model_config.get("use_remove_padding", False),
            use_dynamic_bsz=self.engine_config.use_dynamic_bsz,
            max_token_len_per_gpu=self.engine_config.infer_max_token_len_per_gpu,
            micro_batch_size_per_gpu=self.engine_config.infer_micro_batch_size_per_gpu,
            use_fused_kernels=False,
        )
        for key, val in default_keys.items():
            if key not in data.keys():
                tu.assign_non_tensor(data, **{key: val})

        with self.engine.eval_mode():
            output = self.engine.infer_batch(data, loss_function=None)

        if not self.engine.is_mp_src_rank_with_outputs():
            return None

        hidden_states = output["model_output"]["hidden_states"]  # nested [local_bsz, j1, H]
        seq_lens = data["input_ids"].offsets().diff().tolist()
        assert hidden_states.shape[0] == len(data), (
            f"teacher hidden batch ({hidden_states.shape[0]}) != micro batch size ({len(data)})"
        )
        artifacts = []
        for i in range(len(data)):
            artifact = export_hidden_to_tq(
                hidden=hidden_states[i],
                seq_len=seq_lens[i],
                teacher_name=self.teacher_key,
                step=global_steps,
                # A fresh uuid per export guarantees key uniqueness inside the
                # per-step partition (rollout.n repeats, duplicated dataset uids,
                # retries); the artifact carries the exact key to the student.
                uid=uuid4().hex,
                prefix=self.tq_prefix,
            )
            artifacts.append(artifact)
        return TensorDict({"teacher_full_vocab_artifact": NonTensorStack(*artifacts)}, batch_size=len(data))


class TeacherEngineManager:
    """Manages one ``DistillTeacherWorker`` worker group per teacher model.

    Full-vocab counterpart of ``MultiTeacherModelManager``: splits the teacher
    resource pool per teacher and spins up a forward-only engine worker group
    on each sub-pool.
    """

    def __init__(self, config: DictConfig, resource_pool: RayResourcePool):
        self.config = config
        self.distillation_config: DistillationConfig = omega_conf_to_dataclass(config.distillation)
        distillation_loss = self.distillation_config.distillation_loss
        if not distillation_loss.loss_settings.use_full_vocab:
            raise ValueError("TeacherEngineManager requires a full-vocab distillation loss_mode.")
        # The prefix isolates this run's TQ partitions from other runs sharing
        # the same TQ cluster.
        self.tq_prefix = resolve_partition_prefix(distillation_loss.full_vocab_experiment_name)
        self.resource_pool = resource_pool
        self.teacher_wgs: dict[str, RayWorkerGroup] = {}
        self._initialize_teacher_worker_groups()

    def _initialize_teacher_worker_groups(self):
        teacher_models = self.distillation_config.teacher_models
        split_sizes = [teacher.world_size for teacher in teacher_models.values()]
        split_pools = split_resource_pool(self.resource_pool, split_size=split_sizes)

        for (key, teacher_config), teacher_pool in zip(teacher_models.items(), split_pools, strict=True):
            engine_config = omega_conf_to_dataclass(teacher_config.engine)
            engine_config.forward_only = True
            worker_config = TrainingWorkerConfig(
                model_type="distill_teacher",
                # Deferred instantiation on the worker (see DistillTeacherWorker).
                model_config=OmegaConf.create(
                    {
                        "_target_": "verl.workers.config.HFModelConfig",
                        "path": teacher_config.model_path,
                    }
                ),
                engine_config=engine_config,
                optimizer_config=OptimizerConfig(),
                checkpoint_config=CheckpointConfig(),
                extra_context={"teacher_key": key, "tq_prefix": self.tq_prefix},
            )
            worker_group = RayWorkerGroup(
                resource_pool=teacher_pool,
                ray_cls_with_init=RayClassWithInitArgs(ray.remote(DistillTeacherWorker), config=worker_config),
                device_name=self.config.trainer.device,
                name_prefix=f"distill_teacher_{key}",
            )
            worker_group.reset()  # initialize engines (forward_only: no optimizer, offload after load)
            self.teacher_wgs[key] = worker_group

    def compute_teacher_hidden(self, batch: KVBatchMeta, global_steps: int) -> KVBatchMeta:
        """Compute and export teacher hidden states for the batch; append artifact column.

        The artifact column (``teacher_full_vocab_artifact``) is appended to the
        samples' TQ records by the workers; the returned meta is the input batch
        unchanged (its implicit field set already covers the new column).
        """
        batch.extra_info["global_steps"] = self._normalize_step(global_steps)
        teacher_key_field = self.distillation_config.teacher_key

        if len(self.teacher_wgs) == 1:
            # Single teacher: everything routes to the one worker group.
            wg = next(iter(self.teacher_wgs.values()))
            output = wg.compute_teacher_hidden(batch)
            assert len(output) == len(batch)
            return batch

        # Multi-teacher: read the routing column and split the batch per teacher.
        routing = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=[teacher_key_field])
        routing_values = routing[teacher_key_field]
        if hasattr(routing_values, "tolist"):
            routing_values = routing_values.tolist()
        routing_values = [v.item() if hasattr(v, "item") else v for v in list(routing_values)]

        for teacher_key, wg in self.teacher_wgs.items():
            indices = [i for i, value in enumerate(routing_values) if value == teacher_key]
            if not indices:
                continue
            sub_batch = KVBatchMeta(
                keys=[batch.keys[i] for i in indices],
                tags=[batch.tags[i] for i in indices],
                partition_id=batch.partition_id,
                extra_info=batch.extra_info,
            )
            output = wg.compute_teacher_hidden(sub_batch)
            assert len(output) == len(sub_batch)
        missing = set(routing_values) - set(self.teacher_wgs)
        if missing:
            raise ValueError(
                f"No teacher configured for routing keys {sorted(missing)}. "
                f"Configured teachers: {sorted(self.teacher_wgs)}."
            )
        return batch

    @staticmethod
    def _normalize_step(global_steps) -> int:
        return int(global_steps.item() if hasattr(global_steps, "item") else global_steps)
