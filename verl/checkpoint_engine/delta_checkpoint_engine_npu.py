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
"""Ascend (NPU) variant of the sharded-delta checkpoint engine.

A transport-only subclass of :class:`DeltaShardedCheckpointEngine`: the delta
protocol -- shard byte-diff, batched sparse gather, flush bucketing, wire encode,
and the receive side -- is inherited unchanged. What changes is how a flush
crosses from trainer rank 0 to the rollout workers: plain torch tensors
broadcast over a vLLM ``StatelessProcessGroup`` (HCCL), the same scheme
:class:`~verl.checkpoint_engine.hccl_checkpoint_engine.HCCLCheckpointEngine`
uses, instead of cupy staging + ``ray.util.collective`` NCCL.

Mirroring how ``HCCLCheckpointEngine`` takes over the ``"nccl"`` backend name on
Ascend, this class takes over the ``"delta_sharded"`` backend name on Ascend;
the CUDA engine skips its own registration there, so configs keep working
unchanged on both platforms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from verl.utils.device import get_device_name, get_torch_device, is_torch_npu_available
from verl.utils.net_utils import get_free_port

from .base import CheckpointEngineRegistry
from .delta_checkpoint_engine import DeltaShardedCheckpointEngine
from .nccl_checkpoint_engine import MasterMetadata, WorkerMetadata

if not is_torch_npu_available(check_device=False):
    raise ImportError(
        "HCCLDeltaShardedCheckpointEngine is unavailable because the torch.npu module is not available."
    )

logger = logging.getLogger(__name__)


@dataclass
class HCCLDeltaMasterMetadata(MasterMetadata):
    """The master additionally advertises the HCCL rendezvous endpoint
    (``dist_ip`` / ``dist_port``), exactly like HCCLCheckpointEngine's metadata."""

    dist_ip: str = ""
    dist_port: int = 0


@CheckpointEngineRegistry.register("delta_sharded")
class HCCLDeltaShardedCheckpointEngine(DeltaShardedCheckpointEngine):
    """``delta_sharded`` backend on Ascend: HCCL transport, inherited delta protocol."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # HCCL transport handle, created lazily in init_process_group.
        self.pyhccl = None
        if self.is_master:
            # HCCL rendezvous endpoint, advertised through HCCLDeltaMasterMetadata.
            self.dist_port, _ = get_free_port(self.ip)

    def prepare(self) -> WorkerMetadata:
        # Same single-sender topology as the base engine; the master metadata
        # additionally carries the HCCL rendezvous endpoint.
        master = None
        if self.is_master:
            master = HCCLDeltaMasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.listen_port,
                multi_sender=False,
                dist_ip=self.ip,
                dist_port=self.dist_port,
            )
        return WorkerMetadata(node_id=self.get_node_id(), master=master)

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata, num_senders: int):
        """Stand up the HCCL group (a vLLM StatelessProcessGroup, the same scheme
        HCCLCheckpointEngine uses) instead of the parent's ray NCCL group."""
        from verl.utils.distributed import stateless_init_process_group

        self.num_senders = num_senders
        # Actor workers left out of the group are given rank -1.
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            return
        if self.rebuild_group or self.pyhccl is None:
            self.pyhccl = stateless_init_process_group(
                master_metadata.dist_ip, master_metadata.dist_port, rank, world_size, torch.npu.current_device()
            )
        self.rank = rank
        self.world_size = world_size
        # Only consumers read the flush metadata stream; the sender never does.
        if self.rank >= num_senders:
            self._connect_zmq_client(master_metadata)
        # barrier (the same all_reduce handshake HCCLCheckpointEngine uses)
        signal = torch.tensor([1], dtype=torch.int8, device=get_device_name())
        self.pyhccl.all_reduce(signal)
        logger.info(f"init_process_group (hccl) rank: {self.rank}, world_size: {self.world_size}")

    def finalize(self):
        if self.rebuild_group and getattr(self, "rank", None) is not None and self.rank >= 0:
            if self.pyhccl is not None:
                self.pyhccl.close()
                self.pyhccl = None
            self.rank = None
            self.world_size = None
        get_torch_device().empty_cache()

    # ---- transport overrides (HCCL) ----

    def _bcast_send(self, t: torch.Tensor) -> None:
        # c10d tracks the stream for torch tensors, so no cupy-style staging is
        # needed; synchronize so the buffer freed on return is not reused under
        # an in-flight broadcast (mirrors HCCLCheckpointEngine).
        self.pyhccl.broadcast(t, src=0)
        torch.npu.synchronize()

    def _bcast_recv(self, t: torch.Tensor) -> None:
        self.pyhccl.broadcast(t, src=0)
