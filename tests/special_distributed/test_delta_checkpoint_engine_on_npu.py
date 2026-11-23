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
"""ST for the sharded-delta checkpoint-engine path on Ascend NPU (HCCL).

The CUDA ``DeltaShardedCheckpointEngine`` (verl/checkpoint_engine/delta_checkpoint_engine.py)
is cupy + ray NCCL bound and cannot be imported on Ascend. What CAN and must be validated
on NPU is the portable core it is built from -- this test drives that core end to end over
HCCL, replicating the engine's exact wire semantics:

    trainer FSDP shard -> shard_delta_indices (byte-diff vs snapshot)
        -> translate_flat_indices -> gather_slot_entries_to_rank0 (batched, HCCL)
        -> indices-encode (same layout as DeltaShardedCheckpointEngine._assemble_flush:
           int32 positions blob + values + per-param manifest + checksum)
        -> HCCL broadcast (stands in for the cupy/NCCL broadcast)
        -> delta_loader.apply_delta (the production SGLang receiver: checksum,
           _decode_one densify, NaN-masked copy_ apply)
        -> bit-exact comparison against the new weights

Run on an Ascend machine (any world size; 8 recommended)::

    torchrun --standalone --nproc-per-node=8 tests/special_distributed/test_delta_checkpoint_engine_on_npu.py

Cases:
    1. on-device byte-diff primitive (shard_delta_indices), bf16 + fp32
    2. batched sparse gather vs full-gather diff baseline: 1D Shard(0) with uneven
       shapes, max_round_bytes sub-round determinism, 2D FSDP x SP(Replicate) contribution
    3. full wire round-trip of a ~1%-changed delta -> apply_delta -> bit-exact equality
    4. empty delta round (zero changed elements end to end) is a no-op
    5. checksum corruption is caught loud (RuntimeError from apply_delta)
    6. dense seed flush (values-only "dense" encoding) rebuilds weights from zeros
    7. verify_every sweep: passes on an in-sync server, raises on a drifted one

The device helpers come from verl.utils.device, so the same file also runs on CUDA
(``get_nccl_backend()`` -> nccl); on Ascend it exercises HCCL + torch.npu.
"""

import json
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, Replicate, distribute_tensor

from verl.checkpoint_engine.delta_sync.encode import checksum as wire_checksum
from verl.checkpoint_engine.delta_sync.sparse_gather import gather_slot_entries_to_rank0, shard_delta_indices
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.workers.engine.spec import ShardSpec, derive_dtensor_placement, translate_flat_indices
from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeRolloutModel:
    """Minimal stand-in for the inference engine's model on the receive side.

    SGLang's ``load_weights`` ultimately lands on ``param.copy_(loaded)``; that is
    the only behavior ``apply_delta`` (incl. its NaN-masked copy patch and the
    verify sweep's copy_ interception) relies on.
    """

    def __init__(self, params: dict[str, torch.Tensor]):
        self.params = params

    def load_weights(self, chunk):
        for name, t in chunk:
            self.params[name].copy_(t)


def _bcast_bytes(payload: bytes | None, dev, src: int = 0) -> bytes:
    """Broadcast a JSON spec blob; stands in for the production zmq side-channel."""
    n = torch.tensor([len(payload) if payload is not None else 0], dtype=torch.int64, device=dev)
    dist.broadcast(n, src=src)
    if dist.get_rank() == src:
        buf = torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(dev)
    else:
        buf = torch.empty(int(n.item()), dtype=torch.uint8, device=dev)
    dist.broadcast(buf, src=src)
    return bytes(buf.cpu())


def _bcast_tensor(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    dist.broadcast(t, src=src)
    return t


def _spec_tensor(spec: dict, dev) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8).to(dev)


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """bf16/fp32 equality by bits, so NaN and rounding differences are caught."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.view(int_dtype), b.view(int_dtype)))


def _assemble_indices_flush(entries: list[tuple[str, str, tuple, torch.Tensor, torch.Tensor]], dev):
    """Replicates DeltaShardedCheckpointEngine._assemble_flush (indices encoding):
    int32 positions concatenated and viewed as a uint8 blob, values concatenated,
    per-param manifest with byte/element offsets, plus the wire checksum."""
    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for name, dtype_str, shape, idx, val in entries:
        nnz = int(idx.numel())
        idx_pieces.append(idx.to(torch.int32))
        val_pieces.append(val)
        params.append(
            {
                "name": name,
                "dtype": dtype_str,
                "shape": list(shape),
                "pos_start": pos_off,
                "pos_end": pos_off + nnz * 4,
                "pos_width": 4,
                "val_start": val_off,
                "val_end": val_off + nnz,
            }
        )
        pos_off += nnz * 4
        val_off += nnz
    positions_u8 = (
        torch.cat(idx_pieces).contiguous().view(torch.uint8)
        if pos_off
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )
    values = (
        torch.cat(val_pieces)
        if val_off
        else torch.empty(0, dtype=getattr(torch, entries[0][1]), device=dev)
    )
    spec = {"encoding": "indices", "params": params, "checksum": int(wire_checksum(positions_u8, values))}
    return spec, positions_u8, values


def _sender_sharded_delta(dt_new, dt_old, dev):
    """Trainer side of one parameter: local byte-diff -> global positions."""
    place, contributes, pg = derive_dtensor_placement(ShardSpec.from_param(dt_new))
    if contributes:
        lidx, lval = shard_delta_indices(dt_new.to_local().reshape(-1), dt_old.to_local().reshape(-1), 0)
        gidx = translate_flat_indices(lidx, place)
    else:
        gidx = torch.empty(0, dtype=torch.int64, device=dev)
        lval = dt_new.to_local().reshape(-1)
        lval = torch.empty(0, dtype=lval.dtype, device=dev)
    return gidx, lval, pg, contributes


def _agree(ok: bool, dev) -> bool:
    """All ranks must report the same verdict for a case (MIN-reduced)."""
    t = torch.tensor([0 if ok else 1], dtype=torch.int64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return bool(t.item() == 0)


def _recv_apply(spec_bytes: bytes, positions: torch.Tensor, values: torch.Tensor, model: _FakeRolloutModel):
    """Rollout side: feed the flush through the production delta_loader."""
    spec_t = _spec_tensor(json.loads(spec_bytes.decode()), values.device)
    named = [("__delta_spec__", spec_t), ("__positions__", positions), ("__values__", values)]
    apply_delta(model, named)


# ---------------------------------------------------------------------------
# cases
# ---------------------------------------------------------------------------


def case_primitive_diff(dev) -> bool:
    """1. On-device byte-diff primitive, bf16 + fp32 (mirrors the CPU unit test)."""
    ok = True
    for dtype in (torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        shard = torch.randn(1000, dtype=dtype, device=dev)
        new = shard.clone()
        changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64, device=dev)
        new[changed] = new[changed] + 0.5
        gidx, gval = shard_delta_indices(new, shard, offset=4096)
        order = torch.argsort(gidx)
        pos_ok = torch.equal(gidx.sort().values, (changed + 4096).sort().values)
        int_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
        val_ok = torch.equal(gval[order].view(int_dtype), new[gidx[order] - 4096].view(int_dtype))
        empty_ok = shard_delta_indices(shard.clone(), shard, 0)[0].numel() == 0
        ok = ok and pos_ok and val_ok and empty_ok
        print(f"  primitive diff dtype={dtype}: pos={pos_ok} val={val_ok} empty={empty_ok}", flush=True)
    return ok


def case_gather_vs_baseline(rank, world, dev) -> bool:
    """2. Batched sparse gather over the real backend vs the full-gather diff baseline."""
    ok = True
    mesh = init_device_mesh(get_device_name(), (world,))

    def run_one(shape, si, mesh, placements, max_round_bytes=None):
        torch.manual_seed(si)
        full_old = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
        full_new = full_old.clone()
        g = torch.Generator(device=dev).manual_seed(1000 + si)
        numel = full_old.numel()
        k = max(1, numel // 100)
        pert = torch.randint(0, numel, (k,), device=dev, generator=g)
        full_new.view(-1)[pert] += torch.randn(k, dtype=torch.bfloat16, device=dev, generator=g) * 0.1

        dt_old = distribute_tensor(full_old, mesh, placements)
        dt_new = distribute_tensor(full_new, mesh, placements)
        gidx, gval, pg, contributes = _sender_sharded_delta(dt_new, dt_old, dev)
        counts = torch.tensor([int(gidx.numel())], dtype=torch.int64, device=dev)
        gathered = gather_slot_entries_to_rank0(gidx, gval, counts, group=pg, max_round_bytes=max_round_bytes)

        fo, fn = dt_old.full_tensor().reshape(-1), dt_new.full_tensor().reshape(-1)
        b_idx = (fn.view(torch.int16) != fo.view(torch.int16)).nonzero(as_tuple=False).view(-1)
        b_val = fn[b_idx]

        # Only a CONTRIBUTING gather-group leader can be compared against the full-diff
        # baseline. A rank on a Replicate-only coordinate contributes nothing and its
        # (degenerate) group correctly returns an empty delta.
        if gathered is None or not contributes:
            return True
        (sh_idx, sh_val), = gathered
        so, bo = torch.argsort(sh_idx), torch.argsort(b_idx)
        good = (
            torch.equal(sh_idx[so], b_idx[bo])
            and torch.equal(sh_val[so].view(torch.int16), b_val[bo].view(torch.int16))
            and sh_idx.numel() == b_idx.numel()
        )
        print(
            f"  gather shape={shape} sub_rounds={max_round_bytes is not None}: "
            f"nnz={sh_idx.numel()}/{b_idx.numel()} -> {'PASS' if good else 'FAIL'}",
            flush=True,
        )
        return good

    for si, shape in enumerate([(4096, 1024), (7, 3), (30001, 8), (128,)]):
        ok = run_one(shape, si, mesh, [Shard(0)]) and ok
    # sub-round determinism: byte-budgeted rounds must equal the single-round result
    ok = run_one((4096, 1024), 50, mesh, [Shard(0)], max_round_bytes=4096) and ok
    # 2D FSDP x SP(replicate): only sp-coord-0 ranks contribute (no double count)
    if world % 2 == 0:
        mesh2d = init_device_mesh(get_device_name(), (world // 2, 2), mesh_dim_names=("fsdp", "sp"))
        for si, shape in enumerate([(4096, 1024), (7, 3)]):
            ok = run_one(shape, 60 + si, mesh2d, [Shard(0), Replicate()]) and ok
    return ok


def _build_delta_fixture(rank, world, dev):
    """A tiny 'model' with FSDP-sharded trainer views and full rollout copies."""
    mesh = init_device_mesh(get_device_name(), (world,))
    shapes = {
        "model.embed_tokens.weight": (32000, 64),
        "model.layers.0.mlp.gate_up_proj.weight": (256, 64),
        "model.layers.0.mlp.down_proj.weight": (64, 256),
        "model.layers.0.input_layernorm.weight": (64,),
    }
    full_old, full_new = {}, {}
    for i, (name, shape) in enumerate(shapes.items()):
        torch.manual_seed(2024 + i)
        old = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
        new = old.clone()
        g = torch.Generator(device=dev).manual_seed(9000 + i)
        k = max(1, old.numel() // 100)  # ~1% sparsity, the steady-state regime
        pert = torch.randint(0, old.numel(), (k,), device=dev, generator=g)
        new.view(-1)[pert] += torch.randn(k, dtype=torch.bfloat16, device=dev, generator=g) * 0.1
        full_old[name], full_new[name] = old, new
    dt_old = {n: distribute_tensor(t, mesh, [Shard(0)]) for n, t in full_old.items()}
    dt_new = {n: distribute_tensor(t, mesh, [Shard(0)]) for n, t in full_new.items()}
    return full_old, full_new, dt_old, dt_new


def _gather_model_delta(dt_old, dt_new, dev, empty: bool):
    """One batched gather round for the whole model, exactly like the engine's
    _GatherQueue over a single process group."""
    idx_parts, val_parts, cnts = [], [], []
    names = list(dt_new)
    pg = None
    for name in names:
        gidx, gval, pg, _ = _sender_sharded_delta(dt_new[name], dt_old[name], dev)
        if empty:
            gidx = torch.empty(0, dtype=torch.int64, device=dev)
            gval = torch.empty(0, dtype=torch.bfloat16, device=dev)
        idx_parts.append(gidx)
        val_parts.append(gval)
        cnts.append(int(gidx.numel()))
    counts = torch.tensor(cnts, dtype=torch.int64, device=dev)
    gathered = gather_slot_entries_to_rank0(
        torch.cat(idx_parts), torch.cat(val_parts), counts, group=pg
    )
    return names, gathered


def case_delta_roundtrip(rank, world, dev, empty: bool = False) -> bool:
    """3/4. Full wire round-trip: encode -> HCCL broadcast -> apply_delta -> bit-exact."""
    full_old, full_new, dt_old, dt_new = _build_delta_fixture(rank, world, dev)
    names, gathered = _gather_model_delta(dt_old, dt_new, dev, empty=empty)

    if rank == 0:
        entries = [
            (name, "bfloat16", tuple(full_new[name].shape), idx, val)
            for name, (idx, val) in zip(names, gathered, strict=True)
        ]
        spec, positions_u8, values = _assemble_indices_flush(entries, dev)
        spec_bytes = json.dumps(spec).encode()
        nnz = int(values.numel())
    else:
        spec_bytes, positions_u8, values, nnz = None, None, None, None

    nnz_t = torch.tensor([nnz or 0], dtype=torch.int64, device=dev)
    dist.broadcast(nnz_t, src=0)
    if rank != 0:
        positions_u8 = torch.empty(0, dtype=torch.uint8, device=dev)  # int32 nnz * 4 bytes
        values = torch.empty(int(nnz_t.item()), dtype=torch.bfloat16, device=dev)
    # positions blob size is derivable from the spec, broadcast it explicitly instead
    pos_len = torch.tensor(
        [int(positions_u8.numel()) if rank == 0 else 0], dtype=torch.int64, device=dev
    )
    dist.broadcast(pos_len, src=0)
    if rank != 0:
        positions_u8 = torch.empty(int(pos_len.item()), dtype=torch.uint8, device=dev)
    _bcast_tensor(positions_u8, src=0)
    _bcast_tensor(values, src=0)
    spec_bytes = _bcast_bytes(spec_bytes, dev, src=0)

    # every rank acts as a rollout replica holding the OLD weights
    model = _FakeRolloutModel({n: t.clone() for n, t in full_old.items()})
    _recv_apply(spec_bytes, positions_u8, values, model)

    good = True
    for n in names:
        target = full_old[n] if empty else full_new[n]
        if not _bitwise_equal(model.params[n], target):
            good = False
            print(f"  MISMATCH on {n}", flush=True)
    nnz_val = int(nnz_t.item())
    if empty:
        good = good and nnz_val == 0
    else:
        good = good and nnz_val > 0
    if rank == 0:
        print(f"  roundtrip empty={empty}: nnz={nnz_val} -> {'PASS' if good else 'FAIL'}", flush=True)
    return good


def case_checksum_corruption(rank, world, dev) -> bool:
    """5. One corrupted value byte must trip the wire checksum in the loader."""
    full_old, full_new, dt_old, dt_new = _build_delta_fixture(rank, world, dev)
    names, gathered = _gather_model_delta(dt_old, dt_new, dev, empty=False)

    if rank == 0:
        entries = [
            (name, "bfloat16", tuple(full_new[name].shape), idx, val)
            for name, (idx, val) in zip(names, gathered, strict=True)
        ]
        spec, positions_u8, values = _assemble_indices_flush(entries, dev)
        spec_bytes = json.dumps(spec).encode()
    else:
        spec_bytes = None
        positions_u8 = torch.empty(0, dtype=torch.uint8, device=dev)
        values = torch.empty(0, dtype=torch.bfloat16, device=dev)

    sizes = torch.tensor(
        [int(positions_u8.numel()), int(values.numel())] if rank == 0 else [0, 0],
        dtype=torch.int64,
        device=dev,
    )
    dist.broadcast(sizes, src=0)
    if rank != 0:
        positions_u8 = torch.empty(int(sizes[0].item()), dtype=torch.uint8, device=dev)
        values = torch.empty(int(sizes[1].item()), dtype=torch.bfloat16, device=dev)
    _bcast_tensor(positions_u8, src=0)
    _bcast_tensor(values, src=0)
    spec_bytes = _bcast_bytes(spec_bytes, dev, src=0)

    # corrupt AFTER the broadcast: receiver-side bit flip in the values payload
    values.view(torch.int16)[0] += 1
    model = _FakeRolloutModel({n: t.clone() for n, t in full_old.items()})
    raised = False
    try:
        _recv_apply(spec_bytes, positions_u8, values, model)
    except RuntimeError as e:
        raised = "checksum" in str(e)
    print(f"  checksum corruption: raised={raised} -> {'PASS' if raised else 'FAIL'}", flush=True)
    return raised


def _make_dense_seed(full_new, dev, verify: bool, is_last: bool = True):
    """Values-only seed flush, same layout as DeltaShardedCheckpointEngine._publish_values_flush."""
    params, val_parts, val_off = [], [], 0
    for name, t in full_new.items():
        flat = t.reshape(-1)
        params.append(
            {
                "name": name,
                "dtype": "bfloat16",
                "shape": list(t.shape),
                "pos_start": 0,
                "pos_end": 0,
                "pos_width": 4,
                "val_start": val_off,
                "val_end": val_off + int(flat.numel()),
            }
        )
        val_parts.append(flat)
        val_off += int(flat.numel())
    values = torch.cat(val_parts).contiguous()
    empty_pos = torch.empty(0, dtype=torch.uint8, device=dev)
    spec = {
        "encoding": "dense",
        "verify": verify,
        "is_last": is_last,
        "params": params,
        "checksum": int(wire_checksum(empty_pos, values)),
    }
    return spec, empty_pos, values


def case_dense_seed(rank, world, dev) -> bool:
    """6. The first-sync seed: a dense values-only flush rebuilds weights from zeros."""
    _, full_new, _, _ = _build_delta_fixture(rank, world, dev)
    spec, empty_pos, values = _make_dense_seed(full_new, dev, verify=False)
    model = _FakeRolloutModel({n: torch.zeros_like(t) for n, t in full_new.items()})
    _recv_apply(json.dumps(spec).encode(), empty_pos, values, model)
    good = all(_bitwise_equal(model.params[n], full_new[n]) for n in full_new)
    print(f"  dense seed: {'PASS' if good else 'FAIL'}", flush=True)
    return good


def case_verify_sweep(rank, world, dev) -> bool:
    """7. verify_every idempotence sweep: in-sync passes, drifted server raises."""
    _, full_new, _, _ = _build_delta_fixture(rank, world, dev)

    # in-sync server: replaying the full weights must be a no-op
    spec, empty_pos, values = _make_dense_seed(full_new, dev, verify=True, is_last=True)
    in_sync = _FakeRolloutModel({n: t.clone() for n, t in full_new.items()})
    ok_clean = True
    try:
        _recv_apply(json.dumps(spec).encode(), empty_pos, values, in_sync)
    except RuntimeError:
        ok_clean = False

    # drifted server: one element disagrees with the trainer export -> must raise
    drifted = _FakeRolloutModel({n: t.clone() for n, t in full_new.items()})
    victim = next(iter(drifted.params))
    drifted.params[victim].view(-1)[0] += 1.0
    raised = False
    try:
        _recv_apply(json.dumps(spec).encode(), empty_pos, values, drifted)
    except RuntimeError as e:
        raised = "verification FAILED" in str(e)
    good = ok_clean and raised
    print(f"  verify sweep: clean={ok_clean} drift_raised={raised} -> {'PASS' if good else 'FAIL'}", flush=True)
    return good


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    dist.init_process_group(get_nccl_backend())
    rank, world = dist.get_rank(), dist.get_world_size()
    get_torch_device().set_device(rank)
    dev = torch.device(get_device_name(), rank)
    if rank == 0:
        print(f"device={get_device_name()} backend={get_nccl_backend()} world={world}", flush=True)
    if not hasattr(torch, "hash_tensor"):
        # The wire checksum (delta_sync.encode.checksum) is built on torch.hash_tensor;
        # without it the whole delta protocol cannot run on this stack.
        if rank == 0:
            print(
                "FATAL: torch.hash_tensor missing -- delta_sync checksum requires a "
                "torch/torch_npu version that provides it.",
                flush=True,
            )
        dist.destroy_process_group()
        sys.exit(2)

    results = {}
    if rank == 0:
        results["primitive_diff"] = _agree(case_primitive_diff(dev), dev)
    else:
        results["primitive_diff"] = _agree(True, dev)
    results["gather_vs_baseline"] = _agree(case_gather_vs_baseline(rank, world, dev), dev)
    dist.barrier()
    results["delta_roundtrip"] = _agree(case_delta_roundtrip(rank, world, dev, empty=False), dev)
    dist.barrier()
    results["empty_delta_noop"] = _agree(case_delta_roundtrip(rank, world, dev, empty=True), dev)
    dist.barrier()
    results["checksum_corruption"] = _agree(case_checksum_corruption(rank, world, dev), dev)
    dist.barrier()
    results["dense_seed"] = _agree(case_dense_seed(rank, world, dev), dev)
    dist.barrier()
    results["verify_sweep"] = _agree(case_verify_sweep(rank, world, dev), dev)
    dist.barrier()

    all_ok = all(results.values())
    if rank == 0:
        print("=" * 50, flush=True)
        for name, ok in results.items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"OVERALL: {'ALL PASS' if all_ok else 'FAIL'}", flush=True)
        print("=" * 50, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
