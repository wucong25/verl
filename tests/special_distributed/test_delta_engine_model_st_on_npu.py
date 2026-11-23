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
"""End-to-end ST of the sharded-delta weight-sync path on Ascend NPU, with a REAL model.

Unlike tests/special_distributed/test_delta_checkpoint_engine_on_npu.py (synthetic
tensors), this script loads a real HF checkpoint, shards it FSDP2-style (DTensor
Shard(0)) as the trainer view, and drives the production delta protocol:

    sync 0 (seed)   : full weights, values-only "dense" flush, rollout starts from zeros
    sync 1..N       : a simulated training step perturbs ~`--sparsity` of all elements;
                      each rank byte-diffs its OWN shard (shard_delta_indices), sparse-
                      gathers to rank 0 (gather_slot_entries_to_rank0 over HCCL), rank 0
                      encodes exactly like DeltaShardedCheckpointEngine._assemble_flush
                      (int32 positions blob + values + manifest + checksum), HCCL-
                      broadcasts, and every rank applies via the PRODUCTION receiver
                      verl.workers.rollout.sglang_rollout.delta_loader.apply_delta
    verify          : every `--verify-every` steady syncs, a dense idempotence sweep
                      (the verify_every mechanism) must pass; a final negative control
                      corrupts one rollout element and the sweep MUST raise

Pass criteria: after every sync the rollout replica is BIT-EQUAL to the trainer
weights; exit code 0 iff everything passes.

Run (Ascend, world = NPUs to use)::

    torchrun --standalone --nproc-per-node=8 \
        tests/special_distributed/test_delta_engine_model_st_on_npu.py \
        --model-path /path/to/Qwen2.5-0.5B

Options: --sparsity (default 0.01), --syncs (default 3), --verify-every (default 2).
Notes: every rank holds trainer + rollout copies in host/NPU memory, so pick a model
that fits ~4x; the wire itself is what is under test, at whatever scale you point it.
"""

import argparse
import json
import sys
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from verl.checkpoint_engine.delta_sync.encode import checksum as wire_checksum
from verl.checkpoint_engine.delta_sync.sparse_gather import gather_slot_entries_to_rank0, shard_delta_indices
from verl.utils.device import get_device_name, get_nccl_backend, get_torch_device
from verl.workers.engine.spec import ShardSpec, derive_dtensor_placement, translate_flat_indices
from verl.workers.rollout.sglang_rollout.delta_loader import apply_delta


# ---------------------------------------------------------------------------
# helpers (validated in tests/special_distributed/test_delta_checkpoint_engine_on_npu.py)
# ---------------------------------------------------------------------------


class _FakeRolloutModel:
    """Stand-in for the inference engine: sglang's load_weights lands on param.copy_,
    which is all apply_delta (masked copy / verify sweep interception) relies on."""

    def __init__(self, params: dict[str, torch.Tensor]):
        self.params = params

    def load_weights(self, chunk):
        for name, t in chunk:
            self.params[name].copy_(t)


def _bcast_bytes(payload: bytes | None, dev, src: int = 0) -> bytes:
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


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    int_dtype = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[a.element_size()]
    return bool(torch.equal(a.view(int_dtype), b.view(int_dtype)))


def _assemble_indices_flush(entries, dev):
    """Same layout as DeltaShardedCheckpointEngine._assemble_flush (indices encoding)."""
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
        torch.cat(val_pieces) if val_off else torch.empty(0, dtype=getattr(torch, entries[0][1]), device=dev)
    )
    spec = {"encoding": "indices", "params": params, "checksum": int(wire_checksum(positions_u8, values))}
    return spec, positions_u8, values


def _make_dense_seed(full: dict[str, torch.Tensor], dev, verify: bool):
    """Values-only flush, same layout as _publish_values_flush (the seed / verify sweep)."""
    params, val_parts, val_off = [], [], 0
    for name, t in full.items():
        flat = t.reshape(-1)
        params.append(
            {
                "name": name,
                "dtype": str(t.dtype).replace("torch.", ""),
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
        "is_last": True,
        "params": params,
        "checksum": int(wire_checksum(empty_pos, values)),
    }
    return spec, empty_pos, values


def _sender_sharded_delta(dt_new, dt_old, dev):
    """Trainer side of one parameter: local byte-diff -> full-tensor flat positions."""
    place, contributes, pg = derive_dtensor_placement(ShardSpec.from_param(dt_new))
    if contributes:
        lidx, lval = shard_delta_indices(dt_new.to_local().reshape(-1), dt_old.to_local().reshape(-1), 0)
        gidx = translate_flat_indices(lidx, place)
    else:
        gidx = torch.empty(0, dtype=torch.int64, device=dev)
        lval = torch.empty(0, dtype=dt_new.to_local().dtype, device=dev)
    return gidx, lval, pg


def _recv_apply(spec_bytes: bytes, positions: torch.Tensor, values: torch.Tensor, model: _FakeRolloutModel):
    spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8).to(values.device)
    apply_delta(model, [("__delta_spec__", spec_t), ("__positions__", positions), ("__values__", values)])


def _agree(ok: bool, dev) -> bool:
    t = torch.tensor([0 if ok else 1], dtype=torch.int64, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return bool(t.item() == 0)


def _log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# model fixture
# ---------------------------------------------------------------------------


def load_model_params(model_path: str, dev) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    params = {n: p.detach().clone().to(dev) for n, p in model.named_parameters()}
    del model
    return params


def simulate_train_step(params: dict[str, torch.Tensor], sparsity: float, seed: int, dev):
    """Perturb ~sparsity of all elements in place-detached copies (deterministic)."""
    new = {}
    for i, (name, t) in enumerate(params.items()):
        n = t.clone()
        numel = n.numel()
        k = max(1, int(numel * sparsity))
        g = torch.Generator(device=dev).manual_seed(seed * 100003 + i)
        idx = torch.randint(0, numel, (k,), device=dev, generator=g)
        noise = torch.randn(k, dtype=torch.float32, device=dev, generator=g) * 0.02
        flat = n.view(-1)
        flat[idx] = (flat[idx].float() + noise).to(n.dtype)
        new[name] = n
    return new


# ---------------------------------------------------------------------------
# sync drivers
# ---------------------------------------------------------------------------


def seed_sync(rank, dev, trainer: dict, rollout: _FakeRolloutModel) -> bool:
    """Sync 0: full weights over the dense values-only wire."""
    spec, empty_pos, values = _make_dense_seed(trainer, dev, verify=False)
    total = torch.tensor([int(values.numel())], dtype=torch.int64, device=dev)
    dist.broadcast(total, src=0)
    _bcast_tensor(values, src=0)
    spec_bytes = _bcast_bytes(json.dumps(spec).encode(), dev, src=0)
    _recv_apply(spec_bytes, empty_pos, values, rollout)
    ok = all(_bitwise_equal(rollout.params[n], trainer[n]) for n in trainer)
    _log(rank, f"  seed sync: {values.numel() * 2 / (1 << 20):.1f} MB dense -> {'PASS' if ok else 'FAIL'}")
    return ok


def delta_sync(rank, dev, mesh, snap: dict, current: dict, rollout: _FakeRolloutModel) -> tuple[bool, dict]:
    """One steady sync: sharded diff -> batched sparse gather -> indices flush -> apply."""
    t0 = time.time()
    idx_parts, val_parts, cnts, pg = [], [], [], None
    names = list(current)
    for name in names:
        dt_old = distribute_tensor(snap[name], mesh, [Shard(0)])
        dt_new = distribute_tensor(current[name], mesh, [Shard(0)])
        gidx, gval, pg = _sender_sharded_delta(dt_new, dt_old, dev)
        idx_parts.append(gidx)
        val_parts.append(gval)
        cnts.append(int(gidx.numel()))
    counts = torch.tensor(cnts, dtype=torch.int64, device=dev)
    gathered = gather_slot_entries_to_rank0(torch.cat(idx_parts), torch.cat(val_parts), counts, group=pg)

    if rank == 0:
        entries = [
            (name, str(current[name].dtype).replace("torch.", ""), tuple(current[name].shape), idx, val)
            for name, (idx, val) in zip(names, gathered, strict=True)
        ]
        spec, positions_u8, values = _assemble_indices_flush(entries, dev)
        spec_bytes = json.dumps(spec).encode()
        sizes = [int(positions_u8.numel()), int(values.numel())]
    else:
        spec_bytes, positions_u8, values, sizes = None, None, None, [0, 0]

    sizes_t = torch.tensor(sizes, dtype=torch.int64, device=dev)
    dist.broadcast(sizes_t, src=0)
    if rank != 0:
        positions_u8 = torch.empty(int(sizes_t[0].item()), dtype=torch.uint8, device=dev)
        values = torch.empty(int(sizes_t[1].item()), dtype=current[names[0]].dtype, device=dev)
    _bcast_tensor(positions_u8, src=0)
    _bcast_tensor(values, src=0)
    spec_bytes = _bcast_bytes(spec_bytes, dev, src=0)

    _recv_apply(spec_bytes, positions_u8, values, rollout)
    ok = all(_bitwise_equal(rollout.params[n], current[n]) for n in names)
    wire_mb = (int(sizes_t[0].item()) + int(sizes_t[1].item()) * 2) / (1 << 20)
    full_mb = sum(t.numel() * 2 for t in current.values()) / (1 << 20)
    stats = {
        "nnz": int(sizes_t[1].item()),
        "wire_mb": wire_mb,
        "full_mb": full_mb,
        "elapsed_s": time.time() - t0,
    }
    _log(
        rank,
        f"  delta sync: nnz={stats['nnz']} wire={wire_mb:.2f} MB / full={full_mb:.1f} MB "
        f"({100 * wire_mb / max(full_mb, 1e-9):.3f}%) {stats['elapsed_s']:.2f}s -> {'PASS' if ok else 'FAIL'}",
    )
    return ok, stats


def verify_sweep(rank, dev, trainer: dict, rollout: _FakeRolloutModel, expect_fail: bool) -> bool:
    """Dense idempotence sweep (the verify_every mechanism); expect_fail = drifted server."""
    spec, empty_pos, values = _make_dense_seed(trainer, dev, verify=True)
    raised = False
    try:
        _recv_apply(json.dumps(spec).encode(), empty_pos, values, rollout)
    except RuntimeError as e:
        raised = "verification FAILED" in str(e)
    ok = raised == expect_fail
    _log(rank, f"  verify sweep (expect_fail={expect_fail}): raised={raised} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--sparsity", type=float, default=0.01)
    ap.add_argument("--syncs", type=int, default=3)
    ap.add_argument("--verify-every", type=int, default=2)
    args = ap.parse_args()

    dist.init_process_group(get_nccl_backend())
    rank, world = dist.get_rank(), dist.get_world_size()
    get_torch_device().set_device(rank)
    dev = torch.device(get_device_name(), rank)
    _log(rank, f"device={get_device_name()} backend={get_nccl_backend()} world={world} model={args.model_path}")

    if not hasattr(torch, "hash_tensor"):
        _log(
            rank,
            "FATAL: torch.hash_tensor missing -- delta_sync checksum requires a torch/torch_npu "
            "version that provides it.",
        )
        dist.destroy_process_group()
        sys.exit(2)

    base = load_model_params(args.model_path, dev)
    n_params = sum(t.numel() for t in base.values())
    _log(rank, f"loaded {len(base)} params, {n_params / 1e6:.1f}M elements, bf16")

    mesh = init_device_mesh(get_device_name(), (world,))
    rollout = _FakeRolloutModel({n: torch.zeros_like(t) for n, t in base.items()})
    results = {}

    # ---- sync 0: seed (rollout goes from zeros to full trainer weights) ----
    results["seed"] = _agree(seed_sync(rank, dev, base, rollout), dev)
    snapshot = {n: t.clone() for n, t in base.items()}  # prime_delta_snapshots equivalent

    # ---- sync 1..N: simulated training steps, delta wire ----
    current = base
    for step in range(1, args.syncs + 1):
        current = simulate_train_step(current, args.sparsity, seed=step, dev=dev)
        ok, _ = delta_sync(rank, dev, mesh, snapshot, current, rollout)
        results[f"delta_step_{step}"] = _agree(ok, dev)
        snapshot = {n: t.clone() for n, t in current.items()}
        if args.verify_every > 0 and step % args.verify_every == 0:
            results[f"verify_step_{step}"] = _agree(verify_sweep(rank, dev, current, rollout, False), dev)
        dist.barrier()

    # ---- negative control: drift one rollout element, the sweep MUST raise ----
    victim = next(iter(rollout.params))
    rollout.params[victim].view(-1)[0] += 1.0
    results["verify_negative"] = _agree(verify_sweep(rank, dev, current, rollout, True), dev)

    dist.barrier()
    all_ok = all(results.values())
    if rank == 0:
        print("=" * 60, flush=True)
        for name, ok in results.items():
            print(f"  {name}: {'PASS' if ok else 'FAIL'}", flush=True)
        print(f"OVERALL: {'ALL PASS' if all_ok else 'FAIL'}", flush=True)
        print("=" * 60, flush=True)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
