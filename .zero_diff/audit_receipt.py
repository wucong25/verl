"""Independently verify saved tensors, weight changes, and resident identities."""
import datetime
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1]).resolve()
allowed = Path('/home/w00848261/verl/.zero_diff/runs')
assert root.is_relative_to(allowed), root
report = json.loads((root / 'result.json').read_text())
assert report.get('passed') is True and 'error' not in report
assert report['optimizer_steps'] == 2 and len(report['phases']) == 3
initial = report['resident_initial']
assert len(initial) == 8 and len({r['pid'] for r in initial}) == 8
checks = []
for version, phase in enumerate(report['phases']):
    assert phase['version'] == version and phase['resident'] == initial
    tensors = torch.load(root / f'dual-theta{version}.pt', map_location='cpu', weights_only=True)
    rollout = tensors['rollout']
    assert rollout.dtype == torch.float32 and rollout.numel() > 0
    assert torch.isfinite(rollout).all()
    for name in ('recompute', 'grad_forward'):
        value = tensors[name]
        assert value.dtype == torch.float32 and value.shape == rollout.shape
        assert torch.isfinite(value).all() and torch.equal(value, rollout), (version, name)
    for name in ('pd', 'mixed_batch', 'recompute', 'grad_forward', 'grad_vs_recompute'):
        result = phase[name]
        assert result['finite'] and result['count'] > 0
        assert result['dtype'] == 'torch.float32' and result['mismatch'] == 0 and result['max_abs'] == 0
    inference_path = root / f'inference-theta{version}.pt'
    if report['scope'].startswith('40-layer'):
        assert inference_path.exists(), inference_path
    if inference_path.exists():
        inference = torch.load(inference_path, map_location='cpu', weights_only=True)
        reference = inference['rollout']
        assert reference.dtype == torch.float32 and reference.numel() > 0
        assert torch.isfinite(reference).all()
        for name in ('batched_prefill', 'single_prefill'):
            value = inference[name]
            assert value.dtype == reference.dtype and torch.equal(reference, value), (version, name)
        assert torch.equal(rollout, reference.repeat(2, 1))
    checks.append({'version': version, 'raw_tensor_count_per_native_path': rollout.numel(), 'exact_equal': True})

before_rollout = json.loads((root / 'rollout-weights-before.json').read_text())
after_rollout = json.loads((root / 'rollout-weights-after.json').read_text())
assert len(before_rollout) == len(after_rollout) == 8
for before, after in zip(before_rollout, after_rollout):
    assert before['rank'] == after['rank'] and before['parameters'] == after['parameters']
updates = []
for version in range(2):
    before = json.loads((root / f'parameters-theta{version}.json').read_text())
    after = json.loads((root / f'parameters-theta{version + 1}.json').read_text())
    assert len(before) == len(after) == 16
    changed = []
    for old, new in zip(before, after):
        assert old['rank'] == new['rank']
        assert old['parameters'].keys() == new['parameters'].keys()
        count = sum(old['parameters'][name]['sha256'] != item['sha256'] for name, item in new['parameters'].items())
        assert count > 0, (version, old['rank'])
        changed.append(count)
    phase = report['phases'][version]
    assert changed == phase['changed_parameters']
    norm = torch.as_tensor(phase['optimizer']['grad_norm'], dtype=torch.float64)
    assert torch.isfinite(norm).all() and (norm > 0).all()
    sync = phase['weight_sync']
    assert len(sync) == 16 and sum(bool(r['sent']) for r in sync) == 8
    assert all(r['tensor_count'] > 0 for r in sync)
    assert len({r['name_digest'] for r in sync}) == 1
    updates.append({'from_version': version, 'changed_parameters_per_rank': changed, 'grad_norm': norm.tolist(), 'device_ipc_receipts': 16})

receipt = {'audited_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'run': str(root), 'passed': True, 'raw_tensor_checks': checks, 'initial_sync_all_parameter_hashes_unchanged': True, 'resident_identities_unchanged': True, 'updates': updates}
(root / 'independent-audit.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt, indent=2))
