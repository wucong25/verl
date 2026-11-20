#!/usr/bin/env bash
set -eo pipefail
cd /home/w00848261/verl
source .zero_diff/env.sh
export HCCL_NPU_SOCKET_PORT_RANGE=auto
export HCCL_HOST_SOCKET_PORT_RANGE=auto
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export PYTHONPATH=/home/w00848261/verl/.zero_diff:$PYTHONPATH
export HCCL_CONNECT_TIMEOUT=180
if test -e .zero_diff/runs/full-online-001/result.json; then
    echo "full-online-001 already has a result; inspect it before selecting a new run ID" >&2
    exit 2
fi
python .zero_diff/stamp_run.py full-online-001 .zero_diff/full_online_001.py .zero_diff/online_worker_008.py .zero_diff/weight_audit_007.py .zero_diff/case_manifest_full_dataset.json .zero_diff/env.sh .zero_diff/run_full_v1.sh .zero_diff/review/full-v1.md
python .zero_diff/full_online_001.py > .zero_diff/runs/full-online-001.log 2>&1
python - <<'VERIFY'
import json
from pathlib import Path
p=Path('.zero_diff/runs/full-online-001/result.json')
d=json.loads(p.read_text())
assert d.get('passed') is True, d.get('error', 'Strict acceptance did not pass')
assert d['optimizer_steps']==2 and len(d['phases'])==3
for phase in d['phases']:
    for name in ('pd','mixed_batch','recompute','grad_forward','grad_vs_recompute'):
        result=phase[name]
        assert result['finite'] and result['mismatch']==0 and result['max_abs']==0
        assert result['dtype']=='torch.float32' and result['count']>0
    assert phase['resident']==d['resident_initial']
print('FULL MODEL STRICT ACCEPTANCE PASSED',p)
VERIFY
