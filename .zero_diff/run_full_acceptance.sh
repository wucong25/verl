#!/usr/bin/env bash
set -eo pipefail
cd /home/w00848261/verl
run_name="${1:?Usage: bash .zero_diff/run_full_acceptance.sh NEW_RUN_NAME}"
if [[ ! "$run_name" =~ ^[a-z0-9][a-z0-9_-]{0,31}$ ]]; then
    echo 'Invalid run name' >&2
    exit 2
fi
run_root="/home/w00848261/verl/.zero_diff/runs/$run_name"
if [[ -e "$run_root" ]]; then
    echo "Run directory already exists: $run_root; select a new name" >&2
    exit 2
fi
source .zero_diff/env.sh
export HCCL_NPU_SOCKET_PORT_RANGE=auto HCCL_HOST_SOCKET_PORT_RANGE=auto
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_BATCH_INVARIANT=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 HCCL_CONNECT_TIMEOUT=180
export PYTHONPATH=/home/w00848261/verl/.zero_diff:$PYTHONPATH
python .zero_diff/stamp_run.py "$run_name" .zero_diff/full_acceptance.py .zero_diff/online_worker_008.py .zero_diff/weight_audit_007.py .zero_diff/case_manifest_full_dataset.json .zero_diff/env.sh .zero_diff/run_full_acceptance.sh .zero_diff/review/full-v1.md .zero_diff/audit_receipt.py .zero_diff/audit_full_geometry.py
python .zero_diff/full_acceptance.py --run-name "$run_name" > "$run_root/driver.log" 2>&1
python .zero_diff/audit_receipt.py "$run_root" > "$run_root/independent-audit.log" 2>&1
python .zero_diff/audit_full_geometry.py "$run_root" > "$run_root/model-geometry.log" 2>&1
printf 'STRICT FULL MODEL ACCEPTANCE PASSED: %s\n' "$run_root"
