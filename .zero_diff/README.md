# Qwen3.5-35B-A3B Ascend zero-mismatch work

This directory contains the complete diagnostic and acceptance source code for the Qwen3.5-35B-A3B Megatron/vLLM alignment work. The final 40-layer numerical acceptance passed for all three weight versions around two real native optimizer updates. See [delivery/README.md](delivery/README.md) for the measured scope, limitations, timings and results, and [delivery/verification-summary.json](delivery/verification-summary.json) for a compact machine-readable receipt.

## Repository layout

- The implementation and tests for verl are committed at their normal paths in this repository.
- The complete implementation and tests for the separate vllm-ascend repository are included in [delivery/vllm-ascend.patch](delivery/vllm-ascend.patch). Its base commit and SHA256 are recorded in [delivery/patch-manifest.json](delivery/patch-manifest.json). Apply it from the matching vllm-ascend checkout with `git apply --check /path/to/verl/.zero_diff/delivery/vllm-ascend.patch` followed by `git apply`.
- vllm itself has no code changes; its required base commit is also recorded in the manifest.
- `full_acceptance.py`, `run_full_acceptance.sh`, `online_worker_008.py`, `weight_audit_007.py`, and the two `audit_*.py` scripts form the current acceptance harness.
- Numbered probes, replay scripts and hooks are the complete historical diagnostic source archive. Some intentionally reproduce past failures, depend on server-side traces, or use superseded approaches. They are not production launchers.
- Model checkpoints, dataset tokens, generated tensor traces, cache contents, credentials and handoff records are not versioned. The original server retains the full audited evidence under `runs/full-online-002/`.

## Reproduction in the supplied container

The harness preserves the tested deployment paths: verl at `/home/w00848261/verl`, vllm at `/vllm`, vllm-ascend at `/vllm-ascend`, weights at `/mnt/weight/Qwen3.5-35B-A3B`, and DAPO at `/mnt/share/datasets/dapo-math-17k/dapo-math-17k.parquet`. Source `env.sh` before Python/NPU work; it sources both CANN and ATB.

For a fresh checkout, create a local Ray authentication token without committing it, and reconstruct the exact four input prefixes from the dataset:

```bash
cd /home/w00848261/verl
source .zero_diff/env.sh
python - <<'PYSETUP'
import os, secrets
from pathlib import Path
path = Path(os.environ["RAY_AUTH_TOKEN_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
if not path.exists():
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(secrets.token_hex(32))
PYSETUP
python .zero_diff/make_case_manifest_full.py
bash .zero_diff/run_full_acceptance.sh full-v1-repeat-001
```

Use a new run name each time. This executes the approved numerical update probe, not a reward/accuracy benchmark. The original fixed launcher `run_full_v1_002.sh` is the one that completed the reported full-model run; the repeatable launcher has the same numerical/configuration AST after substituting only the run directory and IPC job.

This work used OpenAI Codex assistance. No upstream PR is included in this publication.
