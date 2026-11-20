#!/usr/bin/env bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/home/w00848261/verl:${PYTHONPATH:-}
export TMPDIR=/home/w00848261/verl/.zero_diff/tmp
export XDG_CACHE_HOME=/home/w00848261/verl/.zero_diff/cache
export ASCEND_WORK_PATH=$XDG_CACHE_HOME/ascend
export ASCEND_CACHE_PATH=$XDG_CACHE_HOME/ascend
export HF_HOME=$XDG_CACHE_HOME/hf
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton
export TORCH_EXTENSIONS_DIR=$XDG_CACHE_HOME/torch_extensions
export UV_CACHE_DIR=$XDG_CACHE_HOME/uv
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TORCHINDUCTOR_CACHE_DIR=$XDG_CACHE_HOME/inductor
export NUMBA_CACHE_DIR=$XDG_CACHE_HOME/numba
export VLLM_RPC_BASE_PATH=$TMPDIR

export RAY_AUTH_MODE=token
export RAY_AUTH_TOKEN_PATH=/home/w00848261/verl/.zero_diff/cache/ray_auth_token
