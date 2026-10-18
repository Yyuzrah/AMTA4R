# Source this before every command of the harness-switch experiment. Sets HARNESS_DIR to the experiment folder
# (extensions/harness_switch) and keeps every cache, temporary file and home-directory write inside $HARNESS_DIR/cache.
# APPTAINER_BIN: bin/ directory of the Apptainer installation (default: the directory of `apptainer` on PATH).
# HF_HOME: set it to a Hugging Face cache that holds Qwen/Qwen3-8B; with HF_HUB_OFFLINE=1 nothing is downloaded.
export HARNESS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HOME=$HARNESS_DIR/cache/home
export TMPDIR=$HARNESS_DIR/cache/tmp UV_CACHE_DIR=$HARNESS_DIR/cache/uv PIP_CACHE_DIR=$HARNESS_DIR/cache/pip PYTHONPYCACHEPREFIX=$HARNESS_DIR/cache/pycache
export XDG_CACHE_HOME=$HARNESS_DIR/cache/xdg/cache XDG_CONFIG_HOME=$HARNESS_DIR/cache/xdg/config XDG_DATA_HOME=$HARNESS_DIR/cache/xdg/data XDG_STATE_HOME=$HARNESS_DIR/cache/xdg/state
export MSWEA_GLOBAL_CONFIG_DIR=$HARNESS_DIR/cache/mswea MSWEA_COST_TRACKING=ignore_errors
export APPTAINER_CACHEDIR=$HARNESS_DIR/cache/apptainer APPTAINER_TMPDIR=$HARNESS_DIR/cache/apptainer_tmp
export APPTAINER_BIN=${APPTAINER_BIN:-$(dirname "$(command -v apptainer)")}
export PATH=$APPTAINER_BIN:$PATH
export VLLM_CACHE_ROOT=$HARNESS_DIR/cache/vllm VLLM_CONFIG_ROOT=$HARNESS_DIR/cache/vllm_config FLASHINFER_WORKSPACE_BASE=$HARNESS_DIR/cache/flashinfer
export TRITON_CACHE_DIR=$HARNESS_DIR/cache/triton TORCHINDUCTOR_CACHE_DIR=$HARNESS_DIR/cache/torchinductor CUDA_CACHE_PATH=$HARNESS_DIR/cache/nv
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
mkdir -p $HOME $TMPDIR $UV_CACHE_DIR $PIP_CACHE_DIR $PYTHONPYCACHEPREFIX $XDG_CACHE_HOME $XDG_CONFIG_HOME $XDG_DATA_HOME $XDG_STATE_HOME $MSWEA_GLOBAL_CONFIG_DIR $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $VLLM_CACHE_ROOT $VLLM_CONFIG_ROOT $FLASHINFER_WORKSPACE_BASE $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $CUDA_CACHE_PATH
