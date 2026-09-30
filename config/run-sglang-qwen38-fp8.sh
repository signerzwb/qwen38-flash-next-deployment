#!/usr/bin/env bash
# FP8 + vision launcher. It reuses the tuned v2.3 conf and only overrides the
# knobs that differ from the NVFP4 run: model path, context size, memory
# fraction and HiCache (off, see the 08-25 A/B: HiCache cost -59% prefill).
set -euo pipefail

export SGLANG_API_KEY="${SGLANG_API_KEY:-REPLACE_WITH_YOUR_API_KEY}"

# uvicorn closes idle keep-alive connections after 5s by default. A client that
# then reuses a pooled connection gets "error sending request" against the
# public endpoint, which is one of the disconnect flavours seen in the field.
export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-300}"

# Optional tuning overrides, so experiments never touch this launcher.
if [[ -f /root/sglang-qwen38-fp8.tune.sh ]]; then
  source /root/sglang-qwen38-fp8.tune.sh
fi

export MODEL_PATH="/root/LLM/Qwen3.8-27B-FP8"
export CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-262144}"
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
export HICACHE_SIZE="${HICACHE_SIZE:-0}"

source /root/sglang-qwen38-v23-single48g.conf.sh

# Post-conf overrides (the tune file is sourced earlier, the conf sets the rest).
if [[ -n "${FP8_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="$FP8_ALLOC_CONF"
fi
if [[ -n "${FP8_POST_ARGS:-}" ]]; then
  read -r -a _post <<< "$FP8_POST_ARGS"
  SGLANG_ARGS+=("${_post[@]}")
fi

cuda_home=/root/sglang-env-v23/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$cuda_home"
export PATH="/root/sglang-env-v23/bin:$cuda_home/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$cuda_home/lib64:/usr/lib/wsl/lib"

ulimit -s 65536
exec /root/sglang-env-v23/bin/python -m sglang.launch_server "${SGLANG_ARGS[@]}"
