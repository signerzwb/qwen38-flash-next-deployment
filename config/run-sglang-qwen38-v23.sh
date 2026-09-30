#!/usr/bin/env bash
set -euo pipefail

export SGLANG_API_KEY="${SGLANG_API_KEY:-REPLACE_WITH_YOUR_API_KEY}"
source /root/sglang-qwen38-v23-single48g.conf.sh

cuda_home=/root/sglang-env-v23/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME="$cuda_home"
export PATH="/root/sglang-env-v23/bin:$cuda_home/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="$cuda_home/lib64:/usr/lib/wsl/lib"

ulimit -s 65536
exec /root/sglang-env-v23/bin/python -m sglang.launch_server "${SGLANG_ARGS[@]}"
