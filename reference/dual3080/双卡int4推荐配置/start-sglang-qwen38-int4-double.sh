#!/usr/bin/env bash
# 启动 sglang（双 3080 int4 推荐配置）。用法：
#   bash start-sglang-qwen38-int4-double.sh          # 前台
#   bash start-sglang-qwen38-int4-double.sh --print-cmd   # 先打印命令再自己跑
# 后台加日志：nohup bash start-sglang-qwen38-int4-double.sh > /root/sglang-3080.log 2>&1 &
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${CONF:-$SCRIPT_DIR/sglang_qwen38_27b_int4_double.conf.sh}"

if [[ -f "$CONF" ]]; then
  source "$CONF"
else
  echo "conf not found: $CONF" >&2
  exit 1
fi

command=("$SGLANG_VENV/bin/python" -m sglang.launch_server "${SGLANG_ARGS[@]}")

if [[ "${1:-}" == "--print-cmd" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$SGLANG_VENV/bin/python" ]]; then
  echo "SGLang Python not executable: $SGLANG_VENV/bin/python" >&2
  exit 1
fi

ulimit -s 65536
source "$SGLANG_VENV/bin/activate"
exec "${command[@]}" "$@"