#!/usr/bin/env bash
set -euo pipefail

source /root/sglang_qwen38_27b_dflash2.conf.sh

readonly PID_FILE="${ENCODER_PID_FILE:-/root/sglang-qwen38-dflash2/encoder.pid}"
readonly LOG_FILE="${ENCODER_LOG_FILE:-/root/sglang-qwen38-dflash2/server-encoder-gpu1.log}"
readonly HEALTH_URL="${ENCODER_URL}/health"

command=(
  env
  "CUDA_VISIBLE_DEVICES=$ENCODER_GPU"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "$SGLANG_VENV/bin/python"
  -m
  sglang.launch_server
  "${ENCODER_ARGS[@]}"
)

if [[ "${1:-}" == "--print-cmd" ]]; then
  if (( $# != 1 )); then
    echo "--print-cmd does not accept extra arguments." >&2
    exit 2
  fi
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if (( $# != 0 )); then
  echo "Usage: $0 [--print-cmd]" >&2
  exit 2
fi

if [[ ! -x "$SGLANG_VENV/bin/python" ]]; then
  echo "SGLang Python is not executable: $SGLANG_VENV/bin/python" >&2
  exit 1
fi

if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
  echo "Encoder is already healthy at $HEALTH_URL."
  exit 0
fi

if fuser -n tcp "$ENCODER_PORT" >/dev/null 2>&1; then
  echo "TCP port $ENCODER_PORT is busy but the encoder is not healthy." >&2
  exit 1
fi

mkdir -p "$(dirname "$PID_FILE")" "$(dirname "$LOG_FILE")"
ulimit -s 65536
nohup "${command[@]}" >"$LOG_FILE" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"

for _ in {1..180}; do
  if curl -fsS --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Encoder is healthy at $HEALTH_URL (pid $pid, GPU $ENCODER_GPU)."
    exit 0
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Encoder exited before becoming healthy; see $LOG_FILE." >&2
    exit 1
  fi
  sleep 1
done

echo "Encoder did not become healthy within 180 seconds; see $LOG_FILE." >&2
exit 1
