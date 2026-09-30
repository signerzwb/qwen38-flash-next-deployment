#!/usr/bin/env bash
set -euo pipefail

source /root/sglang_qwen38_27b_dflash2.conf.sh

readonly PID_FILE="${ENCODER_PID_FILE:-/root/sglang-qwen38-dflash2/encoder.pid}"
mapfile -t pids < <(fuser -n tcp "$ENCODER_PORT" 2>/dev/null | xargs -r -n1 printf '%s\n')

if (( ${#pids[@]} == 0 )); then
  rm -f "$PID_FILE"
  echo "No process is listening on TCP port $ENCODER_PORT."
  exit 0
fi

matched=()
for pid in "${pids[@]}"; do
  if [[ ! -r "/proc/$pid/cmdline" ]]; then
    continue
  fi
  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  if [[ "$cmdline" == *sglang* && "$cmdline" == *"$ENCODER_PORT"* && "$cmdline" == *encoder-only* ]]; then
    matched+=("$pid")
  else
    echo "Refusing to stop unrelated listener PID $pid: $cmdline" >&2
    exit 2
  fi
done

if (( ${#matched[@]} == 0 )); then
  echo "No matching SGLang encoder listener found on TCP port $ENCODER_PORT." >&2
  exit 2
fi

kill -TERM "${matched[@]}"
for _ in {1..60}; do
  alive=()
  for pid in "${matched[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  if (( ${#alive[@]} == 0 )); then
    rm -f "$PID_FILE"
    echo "Stopped SGLang encoder on TCP port $ENCODER_PORT."
    exit 0
  fi
  sleep 1
done

echo "SGLang encoder did not stop within 60 seconds; still running: ${alive[*]}" >&2
exit 1
