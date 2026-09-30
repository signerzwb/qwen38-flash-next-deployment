#!/usr/bin/env bash
set -euo pipefail

readonly PORT="8778"

mapfile -t pids < <(fuser -n tcp "$PORT" 2>/dev/null | xargs -r -n1 printf '%s\n')

if (( ${#pids[@]} == 0 )); then
  echo "No process is listening on TCP port $PORT."
  exit 0
fi

matched=()
for pid in "${pids[@]}"; do
  if [[ ! -r "/proc/$pid/cmdline" ]]; then
    continue
  fi

  cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
  if [[ "$cmdline" == *sglang* && "$cmdline" == *"$PORT"* ]]; then
    matched+=("$pid")
  else
    echo "Refusing to stop unrelated listener PID $pid: $cmdline" >&2
    exit 2
  fi
done

if (( ${#matched[@]} == 0 )); then
  echo "No matching SGLang listener found on TCP port $PORT." >&2
  exit 2
fi

kill -TERM "${matched[@]}"

for _ in {1..30}; do
  alive=()
  for pid in "${matched[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done

  if (( ${#alive[@]} == 0 )); then
    echo "Stopped SGLang on TCP port $PORT."
    exit 0
  fi
  sleep 1
done

echo "SGLang did not stop within 30 seconds; still running: ${alive[*]}" >&2
exit 1
