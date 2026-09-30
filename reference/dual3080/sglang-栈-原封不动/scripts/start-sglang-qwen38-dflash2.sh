#!/usr/bin/env bash
set -euo pipefail

source /root/sglang_qwen38_27b_dflash2.conf.sh

command=("$SGLANG_VENV/bin/python" -m sglang.launch_server "${SGLANG_ARGS[@]}")
encoder_command=(
  env
  "CUDA_VISIBLE_DEVICES=$ENCODER_GPU"
  "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  "$SGLANG_VENV/bin/python"
  -m
  sglang.launch_server
  "${ENCODER_ARGS[@]}"
)

if [[ "${1:-}" == "--print-cmd" ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi

if [[ "${1:-}" == "--print-encoder-cmd" ]]; then
  if (( $# != 1 )); then
    echo "--print-encoder-cmd does not accept extra arguments." >&2
    exit 2
  fi
  printf '%q ' "${encoder_command[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -x "$SGLANG_VENV/bin/python" ]]; then
  echo "SGLang Python is not executable: $SGLANG_VENV/bin/python" >&2
  exit 1
fi

ulimit -s 65536
source "$SGLANG_VENV/bin/activate"
exec "${command[@]}" "$@"
