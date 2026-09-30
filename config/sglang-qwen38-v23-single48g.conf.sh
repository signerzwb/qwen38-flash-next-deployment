#!/usr/bin/env bash

readonly SGLANG_VENV="${SGLANG_VENV:-/root/sglang-env-v23}"
readonly MODEL_PATH="${MODEL_PATH:-/root/LLM/Qwen3.8-27B-Uncensored-NVFP4-v2}"
readonly DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/LLM/Qwen3.8-27B-DFlash2}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8778}"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Required for NVFP4 on Ada/SM89 and for the WSL HiCache path shipped by v2.3.
export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1
export SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1
export SGLANG_ENABLE_GRAPH_POOL_PRECARVE=1
export SGLANG_ENABLE_GRAPH_POOL_BORROW=1

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.95}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-380000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-380000}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-3}"
# v2.3 uses 20 slots for five-way concurrency; 12 is the matching three-way tier.
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-12}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-512}"
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-3}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
HICACHE_SIZE="${HICACHE_SIZE:-24}"

SGLANG_ARGS=(
  --enable-mixed-chunk
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size 1
  --trust-remote-code
  --enable-multimodal
  --context-length "$CONTEXT_LENGTH"
  --kv-cache-dtype fp8_e4m3
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  # The v2.3 guide mentions HRRN, but its delivered merged patch has no HRRN
  # implementation. Use the documented rollback policy supported by this tree.
  --schedule-policy lpm
  --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --attention-backend flashinfer
  --mamba-backend flashinfer
  --mamba-radix-cache-strategy extra_buffer_lazy
  --mamba-ssm-dtype bfloat16
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
  --mamba-max-states-per-path 6
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --prefill-max-requests 1
  --page-size 64
  --radix-eviction-policy lru
  --disable-prefill-cuda-graph
  --cuda-graph-max-bs-decode "$MAX_RUNNING_REQUESTS"
  --enable-cache-report
  --enable-metrics
  --allow-auto-truncate
  --speculative-algorithm DFLASH
  --speculative-draft-model-path "$DRAFT_MODEL_PATH"
  --speculative-num-draft-tokens 8
  --speculative-draft-window-size 2048
  --speculative-draft-attention-backend flashinfer
  --speculative-draft-kv-cache-dtype fp8_e4m3
  --enable-linear-replayssm-spec
)

if [[ "$HICACHE_SIZE" -gt 0 ]]; then
  SGLANG_ARGS+=(
    --enable-hierarchical-cache
    --hicache-size "$HICACHE_SIZE"
    --hicache-io-backend kernel
    --hicache-mem-layout page_first
    --hicache-write-policy write_back
  )
fi

if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  SGLANG_ARGS+=(--api-key "$SGLANG_API_KEY")
fi

if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$SGLANG_EXTRA_ARGS"
  SGLANG_ARGS+=("${extra_args[@]}")
fi
