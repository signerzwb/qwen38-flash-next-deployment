#!/usr/bin/env bash

readonly SGLANG_VENV="${SGLANG_VENV:-/root/sglang-env}"
readonly MODEL_PATH="${MODEL_PATH:-/root/LLM/Qwen3.8-27B-Uncensored-NVFP4-v2}"
# Encoder always uses the original VL checkpoint (vision tower lives there);
# MODEL_PATH override (NVFP4 etc.) must not leak into the encoder process.
readonly ENCODER_MODEL_PATH="${ENCODER_MODEL_PATH:-/root/LLM/Qwen3.8-27B-Uncensored-NVFP4-v2}"
readonly DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/LLM/Qwen3.8-27B-DFlash2}"
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8778}"
readonly ENCODER_HOST="${ENCODER_HOST:-127.0.0.1}"
readonly ENCODER_PORT="${ENCODER_PORT:-8779}"
readonly ENCODER_GPU="${ENCODER_GPU:-1}"
readonly ENCODER_URL="http://${ENCODER_HOST}:${ENCODER_PORT}"
readonly ENCODER_SERVED_MODEL_NAME="${ENCODER_SERVED_MODEL_NAME:-Qwen3.8-27B-encoder}"
readonly HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-kernel}"
readonly HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first}"

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
# expandable_segments reserves VA in 2MB granules; with the EAGLE draft worker's
# extra CUDA graphs this pushes process GPU-VA past dxgkrnl's ~1TB ceiling and
# every later D3DKMTReserveGpuVirtualAddress fails with EOVERFLOW (scheduler
# spins in mamba alloc_group_end). MTP mode therefore runs without it.
if [[ "${ENABLE_MTP:-0}" != "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
fi
# WSL/CUDA 13 can stall in the page-first MHA staged write-back JIT, while its
# AOT fallback can illegally access pinned host memory during eviction. Use the
# explicit PyTorch D2H memcpy fallback for MHA writes; Mamba has its own
# per-pool safe-copy path.
export SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK="${SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK:-1}"
export SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK="${SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK:-1}"
export SGLANG_ENABLE_GRAPH_POOL_PRECARVE="${SGLANG_ENABLE_GRAPH_POOL_PRECARVE:-1}"
export SGLANG_ENABLE_GRAPH_POOL_BORROW="${SGLANG_ENABLE_GRAPH_POOL_BORROW:-1}"

ENABLE_DFLASH="${ENABLE_DFLASH:-1}"
ENABLE_MTP="${ENABLE_MTP:-0}"
DSPARK_MODEL_PATH="${DSPARK_MODEL_PATH:-/root/LLM/Qwen3.8-27B-DSpark}"
SPECULATIVE_NUM_STEPS="${SPECULATIVE_NUM_STEPS:-3}"
SPECULATIVE_EAGLE_TOPK="${SPECULATIVE_EAGLE_TOPK:-1}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.985}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-500000}"
# A/B 2026-09-01 (qwen3.8, 笔记03): 并发 3->5 + mamba 16->20 + graph bs 3->5.
# 依据: 48h 真实数据显示 KV 池 p99 仅 70.3%(>95% 时刻 0.00%), retracted=0,
# mamba 从未成为约束 -> 并发被人为上限卡死, 而非容量。GLM 首版即 5/20/≤5。
# 验收: 智商冒烟 5/5 且输出哈希与基线逐字节一致; 单流 decode 52.7 vs 51.2 无衰减;
# 顿感比 0.48 vs 0.49 无恶化; 长上下文 K=5 wall 11.83->10.70s (-9.6%), 并行度 6.3->7.5。
# 已否决的替代方案: chunked_prefill_size 4096 (prefill 期 FLA 崩溃, 见笔记03 §3.1);
# prefill_max_requests 3 (无收益); --enable-prefill-delayer (running=1 时不触发, 无效)。
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-20}"
# 512 (default 256): fewer in-flight mamba checkpoints per branch -> shallower
# eviction when a Claude Code stop-hook burst (24 concurrent 64-tok requests)
# forces mamba-slot contention; halves the replay distance on prefix restore.
# Constraints ok: 512 >= num_draft_tokens(8), 512 % page_size(64) == 0.
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-512}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
PREFILL_MAX_REQUESTS="${PREFILL_MAX_REQUESTS:-1}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-5}"
# lpm (longest prefix match) instead of fcfs: when multiple requests queue,
# prefill the one sharing the longest cached prefix first. Main-session
# 200k+ prefixes win the queue over stop-hook 64-tok bursts; compat with
# spec decoding verified (policy only reorders prefill admission).
SCHEDULE_POLICY="${SCHEDULE_POLICY:-hrrn}"
# Decode-friendly scheduling: after a prefill batch, run N decode rounds
# before scheduling the next prefill. 0 = disabled. Mitigates decode
# freeze when a long prefill (another session) arrives mid-decode.
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-3}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-5}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
MAMBA_SKIP_DECODE_LOCK="${MAMBA_SKIP_DECODE_LOCK:-0}"
# HiCache (host-DRAM KV tier): ENABLED 2026-08-28 at user request. Motivation:
# multi-tool rotation (claude ~200k + codex ~100k) exceeds the 262k L1 pool, so
# switching tools forced full re-prefill (~47s @200k). L2 restores evicted
# prefixes from DRAM instead, and evicted GDN states are backed up/restored via
# MambaPoolHost. HICACHE_SIZE is split across THREE host pools (main KV /
# mamba / DFlash draft) proportionally to device pool bytes — measured split:
# 70.9% main KV (32,779 B/tok fp8), 29.3% mamba, 22.2% draft (sums >100% due
# to per-pool page rounding). 14 GB -> main KV L2 ~303k tokens (> L1 262k, as
# the code requires for full effectiveness). The earlier "prefill degradation"
# "prefill degradation" measurement was eviction write-back debt on the WSL
# slow D2H path (write_back confines it to evictions); the L2-restore benefit
# was never measured before — retest on first use.
HICACHE_SIZE="${HICACHE_SIZE:-24}"
HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_back}"

if [[ "$MAMBA_SKIP_DECODE_LOCK" == "1" ]]; then
  export SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1
fi

SGLANG_ARGS=(
  --enable-mixed-chunk
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size 1
  --trust-remote-code
  --enable-multimodal
  --language-only
  --encoder-urls "$ENCODER_URL"
  --encoder-transfer-backend zmq_to_scheduler
  --context-length 262144
  --kv-cache-dtype fp8_e4m3
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --schedule-policy "$SCHEDULE_POLICY"
  --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --attention-backend flashinfer
  --mamba-backend flashinfer
  --mamba-radix-cache-strategy extra_buffer_lazy
  --mamba-ssm-dtype bfloat16
  --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
  # PR 机制: 每链只留最新 6 个 mamba 状态(保尾弃头, KV 保留),
  # 防 LRU 把活跃大会话链啃穿 -> 84s 全量重算(9-03 事故)
  --mamba-max-states-per-path 6
  --mamba-track-interval "$MAMBA_TRACK_INTERVAL"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_coder
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --prefill-max-requests "$PREFILL_MAX_REQUESTS"
  --page-size 64
  --radix-eviction-policy lru
  --disable-prefill-cuda-graph
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
)

if [[ "$HICACHE_SIZE" -gt 0 ]]; then
  SGLANG_ARGS+=(
    --enable-hierarchical-cache
    --hicache-size "$HICACHE_SIZE"
    --hicache-io-backend "$HICACHE_IO_BACKEND"
    --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
    --hicache-write-policy "$HICACHE_WRITE_POLICY"
  )
fi
SGLANG_ARGS+=(
  --enable-cache-report
  --enable-metrics
)

if [[ "$ENABLE_DFLASH" == "1" ]]; then
  SGLANG_ARGS+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    --speculative-draft-window-size 2048
    --speculative-draft-attention-backend flashinfer
    --speculative-draft-kv-cache-dtype fp8_e4m3
  )
  # ReplaySSM spec-verify (RFC #28511 Part B) via hand-ported PR #36683:
  # DFlash commit routes through commit_mamba_states_after_verify
  # (fold-every-commit from the per-slot ring) instead of scattering
  # per-draft full-state snapshots. A/B (2026-08-29): frees ~2G VRAM
  # (idle 47.4->45.3G, decode peak ~-1.7G; NOT the ~8G the drafter
  # claimed), decode -3% (80->78 t/s single-stream), t2/t3 golden
  # identical, t1 semantic-equivalent drift@327/512 (bf16 re-quant
  # fold). Keep 12 slots: freed 2G is the safety margin, 16 slots
  # (+1.1G) would leave no headroom. Revisit fp32 SSM (+3.2G pool) or
  # 16 slots after multi-day drift observation.
  ENABLE_LINEAR_REPLAYSSM_SPEC="${ENABLE_LINEAR_REPLAYSSM_SPEC:-1}"
  if [[ "$ENABLE_LINEAR_REPLAYSSM_SPEC" == "1" ]]; then
    SGLANG_ARGS+=(--enable-linear-replayssm-spec)
  fi
elif [[ "$ENABLE_DSPARK" == "1" ]]; then
  # RadixArk trained DSpark drafter (chain draft, confidence-gated depth).
  # Verify window = block_size(7, auto-inferred) + 1 = 8, matching DFlash K=8.
  SGLANG_ARGS+=(
    --speculative-algorithm DSPARK
    --speculative-draft-model-path "$DSPARK_MODEL_PATH"
    --speculative-draft-model-quantization unquant
    --speculative-draft-attention-backend flashinfer
  )
elif [[ "$ENABLE_MTP" == "1" ]]; then
  # In-checkpoint MTP head (mtp.* weights in the target FP8 checkpoint),
  # EAGLE algorithm, linear-chain topk=1. Draft KV pool is a separate
  # 1-layer full-attn fp8 pool sized like the main pool (~0.55GB extra).
  # MTP_TEST_MEM_FRACTION: the draft pool allocation happens after the main
  # pools; on WSL/dxgkrnl a tight 0.985 leaves the post-pool small allocations
  # failing in D3DKMTReserveGpuVirtualAddress (scheduler spin). Reserve slack.
  SGLANG_ARGS+=(
    --speculative-algorithm EAGLE
    --speculative-draft-model-path "$MODEL_PATH"
    --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK"
    --speculative-num-draft-tokens $((SPECULATIVE_NUM_STEPS + SPECULATIVE_EAGLE_TOPK))
    --mem-fraction-static "${MTP_MEM_FRACTION_STATIC:-0.94}"
  )
elif [[ "$ENABLE_MTP" == "1" || "$ENABLE_DSPARK" == "1" ]]; then
  echo "ENABLE_DFLASH=0 required when ENABLE_MTP/ENABLE_DSPARK=1; got ENABLE_DFLASH=$ENABLE_DFLASH" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  SGLANG_ARGS+=(--api-key "$SGLANG_API_KEY")
fi

# 实验钩子(qwen3.8 2026-09-01 加入): 追加任意 sglang 参数, 默认空=无副作用。
# 用法: SGLANG_EXTRA_ARGS="--enable-prefill-delayer" bash restart-main-sglang.sh
if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
  read -r -a _extra_args <<< "$SGLANG_EXTRA_ARGS"
  SGLANG_ARGS+=("${_extra_args[@]}")
fi

ENCODER_ARGS=(
  --model-path "$ENCODER_MODEL_PATH"
  --served-model-name "$ENCODER_SERVED_MODEL_NAME"
  --host "$ENCODER_HOST"
  --port "$ENCODER_PORT"
  --tp-size 1
  --trust-remote-code
  --encoder-only
  --encoder-transfer-backend zmq_to_scheduler
)

# NVFP4 checkpoint (SM89/4090): fused SiLU+mul+FP4-quant kernel has no Ada
# backend in flashinfer (raises "Invalid backend: 89"). Fusion is a pure
# optimization; disable it and MLP down_proj falls back to standard path
# (act_fn -> Marlin W4A16). Harmless for FP8 checkpoints (gate never fires).
export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1
