#!/usr/bin/env bash
# ============================================================
# sglang · Qwen3.8-27B int4 · 双 RTX 3080 推荐配置
# 说明：本文件是从「单卡 4090 + FP8」的经验外推到「双 3080 + int4」。
#       默认 TP=1 / 不开投机，先跑通基本盘；稳了再逐项改大。
#       所有路径、显存数字都要按你机器实际替换。
# ============================================================

readonly SGLANG_VENV="${SGLANG_VENV:-/root/sglang-env}"
# >>> 改成你的 int4 模型实际路径 <<<
readonly MODEL_PATH="${MODEL_PATH:-/root/LLM/Qwen3.8-27B-INT4}"
# >>> 投机草稿模型路径（后期上 DFLASH 时才有用，先留空）<<<
readonly DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/root/LLM/Qwen3.8-27B-DFlash2}"
# sglang 只支持单个模型名，别名都不认，改完客户端要跟着改
readonly SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gpt5.6 sol}"
readonly HOST="${HOST:-0.0.0.0}"
readonly PORT="${PORT:-8778}"

# >>> 关键开关 <<<
# TP_SIZE：先 1（单卡），稳了再 2（双卡分片）
TP_SIZE="${TP_SIZE:-1}"
# 上下文长度：先 131072，显存够再往 262144 拉
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-131072}"
# 显存占比：双卡偏保守，别用 4090 的 0.95
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
# 投机：默认关，最后一步才开（ENABLE_DFLASH=1 后配好草稿模型）
ENABLE_DFLASH="${ENABLE_DFLASH:-0}"
# 只要你一张卡（TP=1）时，把这改成你实际用的那张卡的号，比如 0；TP=2 时用 0,1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# DeltaNet 工作区是临时分配，不开会碎片化 OOM（TP=2 时如遇虚拟地址报错可去掉）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4}"
SCHEDULE_POLICY="${SCHEDULE_POLICY:-lpm}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-2048}"
PREFILL_MAX_REQUESTS="${PREFILL_MAX_REQUESTS:-1}"
PAGE_SIZE="${PAGE_SIZE:-64}"
CUDA_GRAPH_MAX_BS_DECODE="${CUDA_GRAPH_MAX_BS_DECODE:-4}"
SPECULATIVE_NUM_DRAFT_TOKENS="${SPECULATIVE_NUM_DRAFT_TOKENS:-8}"
MAX_MAMBA_CACHE_SIZE="${MAX_MAMBA_CACHE_SIZE:-12}"
MAMBA_TRACK_INTERVAL="${MAMBA_TRACK_INTERVAL:-512}"

SGLANG_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$PORT"
  --tp-size "$TP_SIZE"
  --trust-remote-code
  --context-length "$CONTEXT_LENGTH"
  --kv-cache-dtype fp8_e4m3
  --max-running-requests "$MAX_RUNNING_REQUESTS"
  --schedule-policy "$SCHEDULE_POLICY"
  --max-total-tokens "$MAX_TOTAL_TOKENS"
  --mem-fraction-static "$MEM_FRACTION_STATIC"
  --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
  --prefill-max-requests "$PREFILL_MAX_REQUESTS"
  --page-size "$PAGE_SIZE"
  --radix-eviction-policy lru
  --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS_DECODE"
  --enable-prefix-caching
  --enable-cache-report
  --enable-metrics
)

# 混合线性注意力模型相关：如果你的 sglang 构建支持这些 flags 就带上，不支持就去掉 / 降级为默认
# 若该构建支持 mamba / linear attention 后端，可追加：
#   --attention-backend flashinfer
#   --mamba-backend flashinfer
#   --mamba-radix-cache-strategy extra_buffer_lazy
#   --mamba-ssm-dtype bfloat16
#   --max-mamba-cache-size "$MAX_MAMBA_CACHE_SIZE"
#   --mamba-track-interval "$MAMBA_TRACK_INTERVAL"

if [[ "$ENABLE_DFLASH" == "1" ]]; then
  SGLANG_ARGS+=(
    --speculative-algorithm DFLASH
    --speculative-draft-model-path "$DRAFT_MODEL_PATH"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
    --speculative-draft-window-size 2048
    --speculative-draft-attention-backend flashinfer
    --speculative-draft-kv-cache-dtype fp8_e4m3
  )
fi

if [[ -n "${SGLANG_API_KEY:-}" ]]; then
  SGLANG_ARGS+=(--api-key "$SGLANG_API_KEY")
fi