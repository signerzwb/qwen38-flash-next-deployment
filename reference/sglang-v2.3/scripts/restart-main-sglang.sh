#!/usr/bin/env bash
# Restart ONLY the main sglang service (encoder stays up), wait for health, then warm up.
# 补 warmup(笔记07): 默认 server warmup 仅 ~6 token, 走不到 chunked-GDN/DFlash verify/
# fp8 大 shape/多请求 batch 分支 → 28 个 Triton kernel 在第一个真实请求时才 device-load
# (彼时余量常 <1GB, 有 cuModuleLoadData OOM 隐患)。启动后主动打代表性请求集提前加载。
set -u
LOG=/root/sglang-qwen38-dflash2/main-boot.log
WARMUP=/root/sglang-warmup.py
VENV_PY=/root/sglang-env/bin/python

# 编码器前置检查(9-02 事故): WSL 重启后编码器无人拉起, 主服务 bootstrap 连不上
# 8779 会静默逐出并在 600s 后永久剔除; 之后图片请求降级本地编码, 而 --language-only
# 主进程无视觉塔 → AttributeError 团灭。因此主服务启动前必须保证 8779 健康。
if ! curl -fsS --max-time 2 http://127.0.0.1:8779/health >/dev/null 2>&1; then
    echo "ENCODER DOWN, starting it first ..."
    bash /root/start-sglang-qwen38-encoder.sh || {
        echo "ENCODER FAILED TO START, aborting main restart (would get evicted anyway)"
        exit 1
    }
fi

bash /root/stop-sglang-qwen38-dflash2.sh >/dev/null 2>&1
setsid bash -c "nohup bash /root/start-sglang-qwen38-dflash2.sh > ${LOG} 2>&1 &" </dev/null >/dev/null 2>&1 &
for i in $(seq 1 90); do
    sleep 6
    if curl -fsS --max-time 2 http://127.0.0.1:8778/health >/dev/null 2>&1; then
        echo "MAIN HEALTHY after ~$((i*6))s"
        # warmup 非致命: 失败仍视为启动成功(服务已健康), 仅告警不阻断。
        if [[ -f "$WARMUP" ]]; then
            echo "running post-start warmup ..."
            "$VENV_PY" "$WARMUP" || echo "WARN: warmup had failures (non-fatal)"
        fi
        exit 0
    fi
done
echo "MAIN TIMEOUT"
tail -30 "${LOG}"
exit 1
