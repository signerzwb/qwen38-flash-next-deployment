#!/usr/bin/env bash
# Boot the full sglang Qwen3.8 stack: encoder (GPU1, :8779) then LM (GPU0, :8778).
# Fully detached from the calling session.
set -u
LOG_DIR=/root/sglang-qwen38-dflash2

bash /root/start-sglang-qwen38-encoder.sh > "${LOG_DIR}/encoder-boot.log" 2>&1 &
ENC_WAIT=$!
for i in $(seq 1 60); do
    kill -0 "$ENC_WAIT" 2>/dev/null || break
    curl -fsS --max-time 2 http://127.0.0.1:8779/health >/dev/null 2>&1 && break
    sleep 5
done
if ! curl -fsS --max-time 2 http://127.0.0.1:8779/health >/dev/null 2>&1; then
    echo "ENCODER FAILED"; tail -30 "${LOG_DIR}/encoder-boot.log"; exit 1
fi
echo "Encoder healthy."

setsid bash -c "nohup bash /root/start-sglang-qwen38-dflash2.sh > ${LOG_DIR}/main-boot.log 2>&1 &" </dev/null >/dev/null 2>&1 &
for i in $(seq 1 100); do
    sleep 6
    if curl -fsS --max-time 2 http://127.0.0.1:8778/health >/dev/null 2>&1; then
        echo "MAIN HEALTHY after ~$((i*6))s"
        exit 0
    fi
done
echo "MAIN TIMEOUT"; tail -40 "${LOG_DIR}/main-boot.log"
exit 1
