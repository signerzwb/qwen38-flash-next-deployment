#!/usr/bin/env bash
# Restart ONLY the main sglang service (encoder stays up) and wait for health.
set -u
LOG=/root/sglang-qwen38-dflash2/main-boot.log

bash /root/stop-sglang-qwen38-dflash2.sh >/dev/null 2>&1
setsid bash -c "nohup bash /root/start-sglang-qwen38-dflash2.sh > ${LOG} 2>&1 &" </dev/null >/dev/null 2>&1 &
for i in $(seq 1 90); do
    sleep 6
    if curl -fsS --max-time 2 http://127.0.0.1:8778/health >/dev/null 2>&1; then
        echo "MAIN HEALTHY after ~$((i*6))s"
        exit 0
    fi
done
echo "MAIN TIMEOUT"
tail -30 "${LOG}"
exit 1
