#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGLang 压测请求生成器（专门线程打请求，用于验证 kv-monitor 各面板）

用法:  python loadgen.py [SGLANG地址] [并发数] [持续秒数]
默认:  192.168.88.123:8778  2  90

负载设计:
- 全部请求共享同一 ~4k token 中文前缀 → 第 1 个之后 prefill 走 device_hit（验证命中率/绿线）
- 每请求附加 300~6000 token 随机正文 → prompt 长度直方图铺开多桶
- max_tokens 80/200/500 随机 → generation 直方图铺开
- 多 worker 并发 → running 计数 2~3, 可触发排队
"""
import json
import random
import sys
import threading
import time
import urllib.request

SGLANG = sys.argv[1] if len(sys.argv) > 1 else "192.168.88.123:8778"
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
DURATION = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0

STOP = False
DONE = {"n": 0, "err": 0, "lat": []}
DONE_LOCK = threading.Lock()

# ~3000 汉字 ≈ 4k+ token 的共享前缀（重复句保证 token 化稳定）
PREFIX = ("你是一名资深推理引擎工程师，正在协助排查 SGLang 与 vLLM 的调度行为。"
          "请始终先给出结论，再给出证据，证据必须来自日志与指标。"
          "讨论 KV cache 时区分 running 占用与 radix 驻留两种状态；"
          "讨论吞吐时区分 prefill 计算速率与前缀缓存免计算速率。" * 22)


def filler(n_chars):
    pool = "负载正文片段" * n_chars
    return pool[:n_chars]


def build_prompt():
    size = random.choice([300, 800, 2000, 5000])  # 字 → token 量级 300~6000+
    return (PREFIX + "\n\n【附加上下文】" + filler(size * 2) +
            "\n\n请基于以上材料，用不超过三句话总结 KV cache 调优要点。")


def worker():
    rng = random.Random()
    while not STOP:
        prompt = build_prompt()
        max_tokens = rng.choice([80, 200, 500])
        body = json.dumps({
            "model": "Qwen3.8-27B",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            "http://" + SGLANG + "/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
            prompt_tokens = data.get("usage", {}).get("prompt_tokens")
            gen_tokens = data.get("usage", {}).get("completion_tokens")
            with DONE_LOCK:
                DONE["n"] += 1
                DONE["lat"].append(time.time() - t0)
                DONE["lat"] = DONE["lat"][-200:]
            print("[loadgen] ok  prompt=%s gen=%s lat=%.1fs"
                  % (prompt_tokens, gen_tokens, time.time() - t0), flush=True)
        except Exception as e:
            with DONE_LOCK:
                DONE["err"] += 1
            print("[loadgen] err %s: %s" % (type(e).__name__, e), flush=True)
            time.sleep(1)


def reporter():
    global STOP
    t_end = time.time() + DURATION
    while time.time() < t_end:
        time.sleep(10)
        with DONE_LOCK:
            n, err = DONE["n"], DONE["err"]
            lats = DONE["lat"]
        avg = sum(lats[-50:]) / len(lats[-50:]) if lats else 0
        print("[loadgen] t+%.0fs  ok=%d err=%d avg_lat=%.1fs"
              % (DURATION - (t_end - time.time()), n, err, avg), flush=True)
    STOP = True


def main():
    print("[loadgen] workers=%d duration=%.0fs target=%s" % (WORKERS, DURATION, SGLANG), flush=True)
    ths = [threading.Thread(target=worker, daemon=True) for _ in range(WORKERS)]
    for t in ths:
        t.start()
    reporter()
    for t in ths:
        t.join(timeout=3)
    with DONE_LOCK:
        print("[loadgen] done  ok=%d err=%d" % (DONE["n"], DONE["err"]), flush=True)


if __name__ == "__main__":
    main()
