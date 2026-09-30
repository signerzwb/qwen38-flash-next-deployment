#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sglang 启动后补 warmup — 提前触发 serving-time lazy kernel 加载。

背景(笔记07): 默认 warmup 仅 "The capital city of France is"(~6 token)+8 decode,
走不到 chunked-GDN prefill / DFlash 完整 verify / fp8 大 shape / 多请求 batch KV 簿记,
导致 27 个 Triton kernel 在第一个真实请求时才 device-load(彼时余量已 <1GB, OOM 隐患)。

本脚本用 nonce 前缀(不污染真实 radix)+ 覆盖 5 组分支的请求集, 在服务刚起、
余量最高时把这些 kernel 全部提前加载。结束 flush_cache 清污染。

用法: /root/sglang-env/bin/python sglang-warmup.py [--base-url http://127.0.0.1:8778]
退出码 0=全部请求成功; 非0=有失败(调用方决定是否继续)。"""
import argparse, json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

def post(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def gen(base, text, max_new):
    return post(base + "/generate", {
        "text": text,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new},
    })

# 一段稳定文本, 重复凑长 prefill(触发 chunked GDN / fp8 大 shape / KV 簿记)
UNIT = ("The quick brown fox jumps over the lazy dog while the system processes "
        "a long contextual passage to exercise chunked prefill and gated delta "
        "attention kernels across multiple 2048-token chunks in sequence. ")

def nonce():
    # 时间戳+计数做前缀, 保证 warmup 前缀唯一, 不与真实会话 radix 共享
    return f"[warmup-{time.time_ns()}] "

def run_case(name, base, text, max_new):
    t0 = time.time()
    try:
        r = gen(base, text, max_new)
        pt = r.get("meta_info", {}).get("prompt_tokens", "?")
        print(f"  [{name}] OK prompt_tokens={pt} {time.time()-t0:.1f}s")
        return True
    except Exception as e:
        print(f"  [{name}] FAIL {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8778")
    ap.add_argument("--long-prefill-tokens", type=int, default=120,
                    help="UNIT 重复次数(每次~30 token), 120≈3600 token 覆盖多 chunk")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    # 等 health(最多 120s)
    for _ in range(120):
        try:
            urllib.request.urlopen(base + "/health", timeout=3); break
        except Exception:
            time.sleep(1)
    else:
        print("health 超时, 放弃 warmup"); return 2

    long_prompt = nonce() + UNIT * args.long_prefill_tokens
    ok = True

    print("warmup 请求集:")
    # ① 长 prefill + 短 decode → chunked GDN prefill / fp8 大 shape / KV 簿记
    ok &= run_case("长prefill", base, long_prompt, 8)
    # ② 短 prefill + 长 decode → DFlash verify/accept/bonus/replayssm-fold 多步
    ok &= run_case("长decode", base, nonce() + "Explain the theory of relativity in detail.", 256)
    # ③ 长 prefill + 长 decode → compact draft rebuild(window 截断)
    ok &= run_case("compact-rebuild", base, long_prompt, 128)

    # ④ 并发 5 路(=max_running) → batch KV 簿记 / assign_req_to_token / get_last_loc
    print("  [并发5路] ...")
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(gen, base, nonce() + UNIT * 30, 64) for _ in range(5)]
        n_ok = 0
        for f in futs:
            try:
                f.result(); n_ok += 1
            except Exception as e:
                print(f"    并发路失败: {e}")
        print(f"  [并发5路] {n_ok}/5 成功")
        ok &= (n_ok == 5)

    # flush 清 warmup 前缀污染(该端点返回空 body, 不解析 JSON)
    try:
        req = urllib.request.Request(base + "/flush_cache", data=b"{}",
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"  flush_cache OK (HTTP {r.status})")
    except Exception as e:
        print(f"  flush_cache 失败(非致命): {e}")

    print("warmup 完成:", "全部成功" if ok else "有失败")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
