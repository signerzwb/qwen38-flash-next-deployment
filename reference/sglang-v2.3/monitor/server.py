#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGLang KV Cache 实时监控后端（仅标准库，零依赖）

用法:  python server.py [SGLANG地址] [监听端口]
默认:  192.168.88.123:8778  →  监听 0.0.0.0:8917

数据源: SGLang /metrics (Prometheus 文本) + /get_server_info
页面:   /  →  仪表盘 (index.html)
API:    /api/snapshot  最新采样   /api/history  环形缓冲历史
"""
import json
import os
import re
import subprocess
import sys
import time
import threading
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SGLANG = sys.argv[1] if len(sys.argv) > 1 else "192.168.88.123:8778"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8917
INTERVAL = 1.0          # 采样间隔 (s)
MAX_SAMPLES = 1800      # 环形缓冲: 30 分钟 @1s
HERE = Path(__file__).resolve().parent

LAB = re.compile(r'(\w+)="([^"]*)"')


def get(url, timeout=4):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse(text):
    """Prometheus 文本 → {metric_name: [(labels_dict, value), ...]}"""
    d = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        body, _, val = line.rpartition(" ")
        try:
            v = float(val)
        except ValueError:
            continue
        name, labs = body, {}
        i = body.find("{")
        if i > 0:
            name = body[:i]
            labs = dict(LAB.findall(body[i + 1:-1]))
        if ":" in name and not name.startswith("__name__"):
            name = name.split(":", 1)[1]  # 去掉 "sglang:" 等命名空间前缀
        d.setdefault(name, []).append((labs, v))
    return d


def g1(d, n):
    s = d.get(n)
    return s[0][1] if s else None


def gl(d, n, k, v):
    for labs, val in d.get(n, []):
        if labs.get(k) == v:
            return val
    return None


def hist(d, n):
    """histogram → {"buckets": [[le_str, cum], ...], "sum":, "count":}"""
    buckets = []
    for labs, v in d.get(n + "_bucket", []):
        le = labs.get("le", "+Inf")
        try:
            f = float(le)
        except ValueError:
            f = float("inf")
        buckets.append((f, le, v))
    buckets.sort()
    return {"buckets": [[le, v] for _, le, v in buckets],
            "sum": g1(d, n + "_sum"),
            "count": g1(d, n + "_count")}


def rate(prev, cur, dt):
    """counter 差分速率; 回绕/缺数据返回 None"""
    if prev is None or cur is None or dt <= 0:
        return None
    dc = cur - prev
    return dc / dt if dc >= 0 else None


def compute(d, prev, dt, prev_s=None):
    s = {"t": time.time()}

    mx = g1(d, "max_total_num_tokens")
    used = g1(d, "kv_used_tokens")
    evict = g1(d, "kv_evictable_tokens")
    s["kv"] = {
        "max": mx, "used": used, "evictable": evict,
        "free": max(0.0, mx - used - evict) if None not in (mx, used, evict) else None,
        "usage_pct": (used / mx * 100.0) if mx else None,
        "mem_gb": g1(d, "kv_cache_memory_usage_gb"),
        "weight_gb": g1(d, "weight_memory_usage_gb"),
        "graph_gb": g1(d, "graph_memory_usage_gb"),
    }
    s["gen_tps"] = g1(d, "gen_throughput")
    s["running"] = g1(d, "num_running_reqs")
    s["queued"] = g1(d, "num_queue_reqs")
    s["retracted"] = g1(d, "num_retracted_reqs")

    # 分层缓存 (hicache) 与 mamba 状态
    backup_kv = gl(d, "hicache_backup_tokens_total", "pool", "kv")
    backup_mamba = gl(d, "hicache_backup_tokens_total", "pool", "mamba")
    s["hicache"] = {
        "host_total": g1(d, "hicache_host_total_tokens"),
        "host_used": g1(d, "hicache_host_used_tokens"),
        "backup_tok": backup_kv,
        "backup_mamba": backup_mamba,   # 注意: mamba 池计的是槽备份次数, 非 token 数
        "backup_bytes": g1(d, "hicache_backup_bytes_total"),
        "backup_dur_sum": g1(d, "hicache_backup_duration_seconds_sum"),
        "backup_dur_n": g1(d, "hicache_backup_duration_seconds_count"),
    }
    # L2→GPU 回取 (host_hit 恢复路径的量与耗时)
    load_kv = gl(d, "load_back_tokens_total", "pool", "kv")
    load_mamba = gl(d, "load_back_tokens_total", "pool", "mamba")
    s["hicache"]["load_tok"] = load_kv
    s["hicache"]["load_mamba"] = load_mamba
    s["hicache"]["load_bytes"] = g1(d, "load_back_bytes_total")
    s["hicache"]["load_dur_sum"] = g1(d, "load_back_duration_seconds_sum")
    s["hicache"]["load_dur_n"] = g1(d, "load_back_duration_seconds_count")
    s["hicache"]["load_dur_hist"] = hist(d, "load_back_duration_seconds")
    drop_kv = gl(d, "hicache_dropped_tokens_total", "pool", "kv")
    drop_mamba = gl(d, "hicache_dropped_tokens_total", "pool", "mamba")
    s["hicache_drop"] = (drop_kv or 0.0) + (drop_mamba or 0.0)
    # 逐出总闸: L1 逐出的 token 槽位累计（无论备份成功与否）
    s["evicted"] = g1(d, "evicted_tokens_total")
    # 引擎自报前缀命中率（滑动窗口, 与前端 30min 窗口差分互为交叉验证）
    s["engine_hit_rate"] = g1(d, "cache_hit_rate")
    s["mamba"] = {
        "total": (g1(d, "mamba_used_tokens") or 0.0) + (g1(d, "mamba_evictable_tokens") or 0.0) + (g1(d, "mamba_available_tokens") or 0.0),
        "used": g1(d, "mamba_used_tokens"),
        "evictable": g1(d, "mamba_evictable_tokens"),
        "free": g1(d, "mamba_available_tokens"),
    }
    s["spec_accept"] = g1(d, "spec_accept_length")
    s["spec_rate"] = g1(d, "spec_accept_rate")
    # 排队时间（HRRN 调度的核心观测量）：请求从入队到被调度的等待
    s["queue_t"] = {"sum": g1(d, "queue_time_seconds_sum"),
                    "count": g1(d, "queue_time_seconds_count")}
    # 端到端延迟（含 decode），与 TTFT 互补
    s["e2e"] = {"sum": g1(d, "e2e_request_latency_seconds_sum"),
                "count": g1(d, "e2e_request_latency_seconds_count")}
    # 分阶段耗时: request_process(网关处理) / prefill_forward / chunked_prefill
    s["stage"] = {st: {"sum": gl(d, "per_stage_req_latency_seconds_sum", "stage", st),
                       "count": gl(d, "per_stage_req_latency_seconds_count", "stage", st)}
                  for st in ("request_process", "prefill_forward", "chunked_prefill")}
    # 引擎侧请求计数（区分流式/非流式），用于核对面板窗口
    s["req_total"] = g1(d, "num_requests_total")

    if prev:
        # 保留结账计数差分供长周期统计(API 暴露, 面板不显示):
        # spec_verify_calls_total 是"按请求结账"counter(请求结束才一次性
        # inc 一生的轮数), 差分出来的是结账脉冲不是节奏。
        s["spec_verify_tps"] = rate(g1(prev, "spec_verify_calls_total"),
                                    g1(d, "spec_verify_calls_total"), dt)
        s["pre_tps"] = rate(gl(prev, "realtime_tokens_total", "mode", "prefill_compute"),
                            gl(d, "realtime_tokens_total", "mode", "prefill_compute"), dt)
        s["pre_cache_tps"] = rate(gl(prev, "realtime_tokens_total", "mode", "prefill_cache"),
                                  gl(d, "realtime_tokens_total", "mode", "prefill_cache"), dt)
        s["dec_tps"] = rate(gl(prev, "realtime_tokens_total", "mode", "decode"),
                            gl(d, "realtime_tokens_total", "mode", "decode"), dt)
        # 真·实时 verify 节奏 = 引擎自报 decode 吞吐 / 接受长度。
        # 用 gen_throughput gauge（引擎在 decode 日志点按窗口平滑好）而非
        # realtime_tokens_total 差分——后者同为 40 步日志点脉冲, 中间帧
        # dc=0 会把节奏打成 0; accept 优先用上一帧值与吞吐窗口对齐。
        acc = (prev_s.get("spec_accept") if prev_s else None) or s["spec_accept"]
        tp = s["gen_tps"]
        s["verify_hz"] = (tp / acc) if (tp and acc) else (0.0 if tp else None)

        hb = s["hicache"]["backup_tok"]
        pb = gl(prev, "hicache_backup_tokens_total", "pool", "kv")
        s["hicache_backup_tps"] = rate(pb, hb, dt) if hb is not None else None
        s["hicache_backup_gbps"] = rate(g1(prev, "hicache_backup_bytes_total"),
                                        s["hicache"]["backup_bytes"], dt)
        lb = s["hicache"]["load_tok"]
        plb = gl(prev, "load_back_tokens_total", "pool", "kv")
        s["hicache_load_tps"] = rate(plb, lb, dt) if lb is not None else None
        s["hicache_load_gbps"] = rate(g1(prev, "load_back_bytes_total"),
                                      s["hicache"]["load_bytes"], dt)
        # 逐出速率: L1 每秒被逐出的 token 槽位（备份+drop 的总源头）
        s["evict_tps"] = rate(g1(prev, "evicted_tokens_total"), s["evicted"], dt)

    # 窗口类指标（命中率/TTFT/ITL/直方图）的累计原值，由前端按 30 分钟窗口差分
    s["eff"] = {m: gl(d, "prefill_effective_tokens_total", "mode", m)
                for m in ("input", "device_hit", "host_hit", "storage_hit")}
    for src, key in (("time_to_first_token_seconds", "ttft"),
                     ("inter_token_latency_seconds", "itl")):
        s[key + "_sum"] = g1(d, src + "_sum")
        s[key + "_count"] = g1(d, src + "_count")

    s["histo"] = {"p": hist(d, "prompt_tokens_histogram"),
                  "g": hist(d, "generation_tokens_histogram")}
    return s


STATE = {"samples": deque(maxlen=MAX_SAMPLES), "info": None, "error": None, "gpus": None,
         "prefill_events": deque(maxlen=200), "draft_graph": None, "boot_mem": None}
LOCK = threading.Lock()

INFO_KEYS = ["model_path", "context_length", "kv_cache_dtype", "tp_size", "dp_size",
             "max_running_requests", "max_total_tokens", "chunked_prefill_size",
             "max_prefill_tokens", "page_size", "mem_fraction_static",
             "schedule_policy", "disable_radix_cache", "radix_eviction_policy",
             "retraction_policy", "attention_backend",
             "enable_hierarchical_cache", "hicache_host_memory_mode",
             "hicache_io_backend", "hicache_write_policy", "hicache_storage_backend",
             "max_mamba_cache_size", "mamba_ssm_dtype",
             "speculative_num_draft_tokens", "speculative_draft_window_size",
             "speculative_draft_model_path",
             "enable_linear_replayssm_spec", "enable_mixed_chunk",
             "prefill_decode_interval"]

# 请求级 prefill 归因：tail sglang 主服务日志里的 PREFILL-REQ 行。
# 从 Windows 侧读 WSL UNC 路径（\\wsl.localhost\...），只做增量 tail，
# 失败静默（面板不显示归属，仅 /api/prefill_events 暴露原始事件）。
SGLANG_LOG = r"\\wsl.localhost\ubuntu2204\root\sglang-qwen38-dflash2\main-boot.log"
PREFILL_RE = re.compile(
    r"\[(?P<ts>[\d\- :]+)\] PREFILL-REQ rid=(?P<rid>\S+) input=(?P<input>\d+) "
    r"chunk=(?P<start>\d+)\+(?P<len>\d+) end=(?P<end>\d+) "
    r"hit_device=(?P<hd>\d+) hit_host=(?P<hh>\d+) "
    r"salt=(?P<salt>\S+) mamba_branch=(?P<mb>\S+) mamba_host_hit=(?P<mh>\d+)")


def prefill_events_thread():
    """tail SGLANG_LOG，解析 PREFILL-REQ 行进 STATE["prefill_events"]（deque 200）。
    断点续读：同 inode 偏移继续；日志轮转/截断（size 变小）则重开。
    9-03 修复：服务重启会截断重写日志；若新日志快速增长越过旧偏移，
    size<pos 判据永远不触发 → 线程永远读旧 EOF。加身份指纹（创建时间）
    检测文件被替换/截断，强制重开回读。"""
    pos = None
    identity = None
    while True:
        try:
            st = os.stat(SGLANG_LOG)
            cur_identity = (st.st_ctime, st.st_size // (4 * 1048576))
            with open(SGLANG_LOG, "rb") as f:
                size = f.seek(0, 2)
                if pos is None or size < pos or (identity is not None and cur_identity[0] != identity):
                    pos = max(0, size - 2 * 1048576)   # 首次/轮转/替换：回读最近 2MB
                identity = cur_identity[0]
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
                for line in chunk.splitlines():
                    m = PREFILL_RE.search(line.decode("utf-8", "replace"))
                    if m:
                        with LOCK:
                            STATE["prefill_events"].append(m.groupdict())
        except FileNotFoundError:
            pos = None
            identity = None
        except Exception:
            pass
        time.sleep(2.0)


# draft CUDA graph 状态: 从 boot log 一次性判定(每 60s 重扫直至命中)。
# 禁用行 / folded 行 / 都没有(可能 eager fallback 无日志) → "off"/"on"/None
DRAFT_GRAPH_OFF_RE = re.compile(r"Disable DFLASH draft cuda graph")
DRAFT_GRAPH_ON_RE = re.compile(r"DFLASH (?:selector decode .*|draft greedy head.*)folded into the draft cuda graph")

# 编码器(8779)健康线(9-02 事故): 编码器缺位时主服务 bootstrap 会静默逐出
# (600s 后 permanently dropped), 纯文本流量零感知, 直到一张图片触发降级
# → --language-only 主进程 AttributeError 团灭。三路信号:
#   enc_ok      8779 /health 实时探测
#   enc_evicted 主日志出现 Health check evicted/dropped (逐出告警, 即使后来
#               重新注册也保留——发生过逐出就该查)
#   enc_down_s  8779 连续不可达秒数(>0 即面板告警)
ENCODER_URL = "http://127.0.0.1:8779/health"
ENC_EVICT_RE = re.compile(r"Health check (?:evicted|permanently dropped) .*encoder", re.I)

# ---- 启动日志显存实测（替代硬编码常量）----
# GPU0 的显存组成里，权重/KV池有指标直供，但 mamba 槽 / draft 权重 / draft KV
# 三者只有启动日志有精确值。早期版本用 0.27GB/槽 的硬编码常量，在
# max_mamba_cache_size 20 时错算 3.8GB（实测 81MB/槽）；draft 的 8.4GB 更是
# 整块落进"工作区"倒推。这里改为逐行解析 boot log，拿不到才回退常量。
BOOT_MAIN_W_RE = re.compile(
    r"Load weight end\..*?type=(\w+).*?mem usage=([\d.]+) GB")
BOOT_MAMBA_RE = re.compile(
    r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+), "
    r"conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB "
    r"intermediate_ssm_state_cache size: ([\d.]+)GB "
    r"intermediate_conv_window_cache size: ([\d.]+)GB")
BOOT_KV_RE = re.compile(
    r"KV Cache is allocated\. dtype: \S+, #tokens: (\d+), "
    r"K size: ([\d.]+) GB, V size: ([\d.]+) GB")
BOOT_GRAPH_RE = re.compile(r"Capture (\w+) (?:verify )?CUDA graph end\..*?mem usage=([\d.]+) GB")


def parse_boot_mem():
    """扫 boot log 尾部，返回实测显存组成（GB）。

    同一个进程会打印两段 KV Cache / Load weight：先主模型后 draft。按出现
    顺序归属，不靠名字猜。
    """
    try:
        with open(SGLANG_LOG, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 4 * 1048576))
            text = f.read().decode("utf-8", "replace")
    except Exception:
        return None

    out = {"main_w": None, "draft_w": None, "mamba_per_slot": None,
           "mamba_total": None, "kv_main": None, "kv_draft": None,
           "graph": 0.0}

    for m in BOOT_MAIN_W_RE.finditer(text):
        typ, gb = m.group(1), float(m.group(2))
        if typ == "DFlash2DraftModel" or "Draft" in typ:
            out["draft_w"] = gb
        elif out["main_w"] is None:
            out["main_w"] = gb

    m = None
    for m in BOOT_MAMBA_RE.finditer(text):
        pass
    if m:
        slots = int(m.group(1))
        tot = sum(float(m.group(i)) for i in (2, 3, 4, 5))
        out["mamba_total"] = tot
        out["mamba_per_slot"] = (tot / slots) if slots else None

    kv = [(int(a), float(b) + float(c)) for a, b, c in BOOT_KV_RE.findall(text)]
    if kv:
        out["kv_main"] = kv[0][1]
        if len(kv) > 1:
            out["kv_draft"] = kv[1][1]

    for name, gb in BOOT_GRAPH_RE.findall(text):
        try:
            out["graph"] += float(gb)
        except ValueError:
            pass

    return out if any(out[k] is not None for k in ("main_w", "mamba_per_slot")) else None


def boot_mem_thread():
    """boot log 每次被重写（服务重启）就重扫一次显存组成。"""
    last_id = None
    while True:
        try:
            st = os.stat(SGLANG_LOG)
            if st.st_ctime != last_id:
                got = parse_boot_mem()
                if got:
                    with LOCK:
                        STATE["boot_mem"] = got
                    last_id = st.st_ctime
        except Exception:
            pass
        time.sleep(15.0)


def encoder_thread():
    while True:
        enc_ok = False
        try:
            urllib.request.urlopen(ENCODER_URL, timeout=2).read(1)
            enc_ok = True
        except Exception:
            pass
        try:
            with open(SGLANG_LOG, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4 * 1048576))
                text = f.read().decode("utf-8", "replace")
            evicted = bool(ENC_EVICT_RE.search(text))
        except Exception:
            evicted = None
        with LOCK:
            st = STATE.get("encoder")
            if st is None or not enc_ok:
                # 首次记录或当前不可达: 重置连续不可达计时
                down_since = (st or {}).get("down_since") or time.time()
                down_s = time.time() - down_since if not enc_ok else 0.0
                STATE["encoder"] = {"ok": enc_ok, "evicted": evicted,
                                    "down_s": down_s,
                                    "down_since": down_since if not enc_ok else None,
                                    "t": time.time()}
            else:
                st["ok"] = True
                st["evicted"] = evicted if evicted is not None else st["evicted"]
                st["down_s"] = 0.0
                st["down_since"] = None
                st["t"] = time.time()
        time.sleep(5.0)


def attrib_hit_rate_snapshot():
    """按 PREFILL-REQ 归因行算窗口命中率: 每请求 rid 去重后
    (hit_device+hit_host 截断到 input) 求和 / input 求和。
    天然 <=100%, 不受引擎计数器 chunk 重试重复累加的口径失真影响。"""
    with LOCK:
        events = list(STATE["prefill_events"])
    per = {}
    for e in events:
        rid = e.get("rid")
        if not rid:
            continue
        try:
            inp = int(e.get("input") or 0)
            hd = int(e.get("hd") or 0)
            hh = int(e.get("hh") or 0)
        except (TypeError, ValueError):
            continue
        cur = per.get(rid)
        if cur is None:
            per[rid] = [inp, min(hd + hh, inp)]
        else:
            if min(hd + hh, inp) > cur[1]:
                cur[1] = min(hd + hh, inp)
    tot_in = sum(v[0] for v in per.values())
    tot_hit = sum(v[1] for v in per.values())
    return (tot_hit / tot_in * 100.0) if tot_in > 0 else None


def draft_graph_thread():
    deadline = time.time() + 600   # 服务重启后日志尚未出现目标行, 最多扫 10 分钟
    while time.time() < deadline:
        try:
            with open(SGLANG_LOG, "rb") as f:
                f.seek(max(0, f.seek(0, 2) - 4 * 1048576))   # 末 4MB 足够覆盖启动段
                text = f.read().decode("utf-8", "replace")
            if DRAFT_GRAPH_OFF_RE.search(text):
                STATE["draft_graph"] = "off"
                return
            if DRAFT_GRAPH_ON_RE.search(text):
                STATE["draft_graph"] = "on"
                return
        except Exception:
            pass
        time.sleep(60)


def poll_gpus():
    """nvidia-smi 卡级实测 + Windows GPU 计数器（vmwp = WSL 进程之和）, 每 5s"""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=6)
        gpus = []
        for line in out.stdout.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            gpus.append({"idx": int(p[0]), "name": p[1], "total_mb": int(p[2]),
                         "used_mb": int(p[3]), "util": int(p[4]), "mem_util": int(p[5]),
                         "temp": int(p[6]), "power": float(p[7])})
        if gpus:
            wsl_mb = poll_wsl_gpu()
            # LUID→GPU 匹配: vmwp 在某 LUID 的占用必 ≤ 该 GPU 的 used, 取差值最小的配对
            pairs = []
            for luid, b in wsl_mb.items():
                mb = b / 1048576
                for g in gpus:
                    if mb <= g["used_mb"] + 1024:
                        pairs.append((abs(g["used_mb"] - mb), luid, g, mb))
            pairs.sort()
            seen_l, seen_g = set(), set()
            for d, luid, g, mb in pairs:
                if luid in seen_l or g["idx"] in seen_g:
                    continue
                # 阈值放宽到 8GB: GPU1 上 Windows 侧 WSLg 显示栈会占 ~4GB 且会涨
                if d > 8 * 1024:
                    break
                g["wsl_mb"] = int(mb)
                seen_l.add(luid); seen_g.add(g["idx"])
            for g in gpus:
                g.setdefault("wsl_mb", None)
        return gpus
    except Exception:
        return None


def poll_wsl_gpu():
    """Windows GPU Process Memory 计数器 → vmwp(WSL 通道) 按 LUID 的显存字节数"""
    ps1 = HERE / "gpu_mem.ps1"
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)],
            capture_output=True, text=True, timeout=20)
        j = json.loads(out.stdout.strip().splitlines()[-1])
        return {k: int(v) for k, v in (j.get("wsl") or {}).items()}
    except Exception:
        return {}   # 计数器失败不影响 nvidia-smi 卡级数据


def sampler():
    prev, prev_t = None, None
    prev_s = None   # 上一帧 sample（verify_hz 用它的 spec_accept 对齐吞吐窗口）
    info_t = 0.0
    flush_t = time.time()
    stall_since = None   # running>0 且吞吐≈0 的起始时刻 → decode 冻结检测
    while True:
        t0 = time.time()
        try:
            d = parse(get("http://" + SGLANG + "/metrics"))
            dt = (t0 - prev_t) if prev is not None else 0.0
            s = compute(d, prev, dt, prev_s)
            s["attrib_hit_rate"] = attrib_hit_rate_snapshot()
            prev = d
            prev_s = s
            STATE["error"] = None
            # decode 冻结检测: 有请求在跑但 decode 1s 差分为 0 持续超过 2s。
            # 用 dec_tps (realtime_tokens_total{mode=decode} 差分) 而非 gen_tps:
            # gen_tps 是 40 步窗口 gauge, 真冻结时 forward 停摆 → gauge 停在最后
            # 非零值 → gt<0.1 不成立 → 漏报(scheduler.py 注释自认)。dec_tps 是
            # 1s 差分, 冻结时天然为 0, 配合 running>0 准确判停摆。
            # gt is None(缺数据/首帧) 不判冻结, 避免误报。
            running = s.get("running") or 0
            dtp = s.get("dec_tps")
            if running > 0 and dtp is not None and dtp < 0.1:
                if stall_since is None:
                    stall_since = t0
                s["stall_s"] = t0 - stall_since
            else:
                stall_since = None
                s["stall_s"] = None
            if time.time() - info_t > 120:
                try:
                    info = json.loads(get("http://" + SGLANG + "/get_server_info", 6))
                    STATE["info"] = {k: info[k] for k in INFO_KEYS if info.get(k) is not None}
                except Exception:
                    pass
                info_t = time.time()
        except Exception as e:
            s = {"t": t0, "stale": True}
            prev = None
            prev_s = None
            STATE["error"] = "%s: %s" % (type(e).__name__, e)
        prev_t = t0
        with LOCK:
            s["gpus"] = STATE["gpus"]
            STATE["samples"].append(s)
            _samples = list(STATE["samples"])
        # 历史落盘: 每 10 分钟把整窗 JSONL 快照写入 history/（覆盖同小时文件,
        # 幂等可重跑; 读侧只取 <=now-缓冲 的旧段, 与内存缓冲无缝拼接）
        if time.time() - flush_t >= 600:
            flush_t = time.time()
            try:
                flush_history(_samples)
            except Exception:
                pass
        time.sleep(max(0.0, INTERVAL - (time.time() - t0)))


HIST_DIR = HERE / "history"
HIST_KEEP_HOURS = 48   # 磁盘历史保留时长
_hist_flushed_t = -1.0  # 已落盘的最大样本 t（sampler 单线程访问, 无需锁）


def flush_history(samples):
    """把缓冲中 t > _hist_flushed_t 的增量按样本小时**直接追加**到
    history/YYYYMMDD_HH.jsonl。

    为什么不能整窗覆盖 / 不能 tmp+replace 累积：每次 flush 的 30min 窗口跨小时时
    会同时写到两个文件，而 .tmp.replace() 会消耗掉 .tmp → 下一个 flush 又从空 .tmp
    开始，跨小时边界的小时文件只剩最后一次 flush 的残片，恢复时留 30~50min 空洞。
    改为「按小时直接追加 + 仅写 t>已落盘增量」，相邻 flush 无缝衔接、幂等（重复 t
    因 t>last 判断天然去重）。追加非原子，但每行独立、load_history() 逐行
    try/except 容错，最坏丢崩溃瞬间的半个 flush（≤10min），无跨小时空洞。"""
    global _hist_flushed_t
    HIST_DIR.mkdir(exist_ok=True)
    by_hour = {}
    for p in samples:
        if p["t"] <= _hist_flushed_t:
            continue
        day = time.strftime("%Y%m%d_%H", time.localtime(p["t"]))
        by_hour.setdefault(day, []).append(p)
    for day, pts in by_hour.items():
        with (HIST_DIR / (day + ".jsonl")).open("a", encoding="utf-8") as f:
            f.write("".join(json.dumps(p, ensure_ascii=False, separators=(",", ":")) + "\n"
                            for p in pts))
    _hist_flushed_t = max(_hist_flushed_t, max(p["t"] for p in samples))
    # 清理过期文件（按文件名 = 起始小时判断, 低频无害）
    cutoff = time.time() - HIST_KEEP_HOURS * 3600
    for f in HIST_DIR.glob("*.jsonl"):
        try:
            if time.mktime(time.strptime(f.stem, "%Y%m%d_%H")) + 3600 < cutoff:
                f.unlink()
        except ValueError:
            pass


def load_history():
    """启动时恢复历史: 加载所有 JSONL 段, 按 t 去重拼接, 只留最近缓冲窗。
    恢复后 sampler 从 prev=None 起步 → 窗口类指标（直方图/命中率/TTFT/ITL）
    以最早恢复样本为新基线重新累积, 速率类首拍无差分; 落盘 10min 一次, 崩溃最坏丢 10min。"""
    pts = []
    if HIST_DIR.exists():
        for f in sorted(HIST_DIR.glob("*.jsonl")):
            try:
                for line in f.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        pts.append(json.loads(line))
            except Exception:
                continue
    pts.sort(key=lambda p: p.get("t", 0))
    # 去重（t 唯一）+ 只留最近缓冲窗
    seen, dedup = set(), []
    for p in pts:
        k = p.get("t")
        if k not in seen:
            seen.add(k)
            dedup.append(p)
    return deque(dedup[-MAX_SAMPLES:], maxlen=MAX_SAMPLES)


def gpus_thread():
    """独立线程刷 GPU 数据: nvidia-smi(~1s) + PowerShell 计数器(~2s) 不阻塞 1s 采样循环"""
    while True:
        g = poll_gpus()
        if g:
            STATE["gpus"] = g
        time.sleep(5.0)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj):
        self._send(200, json.dumps(obj, ensure_ascii=False).encode(),
                   "application/json; charset=utf-8")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/snapshot":
            with LOCK:
                latest = STATE["samples"][-1] if STATE["samples"] else None
                self._json({"sample": latest, "info": STATE["info"], "error": STATE["error"],
                            "draft_graph": STATE["draft_graph"],
                            "boot_mem": STATE["boot_mem"],
                            "encoder": STATE.get("encoder")})
        elif p == "/api/history":
            with LOCK:
                self._json({"samples": list(STATE["samples"])})
        elif p == "/api/prefill_events":
            with LOCK:
                self._json({"events": list(STATE["prefill_events"])})
        elif p in ("/", "/index.html"):
            body = (HERE / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")


def main():
    STATE["samples"] = load_history()   # 启动恢复磁盘历史(空目录时仍为空 deque)
    # 已恢复的尾段已在盘上, 把落盘水位推到其最大 t, 首个 flush 只追加新样本
    global _hist_flushed_t
    if STATE["samples"]:
        _hist_flushed_t = STATE["samples"][-1]["t"]
    threading.Thread(target=sampler, daemon=True).start()
    threading.Thread(target=gpus_thread, daemon=True).start()
    threading.Thread(target=prefill_events_thread, daemon=True).start()
    threading.Thread(target=draft_graph_thread, daemon=True).start()
    threading.Thread(target=encoder_thread, daemon=True).start()
    threading.Thread(target=boot_mem_thread, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print("[kv-monitor] http://0.0.0.0:%d  →  sglang=%s  interval=%.1fs  buffer=%ds  history=%s"
          % (PORT, SGLANG, INTERVAL, MAX_SAMPLES, HIST_DIR), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
