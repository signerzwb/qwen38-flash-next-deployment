# SGLang KV Cache Monitor — 设计与实现文档

版本：v1.2（2026-08-29）

## 1. 项目定位

针对本地 WSL 中 SGLang 推理服务的**轻量级实时 KV cache 可视化面板**。设计目标：

- **零第三方依赖**：后端纯 Python 标准库（3.8+），前端单文件 HTML（原生 Canvas 2D，无图表库）
- **单进程可部署**：`python server.py` 一条命令启动，自带静态页面 + JSON API
- **低开销**：1s 采样、30 分钟环形缓冲（1800 样本 × ~1KB ≈ 2MB 内存）
- **只读**：不写 SGLang，只拉 `/metrics` 与 `/get_server_info`，可长期挂着

监控对象（当前部署，2026-09-09 更新）：**Qwen3.8-27B-Uncensored-NVFP4-v2**（混合精度自量化：MLP+lm_head → NVFP4，GDN/attn 投影 → FP8，embed/norms → BF16），视觉塔分离双卡（GPU0 4090 48G 语言模型 @192.168.88.123:8778，GPU1 4070 Ti 视觉 encoder @:8779），KV cache fp8_e4m3，池 **500,000 tokens**（max_total_num_tokens 动态读取，无需随池调整），HiCache 分层缓存（L2 DRAM 主池 655k tokens @ HICACHE_SIZE=24）+ DFlash 投机解码（K=8）+ ReplaySSM fold-every-commit（手搓 PR #36683）+ **HRRN 调度**（2026-09-09 摘 PR #32911，短请求 TTFT 3× 提速）+ mamba 混合架构（**20 槽**）。GPU0 显存组成见 §7。

## 2. 总体架构

```
┌────────────────────────────── WSL (192.168.88.123) ──────────────────────────────┐
│  SGLang :8778  /metrics (Prometheus 文本, chunked)   /get_server_info (JSON)      │
└───────────────▲──────────────────────────────────────────────────────────────────┘
                │ urllib (1s)                      独立线程 (5s)
┌───────────────┴────────────────── Windows 主机 ────────────────────────────────────┐
│  server.py (ThreadingHTTPServer :8917)                                             │
│  ┌─────────────────┐   ┌────────────────────┐   ┌──────────────────────────────┐   │
│  │ sampler 线程    │   │ gpus_thread 线程    │   │ HTTP handler (每请求一线程)   │   │
│  │ 拉 /metrics     │   │ nvidia-smi 卡级     │   │ GET /            → index.html │   │
│  │ compute()→样本  │   │ poll_wsl_gpu:       │   │ GET /api/snapshot → 最新样本  │   │
│  │ 1s/次 → 环形缓冲│   │ powershell→gpu_mem  │   │ GET /api/history  → 全缓冲    │   │
│  └────────┬────────┘   │ 计数器(vmwp 聚合)   │   └──────────────▲───────────────┘   │
│           │  LOCK       └──────────┬─────────┘                  │ fetch (2s/次)     │
│           └──► STATE {samples deque, gpus, info, error} ────────┘                  │
└────────────────────────────────────────────────────────────────────────────────────┘
                │
        浏览器 index.html：2s 轮询 /api/snapshot 增量追加，30 分钟窗口差分，Canvas 重绘
```

### 2.1 线程模型（v1.0 复核后定稿）

| 线程 | 周期 | 职责 | 阻塞特性 |
|------|------|------|----------|
| `sampler` | 1s | 拉 `/metrics` → `parse` → `compute` → 追加环形缓冲；每 120s 拉一次 `/get_server_info` | HTTP 超时 4s，异常时推 stale 占位样本 |
| `gpus_thread` | 5s | `nvidia-smi`（~1s）+ `powershell gpu_mem.ps1`（~2s）→ LUID 匹配 → 写 `STATE["gpus"]` | 独立线程，**不阻塞采样**（v1.0 修复：原先内联在 1s 循环里，每 5s 卡 3s） |
| HTTP handler | — | ThreadingHTTPServer，每连接一线程，读 STATE 持 `LOCK` | 只读快照，毫秒级 |

sampler 每个样本附加 `s["gpus"] = STATE["gpus"]`（引用快照），使历史曲线里每点自带当时 GPU 数据。

### 2.2 关键设计决策

1. **窗口类指标用"缓冲内最早有效样本 → 当前"的差分，而非 1 秒差分。**
   SGLang 的 `prompt_tokens_histogram`、`prefill_effective_tokens_total`、TTFT/ITL 的 `_sum/_count` 都是**进程启动以来的累计值**。前端若用相邻 2 秒差分，请求不密集时几乎每窗口都是 0，表现为"柱状图闪一下就消失"。定稿方案：后端只透传累计原值（`eff` 四 mode、`ttft_sum/count`、`itl_sum/count`、`histo` 全桶），前端以 30 分钟环形缓冲内**第一个有效样本为基线**做窗口差分；计数器回退（SGLang 重启）时 `count` 比较失败自动重锚到全量。

2. **GPU 进程级显存从 Windows 侧读。**
   WSL2 内 `nvidia-smi --query-compute-apps` 返回 `[N/A]`（虚拟机 guest 无进程视图）。Windows 侧性能计数器 `\GPU Process Memory(*)\Dedicated Usage` 可见，且 WSL 内**所有** CUDA 进程在 Windows 侧归因到 `vmwp.exe`（WSL 虚拟机 worker 进程）。因此：
   - vmwp 在某 LUID 的 Dedicated 字节数 = WSL 侧该卡的 CUDA 显存总和
   - LUID → nvidia-smi GPU index 无稳定映射关系（LUID device id ≠ smi 序号），用**贪心数值匹配**：对每个 (luid, gpu) 候选对按 `|gpu.used_mb − wsl_mb|` 排序取最小差配对，约束 `wsl_mb ≤ used_mb + 1GB`（WSL 占用不可能超过卡总占用）、差值阈值 8GB（容纳 GPU1 上 Windows 侧 WSLg 显示栈 ~4GB 且可能增长）、luid/gpu 各配一次。

3. **显存组成 = 启动日志实测 + 指标**。
   nvidia-smi 只给总占用。组成拆分的每一项都有权威来源：主权重/KV 池取 SGLang 指标（`weight_memory_usage_gb`、`kv_cache_memory_usage_gb`）；**draft 权重、draft KV 池、mamba 槽、CUDA graph 取启动日志实测**（`boot_mem_thread` 解析，见 §7）；视觉塔权重 0.86GB（从 safetensors header 实测）。剩余倒推段标"工作区/框架"。

   ⚠️ **历史教训**：早期版本用 `0.27GB/槽` 硬编码（12 槽时代的实测值）。升到 20 槽后该项错算成 5.4GB（实际 1.59GB = **79.5MB/槽**），3.8GB 被从"工作区"错记到 mamba；draft 的 8.4GB（权重 3.65 + KV 4.76）更是整块落进倒推。**任何"按槽/按部署规模"的常量都会随配置漂移，必须从日志或指标动态取。**

4. **stale 占位样本而非丢弃。**
   SGLang 断线时 sampler 推 `{"t":…, "stale":true}`（无 `kv` 字段），前端过滤条件 `p.kv && p.kv.max != null` 把断线段在时间轴上留空（曲线断开），而不是把缓冲压缩对齐——这样恢复后时间轴刻度不跳。

## 3. 后端数据结构

### 3.1 样本（sample）schema

`sampler` 每次产出如下 dict，追加进 `STATE["samples"]`（`deque(maxlen=1800)`）：

```python
{
  "t": 1724900000.123,             # time.time() 采样时刻
  "stale": True,                   # 仅断线占位样本有；有效样本无此键（前端判 p.kv）

  # —— KV cache 核心（gauge/堆叠图）——
  "kv": {
    "max": 320000,                 # max_total_num_tokens（动态读取, 示例值 2026-08-29）
    "used": 57340.0,               # kv_used_tokens（running 请求占用）
    "evictable": 201200.0,         # kv_evictable_tokens（radix 驻留可驱逐）
    "free": 3604.0,                # max - used - evictable（派生）
    "usage_pct": 21.9,             # used/max*100（派生）
    "mem_gb": 8.0,                 # kv_cache_memory_usage_gb（KV 池显存）
    "weight_gb": 27.5,             # weight_memory_usage_gb
    "graph_gb": 1.2,               # graph_memory_usage_gb（cuda graph 预留）
  },

  # —— 调度/吞吐 ——
  "gen_tps": 28.1,                 # gen_throughput（引擎自报 decode tok/s）
  "running": 3.0,                  # num_running_reqs（仅 decode 期，prefill 中不计入）
  "queued": 0.0,                   # num_queue_reqs
  "retracted": 0.0,                # num_retracted_reqs（显存压力回撤）

  # —— HiCache 分层缓存（L2 DRAM 备份）——
  "hicache": {
    "host_total": 302848.0,        # hicache_host_total_tokens（L2 池容量）
    "host_used": 302000.0,         # hicache_host_used_tokens
    "backup_tok": 123456.0,        # hicache_backup_tokens_total{pool=kv}（累计，GPU→DRAM）
    "backup_mamba": 456.0,         # hicache_backup_tokens_total{pool=mamba}（= 槽备份次数, 非 token）
    "backup_bytes": 987654321.0,   # hicache_backup_bytes_total（累计）
    "backup_dur_sum": 12.3,        # hicache_backup_duration_seconds_sum
    "backup_dur_n": 46.0,          # hicache_backup_duration_seconds_count
    "load_tok": 119680.0,          # load_back_tokens_total{pool=kv}（L2→GPU 回取累计, host_hit 路径）
    "load_mamba": 530.0,           # load_back_tokens_total{pool=mamba}
    "load_bytes": 4.67e+10,        # load_back_bytes_total（全池合计, 含 draft/sidecar）
    "load_dur_sum": 67.49,         # load_back_duration_seconds_sum
    "load_dur_n": 120.0,           # load_back_duration_seconds_count
    "load_dur_hist": {…},          # load_back_duration_seconds 全桶（前端窗口差分算 P99）
  },
  "hicache_backup_tps": 512.0,     # backup_tok 差分速率（仅 dt>0 时）
  "hicache_backup_gbps": 0.42,     # backup_bytes 差分速率（前端 /1e9 画右轴 GB/s）
  "hicache_load_tps": 30.0,        # load_tok 差分速率（回取 tok/s）
  "hicache_load_gbps": 0.1,        # load_bytes 差分速率（回取 GB/s）
  "hicache_drop": 0.0,             # dropped_tokens_total 全 reason 合计（L2 驱逐告警）
  "evicted": 672960.0,             # evicted_tokens_total（L1 逐出槽位累计, 备份+drop 总源头）
  "evict_tps": 12.0,               # evicted 1s 差分速率（L1 逐出 tok/s）
  "engine_hit_rate": 0.87,         # cache_hit_rate gauge（引擎滑动窗口自报, 交叉验证用）

  # —— mamba 状态槽（L1 GPU）——
  "mamba": {
    "total": 12.0,                 # used+evictable+available（= max_mamba_cache_size）
    "used": 3.0,                   # mamba_used_tokens（≈ 请求数 × 3 槽/请求）
    "evictable": 9.0,              # mamba_evictable_tokens
    "free": 0.0,                   # mamba_available_tokens
  },

  # —— 投机解码（DFlash draft）——
  "spec_accept": 3.75,             # spec_accept_length（平均接受草稿 token 数 / 8）
  "spec_rate": 0.47,               # spec_accept_rate
  "verify_hz": 12.4,               # dec_tps / spec_accept（真·实时 verify 节奏, 2026-08-29 v1.3.1 改; 旧 spec_verify_tps 为结账脉冲语义, 仅 API 保留）

  # —— prefill/decode 速率（计数器差分，1s 周期，天然适合短窗口）——
  "pre_tps": 12000.0,              # realtime_tokens_total{mode=prefill_compute} 差分
  "pre_cache_tps": 5500.0,         # {mode=prefill_cache}（前缀复用，免计算）
  "dec_tps": 85.0,                 # {mode=decode}

  # —— 窗口类指标累计原值（前端 30 分钟窗口差分，见 §2.2-1）——
  "eff": {"input":…, "device_hit":…, "host_hit":…, "storage_hit":…},
         # prefill_effective_tokens_total 四 mode；
         # 命中率 = (device+host+storage)/(四者和)
  "ttft_sum":…, "ttft_count":…,   # time_to_first_token_seconds
  "itl_sum":…,  "itl_count":…,    # inter_token_latency_seconds

  # —— token 长度直方图（全桶透传）——
  "histo": {
    "p": {"buckets": [["0.05",1],["0.1",1],…,"1100000.0",N],["+Inf",N]],  # 36 桶 le 升序
          "sum":…, "count":N},        # prompt_tokens_histogram
    "g": {…},                            # generation_tokens_histogram
  },

  # —— GPU（每样本引用当时 STATE["gpus"]）——
  "gpus": [
    {"idx":0, "name":"NVIDIA GeForce RTX 4090", "total_mb":49140, "used_mb":48042,
     "util":88, "mem_util":12, "temp":57, "power":309.0, "wsl_mb":48074},
    {"idx":1, "name":"NVIDIA GeForce RTX 4070 Ti", "total_mb":12280, "used_mb":6081,
     "util":52, "mem_util":3, "temp":39, "power":45.0, "wsl_mb":2005},
  ],
}
```

字段来源映射：

| 样本字段 | Prometheus 指标（已剥 `sglang:` 前缀） |
|---|---|
| kv.* | `max_total_num_tokens` / `kv_used_tokens` / `kv_evictable_tokens` / `*_memory_usage_gb` |
| pre_*_tps, dec_tps | `realtime_tokens_total{mode=…}` 差分 |
| hicache.* | `hicache_host_total_tokens` / `hicache_host_used_tokens` / `hicache_backup_*` / `hicache_dropped_tokens_total` / `load_back_tokens_total{pool}` / `load_back_bytes_total` / `load_back_duration_seconds*` |
| evicted, evict_tps | `evicted_tokens_total`（累计 + 1s 差分速率） |
| engine_hit_rate | `cache_hit_rate`（gauge, 引擎自报） |
| mamba.* | `mamba_used/evictable/available_tokens` |
| spec_* / verify_hz | `spec_accept_length` / `spec_accept_rate` / 派生（decode 吞吐 ÷ 接受长度）；`spec_verify_calls_total` 仅 API（按请求结账脉冲） |
| eff / ttft / itl / histo | `prefill_effective_tokens_total` / `time_to_first_token_seconds_*` / `inter_token_latency_seconds_*` / `prompt_tokens_histogram` / `generation_tokens_histogram` |
| gpus | nvidia-smi + Windows GPU Process Memory 计数器（非 Prometheus） |

### 3.2 全局状态

```python
STATE = {
  "samples": deque(maxlen=1800),  # 环形缓冲，30 min @1s
  "info":    {...} | None,        # /get_server_info 白名单字段，120s 刷新
  "error":   "URLError: …" | None,# 最近一次采样异常（前端横幅显示）
  "gpus":    [...] | None,        # gpus_thread 5s 刷新
}
LOCK = threading.Lock()           # 保护 samples 读写；gpus 为整体引用替换
```

### 3.3 API

| 端点 | 返回 | 说明 |
|------|------|------|
| `GET /` | index.html | 单文件页面 |
| `GET /api/snapshot` | `{"sample": 最新样本, "info": 配置快照, "error": 采样错误}` | 前端 2s 增量轮询 |
| `GET /api/history` | `{"samples": [全缓冲]}` | 页面初始加载一次性拉取 |

### 3.4 Prometheus 解析（`parse()`）

标准库手写解析：跳过 `#` 注释行，`rpartition(" ")` 切值，`LAB` 正则 `(\w+)="([^"]*)"` 提取标签，指标名剥命名空间前缀（`sglang:kv_used_tokens` → `kv_used_tokens`）。histogram 三件套（`_bucket`/`_sum`/`_count`）由 `hist()` 归并为 `{buckets: [[le_str, cum],…], sum, count}`，桶按 le 数值排序，`+Inf` 保留字符串原样（前端 `fmtLe` 识别）。

`rate(prev, cur, dt)`：计数器差分速率，`cur < prev`（回绕/重启）返回 None，避免画出负速率毛刺。

## 4. GPU 显存组成计算

### 4.1 gpu_mem.ps1 输出

```json
{"totals": {"luid_16547": 51234567890, "luid_175c9": 7000000000},
 "wsl":    {"luid_16547": 50234567890, "luid_175c9": 2100000000}}
```

- `totals[luid]`：该卡上**所有**进程（Windows 本地 + WSL vmwp）的 Dedicated 显存和
- `wsl[luid]`：其中 vmwp.exe（WSL2 通道进程，pid 动态查）的部分 = WSL 侧总和
- 实例名格式 `pid_(\d+)_luid_0x[0-9a-f]+_0x([0-9a-f]+)`，第二组 hex 是 adapter LUID

### 4.2 LUID 贪心匹配

```
候选 = [(|g.used_mb − mb|, luid, g, mb)  for luid,mb in wsl  for g in gpus  if mb ≤ g.used+1GB]
按差值升序扫描；luid 与 gpu 各只能配一次；差值 > 8GB 截断（break）。
配上的 g["wsl_mb"] = int(mb)；没配上的保持 None（前端显示 "--（计数器不可用）"）。
```

### 4.3 前端组成拆分（`renderGPUs`）

**GPU0（语言模型）**：六段堆叠条 —— `主权重 weight_gb` + `draft 权重` + `KV 池 mem_gb` + `draft KV` + `mamba 槽 mamba.total × boot_mem.mamba_per_slot` + `工作区 = used − 前五者（倒推，≥0）`。后四项来自启动日志实测（§7），拿不到时回退常量并在下方行标注"常量回退（日志未就绪）"。下方行：WSL 侧（vmwp 全进程合计）/ mamba 槽位（n 槽 × 实测 MB）/ 显存组成来源 / 功耗。

**GPU1（视觉塔 encoder）**：`视觉塔权重 0.86GB（常量，safetensors 实测）` + `encoder 工作区 = wsl_mb − 0.86（WSL 内非权重部分）` + `Windows 侧 = used − wsl_mb（WSLg 显示渲染等）`。下方行：WSL 侧 / Windows 侧 / 空闲。

**利用率时间线**：`st.filter(p=>p.gpus)` 取各点 GPU0/GPU1 的 `util`，双折线（vmax=100）。

### 4.4 启动日志显存解析（`boot_mem_thread` / `parse_boot_mem`）

GPU0 的 draft 权重、draft KV 池、mamba 槽、CUDA graph 四项 Prometheus 无指标，只有 boot log 有精确值。`parse_boot_mem()` 扫日志尾部 4MB，按出现顺序归属（主模型先于 draft，不靠名字猜）：

| 字段 | 日志行 | 正则 |
|---|---|---|
| `main_w` / `draft_w` | `Load weight end. ... type=X ... mem usage=N GB` | `BOOT_MAIN_W_RE`（`type` 含 `Draft` → draft） |
| `mamba_per_slot` / `mamba_total` | `Mamba Cache is allocated. max_mamba_cache_size: N, conv_state ... ssm_state ...` | `BOOT_MAMBA_RE`（四项求和 ÷ 槽数） |
| `kv_main` / `kv_draft` | `KV Cache is allocated. ... #tokens: N, K size: X GB, V size: Y GB` | `BOOT_KV_RE`（第 1 次=主池，第 2 次=draft 池） |
| `graph` | `Capture * CUDA graph end. ... mem usage=N GB` | `BOOT_GRAPH_RE`（累加） |

线程每 15s 检查 boot log 的 `st_ctime`，服务重启（日志被重写）即重扫；结果经 `/api/snapshot.boot_mem` 下发。

## 5. 前端（index.html，单文件）

### 5.1 数据流

```
init: fetch /api/history → samples 全量
tick: 每 2s fetch /api/snapshot → 若 sample.t > 末条.t 则追加（防重复）
render: 每 tick 全量重绘（~10 个 canvas，纯 2D 填充，<5ms）
```

有效样本过滤：`st = samples.filter(p => p.kv && p.kv.max != null)`；基线 `base = st[0]`（窗口起点）。`maxTok` 取最新有效样本的 `kv.max`（最新是 stale 时回退 `st` 末条）。

### 5.2 面板清单与渲染方式

| 面板 | 渲染 | 数据 |
|------|------|------|
| KV Cache 占用 | 270° 圆弧 gauge（红=used 蓝=evictable）+ 三行拆分 + 双段水平条 | 最新样本 kv.* |
| KV Cache 占用分布 | 堆叠面积（灰 free 底 / 蓝 evictable / 红 used），vmax=池容量 | 全窗口 kv.used/evictable/free |
| Prefill/Decode 速率 | 双轴折线：左轴 prefill（橙=计算 绿虚=前缀复用），右轴 decode（蓝） | 全窗口 pre_tps / pre_cache_tps / dec_tps |
| GPU 显存组成 | 双卡片：分段堆叠条 + 图例数值 + kvRow 明细；下接利用率时间线 | s.gpus + kv/mamba 指标 |
| HiCache 分层缓存 | L2 占用条 + 六行指标（容量/驻留/空闲/drop/备份速率/备份延迟）+ L2 时间序列（面积 + 右轴 GB/s） | hicache.* / hicache_drop / backup_gbps |
| mamba 状态槽 | 三色彩条（红 used / 紫 evictable / 灰 free）"占用 X · 驻留 Y / 共 12 槽" | mamba.* |
| 指标网格 | running/queue/retract/TTFT/ITL/gen_tps 六格 | 最新 + 窗口差分 |
| token 长度分布 | 双直方图（prompt 橙 / generation 蓝），36 桶，标签由数据生成 | histo 窗口差分 |

### 5.3 绘图原语

- `prep(canvas, hCss)`：DPR 感知尺寸设置（`canvas.width = cssW × devicePixelRatio`，`ctx.setTransform` 回缩）
- `timeAxis(ctx, w, h, padL, padB, samples, getT)`：计算 t0/t1 时间域 + 自适应 x 刻度（目标 ~5 条，10s 对齐），返回 `{X(t), Y(v), …, ctx}`；y 刻度 `niceTicks`
- `drawSeries(ax, series, opts)`：`opts.stack`（累计列堆叠面积，自底向上）、`opts.vmax`（固定上限）、`opts.axis` L/R 双轴、`opts.noAxis`（第二组复用画布不重画轴）、`opts.rlabel`、`opts.yfmt`、`opts.dash`
- `bucketize(hist, base)`：窗口差分。桶为累计值，`vals[i] = max(0, (cum[i]−base_cum[i]) − (cum[i−1]−base_cum[i−1]))`；base 为空（重锚）时取全量。`drawHist` 自动算 2 个代表性桶边界标签
- `fmtLe(le, prevLabel)`：桶标签。识别 `"+Inf"` 字符串 → `">" + 上一桶标签去前缀`（避免 "≥InfinityM"）；≥1M → `≥x.xM`；≥1000 → `≥xk`（整千不带小数）

### 5.4 窗口类指标计算（render 内）

```javascript
// 命中率：窗口内 prefill 有效 token 的缓存命中占比
da = sum4(s.eff) − sum4(base.eff);   dh = sumHit(s.eff) − sumHit(base.eff);
hit = dh/da*100  (da>0 时)
// TTFT/ITL：直方图 _sum/_count 差分均值
ttft = (s.ttft_sum − base.ttft_sum)/(s.ttft_count − base.ttft_count)*1000 ms  (count 递增时)
// 直方图：base.histo.count ≤ s.histo.count 才用 base，否则重锚全量
```

## 6. 领域语义速查（面板数值 ↔ 实际行为）

- **red（running 占用）**：正在 decode 的请求的 KV。Claude Code 每发一轮对话就是一个尖峰，回落后转蓝
- **blue（radix 驻留）**：已算完的会话 KV 留在 L1 等前缀复用。空闲时占比越高越好（下一轮 TTFT 直接省掉 prefill）
- **gray（空闲）**：新请求 prefill 的余量。长期 <20k 说明池快满，会触发 retract（`retracted` 计数上涨）或新请求排队
- **prefill 前缀复用（绿虚线）**：这一秒有多少 token 直接命中缓存免计算——衡量 radix/HiCache 价值的核心指标
- **mamba 占用 ~3 槽/请求**：当前状态 + 每 256 token 一个 checkpoint；20 槽 ÷ 3 ≈ 6 并发上限（实际受 `max_running_requests=5` 约束）
- **L2 DRAM 使用率**：write_back 策略下持续爬升属正常（历史会话 KV 备份）；`drop > 0` 才是 L2 压力告警（host_pressure 驱逐会损失未来命中率）
- **spec_accept 3.75/8**：DFlash 草稿平均接受率 47%，越高越省 decode 步数
- **num_running_reqs 只计 decode 期**：prefill 中的请求（chunked 2048/块）短暂不在计数里，所以会出现 running=0 但 mamba>0 的窗口
- **GPU1 Windows 侧 ~4GB**：WSLg 显示栈（窗口合成/渲染）常驻，非泄漏；关显示器/WSLg 可验证
- **Claude Code 计数器 vs 本面板**：app 显示的是 Claude tokenizer 的 payload 尺寸（如 161k），Qwen tokenizer 同内容约 120k，且只在该会话 prefill/decode 时体现为 red；会话间它常驻 blue

## 7. 压测请求生成器（loadgen.py）

```bash
python loadgen.py [SGLANG地址] [并发数] [持续秒数]
```

N 个 worker 线程并发打 `/v1/chat/completions`：全部请求共享 ~4k token 中文前缀（首个请求后其余 prefill 走 device_hit，可看绿虚线）+ 300~6000 token 随机正文（prompt 直方图铺开多桶）+ max_tokens 随机 80/200/500（gen 直方图铺开）。验证面板各区块随负载变化的标准手段。

## 8. 已知限制

1. 窗口类指标 = 30 分钟滚动窗口差分，后端重启后从头累积；进程级累计值直接看 SGLang `/metrics`
2. 直方图重锚（计数器回退）后临时显示全量而非窗口值，下一个窗口自然收敛
3. GPU 组成中 **draft 权重 / draft KV / mamba 槽 / CUDA graph 均从启动日志动态解析**（§4.4），服务重启自动重扫；视觉塔 0.86GB 仍是部署常量（§4.3），换视觉塔需改 `index.html` 的 `renderGPUs`
4. LUID 贪心匹配假设 WSL 占用 ≤ 卡总占用且双卡各配一次；三卡以上或两卡占用极接近时可能配错（当前双卡差 40GB+，安全）
5. 无鉴权、HTTP 明文，仅适合局域网/本机
6. `poll_wsl_gpu` 依赖 Windows 性能计数器与 `vmwp` 进程名；WLM 环境或进程改名后 WSL 侧降级为 `--`（卡级数据不受影响）
7. `spec_accept_length` 等是 SGLang 当前实现的具体指标名，升级 SGLang 大版本时需对照 `/metrics` 重新确认

## 9. 文件清单

| 文件 | 行数 | 职责 |
|------|------|------|
| `server.py` | ~400 | 后端：Prometheus 解析、采样、GPU 轮询、启动日志显存解析、编码器健康、HTTP 服务 |
| `index.html` | ~700 | 前端单页：全部面板 + Canvas 绘图原语 |
| `gpu_mem.ps1` | ~17 | Windows GPU 进程内存计数器 → JSON |
| `loadgen.py` | ~110 | 压测请求生成器（多线程、共享前缀、随机长度） |
| `README.md` | — | 使用说明 + 面板映射表 |
| `DESIGN.md` | — | 本文档 |

## 10. 版本记录

- **v1.3（2026-09-09）**：适配 v2 NVFP4 栈 + 修四类缺陷：
  - **显存组成改为启动日志实测**（修 0.27GB/槽 的过时常量）：实测 20 槽 = 1.59GB（79.5MB/槽），旧常量错算 3.8GB；新增 draft 权重 3.65GB 与 draft KV 4.76GB 两段，工作区倒推从 9.6GB 修正到 1.8GB。`parse_boot_mem` + `boot_mem_thread`（§4.4）
  - **编码器健康上屏**：后端 `encoder_thread` 一直在采集 8779 的 `/health` 与日志逐出记录，但前端从不渲染——9-02 事故的探测器不可见。现接入横幅告警（不可达 / 曾被逐出）
  - **窗口基线回退修复**：监控从 `history/` 恢复旧样本、而 SGLang 服务期间也重启过时，引擎 counter 归零 → 基线大于当前 → 差分出负数（实测 TTFT 显示 **-3006ms**）。改用 `num_requests_total` 单调性定位当前引擎生命周期起点
  - **新增指标**：排队等待 `queue_time_seconds`（HRRN 调度的核心观测量）、分阶段耗时 `per_stage_req_latency_seconds`（网关/prefill/chunk）
  - 文档同步至 NVFP4-v2 / 500k 池 / 20 槽 / HRRN

- **v1.2（2026-08-29）**：四条优化（commit 783591e）：
  - **逐出速率**：`evicted_tokens_total` 1s 差分 → HiCache 面板新增"L1 逐出速率"行 + L2 时间序列右轴红虚线。与 backup/drop 合看可算逐出成功率：`backup/(evict−drop)`
  - **引擎命中率交叉验证**：`cache_hit_rate`（引擎滑动窗口自报）与前端 30min 窗口差分并列显示，两者背离时优先怀疑窗口基线（如缓冲刚重锚）
  - **历史落盘**：`history/YYYYMMDD_HH.jsonl` 每 10 分钟把缓冲中 `t > 已落盘水位` 的增量**按样本所在小时直接追加**（非原子，但每行独立、读侧逐行 try/except 容错）。启动时 `load_history()` 按 t 去重恢复最近 30min；保留 48h，已加入 .gitignore。
    踩坑（经真实参数 600s/1800s/1s 模拟 + 真实函数 end-to-end 双重验证）：每次 flush 的 30min 窗口跨小时时会同时写两个文件，**覆盖式**（首版整窗写单文件）和 **tmp+replace 累积式**（90dac74 首改）都会在跨小时边界留下 30~50min 恢复空洞——前者每小时文件只留最后一次 flush 的残片，后者因 `.tmp.replace()` 消耗 `.tmp` 导致跨 flush 不累积。正解 = 按小时**直接追加** + 只写 `t>已落盘水位` 增量（幂等、无缝）；崩溃最坏丢 ≤10min 尾段，无跨小时空洞
  - **回取 P99**：`load_back_duration_seconds` 直方图窗口差分 + 累计桶线性查找 P99，HiCache 面板新增"L2→GPU 回取 P99（窗口）"行；窗口无回取时显示占位
  - 实现注记：差分一律用 `g1/gl(prev, 原始指标名)` 查 raw parsed dict——不能 `prev.get(sample字段)`（prev 是 raw dict 非 sample）；evict_tps/load_gbps 首版曾犯此错
- **v1.1（2026-08-29）**：外部复核修复（commit 3b5128a）：
  - GPU0 mamba 显存常量 0.293→0.27GB/槽（原从 host 池 4.1G/14 反推，未含备份页开销；启动日志实测 12 槽 ≈3.24GB：conv 0.04+ssm 0.91+intermediate_ssm 2.25+conv_win 0.04）
  - `hicache_backup_tokens_total{pool=mamba}` 语义修正：计的是**槽备份次数**而非 token 数，面板单位"槽"→"次"
  - loadgen.py `STOP=True` 缺 `global` 声明 → DURATION 到点后 worker 永不停（实际 bug）
  - 新增 L2→GPU 回取指标：`load_back_tokens_total{pool}` / `load_back_bytes_total` / `load_back_duration_seconds`（HiCache 收益侧——被逐出前缀从 DRAM 恢复的量与耗时，与备份方向对称）
- **v1.0（2026-08-29）**：全量代码复核后定稿。修复/改进：
  - 直方图 `+Inf` 末桶标签（"≥InfinityM" → ">1.0M"）
  - GPU 轮询拆独立线程（5s），不再阻塞 1s 采样循环（环形缓冲恢复满 30 min）
  - LUID 匹配阈值 5GB→8GB（容纳 WSLg 显示栈增长）
  - `poll_wsl_gpu` 异常降级（计数器挂掉不丢卡级数据）
  - 后端失联时 `maxTok` 回退窗口末条有效样本（时间曲线不再压扁）
  - 文案修正（vmwp 全进程合计）、删除死变量 `histState`、README 同步
