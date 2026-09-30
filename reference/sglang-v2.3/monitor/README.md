# SGLang KV Cache Monitor

针对 SGLang 部署（当前：Qwen3.8-27B-Uncensored-NVFP4-v2 @ 192.168.88.123:8778）的轻量实时可视化面板。纯 Python 标准库 + 单页 HTML，零第三方依赖。

## 启动

```bash
python server.py [SGLANG地址] [监听端口]
# 默认: python server.py 192.168.88.123:8778 8917
# 浏览器打开 http://localhost:8917
```

也可通过 launch.json 的 `kv-monitor` 配置启动（`.claude/launch.json`）。

## 面板内容

| 面板 | 数据源 |
|------|--------|
| GPU 显存组成（GPU0 语言模型 / GPU1 视觉塔，含组成条 + 利用率时间线） | nvidia-smi 卡级实测（独立线程 5s 轮询）+ Windows GPU 计数器读 WSL 侧（vmwp 聚合，gpu_mem.ps1）+ **启动日志实测**（主权重/draft 权重/KV 池/draft KV/mamba 槽/CUDA graph，见下）+ SGLang 指标（权重/KV池） |
| KV 占用仪表 + running/radix/空闲 拆分 | `kv_used/kv_evictable/max_total_num_tokens` |
| KV 占用分布（堆叠时间曲线，30 分钟环形缓冲） | 同上，1s 采样 |
| Prefill / Decode 速率（上下双泳道：prefill 计算+复用堆叠 / decode 独立尺度） | `realtime_tokens_total{mode=prefill_compute\|prefill_cache\|decode}` 差分 |
| 分层缓存 HiCache（L2 DRAM 占用/时间序列/备份速率 GB/s/drop/备份延迟/L1 逐出速率） | `hicache_host_total/used_tokens`、`hicache_backup_tokens_total{pool=kv}`、`hicache_backup_bytes_total`、`hicache_dropped_tokens_total`、`evicted_tokens_total` 差分 |
| mamba 槽位（L1 used/evictable/free + L2 备份次数） | `mamba_used/evictable/available_tokens`、`hicache_backup_tokens_total{pool=mamba}`（= 槽备份次数） |
| L2→GPU 回取（累计 tok / P99 延迟） | `load_back_tokens_total{pool=kv}`、`load_back_duration_seconds`。**注意**：Prometheus 带标签 child 仅在首次 `.inc()` 后导出，`host_hit=0` 时面板显示 `--` 属正常的空状态，非故障 |
| spec decode 统计（接受长度/接受率/verify 节奏，DFlash draft + ReplaySSM fold） | `spec_accept_length/spec_accept_rate`、verify 节奏 = decode 吞吐 ÷ 接受长度（派生，实时）；`spec_verify_calls_total` 仅 API 保留（结账脉冲语义） |
| 前缀缓存命中率（30 分钟窗口 + 引擎自报交叉验证） | `prefill_effective_tokens_total` 四 mode 窗口差分 + `cache_hit_rate` gauge |
| 按请求 token 长度分布（prompt / generation 双直方图） | `prompt_tokens_histogram` / `generation_tokens_histogram` 窗口差分 |
| 请求数 / 排队 / retract / TTFT / ITL / 引擎自报 decode 吞吐 | `num_running_reqs` 等 + 直方图 `_sum/_count` 窗口差分 |
| 排队等待（入队→被调度） | `queue_time_seconds_sum/count` 窗口差分 |
| 分阶段耗时（网关处理 / prefill 前向 / chunk 前向） | `per_stage_req_latency_seconds{stage=...}` 窗口差分 |
| 视觉编码器(8779)健康 | 实时 /health 探测 + 主日志 `Health check evicted` 扫描（9-02 事故探测器） |

## GPU 显存组成：为什么是启动日志而不是硬编码

GPU0 的权重与 KV 池有 Prometheus 指标直供，但 **draft 权重（3.65GB）、draft KV 池（4.76GB）、mamba 槽（1.59GB @ 20 槽）** 三项只有启动日志有精确值。

早期版本用 `0.27GB/槽` 硬编码：那是 12 槽时代的实测值，20 槽时错算成 5.4GB（实际 1.59GB，**79.5MB/槽**），把 3.8GB 从"工作区"错记到 mamba；draft 的 8.4GB 更是整块落进倒推段。

现在 `server.py` 的 `boot_mem_thread` 解析 boot log 的这几行，服务重启即自动重扫：

```
Load weight end. ... type=Qwen3_5ForConditionalGeneration ... mem usage=18.96 GB
Load weight end. ... type=DFlash2DraftModel ... mem usage=3.65 GB
Mamba Cache is allocated. max_mamba_cache_size: 20, conv_state ... ssm_state ...
KV Cache is allocated. dtype: ... #tokens: 499968, K size: 7.63 GB, V size: 7.63 GB
KV Cache is allocated. ... K size: 2.38 GB, V size: 2.38 GB   ← draft 池
```

拿不到日志时回退常量，面板会标注"常量回退（日志未就绪）"。

## 已知限制

- 窗口类指标（直方图 / 命中率 / TTFT / ITL）= 30 分钟滚动缓冲内"最早采样 → 当前"的差分。历史落盘（v1.2）后后端重启会从 `history/` 恢复最近 30 分钟；进程级累计值直接看 SGLang `/metrics`
- `engine_hit_rate`（`cache_hit_rate` gauge）为**离散更新**：仅在 prefill 完成时从滚动窗口刷新（源码 `metrics_reporter.py` `recent_cache_hit_rate`），空闲期停在旧值或 0，压测实测可到 0.99；`spec_verify_calls_total` 为**按请求结账** counter（请求结束才一次性 inc 该请求一生累计的 verify 轮数，源码 `observe_one_finished_request`），其差分是结账脉冲（可瞬时 >100），面板的 verify 节奏行不使用它，改用 `dec_tps ÷ spec_accept`（实时）；仅活跃期参考，勿在空闲期读数、勿用于告警
- 历史落盘每 10 分钟一次，崩溃最坏丢 10 分钟；`history/` 保留 48h 自动清理
- 采样间隔 1s（`INTERVAL`），环形缓冲 30 分钟（`MAX_SAMPLES`），都在 server.py 顶部可调；GPU 轮询在独立线程（5s），不阻塞 1s 采样循环
- 无鉴权，别暴露到公网

## 压测请求生成器

```bash
python loadgen.py [SGLANG地址] [并发数] [持续秒数]
# 例: python loadgen.py 192.168.88.123:8778 3 90
```

多线程并发打 /v1/chat/completions：全部请求共享 ~4k token 中文前缀（第 1 个后 prefill 走 device_hit），
附加 300~6000 token 随机正文（prompt 直方图铺开多桶），max_tokens 随机 80/200/500（gen 直方图铺开）。
