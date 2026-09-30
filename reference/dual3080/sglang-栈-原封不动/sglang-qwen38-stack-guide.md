# Qwen3.8-27B 本地推理栈完整技术方案

**sglang + HiCache 分层缓存 + 视觉塔分离 + DFlash2 投机解码（WSL2 双卡）**

> 版本：2026-08-29 定稿。本文档是整套生产栈从选型、踩坑到定案的完整技术沉淀，面向两类读者：
> 1. 想直接复刻这套部署的朋友；
> 2. 想借助 AI（Claude Code 等）继续优化自己本地 sglang 部署的人——本文的"方法论"和"判读指纹"章节就是给 AI 看的弹药库。
>
> 所有性能数字均为同硬件 A/B 实测，非理论推算。

---

## 0. TL;DR 成果一览

| 维度 | 成果 | 数字 |
|------|------|------|
| 引擎选型 | vLLM → sglang | decode 单路 58-66 t/s（+30%）；冷 prefill 快 **13-17 倍**（16k prompt 3.3s vs 43.7s） |
| Agent 场景 | 子代理冷启动 | 3.3s vs 41.8s（**10 倍+**），多 agent 工作流质变 |
| 投机解码 | DFlash2 K=8 胜出 | decode 80 t/s（ReplaySSM 后 78），accept ~3.5/8 |
| 分层缓存 | HiCache write_back 重启启用 | L2 回取 5.2k tok/s，比全量重算快 **47%**；多会话 300k 轮转零丢失 |
| KV 池 | ReplaySSM 释放显存回填 | 262k → **320k tokens**（+58k），200k+100k 双会话全进 L1 |
| 精度修复 | GDN beta fp32 | 1000 步漂移 4.1e-04 → 4.8e-07（**865 倍**改善），领先上游 |
| 调度 | fcfs → lpm | 主会话 200k 前缀优先 prefill，赢过 hook 风暴排队 |
| 逐出策略 | LFU → LRU | 修复"活跃对话分支被当冷数据逐出"的设计陷阱 |
| 可观测性 | 自研 monitor v1.3.2 | 实时 verify_hz、请求级 prefill 归因（含 mamba 断链指纹） |

一句话定位：**这是一个为"超长上下文多 agent 日常使用"优化的单用户生产栈**，不是跑分栈。所有取舍（并发 3、lpm、LRU、HiCache）都围绕"主会话 200k+ 常驻、多工具轮转、TTFT 敏感"这个真实负载。

---

## 1. 硬件与拓扑

### 1.1 双卡分工

```
┌─────────────────────────────────────────────────────────┐
│ Windows 11 IoT LTSC（宿主）                              │
│   WSL2 (ubuntu2204) 内跑全部推理                          │
│                                                         │
│  GPU0: RTX 4090 48G 魔改卡 —— 语言模型                    │
│    Qwen3.8-27B FP8 权重 + 320k KV 池 + 12 mamba 槽       │
│    + DFlash draft 模型，sglang :8778                     │
│    decode 峰值 47.5G / 48G（mem_fraction_static=0.985）  │
│                                                         │
│  GPU1: RTX 4070 Ti 12G —— 视觉 encoder                   │
│    Qwen3.8 视觉塔（0.86G 权重），sglang :8779             │
│    主服务以 server_url 指向它做视觉推理                    │
└─────────────────────────────────────────────────────────┘
```

为什么拆视觉塔：
- 语言侧 27B FP8 + 大池已经把 GPU0 显存吃满（47.5/48G），视觉塔再挤进来就开不了 320k 池；
- 视觉请求是低频小负载，12G 卡完全够；
- 代价是图像请求多一跳 HTTP，实测可接受。

### 1.2 为什么选 48G 魔改 4090

- 27B FP8 权重 ~27.5G，48G 才装得下"权重 + 大 KV 池 + mamba 槽 + draft 模型 + CUDA graph"全套；
- 该卡是 Duanyll/open-gpu-kernel-modules fork（动态 BAR1 P2P）针对的型号，未来裸机 Linux 多卡 TP 时有现成 P2P 方案（busbw 6.1×）；
- **功耗形态实测**：decode 稳态 ~302W @ 2745MHz 是**显存带宽瓶颈的正常形态**（显存时钟 10501MHz 顶格，SM 算力过剩自动降频），不是"算力被浪费"；计算重相位实测能摸到 396-407W；450W = power.limit 顶格无解锁空间。这 150W 算力盈余正是树形 verify 类优化能兑现的空间（见 §11）。

### 1.3 WSL2 的代价（先说清楚，劝退或劝备）

WSL2 能用、好用，但有硬约束，全部在 §7 详述。最关键的三个：
- GPU VA 上限 ~1TB（dxgkrnl），激进显存特性会把进程 VA 打穿 → EOVERFLOW 死旋；
- 部分 CUDA 内核在 WSL 的 dxgkrnl 路径上不稳（chunked prefill 8192 崩溃 → 2048 稳态）；
- `wsl.exe` 会话退出会杀掉 setsid 的后台进程，长驻服务必须在宿主侧保活的会话里启动。

如果条件允许，裸机 Linux 是更干净的地基（P2P、stable nvidia.ko、无 VA 上限）。本栈的全部 WSL workaround 在裸机上可直接省略。

---

## 2. 软件栈与源码树

### 2.1 目录约定

```
WSL 内：
/root/sglang-source-qwen38-dflash2     # sglang 源码树，分支 qwen38-dflash2-integration
/root/sglang-env                       # 软链 → /root/sglang-envs/qwen38-dflash2-4be18295bda6
/root/boot-sgl-stack.sh                # 完整启动（encoder 先起 → 主服务）
/root/start-sglang-qwen38-dflash2.sh   # 只起主服务
/root/start-sglang-qwen38-encoder.sh   # 只起 encoder
/root/restart-main-sglang.sh           # 只重启主服务（encoder 不动）
/root/sglang_qwen38_27b_dflash2.conf.sh# 全部配置，唯一调参入口
/root/sglang-qwen38-dflash2/main-boot.log  # 主服务日志（monitor 在 tail 这个）
/root/ab-results.md                    # 全部 A/B 测试数据存档
Windows 侧：
C:\...\sglang-kv-monitor\              # 监控面板（server.py + index.html）
```

### 2.2 源码树上的提交链（按依赖顺序，朋友复刻时的 cherry-pick 清单）

本树的哲学：**跑 beta 版，要前沿性能。未合并的上游 PR 只要测试成本不高就先摘**。历史印证：#35957、GDN beta 修复都是抢跑成功案例。

| 提交 | 内容 | 来源/性质 |
|------|------|----------|
| `0d1e7f27f0` | #35957 retraction 丢 recurrent state 修复 | 上游已合并，cherry-pick |
| `aa90aa170e` | HiCache WSL workarounds（两个环境变量开关，见 §7） | 自研 workaround |
| `10f7ffb361` | #36415 mamba 准入槽未扣账 + schedule_policy mamba_gap_reserve | 上游 open PR，git apply |
| `2c65c59e78` | #36696 mamba radix split 注册槽用 node 自己的 key，修 eviction "parent does not have child key" 生产崩溃 | 上游 open PR |
| `d0227ce92c` | #36266 COW 内核启动预热（实测预热耗时 1.08s） | 上游 open PR |
| `50bf829548` | **#36683 ReplaySSM fold-every-commit**（4 文件：dflash_worker_v2/dspark_worker_v2/kv_cache_configurator + 护栏改） | 上游 open PR 抢跑手搓，翻案级收益 |
| `e4cfca2e14` → `0d02788fcd` → `909c3bec0f` → `3dcd0cf83d` | 请求级 prefill 归因日志（PREFILL-REQ 行，4 连迭代，见 §8.3） | 自研 |
| 另有 | #35496 量化 lm_head | 上游，cherry-pick |

**已核实不需要摘的**：#34234（DFlash draft KV 按实几何预算）已在树内。

**盯防清单**（上游合并后重新摘）：
- **#36683 正式合并版**（替换手搓版）；
- **#33639**（mamba branching 进 unified radix + HiCache）——GDN 断链根治，最关键基础设施；
- #33777（write_back 回收重复 host 副本，L2 有效容量↑）、#36705（host 池 mmap 去重，分配 -13%）、#35931/#36738（load-back 竞态修复）、#34592（GDN verify backend 尊重显式配置）、#36729（hybrid 共享字节预算）；
- vLLM 侧对照观察：#53877（GDN beta，我们本地修复继续领先）、#51725（自适应投机预算）、#52244（GDN 哈希对齐零命中现象）。

### 2.3 conf.sh 三模式投机切换

配置文件设计成一套参数三种投机模式，环境变量一键切换：

```bash
ENABLE_DFLASH=1 ENABLE_MTP=0 ENABLE_DSPARK=0   # 生产：DFlash2 K=8
ENABLE_DFLASH=0 ENABLE_MTP=1 ENABLE_DFLASH=0   # EAGLE MTP 对照
ENABLE_DFLASH=0 ENABLE_MTP=0 ENABLE_DSPARK=1   # RadixArk DSpark 对照
```

这是 A/B 方法学的一部分：任何切换都只需要改两个变量 + 重启主服务，杜绝"参数改了但忘了回"的事故。

---

## 3. 核心配置逐项解读

配置全貌（`/root/sglang_qwen38_27b_dflash2.conf.sh`，每个值都有 why）：

```bash
# —— 池与显存 ——
MAX_TOTAL_TOKENS=320000        # KV 池（见 3.1）
MEM_FRACTION_STATIC=0.985      # 27B+大池场景顶格；sglang 有 Triton 后加载 OOM 警告但实测 4/4 冒烟通过
HICACHE_SIZE=24                # L2 host 池（见 §5）

# —— 混合架构（GDN 线性注意力 + 全注意力）——
MAX_MAMBA_CACHE_SIZE=12        # mamba 状态槽（见 3.2）
MAMBA_TRACK_INTERVAL=512       # SSM checkpoint 间隔（见 3.3）

# —— 调度 ——
MAX_RUNNING_REQUESTS=3         # 并发上限 = mamba 槽约束的工程取舍
SCHEDULE_POLICY=lpm            # 前缀最长优先（见 3.4）
CHUNKED_PREFILL_SIZE=2048      # WSL 稳态上限（8192 崩溃，见 §7）
page-size 64                   # 大 page 减少 radix 元数据与 HiCache 页表开销

# —— 逐出 ——
--radix-eviction-policy lru    # 见 3.5，LFU 陷阱

# —— WSL workaround（§7 详述）——
SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1
SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1
# MTP 模式下去掉 expandable_segments（VA 上限，§7）
```

### 3.1 为什么是 320k：池账本

池的动态构成（monitor 实测）：

```
320,000 = running ~180.7k（活跃请求 KV）
        + radix 驻留 ~137k（算完等复用，可逐出）
        + free ~2.2k
```

三个关键认知：
1. **radix 涨满是设计使然**——空闲空间给缓存是零成本的，"radix 用满但 running 只有 180k"不是故障；
2. 262k → 320k 的 58k 来自 ReplaySSM 释放的 ~2G 显存（§6.2），全部回填给 KV：claude 200k + codex 100k = 300k 现在全进 L1 轮转零回取；
3. decode 峰值 47.5G/48G，空闲余量仅 0.38G。**若将来 OOM，先回 262k**（回滚开关就一行）。

### 3.2 为什么并发是 3：mamba 槽账本

Qwen3.8 是 GDN 混合架构，每个请求占 ~3 个 mamba 状态槽（当前状态 + checkpoint）。12 槽 ÷ 3 ≈ 4 并发理论上限，取 `MAX_RUNNING_REQUESTS=3` 留安全垫。

**坑**：mamba 槽是稀缺资源，槽竞争会把主前缀的整棵 radix 树变无效（GDN 状态链断，见 §8.4 实锤案例）。#36415（准入槽扣账）和 #36696（split 崩溃）就是修这一层的。

### 3.3 MAMBA_TRACK_INTERVAL=512

每 512 token 存一个 SSM checkpoint，决定"从最近 checkpoint 重放"的粒度。512 是 8-29 从 256 调上去的：治标逻辑是 hook 风暴逐出后树更浅、恢复更快。待做实验：256 档（断链短期缓解）。

### 3.4 SCHEDULE_POLICY=lpm

fcfs → lpm（Longest Prefix Match 优先）。真实收益场景：主会话 200k 前缀 vs Claude Code stop-hook 风暴（24 并发 64-token 小请求）排队时，lpm 让主会话先 prefill——前缀最长的请求对"重新 prefill 的痛"最敏感。

### 3.5 radix-eviction-policy：LFU 陷阱与 LRU 定案

这是本栈最有教学价值的一个坑，值得展开：

**LFU 的实现陷阱**（`radix_cache.py:736`）：`hit_count` 只在 **insert 时 +1**，match 命中**不加分**。后果：单会话持续增长场景下，"最新长出的对话分支" hit_count 永远最低 → 被 LFU 当冷数据优先逐出——**恰好逐到用户最活跃的对话部分**。

**实锤案例**：我的带载测试请求（2.5k prompt + 4.5k decode）在池剩 2.2k free 时，把主窗口 24k 的活跃分支挤掉了，触发 24k evict 风暴 + 全量重 prefill。

**LRU 为什么对**：按最后访问时间逐，活跃分支刚被 match 刷新，必然最后被逐。

**运维教训**：生产时段（用户在用）不打长 decode 带载测试；短探测（/metrics）无影响。

---

## 4. 投机解码：三模式对比与选型

### 4.1 全量 A/B 数据（同硬件同负载）

| 方案 | accept len | 单路 decode | 结论 |
|------|-----------|------------|------|
| **DFlash2 K=8**（生产） | ~3.5（域内创作文本实测 1.6-2.8） | 80 t/s（ReplaySSM 后 78） | **胜出** |
| MTP / EAGLE 内置头 | — | 慢 3-4%，长解码慢 ~5% | 否决 |
| MTP 子代理场景 | — | TTFT 差一个量级 | 否决主因 |
| DSpark（RadixArk，draft 2.6G） | ~2.4 | **-45%**（-27%~-50%） | 否决 |
| 草稿 int8 量化 | — | 精度损失不划算 | 不采纳 |

要点解读：
- DFlash 的 accept ~3.5 是 greedy agent 场景均值；**中文网文长 prose + temp 1.0 的创作负载实测只有 1.6-2.8**——创作文本投机命中天然低，这是后续微调 draft 的动机（§11）；
- DSpark 的强项是高熵采样场景，greedy agent 场景不是它的战场；
- DSPARK 分支注意：必须带 `--speculative-draft-model-quantization unquant`（BF16 草稿 + FP8 target 时默认继承会拒载）。

### 4.2 MTP 模式的 VA 大坑

EAGLE 双 worker + `expandable_segments` 会把进程 GPU VA 推过 dxgkrnl 的 **~1TB 上限**：ioctl 全部 EOVERFLOW，scheduler 死旋在 mamba alloc_group_end，表现为进程僵死不报错。conf.sh 已条件化：MTP 模式自动去掉 expandable_segments。详见 §7.1。

### 4.3 ReplaySSM（手搓 #36683）——本栈最大的单笔抢跑

**问题**：DFlash verify 每步要存 per-draft 完整 SSM 快照，spec intermediate 缓冲吃显存。
**方案**：verify 存每步原始输入（ring buffer），commit 时重放折叠（fold-every-commit）。
**实测 A/B**：
- 显存省 ~2G（空闲 47.4→45.3G，decode 峰 -1.7G）——注意隔壁"8.6G scratch"的估算远过于乐观，fold 模式只消除一部分中间缓冲；
- decode -3%（80→78 t/s）——重放有代价；
- 精度：t2/t3 逐字一致，t1 在 327/512 处语义等价分叉（bf16 re-quant fold 的预期漂移，非 bug）；
- **决策**：2G 换 -3% decode 值得——释放的显存全部回填 KV 池（262k→320k），多会话容量 +58k。
- 保持 12 槽并发 3 不动：释放的 2G 当显存安全垫，不上 16 槽。

### 4.4 decode 再提速路线（已定案待执行）

用户问题"decode 还想提，换模型之外有什么办法"的调研结论：

1. **#32053 verify budget**（下一个动手，一晚可 A/B）：`--speculative-dflash-verify-budget` 把 verify 宽度 8→5/6/7。K 仍取 8 个草稿，但 accept ~3.5 时砍掉目标侧 25-35% 的无效 verify 计算。7 文件 +424 行。
2. **#36196 DFlash2 树形草稿**（第二动作）：保留 top-K 兄弟展开成树 verify，上游实测 accept 3.42→4.39@W=4、吞吐 +23%。约束：W=4 draft SSM workspace +7.4G 显存爆；W=2（+2.5G，需池回 262k）是可行档；**且 PR 只验证了 greedy，temp 1.0 下树采样分布未验证是创作栈主风险**。
3. **域内微调 draft**（中期最直接杠杆）：现有 draft 对中文网文长 prose 的 accept 仅 1.6-2.8，对 draft 做 CPT 是命中率的最大杠杆。
4. 已否：#34171 AdaFlash（自适应 verify 长度头）——需训 checkpoint、60 文件、不支持 Qwen3.8-27B。

功耗侧结论（为什么"算力全开"不是路）：decode 302W 是显存带宽瓶颈的正常形态，锁高频不涨 t/s；150W 算力盈余只能靠"一次前向验更多候选"（即树形 verify）兑现。

---

## 5. HiCache 分层缓存：三轮实验与定案

这是本栈最曲折的一章——**结论反转过一次**，把过程写全，因为"测什么"比"结论是什么"更有价值。

### 5.1 三轮实验

| 轮次 | 配置 | 结论 |
|------|------|------|
| 第一轮（8-25） | nonce 全冷测试 | "必须关"——但后来发现**只测到了回写税，收益场景从未测过** |
| 第二轮 | 开 HICACHE_SIZE=14 | 三 host 池按 device 字节比例切分；确认机制健康 |
| 第三轮（定案） | **HICACHE_SIZE=24** | 多会话轮转收益实锤，drop=0，生产开启 |

第一轮的方法论错误值得铭记：全冷测试只暴露"逐出回写税 -20%"，而收益（L2 回取）只在"多会话超 L1 轮转"场景出现——测试设计要先想清楚收益场景再测。

### 5.2 机制与数字

**池切分**（按 device 字节比例自动切三份）：
- 主 KV L2：302,848 tokens（9.92G，> L1 才有意义）
- mamba host：4.08G
- draft host：3.10G

**收益实测**：被逐出前缀从 DRAM 恢复 **5,233 tok/s**，比全量重算（~2.7k tok/s）**快 47%**。
**代价**：逐出回写税约 -20% 一次性（发生在逐出瞬间，不摊薄）。
**容量**：HICACHE_SIZE 14→24 后 host 池 519k tokens，300k 轮转临界解除，drop=0；扩容直接调 HICACHE_SIZE（hicache_size 覆盖 hicache_ratio），GPU 显存不变。

**收益场景判定**：
- 超过 L1 的多工具轮转（claude 200k + codex 100k）→ 收益区；
- ≤262k 单会话 → 无税无益（不触发逐出），开不开无所谓；
- 回关开关：`HICACHE_SIZE=0`。

### 5.3 write_back 机制判读（监控视角）

- **write_back 策略下 evict == backup 帧帧相等 = 零丢失**——逐出即备份，L2 是 L1 的完整镜像延伸；
- **L2 回取是被动触发**：请求前缀走回被逐子树才 load_back，不会自动补货。"图表上 L2 占用涨"不等于"会变快"，只有 load_tps 出现才代表收益兑现；
- **图表判读**：陡升 = 命中（一帧跳完）vs 斜坡 = 真 prefill（~2.5k tok/s），时间尺度差 60 倍，一眼可分。

### 5.4 已否决的变体：KV 放副卡（GPU1）

思路：4070Ti 有 12G 空闲，放 KV 池做"L1.5"。
实测否决，三重不成立：
1. 4090↔4070Ti `can_device_access_peer=False`（无 NVLink + WSL2 dxgkrnl 不开 PCIe P2P）→ 无直连时 GPU0↔GPU1 必须四跳绕宿主，比 DRAM L2 单跳 H2D 更慢；
2. 容量仅 ~190k tokens vs DRAM 519k；
3. sglang 无 GPU 层级缓存概念，改造成本无穷大。

Duanyll fork（动态 BAR1 P2P，22.7-26.3 GB/s 实测）评估过也不适用：①mod 改 Linux nvidia.ko，WSL2 走 dxgkrnl 无模块可 patch；②4070Ti 是 AD104 不在支持列表；③即便 P2P 通，KV 走 PCIe Gen4 与 H2D 读 DRAM 同一根线同带宽。**真正价值 = 未来裸机 Linux + 多张 48G 4090 做 TP/NCCL 时的现成方案**。

其他已否：L3 落盘（radix 树不落盘，重启收益不存在）；session radix cache（CC 的 CPA 请求无 session_id 字段，对 Claude Code 无效）。

---

## 6. 精度工程

### 6.1 GDN beta fp32 修复（865 倍）

**问题**：GDN（gated delta net）的 fused_recurrent.py packed decode 与 fused_gdn_gating.py 两处 beta 用低精度累积，1000 步漂移 4.1e-04。
**修复**：两处 beta 改全程 fp32。
**实测**：漂移降到 4.8e-07（**865 倍改善**），性能零回归（decode 67.3 / 长解码 65.3 / prefill 4244），accept len tail 达 7.25。
**状态**：已进生产。vLLM 同源问题在 #53877（仍 open）——本地修复领先上游。

### 6.2 ReplaySSM bf16 fold 漂移

§4.3 已述：t1 在 327/512 处语义等价分叉，是 bf16 re-quant fold 的预期漂移。**观察数日后的选项**：fp32 SSM（+3.2G 池）或 16 槽——当前两个都不动，2G 留作安全垫。

### 6.3 方法学

精度对照用三条 golden prompt（t1 创作文本 / t2 agent 任务 / t3 代码）逐字对比 + 语义等价判读。任何"动了数值路径"的改动（投机、量化、kernel 修复）都过这一关。

---

## 7. WSL2 特有坑清单（复刻者必读）

按踩坑惨烈度排序：

### 7.1 GPU VA ~1TB 上限（最阴险）
**症状**：EAGLE/MTP + expandable_segments 时进程 GPU VA 推过 dxgkrnl 的 ~1TB 上限，**所有 ioctl 返回 EOVERFLOW**，scheduler 死旋在 mamba alloc_group_end——进程不崩溃、不报错、只是僵死。
**解法**：MTP 模式去掉 expandable_segments（conf.sh 已条件化）。

### 7.2 chunked prefill 8192 崩溃
**症状**：CHUNKED_PREFILL_SIZE=8192 触发 FLA kernel device-not-ready 崩溃。
**解法**：2048 是 WSL 稳态（已验证），也是生产值。副作用见 §5.1 的 13-17x 归因——chunked 税是主因之一，但在 WSL 上没有选择。

### 7.3 wsl.exe 会话退出杀后台
**症状**：`wsl.exe -- bash -c "setsid ... &"` 启动的服务，宿主侧 wsl.exe 会话退出即被杀。
**解法**：必须在宿主侧保活的会话里跑启动脚本（比如常开的终端标签页）。stop 脚本只停主服务不停 encoder（设计如此——encoder 稳定不常动）。

### 7.4 AOT fallback 非法访问 pinned 内存
**症状**：HiCache write_back 的 MHA staged 路径在 WSL 的 AOT fallback 下非法访问 pinned 内存。
**解法**：`SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1` + `SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1`（aa90aa170e 提交封装）。

### 7.5 观测层缺失
`nvidia-smi --query-compute-apps` 在 WSL guest 内返回 `[N/A]`（无进程视图）。解法：Windows 侧读性能计数器，WSL 进程归因到 vmwp.exe，按 LUID 贪心匹配 GPU（monitor §4.2）。

### 7.6 无 P2P
`can_device_access_peer=False`（无 NVLink + dxgkrnl 不开 PCIe P2P）。影响：跨卡 KV 方案否决（§5.4）、未来 TP 需走 NCCL SYS 层。

---

## 8. 可观测性：monitor v1.3.2 与请求级归因

**没有观测就没有优化。** 本栈全部关键决策（LFU 陷阱、HiCache 翻案、mamba 断链）都靠这套工具实锤。

### 8.1 架构

```
WSL sglang :8778  /metrics + /get_server_info + main-boot.log
        │ 1s 采样                 │ UNC 路径 tail（Windows 侧直读 \\wsl.localhost\...）
Windows server.py :8917（纯标准库，零依赖）
        │ /api/snapshot /api/history /api/prefill_events
浏览器 index.html（单文件，原生 Canvas）
```

- 零第三方依赖：后端纯 Python 标准库，前端单文件 HTML；
- 1s 采样、30min 环形缓冲、10min 增量落盘 JSONL（崩溃最坏丢 10min，按小时追加幂等）；
- GPU 进程级显存从 Windows 计数器读（绕过 §7.5）；
- 完整设计文档见 monitor 目录 `DESIGN.md`。

### 8.2 verify_hz 的语义陷阱（v1.3.1 修复）

`spec_verify_calls_total` 是**按请求结账**的 counter（请求结束才一次性加一生的轮数），差分出来是结账脉冲，不是节奏。
**真·实时 verify 节奏 = gen_throughput gauge ÷ spec_accept**，且 accept 用上一帧值与吞吐窗口对齐（`prev_s` 机制）。
**通用教训**：Prometheus counter 分两种语义（事件发生时累加 vs 结账时累加），差分前先确认。

### 8.3 请求级 prefill 归因（PREFILL-REQ 日志）

sglang 树 4 连提交实现（`metrics_reporter.py` report_prefill_stats 内逐请求）：

```
PREFILL-REQ rid=xxx input=6457 chunk=0+2048 end=2048
    hit_device=0 hit_host=0 salt=- mamba_branch=181696 mamba_host_hit=0
```

- floor=4096 按**完整 input 长度**判（第一版按 per-chunk new 判 → chunked 请求全静默的三连坑：first chunk 误判 middle、tail chunk 低于 floor）；
- chunked prefill 逐 chunk 一行，可看推进；
- 环境变量 `SGLANG_PREFILL_LOG_MIN_TOKENS` 可调（0=全打）；
- monitor 侧：tail 日志 → `/api/prefill_events` → 前端折叠表格（同 rid chunk 折叠，**命中取首 chunk**——末 chunk 命中的是同请求前面 chunk 刚算的自己，会显示假 100%）。

### 8.4 归因判读指纹（给 AI 的弹药库）

| 指纹 | 诊断 |
|------|------|
| hit=0 且 mamba_branch=0 | **客户端根不同**：新窗口/不同 system prompt，radix 从第 0 页就分叉 |
| 高 KV 命中 + branch=0 | **GDN 状态链断**：KV 全在但 mamba 状态没了 |
| mamba_branch>0 但命中低 | **mamba 断在中间** |

字段语义：`mamba_branch`（MatchResult.mamba_branching_seqlen）="如果有 mamba 状态在分叉点，KV 本可命中到的长度"。

**实锤案例（本指纹的首战）**：同一主窗口出现两条 180k × 91 秒全量重算。归因日志显示 `mamba_branch=29,568 / 181,696`——**KV 树完整在池（KV 本可命中 181k），但 GDN mamba 状态链断了**，180k 白算 91 秒。根因链：Claude Code stop-hook 24 并发 64-token 小请求 → mamba 槽竞争 → 主前缀整树失效。这正是 LFU→LRU 之外的第二类"重复 prefill"根因，根治在 #36683（已手搓上线）+ #33639（上游合并后摘）。

另一案例：两条 81k 全量重算，池 free 充足、LRU 未逐出、连 25s 前刚存的也没认出 → 第 0 页就分叉 = 某客户端 system prompt 带变化注入（nonce/时间戳类），等 salt 字段复盘确认。

### 8.5 cache_salt 字段

可选请求参数（客户端 JSON 显式传），radix 树命名空间隔离用。`radix_cache.py:227` 只有非 None 才参与树 key。**所有请求显示 '-' 是预期**——没人传 = 全在默认共享命名空间。它不是 bug 探测器，是"将来客户端传了 salt 才有诊断价值"的预留字段。

---

## 9. 运维手册

### 9.1 日常操作

```bash
# 完整启动（encoder 先起，再主服务）
bash /root/boot-sgl-stack.sh
# 只重启主服务（encoder 不动，改 conf.sh 后用这个）
bash /root/restart-main-sglang.sh
```

注意：必须在宿主侧保活的会话里跑（§7.3）。

### 9.2 回滚开关速查

| 场景 | 动作 |
|------|------|
| KV 池 OOM | MAX_TOTAL_TOKENS 回 262000 |
| HiCache 异常 | HICACHE_SIZE=0 |
| DFlash2 异常 | ENABLE_DFLASH=0 ENABLE_MTP=1（注意 MTP 自动去 expandable_segments） |
| 逐出策略回滚 | --radix-eviction-policy lfu |
| 归因日志刷屏 | SGLANG_PREFILL_LOG_MIN_TOKENS 调大 |

### 9.3 生产纪律

- 生产时段（用户在用）不打长 decode 带载测试——带载请求会触发逐出，挤掉活跃会话（LFU 时代实锤过；LRU 时代风险已大减但纪律保留）；
- 短探测（/metrics）无影响，随便打；
- 任何源码改动：`git log` 确认提交链 → 重启主服务 → 冒烟（4/4 通过标准）→ monitor 观察 30min。

---

## 10. 复刻指南

### 10.1 最小硬件

| 档位 | 配置 | 裁剪 |
|------|------|------|
| 完整复刻 | 48G 单卡（魔改 4090 / A6000 / 6000 Ada） | 无 |
| 视觉分离版 | 48G + 任意 12G 副卡 | 无（本文档拓扑） |
| 紧凑版 | 24G（3090/4090） | 池缩到 ~120k，并发 2，DFlash 保留 |
| 最低 | 16G | 只跑 bf16 半精度小模型，本文档大部分优化不适用 |

### 10.2 复刻步骤（按序）

1. 裸 WSL2 + CUDA 驱动通（`nvidia-smi` 在 WSL 内可见两张卡）；
2. 克隆 sglang，建集成分支，按 §2.2 提交链从旧到新 cherry-pick（每摘一个冒烟一次）；
3. 拷 conf.sh 改路径，先跑 `ENABLE_DFLASH=1` 基线；
4. 起监视器（sglang-kv-monitor 三件套），**先于一切优化部署**；
5. 按 §9.1 启动，用 loadgen.py 冒烟；
6. 逐项开启优化，每项 A/B 记录（参照 ab-results.md 的口径）。

### 10.3 可裁剪项（按收益/复杂度排序）

| 优化项 | 收益 | 复杂度 | 建议 |
|--------|------|--------|------|
| GDN beta fp32 修复 | 精度 865 倍 | 两处改 fp32 | **必做** |
| LFU→LRU | 修复活跃会话被逐 | 一个参数 | **必做** |
| lpm 调度 | 长前缀优先 | 一个参数 | **必做** |
| ReplaySSM（#36683） | +2G 显存 → 池 +58k | 4 文件 apply | 强烈建议 |
| HiCache 24 | 多会话轮转 47% 提速 | 一个参数 + 两个 WSL env | 多会话用户必做 |
| 请求级归因日志 | 可诊断性 | 4 提交 | 排障前装好 |
| verify budget（#32053） | decode +10-15%（预估） | 待 A/B | 观望合并 |
| 树形草稿 W=2（#36196） | accept 3.42→~4 | 28 文件 | greedy 场景再考虑 |

### 10.4 借助 AI 继续优化（给朋友的方法论）

1. **先装观测再谈优化**：monitor + 请求级归因日志是 AI 诊断的输入源。没有数据，AI 只能猜；
2. **把 ab-results.md 的口径抄走**：同硬件、同 golden prompt 三条、同负载脚本、单项开关。A/B 不控制变量 = 白测；
3. **判读指纹直接喂给 AI**：§8.4 的三行指纹 + "hit_host vs hit_device"、"陡升 vs 斜坡"这些区分度特征，AI 拿到日志就能归因；
4. **抢跑纪律**：上游 open PR 小的直接 cherry-pick + A/B；翻案级收益的合并与否都自己动手；只有测试成本 > 预期收益才等。每个摘取进提交链（本文 §2.2 就是现成模板）；
5. **警惕"第一轮测试陷阱"**：HiCache 第一轮全冷测试得出"必须关"的错误结论，因为只测了代价没测收益。设计测试前先列出"这个特性的收益场景是什么"。

---

## 11. 遗留问题与下一步

| 事项 | 状态 |
|------|------|
| #32053 verify budget A/B（5/6/7 三档） | 下一个动手 |
| #36196 树形草稿 W=2 | 第二动作（需池回 262k；temp 1.0 验证是前置） |
| mamba_track_interval 256 实验 | GDN 断链短期缓解，已提议未批 |
| #36683 正式合并 | 合并后替换手搓版 |
| #33639 合并 | GDN 断链根治，最高优先盯防 |
| 23:25/23:26 两条 81k 零命中之谜 | 等 salt/mamba 指纹下次复现定案 |
| draft 域内微调（中文创作 accept 提升） | 中期选项 |

---

## 附录 A：sglang vs vLLM 差距归因（13-17x 从哪来）

冷 prefill 13-17x 的差距**不是引擎本质差距**，归因拆解：
- 8x 来自 chunked prefill 税（WSL 上 CHUNKED_PREFILL_SIZE 只能 2048，vLLM 侧等价配置不同）；
- 1.6x 来自每个税点的固定开销差。
换裸机 Linux 消除 chunked 约束后，差距预期收窄到 ~2x 以内。decode 侧 sglang 优势是真实的（+30%），来自调度与投机实现的差异。

## 附录 B：文件与数据索引

- 配置：`/root/sglang_qwen38_27b_dflash2.conf.sh`（全文含注释版注解）
- A/B 数据存档：`/root/ab-results.md`（359 行，全部实验原始数字）
- 监控三件套：`sglang-kv-monitor/{server.py, index.html, DESIGN.md}` + `loadgen.py` + `gpu_mem.ps1`
- 主日志：`/root/sglang-qwen38-dflash2/main-boot.log`（PREFILL-REQ 行在这里）
- 旧 vLLM 栈（保留对照）：`/root/vllm-research/qwen38-pr-stack`

## 附录 C：术语表

| 术语 | 含义 |
|------|------|
| GDN | Gated Delta Net，Qwen3.8 的线性注意力组件，混合架构中的 recurrent 侧 |
| DFlash2 | 自研/集成的投机解码方案，K=8 draft + 目标 verify |
| ReplaySSM | fold-every-commit 策略：verify 存原始输入，commit 时重放折叠 SSM，替代存完整快照 |
| HiCache / L2 | sglang 分层缓存，GPU L1 逐出的 KV 备份到 host DRAM |
| write_back | 逐出即备份策略（vs write_through），evict==backup 帧帧相等 |
| lpm | Longest Prefix Match 调度：前缀最长的请求优先 prefill |
| radix 驻留 | 已算完留池等前缀复用的 KV，可逐出，占满无害 |
| mamba_branch | 分叉点存在 mamba 状态时 KV 本可命中的长度（断链诊断核心字段） |
| accept len | 投机解码平均每轮接受的草稿 token 数（K=8 满分 8） |
| verify budget | 限制目标模型每轮 verify 的候选宽度（省算力不省草稿） |
