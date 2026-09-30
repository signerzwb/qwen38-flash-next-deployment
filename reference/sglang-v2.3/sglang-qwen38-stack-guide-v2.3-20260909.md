# Qwen3.8-27B 本地推理栈完整技术方案

**sglang + HiCache 分层缓存 + 视觉塔分离 + DFlash2 投机解码（WSL2 双卡）**

> 版本：**v2.3，2026-09-09 定稿**（v2.0 见 `sglang-qwen38-stack-guide-v2.0-20260831.md`，v1.0 见 `...-v1.0-20260829.md`，均归档保留）。
>
> **v2.3 相对 v2.0.1 的增量**（9-01 之后的一整轮）：① **模型换代为 NVFP4 混合精度**（自量化，KV 池 310k→**500k**，decode +45-60%）；② **HRRN 调度**取代 lpm（短请求 3× 提速）；③ **PDI=3** 消除 decode 抖动；④ **ReplaySSM #35544 攻坚失败**的完整记录（含"断层比 diff 大小更能预测移植成本"的方法论）；⑤ 补丁集从 32 个逐提交 patch 改为**单一合并 diff**（原方案实测无法应用，见 §2.3）；⑥ 智力三层标尺（L1/L2/L3）+ 拒答率/视觉评测；⑦ 监控面板 v2.2（启动日志显存归因、编码器警告）。
>
> 本文档是整套生产栈从选型、踩坑到定案的完整技术沉淀，面向两类读者：
> 1. 想直接复刻这套部署的朋友；
> 2. 想借助 AI（Claude Code 等）继续优化自己本地 sglang 部署的人——本文的"方法论"和"判读指纹"章节就是给 AI 看的弹药库。
>
> 所有性能数字均为同硬件 A/B 实测，非理论推算。
>
> **v2.0.1 修订（2026-09-01 qwen3.8 独立复核）**：① §0/§8.6 ITL 口径纠正——sum/count 本源是 **per-token 间隔（~17.3ms）**，旧版"57ms 是 verify 轮间隔"错，verify 轮间隔 = per-token × accept；② §0 TTFT 纠正——生产真实 P50≈2.5s/P90≈6.8s/P99≈15.8s，"165-195ms"是 0.8% 极短 prompt 孤例；③ §8.4 mamba_branch 判读表重写——`branch=0` 是不可能出现的死判读（源码把 0 转 None），None 有健康/真无命中两种成因；④ §3.1/§4.3 ReplaySSM "省 2G 属实但非回填 KV 池"（是安全余量，池扩容另有原因）；⑤ §0/§3 并发 3→5、mamba 槽 16→20（A1a 实测固化）。详见 `HANDOFF.md` 第四节、`research-notes/03~06`。

---

## 0. TL;DR 成果一览

| 维度 | 成果 | 数字 |
|------|------|------|
| 引擎选型 | vLLM → sglang | decode 单路 48-82 t/s（冷 prefill 快 13-17 倍，v1.0 实测） |
| 上游跟进 | 8-31 大合并 | fork 点后 644 上游 commits 一次吃下，冲突仅 10 文件 16 块 |
| **模型换代** | **NVFP4 混合精度（自量化）** | 权重 29G→**21.1G**，KV 池 310k→**500k**（+61%），decode **100-125 t/s**（+45-60%） |
| Agent 场景 | 子代理冷启动 | 3.3s vs 41.8s（**10 倍+**），多 agent 工作流质变 |
| 投机解码 | DFlash2 K=8 | accept 2.4-4.7（引擎侧 decode 48-125 t/s 随 accept 波动） |
| mixed chunk | **decode 冻结根除** | 150k prefill 期间 decode 从"完全零输出 20-40s"→ 持续产出 |
| **decode 抖动** | **PDI=3 消除** | 多会话涌入期 ITL p50 从 30ms 抖动到 581ms → **30/30/30ms 零抖动** |
| KV 池 | 500k tokens / 20 mamba 槽 | claude 200k + codex 100k 全进 L1 轮转；31 并发风暴零 OOM |
| 冷 prefill | chunk 2048 定案 | 16k 档 4211 tok/s（FP8）；NVFP4 档 3222 tok/s（Ada 无 FP4 tensor core，Marlin 税） |
| 分层缓存 | HiCache write_back，24G | L2 回取 5.2k tok/s 比全量重算快 47%；host 池 519k tokens drop=0 |
| **调度** | **HRRN（9-09 换）** | 短请求 med 7.41s→**2.48s（3.0×）**，长请求 p90/max 不恶化 |
| 精度修复 | GDN beta fp32 | 1000 步漂移 4.1e-04 → 4.8e-07（**865 倍**），领先上游 |
| **无审查保留** | 拒答率实测 | 敏感类 **8/8 compliance**，对照 4/4 正常 |
| **智力标尺** | L1+L2 双层 | **12/12**（与官方版/FP8 现役同分）；视觉 **5/5** |
| 可观测性 | monitor v2.2 | decode 冻结秒级告警、启动日志显存归因、请求级 prefill 归因、编码器健康探测 |

一句话定位：**这是一个为"超长上下文多 agent 日常使用"优化的单用户生产栈**，不是跑分栈。所有取舍（并发 5、HRRN、LRU、HiCache、chunk 2048、PDI=3）都围绕"主会话 200k+ 常驻、多工具轮转、TTFT 敏感"这个真实负载。

**当前性能基准（v2.3 口径，NVFP4 模型）**：

| 指标 | 数字 | 口径说明 |
|------|------|----------|
| 单路 decode | 100-125 t/s | **引擎侧**，NVFP4 模型；FP8 时代为 48-82 |
| accept len | 2.4-4.7 | 必须同任务形态对照（见 §8.6 陷阱） |
| ITL（per-token） | ~17.3ms/token（≈58 tok/s，FP8 口径） | **ITL sum/count 本源是 per-token 间隔**，非 verify 轮间隔 |
| verify 轮间隔 | = ITL × accept len（accept≈2.8 时约 48ms） | 一轮 verify 打包输出多个已接受 token，轮间隔远大于 token 间隔 |
| TTFT | P50≈2.5s / P90≈6.8s / P99≈15.8s | 253 请求直方图实测；"165-195ms"仅 ≤200ms 的 0.8% 极短 prompt 孤例，非生产均值 |
| 工具轮转 TTFT | 108-130ms | agent 高频动作体感基线 |
| 冷 prefill | 3222 tok/s @4k / 3178 @14k / 2911 @32k | NVFP4 口径；FP8 时代 4211 @16k |
| 命中 prefill | 68k tok/s | radix 命中场景 |
| 并发聚合 | 230.7 t/s @5 路 / 702 t/s @7 路 | 两套负载脚本口径不同，见附录 A |
| GPU util / SM clock | 91-93% / 2745 MHz | 已打满，功耗未触墙（307W p50 / 406W max，墙 450W） |

---

## 1. 硬件与拓扑

### 1.1 双卡分工

```
┌─────────────────────────────────────────────────────────┐
│ Windows 11 IoT LTSC（宿主）                              │
│   WSL2 (ubuntu2204) 内跑全部推理                          │
│                                                         │
│  GPU0: RTX 4090 48G 魔改卡 —— 语言模型                    │
│    Qwen3.8-27B-Uncensored-NVFP4-v2（21.1G）              │
│    + 500k KV 池 + 20 mamba 槽 + DFlash draft 模型，      │
│    sglang :8778                                         │
│    decode 峰值 ~47.5G / 48G（mem_fraction_static=0.985） │
│                                                         │
│  GPU1: RTX 4070 Ti 12G —— 视觉 encoder                   │
│    Qwen3.8 视觉塔（0.92G 权重），sglang :8779             │
│    主服务以 server_url 指向它做视觉推理                    │
└─────────────────────────────────────────────────────────┘
```

**模型说明（v2.3 换代）**：现役是**自量化的 NVFP4 混合精度版**
`Qwen3.8-27B-Uncensored-NVFP4-v2`（HF 公开仓库同名，21.1 GB 含视觉塔）。配方对齐
nvidia 官方版：MLP gate/up/down + lm_head = NVFP4，GDN/attn 投影 = FP8 e4m3，
embed/norm/视觉塔 = BF16。收益：权重 29G→21.1G，腾出的显存全给 KV 池
（310k→**500k tokens**），decode **+45-60%**。代价：Ada（SM89）无 FP4 tensor core，
prefill 走 Marlin W4A16 反量化税，冷 prefill 从 4211 → 3222 tok/s（**-23%**，物理极限）。
完整量化复现流程见模型仓库 README；上 Ada 必须 `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1`
（conf.sh 已固化，原因见 §3.7）。

为什么拆视觉塔：
- 语言侧 + 大池已经把 GPU0 显存吃满，视觉塔再挤进来就开不了 500k 池；
- 视觉请求是低频小负载，12G 卡完全够；
- 代价是图像请求多一跳 HTTP + **跨进程 pickle 序列化**（8-31 生产崩溃的根因载体，见 §9.4）。

### 1.2 为什么选 48G 魔改 4090

- 27B FP8 权重 ~27.5G，48G 才装得下"权重 + 大 KV 池 + mamba 槽 + draft 模型 + CUDA graph"全套；
- 该卡是 Duanyll/open-gpu-kernel-modules fork（动态 BAR1 P2P）针对的型号，未来裸机 Linux 多卡 TP 时有现成 P2P 方案（busbw 6.1×）。

### 1.3 分相位 GPU 墙（8-31 实测：两堵墙都摸到了）

用户问"prefill 能跑满功耗、decode 跑不满，算力和带宽用满了吗"——按相位分开测（50k 冷 prefill vs 单路纯 decode，0.4s 采样 nvidia-smi）：

| 相位 | 功耗 p50 | SM util | 显存 util | 判定 |
|------|---------|---------|-----------|------|
| 冷 prefill | 439W（max 458W，>380W 占 80%） | 100% | 23% | **功耗墙型** |
| 单路 decode | 302W | 91-93% | 83-85%（显存时钟 10501MHz 顶格） | **带宽墙型** |

解读：
- prefill 是计算密集，功耗顶到 439W/450W 墙、SM 100%——算力全用上了；
- decode 是每步读全部权重的访存密集，显存带宽顶格（时钟 10501 顶格）、SM 有余量自动降频——这是 GEMV 形态的物理规律，不是浪费；
- decode 剩下的 ~150W 功耗盈余 = 树形 verify 类"一次前向验更多候选"优化能兑现的空间（§4.5）；
- 450W = power.limit 顶格，无解锁空间；锁高频不涨 t/s。

> **v2.3 复核（9-09，NVFP4 模型）**：单路 decode 实测 **p50 307W / max 406W**，
> GPU util **p50 93%**、SM clock **2745 MHz 顶格**——与 FP8 时代同档。
> **判定：已达硬件最优**。瓶颈是 SM 计算占空比，不是带宽/功耗/调度；
> 剩余优化空间只在上游 kernel 层面。

### 1.4 WSL2 的代价（先说清楚，劝退或劝备）

WSL2 能用、好用，但有硬约束，全部在 §7 详述。最关键的四个：
- GPU VA 上限 ~1TB（dxgkrnl），激进显存特性会把进程 VA 打穿 → EOVERFLOW 死旋；
- 部分 CUDA 内核在 WSL 的 dxgkrnl 路径上不稳（chunked prefill 8192 崩溃 → 2048 稳态）；
- `wsl.exe` 会话退出会杀掉 setsid/nohup 的后台进程，长驻服务必须在宿主侧保活的会话里前台启动；
- 跨进程 pickle（主服务 ↔ encoder）对代码版本错位零容忍（§9.4 生产崩溃）。

如果条件允许，裸机 Linux 是更干净的地基。本栈的全部 WSL workaround 在裸机上可直接省略。

---

## 2. 软件栈与源码树

### 2.1 目录约定

```
WSL 内：
/root/sglang-source-qwen38-dflash2     # sglang 源码树，分支 radar-perf-0906
/root/sglang-env                       # 软链 → /root/sglang-envs/qwen38-dflash2-4be18295bda6
/root/boot-sgl-stack.sh                # 完整启动（encoder 先起 → 主服务）
/root/restart-main-sglang.sh           # 只重启主服务（见 §9.4 的 pickle 雷）
/root/sglang_qwen38_27b_dflash2.conf.sh# 全部配置，唯一调参入口
/root/sglang-warmup.py                 # 28 个 lazy kernel 启动期预热（restart 脚本自动调）
/root/sglang-qwen38-dflash2/main-boot.log  # 主服务日志（monitor 在 tail 这个）
/root/ab-results.md                    # 全部 A/B 测试数据存档
/root/LLM/Qwen3.8-27B-Uncensored-NVFP4-v2  # 现役模型（含视觉塔 + 复现脚本 + README）
Windows 侧：
C:\...\sglang-kv-monitor\              # 监控面板（server.py + index.html）
```

### 2.2 8-31 上游大合并工程（v2.0 核心增量，**历史记录**）

v1.0 时代树落后上游 600+ commits，逐个 cherry-pick 越来越痛。8-31 一次吃下：

- **fork 点** `a5c96362b6`（8-19），到合并时上游领先 **644 commits**；
- 新分支 `qwen38-main-merge` = 本地 29 个自研/抢跑 commit + origin/main 644 commits 的 merge；
- **冲突仅 10 文件 16 块**（预测 72 文件）——提前把上游 refactor 方向摸清是关键；
- 冲突取舍原则：scheduler/dp_attn/unquant/server_args/dflash/dspark 一律取上游，model_runner/pool 保留 #36911（GraphPoolPrecarve 上游没有），dflash_utils/dflash_worker 双边 import 合并。

**merge 后首日实锤三个 PR 的价值**：
- **#36288 mixed chunk base + #36933 spec×mixed chunk**：`--enable-mixed-chunk` + DFLASH 同时生效，150k prefill 期间 decode 从"完全冻结 20-40s"变成持续产出——**decode 冻结根除**；
- #36911 的 GraphPoolPrecarve + SamplingPrewarm 在新树上保留（上游没有），#37120 custom_mask 修复确认对 DFlash 路径是 no-op（上游修的是 EAGLE 路径）。

**踩坑记录**（复刻者必读）：
1. 冲突取 PR 侧后缺 import（get_memory/SamplingObserver——PR 依赖上游新文件 sampling_observer.py，79 行零内部依赖，直接 vendor）；
2. dflash_worker 死 import mamba_track_grid（fork 无此函数，删）；
3. **教训：冲突取 PR 侧必须把每个新符号的可解析性核一遍再重启**；
4. PR 自带 bug 一枚：batch_result_processor.py:339 `req.kv_committed_len` 应为 `req.kv.kv_committed_len`（ReqKvInfo 才有该字段，同文件 760 行写法正确，339 行是笔误）——已修并本地提交（26ac057043），上游合并时大概率也会修。

### 2.3 树上的提交链（38 个 delta，**交付方式已修正**）

**⚠️ v2.3 重要修正：原"逐个 `git am` 打 patch"的方法实测不可用。**
包内旧版 32 个 format-patch 从任何单点 `git am` 都会在**第 1 个就失败**——因为提交链
里有 **3 个 merge commit**（`1aac1cffe0`/`4be18295bd`/`1db40b230f`），且其中两个的
父提交来自**不在上游的 PR 分支**（`origin/pr-34859` 等）。`format-patch --no-merges`
生成的序列天然无法线性重放。

**v2.3 交付方式**：包内 `patches/qwen38-0909-merged.patch` 是
**从 merge-base 到 HEAD 的单一合并 diff**（86 文件 / 325KB）。实测验证：

```bash
git checkout -b qwen38-integration 0da6a66856   # merge-base，上游可达
git apply patches/qwen38-0909-merged.patch      # 一次应用
# 结果 tree = 53bbd2f6d78d...，与生产 HEAD 逐字节一致 ✓
```

`patches/by-commit/` 保留 38 个逐提交 patch，**仅供阅读/追溯**，不用于应用。

| # | 内容 |
|---|------|
| 0001 | Qwen3.8-27B 模型支持（基座） |
| 0002 | 视觉塔分离（language-only 模式跳过 vision tower） |
| 0003 | #35496 量化 lm_head |
| 0004 | #35957 retraction 丢 recurrent state |
| 0005 | HiCache WSL workarounds（两个环境变量开关，**WSL 必装**） |
| 0006-0008 | #36415 准入槽扣账 / #36696 split 崩溃修复 / #36266 COW 预热 |
| 0009 | **#36683 ReplaySSM**（手搓版，最大收益项） |
| 0010-0013 | 请求级 prefill 归因日志（4 连迭代） |
| 0014-0017 | HiCache 三修复 + mmap 去重（#36705/#35931/#36738/#36798） |
| 0018-0019 | #37120 custom_mask 上限 / #36970 verify 循环按 shape 展开 |
| 0020-0022 | #37024 chunked 进度保留 / **#36911** GraphPoolPrecarve + Borrow（三连前二） |
| 0023-0026 | #36911 配套修复（test fixture / import ×2 / vendor sampling_observer） |
| 0027-0028 | **#36288 mixed chunk base + #36933 spec×mixed chunk**（decode 冻结根除） |
| 0029 | #36933 自带笔误修复（kv_committed_len 落在 req.kv） |
| 0030 | **#37613 mamba host 池溢出环 + #37619 alloc-fail 跳过 + #37612 hybrid SSM batch_full 重置**（三连抢跑，单测 10/10） |
| 0031 | **mamba 归因日志 + 5 分钟审计心跳 + BACKUP-ON-EVICT**（9-03 根因修复，见下） |
| 0032 | **#37837 ReplaySSM ring cursor 在 extra_buffer 捐赠路径复位**（4 行，9-04） |
| 0033 | DFlash sampling accept_len 钳位（fold OOB 越界修复） |
| 0034-0036 | **#35544 移植尝试 + 回滚**（失败记录见 §4.8；合并 diff 里 net 效果 = 零） |
| 0037 | **#36267 Qwen3.5 GDN prefill 投影布局优化**（上游已合并） |
| 0038 | **#37324 radix 按 offset 走树**（上游已合并，避免 token 存储 re-slice） |

> **0034-0036 在最终树里是净零**：34 是全量移植、35 是配套补丁、36 是回滚。
> 合并 diff 已自动消解，朋友复刻时**不会**拿到那半次失败移植的残留。

**0031 BACKUP-ON-EVICT 是治本修复**：9-03 实证多会话场景下，活跃会话
chunked prefill 挤占 20 槽 mamba 池，闲置会话链被 LRU 驱逐且内部节点
tombstone 无备份 → 该会话回来 mh=0 全量重算（54k 重算 40s）。修复在
`evict_component` 单点：驱逐前无 host 副本则同步 D→H 备份（~11ms），
回来时 mh=1 从 host 秒回。生产验证：驱逐+回取同帧闭环，回归会话 100% 命中。
配套观测：`MAMBA-EVICT-BACKUP`（备份成功）/`MAMBA-EVICT-DROP`（host 也满，
查溢出环）/`MAMBA-AUDIT`（5 分钟树健康心跳）/归因面板"归因口径命中率"。
conf.sh 对应参数：`--mamba-max-states-per-path 6`（每链保尾弃头）。

**抢跑纪律**（v1.0 原则继续有效）：上游 open PR 小的直接 cherry-pick + A/B；翻案级收益的合并与否都自己动手；只有测试成本 > 预期收益才等。
**摘取方法论**：open PR 用 `git fetch origin pull/N/head` 后摘真 commit（不能摘作者的 merge tip）；上游已合并的直接 cherry-pick 上游 commit；`git log --author=root` 区分自研与抢跑（抢跑摘取后作者仍是上游作者）。

### 2.4 PR 盯防清单（**9-09 全量重扫后**）

**已摘并上线（9-09 批次）**：
- **#32911 HRRN 调度**（MERGED → 已摘，短请求 3× 提速，见 §3.4）
- **#38117 Gumbel-max 采样器**（MERGED → 已摘，1 行，带 kill switch）
- **xgrammar 0.2.1 → 0.2.6**（依赖升级，修 JSON schema 类型校验 bug）

**已摘（9-06 批次，上游均已合并）**：
- #36267（GDN prefill 投影布局）/ #37324（radix offset 走树）/ #36911 / #36933

**9-09 扫描被否决项（记录备查，勿重复评估）**：

| PR | 否决原因 |
|---|---|
| **#38344** GDN prefill 上下文并行 | 需 CP 组（我们 tp=1）、需 PD 分离、显式拒绝 mixed-chunk——三重不适用 |
| #37165 Mamba 延迟初始化 | 触发条件是 DRAFT_EXTEND_V2，DFlash 只走 TARGET_VERIFY |
| #38625 mamba 前缀墓碑 | 我们用 extra_buffer_lazy；实测 cached_tokens 复用正确 |
| #34820 mamba checkpoint dtype | 我们 ssm-dtype=bf16 == 激活 dtype |
| #37103/#38481/#38480/#38482 | 无 PD 分离 / 无 L3 storage / 无 Rust tree core |
| #38159 | 无 SWA 池（纯 GDN + full attention） |
| #38565 确定性 renorm | tp=1 无 rank 分歧；默认 top_p=1.0 |
| #37411 CUDA graph 懒实例化 | 已禁用 prefill CUDA graph；收益是冷启动 |
| #31586 / #36810 | 面向 dLLM / Blackwell，平台不符 |
| #38108/#38139/#37843 | Router 多机场景，单卡无益 |

**高价值盯防**：
- **#37065**（DFlash prefill radix 前缀复用——draft 状态复用，命中场景 TTFT 直接受益；仍 draft，8-29 后冷置）
- #36136（DFlash2 置信度动态验证，+2204 行大盘子）
- #37164（mamba state 入 ReqKvInfo——**注意：它是 #35544 的地基**，见 §4.8）

**降级/放弃**：
- #35544（ReplaySSM checkpoint 物化摊销）——**移植失败，见 §4.8**
- #32053 verify budget 自 7-31 停滞——上游活力已转移

**vLLM 侧对照观察**：#53877（GDN beta，本地修复继续领先）、#51725（自适应投机预算）、#52244（GDN 哈希对齐零命中现象）。

### 2.5 conf.sh 三模式投机切换

```bash
ENABLE_DFLASH=1 ENABLE_MTP=0 ENABLE_DSPARK=0   # 生产：DFlash2 K=8
ENABLE_DFLASH=0 ENABLE_MTP=1 ENABLE_DFLASH=0   # EAGLE MTP 对照
ENABLE_DFLASH=0 ENABLE_MTP=0 ENABLE_DSPARK=1   # RadixArk DSpark 对照
```

任何切换都只需要改两个变量 + 重启主服务，杜绝"参数改了但忘了回"的事故。

---

## 3. 核心配置逐项解读

配置全貌见包内 `conf/sglang_qwen38_27b_dflash2.conf.sh`，此处讲每个值的 why：

```bash
# —— 池与显存 ——
MAX_TOTAL_TOKENS=500000        # KV 池（见 3.1，NVFP4 换代后 310k→500k）
MEM_FRACTION_STATIC=0.985      # 27B+大池场景顶格
HICACHE_SIZE=24                # L2 host 池（见 §5）

# —— 混合架构（GDN 线性注意力 + 全注意力）——
MAX_MAMBA_CACHE_SIZE=20        # mamba 状态槽（见 3.2，12→16 翻案；9-01 A1a 随并发 16→20）
MAMBA_TRACK_INTERVAL=512       # SSM checkpoint 间隔

# —— 调度 ——
MAX_RUNNING_REQUESTS=5         # 并发上限（9-01 A1a 实测固化 3→5：智商哈希逐字节一致、K=5 wall -9.6%）
SCHEDULE_POLICY=hrrn            # 响应比优先（9-09 换，见 3.4；lpm 仍可一行切回）
PREFILL_DECODE_INTERVAL=3      # prefill 后连做 3 轮 decode，消除 ITL 抖动（见 3.4b）
CHUNKED_PREFILL_SIZE=2048      # 定案勿再标定（见 3.6 翻案史）
page-size 64                   # 大 page 减少 radix 元数据与 HiCache 页表开销

# —— mixed chunk（8-31 新增，decode 冻结根除）——
--enable-mixed-chunk           # #36288+#36933，prefill 期间 decode 不再冻结

# —— 逐出 ——
--radix-eviction-policy lru    # LFU 陷阱（v1.0 §3.5，结论不变）

# —— 图池优化（#36911 抢跑）——
SGLANG_ENABLE_GRAPH_POOL_PRECARVE=1
SGLANG_ENABLE_GRAPH_POOL_BORROW=1

# —— NVFP4 模型专用（Ada/SM89）——
SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1   # 见 3.7，不设必崩
```

### 3.1 为什么是 500k：池账本与压测定案（NVFP4 换代）

演进史：262k → 320k（v1.0）→ merge 后重标定 → 310k（v2.0）→ 470k（NVFP4 v1 上线）
→ **500k 定案**（NVFP4 v2 混合配方）。

**换代逻辑**：FP8 权重 29G → NVFP4 混合配方 **21.1G**（含视觉塔），腾出 ~8G
全部给 KV 池。每 token 账本按实测 `31,4xx B/tok`（fp8 KV），310k→500k 需 ~6G。

**500k 压测（三档对比）**：

| 池大小 | 启动 | 压测后显存 | 余量 | `0.00 GiB` 警告 |
|---|---|---|---|---|
| 470k | OK | 45.8G | 3.3G | 0 |
| **500k（现役）** | OK | **47.5G** | **1.6G** | **0** ✅ |
| 520k | OK | 48.1G | 1.0G | **47 条** ⚠️ |

**520k 被否决**：压测时出现 47 条 `free device mem: 0.00 GiB` 警告（Triton kernel
服务期加载无空间，随时 OOM 崩溃，见 §7.7）。**500k 是安全上限。**

> ⚠️ **8-31 复核纠正**：旧版把 262k→320k 归因于"ReplaySSM 释放 2G 回填 KV 池"——
> **"省 2G"属实，但"回填池"不成立**。GLM 原注释明说这 2G 是**安全余量（safety
> margin）**而非回填进 KV 池；池扩容另有原因，#36911 GraphPoolPrecarve 更可能是真正推手。

**310k 压测（#36911 抢跑后，FP8 时代）**：31 并发风暴 + MAIN 150k tokens 同时进场，
GPU0 峰值 48,086 MiB、floor 1,054 MiB（历史最坏 570 MiB 的 1.85 倍安全垫），**零 OOM**。
#36911 的 rehearsal 机制把旧代码 +1.3GiB 瞬态尖峰彻底吸收——**旧口径"静态扩池有不可预测
瞬态尖峰"作废，新代码下静态扩池的地板可预测**。

池的动态构成（monitor 实测）：

```
500,000 = running ~180k（活跃请求 KV）
        + radix 驻留 ~317k（算完等复用，可逐出）
        + free ~2-3k
```

**radix 涨满是设计使然**——空闲空间给缓存是零成本的，"radix 用满但 running 只有 180k"不是故障。若将来 OOM，回滚开关就一行（MAX_TOTAL_TOKENS）。

### 3.2 为什么是 20 槽：mamba 槽翻案史（12 → 16 → 20）

v1.0 结论"12 槽并发 3 留安全垫"被 7-agent workflow 风暴实测推翻：

- 12 槽时代风暴（queued 峰 52）：9-11 槽被 running 死占、机动位只剩 1-3，mamba 状态被挤 → **断链灾难行 24 条、浪费 1.9M tokens（占全部重算浪费 72%）**；
- 且 write_back 模式下 mamba 状态逐出不回写 host（mamba_component.py 注释自认），灾难行全部 mamba_host_hit=0 无 L2 兜底；
- **16 槽后**：available_gpu_mem 反而升到 1.26G（池缩的），灾难行根除；
- **20 槽（9-01 A1a 固化）**：并发 3→5 后，4 并发需 20 槽起步（4×3 死占 + 机动位），
  槽的机动位从 1-3 回到 4-5。验收：智商冒烟 5/5 且输出哈希与基线逐字节一致、
  单流 decode 52.7 vs 51.2 无衰减、长上下文 K=5 wall 11.83→10.70s（-9.6%）、并行度 6.3→7.5。

**坑**：mamba 槽是稀缺资源，槽竞争会把主前缀的整棵 radix 树变无效（GDN 状态链断，§8.4 实锤案例）。#36415（准入槽扣账）和 #36696（split 崩溃）就是修这一层的。
**9-03 追加**：槽竞争还会让**闲置会话链被 LRU 驱逐且内部节点 tombstone 无备份** →
回来 mh=0 全量重算。治本修复 = BACKUP-ON-EVICT（§2.3 提交 0031）+ `--mamba-max-states-per-path 6`。

### 3.3 MAMBA_TRACK_INTERVAL=512

每 512 token 存一个 SSM checkpoint，决定"从最近 checkpoint 重放"的粒度。512 是治标：hook 风暴逐出后树更浅、恢复更快。约束：512 ≥ num_draft_tokens(8)，512 % page_size(64) == 0。#36683 ReplaySSM 合并后此参数的重要性下降。

### 3.4 SCHEDULE_POLICY=hrrn（9-09 从 lpm 换过来）

**演进**：fcfs → lpm（Longest Prefix Match）→ **hrrn**（Highest Response Ratio Next，#32911）。

**lpm 的盲区**（实测暴露）：它只认前缀匹配长度，**对请求长度完全无感**。
5 轮 × 12 路并发 A/B（`--prefill-max-requests 1`）里，lpm 下短请求（60 tok）
中位 **7.41s** 比长请求（2500 tok，5.80s）**还慢**——因为它一直排在长前缀后面。

**HRRN 的排序键**：`等待token / 未缓存token`，即"谁等得久 + 谁算得快"，
比值高者优先，带老化保护避免饥饿。

**决定性 A/B（5 轮 × 12 路并发 = 60 请求）**：

| 请求档 | LPM | HRRN | 变化 |
|---|---|---|---|
| 短 (60 tok) med | 7.41s | **2.48s** | **3.0× 快** |
| 短 p90 | 13.05s | **4.15s** | **3.1× 快** |
| 短 max | 13.36s | **5.10s** | 2.6× 快 |
| 长 (2500 tok) med | 5.80s | 9.03s | 1.56× 慢 |
| 长 p90 | 13.07s | 12.86s | 持平 |
| 长 max | 13.48s | 14.22s | 持平 |

**取舍**：长请求中位数 +3.2s 换短请求 3× 提速，长请求 p90/max 不恶化（无饿死）。
我们的真实负载是 **agent 高频小请求 + 少量长会话**，短请求延迟是体感主导项，故选 HRRN。
**若你的负载以长文档为主，改回 `SCHEDULE_POLICY=lpm` 一行即可**（conf.sh 第 65 行附近）。

单元验证 5 用例全过：短优先 / 老化反超 / 全缓存立即出队 / 降级请求排最后 / hicache 命中算已缓存。

### 3.4b PREFILL_DECODE_INTERVAL=3：消除 decode 抖动（9-09 新增）

**问题**：mixed chunk 保证 prefill 期间 decode 不冻死，但**ITL 会剧烈抖动**——
实测 1 路 decode + 5 个新会话同时涌入时，decode ITL p50 从 30ms 崩到 **581ms**（20 倍），
体感就是"卡住"。

**机制**：`--prefill-decode-interval N` = 每做完一个 prefill chunk，先连做 N 轮 decode
再调度下一个 prefill。调的是**优先级**，不是并行度（单卡上 prefill/decode 抢同一批 SM，
"同时"本质是时分复用）。

| 配置 | decode ITL p50（涌入期） | p90 | 第 5 会话 TTFT |
|---|---|---|---|
| PDI=0（原） | 30ms **但抖动到 581ms** | 603ms | 4.4s |
| **PDI=3（现役）** | **30/30/30ms 零抖动** | 577ms | 4.7-8.2s |
| PDI=8 | 29/29/29ms | 336ms | 6.8-9.7s |

**取舍**：decode 抖动消除，新会话首字多等 0-3 秒。PDI=8 抖动更低但新会话 TTFT 更差，
3 是甜点。回滚：`PREFILL_DECODE_INTERVAL=0`。

### 3.5 radix-eviction-policy：LFU 陷阱与 LRU 定案（v1.0 结论不变）

LFU 的实现陷阱（`radix_cache.py`）：`hit_count` 只在 **insert 时 +1**，match 命中**不加分**。后果：单会话持续增长场景下，"最新长出的对话分支" hit_count 永远最低 → 被 LFU 当冷数据优先逐出——**恰好逐到用户最活跃的对话部分**。

实锤案例：带载测试请求（2.5k prompt + 4.5k decode）在池剩 2.2k free 时，把主窗口 24k 的活跃分支挤掉，触发 24k evict 风暴 + 全量重 prefill。

LRU 按最后访问时间逐，活跃分支刚被 match 刷新，必然最后被逐。

运维教训：生产时段（用户在用）不打长 decode 带载测试；短探测（/metrics）无影响。

### 3.6 CHUNKED_PREFILL_SIZE：1024 翻案史与 2048 定案（勿再标定）

这参数翻过两次案，过程有教学价值：

| 档位 | 冷 prefill @16k | 150k prefill 期间 decode ITL p50 | 定案 |
|------|----------------|--------------------------------|------|
| 8192 | — | — | WSL 崩溃（FLA kernel device-not-ready），不可用 |
| 2048 | 3885→4211 tok/s | 865ms | **生产定案** |
| 1024 | 1323 tok/s（**砍 2/3**） | 455ms | 曾短住生产，翻案出局 |
| 512 | 更差 | 287ms | prefill 拖 26%，出局 |

翻案过程：mixed chunk 上线后 ITL 三轮标定（870/455/287ms），1024 看似甜点（ITL 减半零代价）→ 入住生产 → 用户问"冷 prefill 怎么只有 2k5"引出 A/B → **1024 把冷 prefill 吞吐砍 2/3**（chunk 越小每 token 摊的 kernel/调度开销越大，越长越亏）→ 回 2048 复测 4211 tok/s 归位基线。

**权衡逻辑**：用户场景长对话续写 + 新 agent 冷启动是硬成本，3 倍冷 prefill 差距比 prefill 期间 decode ITL 455→865ms 更疼；且 mixed chunk 保底在，decode 不会冻死。

**定案：2048 勿再标定。** 单路 decode 与 TTFT 两档无差（ITL p50 51ms / TTFT 117-195ms）。

**9-09 复测补充**：chunk 4096 在 NVFP4 模型上 **prefill 吞吐三档相同**（3014/3008/2989 tok/s），
但 decode ITL p50 涨到 370ms（2048 是 30ms）——**chunk 越大 decode 等待窗口越长**。
且 chunk **不额外占显存**（三档 GPU 占用相同），"把显存给 chunk"的前提不成立。结论不变。

### 3.7 NVFP4 模型专用配置（9-09 新增）

现役模型是 NVFP4 混合精度，Ada（SM89）上有两个必知项：

**① `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` 必设，否则启动即崩**

```
ValueError: Invalid backend: 89
```

flashinfer 的 `silu_and_mul_scaled_nvfp4_experts_quantize` 融合 kernel 的 backend 表
只有 `90/100/103/107/110/120/121`，**无 SM89 实现**。崩溃点在 CUDA graph capture 阶段的
MLP forward（`qwen2_moe.py:243 _silu_fp4_quant_fused`）。

关闭融合后 down_proj 走 **Marlin W4A16**（`modelopt_quant.py:1949 apply()` 的 marlin
分支在最前，接收普通 BF16 张量）——融合只是纯优化，关掉不影响正确性。
该开关对 FP8 checkpoint 无影响（gate 不触发），已固化进 conf.sh。

**② prefill 的 Marlin 税是物理极限**

Ada 无 FP4 tensor core，FP4 权重每步都要反量化。实测：

| 相位 | FP8 时代 | NVFP4 | 变化 |
|---|---|---|---|
| 单路 decode（代码任务） | 73-84 t/s | **100-125 t/s** | **+45-60%**（权重带宽省了） |
| 冷 prefill 14k | 4280 tok/s | ~3200 tok/s | **-25%**（反量化税） |

**decode 赢在权重带宽（FP4 权重是 FP8 的一半），prefill 输在反量化计算**。
Blackwell 有原生 FP4 tensor core，prefill 不会吃这个税。

**③ 模型来源**：HF 公开仓库 `Qwen3.8-27B-Uncensored-NVFP4`（含视觉塔、复现脚本、
中英文 README）。若用 FP8 版（`orcarouter/Qwen3.8-27B-Uncensored-FP8`），本节两项都不需要，
但 KV 池只能到 310k、decode 只有 48-82 t/s。

---

## 4. 投机解码：三模式对比与选型

### 4.1 全量 A/B 数据（同硬件同负载）

| 方案 | accept len | 单路 decode | 结论 |
|------|-----------|------------|------|
| **DFlash2 K=8**（生产） | 2.4-4.7（创作文本 1.6-2.8） | 引擎侧 48-125 t/s（随模型/任务形态） | **胜出** |
| MTP / EAGLE 内置头 | — | 慢 3-4%，子代理 TTFT 差一个量级 | 否决 |
| DSpark（RadixArk，draft 2.6G） | ~2.4 | **-45%** | 否决 |

要点解读：
- DFlash 的 accept ~3.5 是 greedy agent 场景均值；**中文网文长 prose + temp 1.0 的创作负载实测只有 1.6-2.8**——创作文本投机命中天然低，这是后续微调 draft 的动机（§11）；
- DSpark 的强项是高熵采样场景，greedy agent 场景不是它的战场；
- DSPARK 分支注意：必须带 `--speculative-draft-model-quantization unquant`（BF16 草稿 + FP8 target 时默认继承会拒载）。

### 4.2 MTP 模式的 VA 大坑

EAGLE 双 worker + `expandable_segments` 会把进程 GPU VA 推过 dxgkrnl 的 **~1TB 上限**：ioctl 全部 EOVERFLOW，scheduler 死旋在 mamba alloc_group_end，表现为进程僵死不报错。conf.sh 已条件化：MTP 模式自动去掉 expandable_segments。详见 §7.1。

### 4.3 ReplaySSM（#36683）——本栈最大的单笔抢跑

**问题**：DFlash verify 每步要存 per-draft 完整 SSM 快照，spec intermediate 缓冲吃显存。
**方案**：verify 存每步原始输入（ring buffer），commit 时重放折叠（fold-every-commit）。
**实测 A/B**：
- 显存省 ~2G（空闲 47.4→45.3G，decode 峰 -1.7G）——注意隔壁"8.6G scratch"的估算远过于乐观，fold 模式只消除一部分中间缓冲；
- decode -3%（80→78 t/s）——重放有代价；
- 精度：t2/t3 逐字一致，t1 在 327/512 处语义等价分叉（bf16 re-quant fold 的预期漂移，非 bug）；
- **决策**：2G 换 -3% decode 值得——释放的显存作为**安全余量**（8-31 纠正：非"回填 KV 池"，见 §3.1 复核注）。
- 8-31 merge 后此 PR 生态已全进上游基座，本地手搓版与上游同源。

### 4.4 mixed chunk（#36288+#36933）——decode 冻结根除

v1.0 时代的痛点：150k 大 prefill 期间 decode **完全零输出 20-40s**（调度器被 prefill chunk 独占）。#36288 mixed chunk base 把 prefill/decode 混批，但与 spec decoding 不兼容；**#36933 [2/N] spec×mixed chunk 补上这块**。

merge 后从上游树摘 #36933 真身 diff（589 行 14 文件 --3way 全干净 apply）——前一天"不可能移植"的判定作废，判死的是被 refactor 污染的旧树，merge 后同源直接吃。

**实测**：150k prefill 期间 decode 从零输出 → 持续产出（2048 档 ITL p50 865ms）；对比旧代码完全冻结，质变。代价与收益的完整权衡见 §3.6。

### 4.5 draft CUDA graph A/B 定案：维持 off（8-31 晚）

**点亮条件**：dflash_worker_v2.py capture 时 avail > 1.0G 硬阈值。310k 池剩 0.46G 不够；**300k 剩 0.87G 也不够；285k 才点亮**（selector+greedy+sampling 全折进图）。

**用户问"我可以接受 300k KV，如果有加速的话"，并要求做 agent 使用场景的多维度评测，不要光数字好看**。三维 agent_bench.py（flush_cache 同起点）：

| 维度 | ON (285k) | OFF (310k) | 差异 |
|------|-----------|------------|------|
| P1 单路代码 decode | 282.5 t/s | 270.9 t/s | +4%（单样本噪声） |
| P2 工具轮转 TTFT ×6 | p50 108ms | 119ms | -9%（噪声量级） |
| P3 7 路并发聚合 | 478 t/s | 497 t/s | -4%（反向噪声） |

**结论：draft graph 加速 ≈0-5%，全维度在噪声内。不值得为它把池从 310k 缩到 285k（-8% 上下文容量）。维持 310k / draft-graph off。** 用户 300k 预算用不上——没有可兑换的加速。

**monitor 配套**：boot log 正则解析 draft graph 状态（"Disable DFLASH draft cuda graph" = off / "DFLASH ... folded into the draft cuda graph" = on），前端徽章 `draft-graph:off|on`。

> **v2.3 复核（9-09）**：本结论在 FP8/310k 语境下取得。NVFP4 换代后池已到 500k，
> 若要重新评估 draft graph，点亮条件（capture 时 avail > 1.0G）需要重新测——但**结论大概率不变**：
> 加速来自把 selector+greedy+sampling 折进图，与模型精度无关，仍是 0-5% 噪声级。
> 除非有人愿意用几十 k 池容量去换，否则不值得重做。

### 4.6 树形草稿 #36196：三门槛评估（等上游）

上游实测 accept 3.42→4.39@W=4、吞吐 +23%，但三个门槛都未过：
1. W=4 draft SSM workspace +7.4G 显存爆；W=2（+2.5G）需池回 262k——放弃 310k 容量换 accept，与本栈定位相反；
2. PR 只验证了 greedy，**temp 1.0 下树采样分布未验证是创作栈主风险**；
3. 28 文件大盘子，维护成本高。

**结论：等上游成熟（尤其 temp 1.0 验证）再评。** decode 剩余 150W 算力盈余（§1.3）的兑现路径仍是它，但不是现在。
**v2.3 补充**：NVFP4 换代后显存压力已缓解（权重少 8G），W=2 的 +2.5G 代价相对可接受，
但第 2 条门槛（temp 1.0 未验证）仍卡着——**创作栈不能拿分布正确性冒险**。

### 4.7 decode 再提速路线图

1. **#37065 DFlash prefill radix 前缀复用**（当前最高优先盯防）：draft 状态复用，命中场景 TTFT 直接受益——正中 agent 场景；
2. #36136 置信度动态验证（大盘子，等稳定）；
3. **域内微调 draft**（中期最直接杠杆）：现有 draft 对中文网文长 prose 的 accept 仅 1.6-2.8，对 draft 做 CPT 是命中率的最大杠杆；
4. 已否：#34171 AdaFlash（需训 checkpoint、60 文件、不支持 Qwen3.8-27B）；#32053 verify budget（停滞 6 周降级）。

### 4.8 #35544 移植失败全记录（9-06，**方法论价值高于结论**）

上游 #35544（ReplaySSM checkpoint 物化摊销，宣称 +3.4% 吞吐）看着和我们的手搓
#36683 同源，diff 只有 850 行，于是动手移植。**结论：放弃。** 三次启动失败定位如下：

| # | 错误 | 根因 | 处理 |
|---|---|---|---|
| 1 | `MemoryPoolConfig has no attribute 'unified_total_bytes'` | 上游 configurator 需要 config 透传新字段 | 已修（补字段默认 None） |
| 2 | `--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA model` | 上游守卫把 GDN×DFlash 判非法 | 已放宽（论证见下） |
| 3 | `'ReqKvInfo' object has no attribute 'holds_kv'` | **地基断层** | **未修——放弃原因** |

**第 3 项是致命的**：上游的 memory_pool/spec_utils 建立在**一次完整的 Req 状态模型
迁移**上（`mamba_pool_idx` 从 Req 本体迁入 `req.kv`、`holds_kv`/`holds_mamba`/
`is_kv_released` 等 property、`mark_released` 语义）。我们树里旧风格读写点
（`req.mamba_pool_idx`）**45 处**，新风格（`req.kv.*`）**12 处**——这是**半次 rebase
的工作量，不是补丁**。

**关键技术判读（下一棒可直接引用）**：

1. **kernel 文件零差异**：`git show` 逐行对比确认我们树的
   `gdn_replayssm_spec_decode.py` 与上游 **同为 1555 行、函数签名逐一相同**。
   环形历史/窗口物化的 kernel 我们 8-29 手搓 #36683 时就带进来了。
   **#35544 的新增价值全在调用层**（cursor 推进式 commit 替代每步物化），
   而调用层依赖新 req 状态模型——**这就是"拆肉摘取"失败的根源：肉（kernel）
   我们本来就有，骨（调用层）拆不下来。**
2. **上游守卫是过严防御，不是真风险**：`gdn_backend.py:866` 上游自注释
   "this branch is unreachable with a None buffer **by construction**"；
   `spec_utils.commit_mamba_states_after_verify` 881/925 行有专门的
   `not replayssm_is_kda` GDN 分支。守卫放宽是安全的（已验证到 CUDA graph
   capture 全过、KV 池 309,952 正常）。
3. **崩溃点全在启动早期（pool 构建阶段）**，不在运行时——说明若做适配，
   风险集中在启动自检，错了立刻崩，不会静默错数据。

**⭐ 方法论（比这个 PR 本身值钱）**：

> **上游 merge 后的 PR 与我们抢跑树之间的断层，比上游 PR 本身的 diff 大小
> 更能预测移植成本。** #35544 diff 只有 850 行，但它脚下的地基 diff 有几千行。

**下一步路线（按性价比）**：
1. **正路**：先摘 #37164（mamba state 入 ReqKvInfo）+ 相邻 req 状态迁移 refactor
   （它们是 #35544 调用层的地基），再摘 #35544 调用层。每步有上游 commit 可循，不是盲改；
2. **等 rebase**（推荐）：下次上游大版本 rebase 时这些 refactor 自动到位，
   #35544 收益白拿。零成本，但要等；
3. **放弃收益**：现栈 100-125 tok/s decode 已是手搓版闭环调优的产物，
   +3.4% 的绝对值 ≈ 3-4 tok/s，日常体验增量有限。

**可回收资产**：`radar-35544` 分支保留了完整移植尝试链（含 unified_total_bytes
补丁、守卫放宽两个版本），路线 1 动工时可直接 cherry-pick。
**上游 issue 候选**：configurator 守卫与 spec_utils GDN 分支自相矛盾，
GDN fold 路径被误伤——这是个真 upstreamable 的发现。

---

## 5. HiCache 分层缓存：定案与运维语义

### 5.1 定案（v1.0 三轮实验的结论继续有效）

- **HICACHE_SIZE=24**（8-31 从 14 上调）：host 池 519k tokens，300k 轮转临界解除，drop=0；
- 扩容直接调 HICACHE_SIZE（hicache_size 覆盖 hicache_ratio），GPU 显存不变；
- 三 host 池按 device 字节比例切分：主 KV L2 ~9.92G / mamba host ~4G / draft host ~3G；
- 收益实测：被逐出前缀从 DRAM 恢复 **5,233 tok/s**，比全量重算（~2.7k）快 **47%**；
- 代价：逐出回写税约 -20% 一次性（发生在逐出瞬间，不摊薄）。

收益场景判定：
- 超过 L1 的多工具轮转（claude 200k + codex 100k）→ 收益区；
- ≤260k 单会话 → 无税无益（不触发逐出）；
- 回关开关：`HICACHE_SIZE=0`。

> **v2.3 复核（9-01 笔记06）**：48h 实测 **HiCache L2 实际只贡献 0.45%** 的命中
> （L1 命中率 93.0% 已吃掉绝大部分）。**维持"低频保险"定位，非扩容依据**——
> 它救的是"超长会话被逐出后回来"这种低频但极痛（47s 全量重算）的场景。
> NVFP4 换代后 L1 池涨到 500k，触发概率进一步下降，但**开关保持 24 不动**
> （显存够、代价低、保险价值在）。

### 5.2 host_used 100% 的语义（8-30 实测定案，告警必读）

**drop 统计的是"L1 逐出时备份到 L2 失败"（失去兜底、将来必全量重算），不是"L2 满"**。host 池内部还有一层 LRU 自逐出（挤冷条目腾位），**host_used 100% ≠ 备份失败**。

实测：风暴期 host_used 钉死 100%（657,344/657,344）而 drop=0 = 自逐出忙碌运转但每次都腾得出位 = write_back 健康。

**真告警条件是三组合**：drop>0 **且** host_used 钉 100% **且** backup_tps 掉下来（真腾不出地了）。单看 host_used 会误报。

### 5.3 write_back 机制判读（监控视角）

- **write_back 策略下 evict == backup 帧帧相等 = 零丢失**——逐出即备份，L2 是 L1 的完整镜像延伸；
- **L2 回取是被动触发**：请求前缀走回被逐子树才 load_back，不会自动补货。"图表上 L2 占用涨"不等于"会变快"，只有 load_tps 出现才代表收益兑现；
- **图表判读**：陡升 = 命中（一帧跳完）vs 斜坡 = 真 prefill（~2.5k tok/s），时间尺度差 60 倍，一眼可分。

### 5.4 已否决的变体（v1.0 结论不变，保留结论防重蹈）

- **KV 放副卡（GPU1）**：4090↔4070Ti `can_device_access_peer=False`（无 NVLink + WSL2 dxgkrnl 不开 PCIe P2P）→ 四跳绕宿主比 DRAM L2 更慢；容量仅 ~190k vs DRAM 519k；sglang 无 GPU 层级缓存概念。Duanyll fork（动态 BAR1 P2P）也不适用：mod 改 Linux nvidia.ko 而 WSL2 走 dxgkrnl 无模块可 patch；4070Ti 是 AD104 不在支持列表。**真正价值 = 未来裸机 Linux + 多张 48G 4090 做 TP/NCCL 的现成方案**。
- L3 落盘：radix 树不落盘，重启收益不存在。
- session radix cache：CC 的 CPA 请求无 session_id 字段，对 Claude Code 无效。

---

## 6. 精度工程

### 6.1 GDN beta fp32 修复（865 倍）

fused_recurrent.py packed decode 与 fused_gdn_gating.py 两处 beta 改全程 fp32。1000 步漂移 4.1e-04 → 4.8e-07，性能零回归，accept len tail 达 7.25。已进生产；vLLM 同源问题 #53877 仍 open——本地修复领先上游。8-31 merge 后确认该修复仍保留在树上（上游尚未合并同源修复）。

### 6.2 ReplaySSM bf16 fold 漂移

t1 在 327/512 处语义等价分叉，是 bf16 re-quant fold 的预期漂移。多日观察无恶化，不动。

### 6.3 方法学

精度对照用三条 golden prompt（t1 创作文本 / t2 agent 任务 / t3 代码）逐字对比 + 语义等价判读。任何"动了数值路径"的改动（投机、量化、kernel 修复）都过这一关。

### 6.4 智力三层标尺（9-04 定型，9-09 扩充）

**跑分好看 ≠ 实际好用**。本栈用三层标尺，各有明确适用边界：

| 层 | 名称 | 判据 | 频率 | 脚本/存档 |
|---|---|---|---|---|
| **L1** | 逐字节 | greedy + 哈希对比（短题 EXACT） | 每次改动 | `iq_baseline.py check` |
| **L2** | 抗污染 | 6 题（AIME 2025 计数/组合/数论/LIS 代码/中文逻辑/严格格式），防训练集污染 | 月度 | `_l2_bench_904.py`，存档 `l2-baseline-20260904.txt` |
| **L3** | 实战盲测 | 真实使用体感 | 持续 | — |

**L1 的边界（重要）**：长 think 题受 KV 命中态与调度时序影响，**轨迹必分叉，不能逐字节对账**——
改用 SEMANTIC 关键词判读。实证：同一题首轮 12k vs 3.2k think 长度差异巨大但结论相同。

**9-09 实测结果（NVFP4 v2 模型）**：

| 维度 | 结果 |
|---|---|
| L1 + L2 智力 | **12/12**（与官方版、v1、FP8 现役基线同分） |
| 拒答率 | 敏感类 **8/8 compliance**，对照 4/4 正常 |
| 视觉 | **5/5**（颜色/计数/空间/文字/形状） |

**L2 判分 bug 修正（9-04 遗留）**：L2-Q1（a³+b³=1000 有序对计数）正确答案是 **0**
（无解：s=a+b 须整除 1000 且 s³∈(1000,4000) → s∈[11,15]，1000 无此因子），
旧判分 `"3" in answer` 把错误答案判 PASS。枚举 + 因子分解双路确认。

**xhigh think 预算陷阱**：xhigh 档 AIME 题 think 可达 5211 chars，4096 max_tokens 下
answer 被挤空（`finish_reason=length`，假 FAIL）。该题改用 medium 档（think 2840 chars 收敛）。

### 6.5 NVFP4 量化工程（9-09 新增，复现见模型仓库 README）

**配方**（对齐 nvidia 官方版，通配符 quantizer_name 精确圈定）：

| 组件 | 格式 | 说明 |
|---|---|---|
| MLP gate/up/down（64 层） | **NVFP4**（W4A4 序列化，group 16） | 主要体积节省 |
| lm_head | **NVFP4** | 激进；过了 12 题智力套件 |
| GDN linear_attn 投影（in_proj_qkv/z, out_proj，48 层） | **FP8** e4m3 per-tensor | 保 Ada decode 速度 |
| self_attn q/k/v/o（16 层） | **FP8** e4m3 per-tensor | 同上 |
| embed_tokens / norms / 视觉塔 | **BF16** | 不动 |

**modelopt 0.46.0 两个坑（各值一次失败的运行）**：

1. **`{"algorithm": "max"}` 是必需的**。缺了它，`mtq.quantize()` → `calibrate()`
   派发到 `NoneCalibrateModeDescriptor`（`_calib_func=None`）→ **forward_loop 静默不执行**，
   校准 0 秒"假完成"，全部 amax=None，GPU 无动静。**症状是"跑得飞快但什么也没做"。**
2. **`_amax` 导出 bug**。FP8 导出器读 `weight_quantizer._amax`，但 `TensorQuantizer`
   现在把它暴露为 `amax` property → AttributeError。复现脚本已 wrap
   `_export_quantized_weight` 临时重绑。

**sglang 量化工作流不可用**：`ModelOptModelLoader._load_modelopt_base_model` 校准 forward
会把 `ForwardBatch` 传进 transformers 原生 `modeling_qwen3_5.py`（`position_ids.ndim`
AttributeError）。→ 直接用 modelopt 原生 API，绕开 sglang 工作流。

**OOM 修复**：54G BF16 权重 > 48G VRAM，`device_map="auto"` 默认塞满 GPU 后校准激活爆显存。
修法 `max_memory={0: "24GiB", "cpu": "80GiB"}`（权重一半 GPU 一半 RAM，为激活留空间）。
校准 512 样本 × 2048 tok，~15.5s/batch（CPU offload 拖慢），全程 ~50 分钟。

**checkpoint 兼容四件套**（部署时逐个踩过，v2 复用）：
1. `--language-only` 分离模式白名单拒收 `Qwen3_5ForCausalLM`
   （`pd_disaggregation_hook.py:259`）→ **config 重建为复合 VL 结构**：
   text 字段整体挪进 `text_config`，vision_config/image_token_id 从 BF16 原版拷贝，
   `architectures=Qwen3_5ForConditionalGeneration`，`model_type=qwen3_5`；
2. processor 缺失 → 拷贝 preprocessor_config.json / video_preprocessor_config.json / chat_template.jinja；
3. tokenizer 缺失 → 拷贝 tokenizer.json / tokenizer_config.json / vocab.json / merges.txt；
4. text 子 config 缺 `vision_start_token_id` → 复合结构顺带解决。

---

## 7. WSL2 特有坑清单（复刻者必读）

按踩坑惨烈度排序：

### 7.1 GPU VA ~1TB 上限（最阴险）
**症状**：EAGLE/MTP + expandable_segments 时进程 GPU VA 推过 dxgkrnl 的 ~1TB 上限，**所有 ioctl 返回 EOVERFLOW**，scheduler 死旋在 mamba alloc_group_end——进程不崩溃、不报错、只是僵死。
**解法**：MTP 模式去掉 expandable_segments（conf.sh 已条件化）。

### 7.2 chunked prefill 8192 崩溃
**症状**：CHUNKED_PREFILL_SIZE=8192 触发 FLA kernel device-not-ready 崩溃。
**解法**：2048 是 WSL 稳态，也是性能定案（§3.6）。

### 7.3 wsl.exe 会话退出杀后台（含 nohup）
**症状**：`wsl bash -lc "nohup ... &"` 启动的服务，宿主侧 wsl.exe 会话退出即被杀——**且日志文件都不建，死得无声无息**。
**解法**：长驻服务必须在宿主侧保活的会话里**前台**跑启动脚本（比如常开的终端标签页）。stop 脚本只停主服务不停 encoder（设计如此——但换代码后必须先杀旧 encoder，见 §9.4）。

### 7.4 HiCache write_back 的 AOT fallback 崩溃
**症状**：MHA staged write-back 的 JIT 在 WSL/CUDA 13 会 stall，AOT fallback 非法访问 pinned host 内存。
**解法**：`SGLANG_DISABLE_HICACHE_MHA_STAGED_WRITE_BACK=1` + `SGLANG_USE_HICACHE_SAFE_PAGE_FIRST_WRITE_BACK=1`（aa90aa170e 提交封装）。

### 7.5 观测层缺失
`nvidia-smi --query-compute-apps` 在 WSL guest 内返回 `[N/A]`（无进程视图）。解法：Windows 侧读性能计数器，WSL 进程归因到 vmwp.exe，按 LUID 贪心匹配 GPU（monitor §4.2）。

### 7.6 无 P2P
`can_device_access_peer=False`。影响：跨卡 KV 方案否决（§5.4）、未来 TP 需走 NCCL SYS 层。

### 7.7 Triton kernel 冷加载余量（**9-01 发现，已根治**）

**症状**：`mem_fraction_static=0.985` 把池外余量压到极限，torch caching allocator 的
high-water mark 在早期服务期吃光剩余空间，而**此时仍有新 kernel specialization 首次加载**
→ `cuModuleLoadData` CUDA OOM。表现为**无规律的请求失败/进程崩**，极易被误判成"偶发"。

**告警时间线**（`main-boot.log`，重启后 1h12m 内共 37 次）：

| 时刻 | 空闲显存 | 事件 |
|---|---|---|
| 23:33:15（启动后 2min） | 0.36 GiB | `alloc_extend_kernel` / `write_req_to_token_pool_triton` / `compute_position_kernel` |
| 23:40:59 | 0.35 GiB | `_fused_norm_rope_kernel_stacked` |
| **00:11:37 / 00:11:48（启动后 38min）** | **0.00 GiB** | `_fused_sigmoid_mul_kernel`、`chunk_fwd_kernel_o` **仍在做 cuModuleLoadData** |

0.00 GiB 是**侥幸没炸**——下一次就是 CUDA OOM，且必然发生在**负载最高、余量最低**的时刻。

**根治方案（9-01 04:00 实施并验证）**：
1. **诊断**：设 `WARNING_THRESHOLD_GB=100` 全量记录（比 `SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`
   更好——不杀进程/拿全量/可 A/B），跑 `warmup.py`（长 prefill 4825 / 长 decode 256 /
   compact-rebuild / 并发 5 路）→ 真实流量探针后冷加载去重**稳定 28 不再增长** = 覆盖完全；
2. **固化**：`restart-main-sglang.sh` health 200 后自动跑 `/root/sglang-warmup.py`
   （**非致命，失败仅告警**）——28 个 lazy kernel 提前到启动期加载（余量 >1GB）；
3. **验证**：默认参数端到端重启，S5 智商冒烟 5/5 哈希逐字节一致，全部 41 条冷加载
   集中在启动 + warmup 的一分钟内，S5 后零新增。

**方法论**：warmup 有效性判据 = **打真实流量后去重数不增长**，关键差异是**时机**不是数量。

**遗留**：多模态图片路径的 kernel 未覆盖（视觉在 GPU1 独立服务，另有一套 warmup）。

---

## 8. 可观测性：monitor v2 与请求级归因

**没有观测就没有优化。** 本栈全部关键决策（LFU 陷阱、HiCache 翻案、mamba 断链、draft graph 定案）都靠这套工具实锤。

### 8.1 架构（v2.2 现状）

```
WSL sglang :8778  /metrics + /get_server_info + main-boot.log
        │ 1s 采样                 │ UNC 路径 tail（Windows 侧直读 \\wsl.localhost\...）
Windows server.py :8917（纯标准库，零依赖）
        │ /api/snapshot /api/history /api/prefill_events
浏览器 index.html（单文件，原生 Canvas）
```

- 零第三方依赖：后端纯 Python 标准库，前端单文件 HTML；
- 1s 采样、30min 环形缓冲、10min 增量落盘 JSONL；
- **monitor 跑在 Windows 宿主侧**（UNC 日志 + powershell 计数器决定）——**WSL 里 curl 127.0.0.1:8917 探不到是方向错误**，宿主侧探；
- 重启方式：taskkill 旧 PID → powershell Start-Process anaconda python server.py <sglang-host:port> 8917。

### 8.2 版本升级记录

**v2.0 三项（8-31）**：
1. **decode 冻结检测**：sampler 派生 `stall_s`（running>0 且 gen_tps<0.1 的持续秒数），前端 Decode 吞吐 KPI 砖红框告警"⚠ decode 冻结 Ns"（>2s 触发）；
2. **INFO_KEYS 补全 + draft-graph 徽章**：enable_mixed_chunk / prefill_decode_interval 进面板；boot log 正则解析 draft cuda graph 状态（`draft_graph` 字段，前端徽章 draft-graph:off|on）；
3. **ITL 口径修正**：TTFT 砖的 ITL 标签改为"verify 轮间隔"，注明一轮打包多 token（见 8.6 陷阱）。

**v2.1 三处 bug 修复（9-01）**：
1. **radix 字段错配**：`enable_radix_cache` → `disable_radix_cache`（`/get_server_info` 只有 disable 键，原键恒被过滤 → 徽章恒显"on"）；
2. **ITL 单位**：标签"verify 轮间隔" → "token 间隔"（sum/count 本源即 per-token）；
3. **冻结检测**：`gen_tps`（40 步窗口 gauge，真冻住会漏报）→ `dec_tps`（1s 差分，冻住天然 0），+"dtp is not None" 消除误报。

**v2.2 显存归因 + 编码器健康（9-03/9-04）**：
1. **启动日志显存归因**（`boot_mem_thread`）：draft 权重 / draft KV 池 / mamba 槽 / CUDA graph
   四项只有启动日志有精确值，早期用 `0.27GB/槽` 硬编码——那是 12 槽时代的实测值，
   20 槽时错算成 5.4GB（实际 **1.59GB，79.5MB/槽**），把 3.8GB 从"工作区"错记到 mamba；
   draft 的 8.4GB 更是整块落进倒推段。现在解析 boot log 这几行，服务重启即自动重扫：
   ```
   Load weight end. ... type=Qwen3_5ForConditionalGeneration ... mem usage=18.96 GB
   Load weight end. ... type=DFlash2DraftModel ... mem usage=3.65 GB
   Mamba Cache is allocated. max_mamba_cache_size: 20, conv_state ... ssm_state ...
   KV Cache is allocated. dtype: ... #tokens: 499968, K size: 7.63 GB, V size: 7.63 GB
   KV Cache is allocated. ... K size: 2.38 GB, V size: 2.38 GB   ← draft 池
   ```
   拿不到日志时回退常量，面板标注"常量回退（日志未就绪）"。
2. **编码器健康探测**：实时 `/health` 探测 8779 + 主日志 `Health check evicted` 扫描
   （9-02 事故探测器）；
3. **排队/分阶段耗时**：`queue_time_seconds`（入队→被调度）、`per_stage_req_latency_seconds`
   （网关处理 / prefill 前向 / chunk 前向）窗口差分。

**v2.2 新增面板行**：GPU 显存组成从 4 段改 6 段（含启动日志实测项）、mamba 审计心跳
（5 分钟树健康）、归因口径前缀命中率。

### 8.3 请求级 prefill 归因（PREFILL-REQ 日志）

sglang 树 4 连提交实现（`metrics_reporter.py` report_prefill_stats 内逐请求）：

```
PREFILL-REQ rid=xxx input=6457 chunk=0+2048 end=2048
    hit_device=0 hit_host=0 salt=- mamba_branch=181696 mamba_host_hit=0
```

- floor=4096 按**完整 input 长度**判（第一版按 per-chunk new 判 → chunked 请求全静默的三连坑）；
- chunked prefill 逐 chunk 一行，可看推进；
- 环境变量 `SGLANG_PREFILL_LOG_MIN_TOKENS` 可调（0=全打）；
- monitor 侧：tail 日志 → `/api/prefill_events` → 前端折叠表格（同 rid chunk 折叠，**命中取首 chunk**——末 chunk 命中的是同请求前面 chunk 刚算的自己，会显示假 100%）。

### 8.4 归因判读指纹（给 AI 的弹药库）

> ⚠️ **8-31 二次复核纠正**：本节旧版三行判读全部失效——源码 `mamba_radix_cache.py:1196-1203` 是
> `chunk_aligned_seqlen if chunk_aligned_seqlen > 0 else None`，**0 会被转成 None**，
> 所以该字段取值域 = `{None} ∪ {正整数}`，日志里**永远不会出现 `mamba_branch=0`**（重启后 1h12m 窗口 148 条 PREFILL-REQ 实测：零 0 值坐实；None = 29 条 19.6%，数值组 = 119 条 80.4%）。
> 旧版拿"salt/客户端根不同"解释 None 也是错的：该窗口 `salt=-` 全部 148 条，此成因根本不存在。

| 指纹 | 诊断 | 动作 |
|------|------|------|
| `branch=None` + hit_device 高 | **mamba 状态完整覆盖匹配 = 健康**（成因①：`len(value) <= best_value_len`） | 无 |
| `branch=None` + hit_device≈0 | **真无命中**（成因②：全匹配 <64 → 对齐到 chunk 后为 0 → 也变 None） | 查前缀/是否新窗口 |
| `branch>0` 且 ≈ hit_device | 断链右端点≈匹配末端 = **mamba 状态几乎全缺**，这段要重算 | 关注 track-interval |
| `branch>0` 且 ≪ hit_device | 断链在中间 | 正常轮转恢复路径 |

**两种 None 不可一律判健康**——换负载（超短 prompt）时成因② 会出现。

**实锤案例**：同一主窗口出现两条 180k × 91 秒全量重算。归因日志显示 `mamba_branch=29,568 / input=181,696`——**KV 树完整在池，但 GDN mamba 状态链断了**（branch>0 即确有断链，此例断在 29,568 处），180k 白算 91 秒。根因链：Claude Code stop-hook 24 并发 64-token 小请求 → mamba 槽竞争 → 主前缀整树失效。16 槽翻案（§3.2）+ ReplaySSM 就是修这一层。

高频样本（可作后续对比基线）：`branch=11008 hit_dev=11008`（8 次，断链≈匹配末端）、`branch=2688 hit_dev=98304 input=167164`（1 次，断链靠后、重算量大）。

另一案例：两条 81k 全量重算，池 free 充足、LRU 未逐出 → 第 0 页就分叉 = 某客户端 system prompt 带变化注入（nonce/时间戳类）。风暴前后对照基线：风暴期命中率 73.4%/灾难连片 vs 移除后 85.5%/100% 全命中连片；**健康形态指纹 = hh 7-10k + mh=1 成对出现**。

**9-03 新增观测设施**（配合 BACKUP-ON-EVICT）：

| 日志/面板 | 含义 | 判据 |
|---|---|---|
| `MAMBA-EVICT-BACKUP` | 驱逐前成功同步 D→H 备份 | 正常运转，计数增长是好事 |
| `MAMBA-EVICT-DROP` | host 池也满，备份失败 | **告警**（该会话回来必全量重算） |
| `MAMBA-EVICT-BACKUP-FAIL` | 备份异常 | **告警**，看堆栈 |
| `MAMBA-AUDIT`（5 分钟） | 树健康心跳：nodes/states/deepest/branch_max/evictable/locked | deepest 长期不变 = 主链未断 |
| 归因面板"归因口径命中率" | `attrib_hit_rate` = 按 rid 去重 `min(hd+hh, inp)` 求和 / input 求和 | 比引擎自报更贴真实体感 |

**9-04 实战验证**：8 路风暴 ×3 波零失败零 OOM，全程 **223 次 MAMBA-EVICT-BACKUP 零 DROP**；
198,446 tok 会话经 6 填充会话挤占后回归，**0.6s / 100% cached / 等效 32 万 tok/s**——
"秒级 LOAD_BACK"实为**亚秒级**。

### 8.5 cache_salt 字段

可选请求参数（客户端 JSON 显式传），radix 树命名空间隔离用。**所有请求显示 '-' 是预期**——没人传 = 全在默认共享命名空间。"将来客户端传了 salt 才有诊断价值"的预留字段。

### 8.6 SSE delta 计数陷阱（8-31 新血泪）

**客户端按 SSE delta 计数会得"17.4 t/s"假回退**。真相：新代码一轮 DFlash verify 接受 ~2.8 token **打包成一个 SSE delta**——delta 率 × accept 才是真吞吐（引擎侧 50-80 t/s 与历史持平）。**ITL 的 sum/count 本源是 per-token 间隔（实测 ~17.3ms/token ≈ 58 tok/s）**；verify 轮间隔 = per-token × accept ≈ 48ms（accept≈2.8），不是 ITL 直读值。（本节旧版曾写"ITL 57ms 是 verify 轮间隔"——口径错误，8-31 已按 monitor 修复同步纠正。）

**连带的判读规则**：accept len 必须同任务形态对照——中文长散文 greedy 创作 accept 天然 1.6-2.8，拿它对照"混合负载中位 3.57"是错位对照；同代码任务对照立即归位（2.7-4.2, 55-84 t/s）。曾因此误判严重回退、连 revert 都没恢复，最后发现是测试设计错误。**单任务样本 ≠ 混合负载基线。**

---

## 9. 运维手册

### 9.1 日常操作

```bash
# 完整启动（encoder 先起，再主服务）
bash /root/boot-sgl-stack.sh
# 只重启主服务（encoder 不动，改 conf.sh 后用这个）
bash /root/restart-main-sglang.sh
```

注意：必须在宿主侧保活的会话里前台跑（§7.3）。

### 9.2 回滚开关速查

| 场景 | 动作 |
|------|------|
| KV 池 OOM | `MAX_TOTAL_TOKENS` 回 470000（或 310000 保守档） |
| HiCache 异常 | `HICACHE_SIZE=0` |
| DFlash2 异常 | `ENABLE_DFLASH=0 ENABLE_MTP=1`（注意 MTP 自动去 expandable_segments） |
| **HRRN 异常** | `SCHEDULE_POLICY=lpm`（一行切回，见 §3.4） |
| **decode 抖动** | `PREFILL_DECODE_INTERVAL=0`（回到 prefill 独占，见 §3.4b） |
| 逐出策略回滚 | `--radix-eviction-policy lfu` |
| 归因日志刷屏 | `SGLANG_PREFILL_LOG_MIN_TOKENS` 调大 |
| mixed chunk 异常 | conf.sh 去掉 `--enable-mixed-chunk`（decode 会退回冻结形态但能跑） |
| BACKUP-ON-EVICT 异常 | 回滚 0031 中 `evict_component` 内备份块（`git revert` 该 patch 会连带掉归因日志，手工摘除更稳）；日志判据 `MAMBA-EVICT-BACKUP-FAIL` 异常堆栈 |
| 溢出环异常 | conf.sh 加 `--mamba-overflow-size 0`（默认 8 行预留，关掉即回 #37613 前行为） |
| **Gumbel 采样异常** | `SGLANG_OPT_USE_GUMBEL_SAMPLE=0`（回到 `torch.multinomial`） |
| **NVFP4 启动崩** | 确认 `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` 在位（见 §3.7） |

### 9.3 生产纪律

- 生产时段（用户在用）不打长 decode 带载测试——带载请求会触发逐出，挤掉活跃会话（LRU 时代风险已大减但纪律保留）；
- 短探测（/metrics）无影响，随便打；
- 任何源码改动：`git log` 确认提交链 → 重启 → 冒烟 → monitor 观察 30min；
- **带图冒烟必须做**（§9.4）：纯文本不触发 encoder 路径，pickle 错位只有 mm 请求能引爆。

### 9.4 8-31 生产崩溃案例：跨进程 pickle 类路径漂移（v2.0 新增，最重要的一课）

**事故时间线**：8-31 上午 merge 完成后主服务用新代码重启，encoder 还是 8-29 旧代码进程。11:11 起稳跑纯文本无恙；13:17 首个带图请求触发 unpickle → 旧 encoder pickle 的 EmbeddingData 类路径 `sglang.srt.disaggregation.encode_receiver` 在新代码里已改名 `disaggregation/encoder/receiver.py` → ModuleNotFoundError → **SIGQUIT 全树死**。

**踩雷滞后 2 小时**——纯文本不走 encoder，pickle 类路径错位只有 mm 请求能引爆。这就是"带图冒烟必须做"的原因。

**根因**：主服务 ↔ encoder 跨进程传数据用 pickle，pickle 序列化的是**类路径引用**不是类定义——两端代码版本必须逐字节同源。

**教训与纪律**：
1. **merge/升级代码后 encoder + 主服务必须同时重启**。`restart-main-sglang.sh` 只重启主服务的日常惯例，在跨进程 pickle 场景是雷；
2. boot 脚本看到 encoder health OK 会直接跳过——换代码后必须**先杀旧 encoder 再 boot**；
3. 重启后用带图探测（mm_probe.py，手工构造 PNG 走 encoder 往返）验证，不能只测纯文本；
4. 重启后用 /get_server_info 复核关键参数实际生效值（本案例发现 conf.sh chunk 之前声称 1024 实际是 512 没落盘）。

### 9.5 风暴压测方法论

- **压测工具必须自带 nvidia-smi 采样器**（storm_with_sampler.py）——用户 monitor（8917）会在 sglang 重启时死掉，不能依赖；
- 合成语料 token 比率：随机双字组合 ≈3.26 token/字符（tokenizer 无压缩），真实中文 ≈1.3-1.5——生成压测语料按 46k 字符 ≈ 150k tokens 换算；
- 压测验收口径：峰值 MiB / floor MiB / OOM 数 / FAILED 数 / MAIN 完成时长。

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
2. **拉模型**：HF `Qwen3.8-27B-Uncensored-NVFP4`（21.1 GB，含视觉塔）。
   想自己量化见模型仓库 README 的三步脚本（quantize → merge vision → composite config）；
   想省事可退回 FP8 版（`orcarouter/Qwen3.8-27B-Uncensored-FP8`），代价是池 310k / decode 减半；
3. **建源码树**：
   ```bash
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git fetch origin 0da6a6685648a415818bfa2e44471cb884009f35   # merge-base，2026-08-31
   git checkout -b qwen38-integration 0da6a66856
   git apply /path/to/patches/qwen38-0909-merged.patch          # 单一合并 diff，一次应用
   # 验证：git rev-parse HEAD^{tree} 应得 53bbd2f6d78d9021a1e0e7021a263fea3081b507
   ```
   > ⚠️ **不要用 `git am` 逐个打 `patches/by-commit/`**——提交链含 3 个 merge 且
   > 部分父提交不在上游，线性重放必失败（v2.3 修正，详见 §2.3）。
   > `by-commit/` 仅供阅读追溯。
4. 拷 conf.sh 改路径（模型路径/IP/端口），先跑 `ENABLE_DFLASH=1` 基线；
   **NVFP4 用户确认 `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` 在位**（§3.7）；
5. 起监视器（monitor 六件套，**在 Windows 宿主侧跑**），**先于一切优化部署**；
6. 按 §9.1 启动，**带图探测走一遍**（§9.4——纯文本不触发 encoder pickle 路径）；
7. 逐项开启优化，每项 A/B 记录（参照 ab-results.md 的口径）。

### 10.3 可裁剪项（按收益/复杂度排序）

| 优化项 | 收益 | 复杂度 | 建议 |
|--------|------|--------|------|
| GDN beta fp32 修复 | 精度 865 倍 | 两处改 fp32 | **必做** |
| LFU→LRU | 修复活跃会话被逐 | 一个参数 | **必做** |
| mixed chunk（#36288+#36933） | decode 冻结根除 | 一个参数（合并 diff 自带） | **必做** |
| NVFP4 混合配方 | decode +45-60%、池 +61% | 换模型（或自量化） | **强烈建议**（Ada 上 prefill -25%） |
| **HRRN 调度** | 短请求 3× 提速 | 一个参数 | **必做**（长文档负载改 lpm） |
| **PDI=3** | decode 抖动消除 | 一个参数 | **多会话用户必做** |
| ReplaySSM（#36683） | +2G 显存 → 安全余量 | 合并 diff 自带 | 强烈建议 |
| HiCache 24 | 多会话轮转 47% 提速 | 一个参数 + 两个 WSL env | 多会话用户必做 |
| 20 mamba 槽 | 风暴断链根除 | 一个参数 | 多 agent 必做 |
| 请求级归因日志 | 可诊断性 | 合并 diff 自带 | 排障前装好 |
| **warmup 补丁（§7.7）** | 消除服务期 kernel OOM | 一个脚本 + restart 挂钩 | **稳定性必做** |
| draft CUDA graph | **无收益**（噪声级） | 需池缩到 285k | **不做** |
| 树形草稿 W=2（#36196） | accept 3.42→~4（greedy 场景） | 28 文件 | 等上游 temp 1.0 验证 |

### 10.4 借助 AI 继续优化（给朋友的方法论）

1. **先装观测再谈优化**：monitor + 请求级归因日志是 AI 诊断的输入源。没有数据，AI 只能猜；
2. **把 ab-results.md 的口径抄走**：同硬件、同 golden prompt 三条、同负载脚本、单项开关。A/B 不控制变量 = 白测；
3. **评测要贴真实负载**：跑分好看 ≠ 实际好用。本栈的 draft graph 决策就是靠 agent 场景三维评测（单路代码/工具轮转 TTFT/多路并发）定案的——纯跑分维度上它"有加速"，agent 场景实测噪声内，所以不做；
4. **判读指纹直接喂给 AI**：§8.4 的三行指纹 + "hit_host vs hit_device"、"陡升 vs 斜坡"、§8.6 的 SSE delta 陷阱，AI 拿到日志就能归因；
5. **抢跑纪律**：上游 open PR 小的直接 cherry-pick + A/B；翻案级收益的合并与否都自己动手；只有测试成本 > 预期收益才等。每个摘取进提交链（§2.3 就是现成模板）；
6. **警惕"第一轮测试陷阱"**：HiCache 第一轮全冷测试得出"必须关"的错误结论，因为只测了代价没测收益。设计测试前先列出"这个特性的收益场景是什么"；
7. **merge/升级后三件事**：encoder+主服务同时重启、带图冒烟、get_server_info 复核参数落盘；
8. **移植成本看断层，不看 diff 大小**（§4.8 教训）：#35544 diff 只有 850 行却移植失败，
   因为它的地基（req 状态模型迁移）有几千行。**摘上游 PR 前先问：它脚下踩的地基，
   我们树里有吗？**
9. **量化/依赖类改动要能复现**：模型量化脚本已随 HF 仓库发布（含两个 modelopt 坑的
   规避代码，§6.5）；依赖升级（xgrammar）留 .bak + dist-info 备份，回滚一条命令。

---

## 11. 聊天模板评测：froggeric/Qwen-Fixed-Chat-Templates（8-31 定案：不切换）

用户发现社区修复版聊天模板，要求 A/B 且"不能降智商"（xhigh 思考强度是日常配置）。

**评测方法**：jinja2 渲染对照（token 序列差异）+ 服务端智力 6 题（数学/逻辑/中文陷阱）+ KV 前缀缓存 + 工具轮转三维度，两模板各跑一遍。

**结论**：
- **xhigh 下两模板渲染的 token 序列逐字节一致**（同一条 xhigh 注入指令，官方默认就是 xhigh、修复版默认 medium 但传参后相同）；
- 智力 6 题两模板答案逐字节一致（think 长度都一样）——**无降智商问题也无改善空间**；
- 该模板的真实卖点我们全都不需要：修的是 Qwen3.8 崩溃（enable_thinking=false 抛异常）/空 think 污染/llama.cpp 吞吐/minijinja 兼容——我们不走 llama.cpp、不关思考、sglang 用 Python jinja；
- 它的"历史 think 保留 → 100% 前缀缓存命中"特性对 CC/CPA 场景无效（CPA 请求每次全量 system prompt，salt 不同根不同，模板层解决不了）。

**生产保持官方模板。** 备份在 tokenizer_config.json.official-bak，切换脚本 `_pr_scan/switch_template.py`（fixed|official），评测脚本 server_ab.py / deep_dive.py / template_render.py。

---

## 12. 遗留问题与下一步

| 事项 | 状态 |
|------|------|
| #37065 DFlash prefill radix 前缀复用 | **最高优先盯防**（8-30 后冷置），稳定即摘 |
| #36136 置信度动态验证 | 盯防（+2204 行大盘子，等稳定） |
| **#37164 mamba state 入 ReqKvInfo** | **#35544 的地基**——摘它才能解锁 #35544 调用层（§4.8） |
| #35544 ReplaySSM checkpoint 物化摊销 | **移植失败**（§4.8）；等下次上游 rebase 自动到位 |
| #36196 树形草稿 | 等上游 temp 1.0 树采样验证 |
| draft 域内微调（中文创作 accept 1.6-2.8 → 3+） | 中期最大杠杆 |
| 81k 零命中之谜（system prompt nonce 注入） | 等 salt 字段下次复现定案 |
| mamba_track_interval 256 实验 | 已提议未批（512 工作良好） |
| 多模态图片路径 kernel warmup 覆盖 | 未做（视觉在 GPU1 独立服务，另套 warmup） |
| 多模态图片路径的 Triton 冷加载 | 遗留（§7.7 末尾） |
| **GPU 功耗/频率实验** | **用户已否决"不碰硬件"**，冻结；软件侧已探底 |
| FP8 旧模型删除 | `/root/LLM/Qwen3.8-27B-Uncensored-FP8`（29G）已无依赖，可删 |
| NVFP4 v1 模型删除 | `/root/LLM/Qwen3.8-27B-Uncensored-NVFP4`（18G）已被 v2 取代，可删 |

---

## 附录 A：agent 场景评测基准与方法论（8-31 建立，9-04 复测）

**设计原则**（用户原话："评测尽量做我 agent 使用的场景评测，多维度的评测，不然光数字好看，但实际使用脱离，也没有用，我不是要跑分，而是实际使用"）：

| 维度 | 脚本 | 覆盖场景 | 基准（9-04 复测） |
|------|------|----------|----------|
| P1 单路代码 | `_pr_scan/agent_bench.py` P1 | LRU+Trie 类代码任务 greedy decode | 264 t/s（基线 270.9） |
| P2 工具轮转 TTFT ×6 | `_pr_scan/agent_bench.py` P2 | agent 高频小请求往返 | p50 133ms（基线 108-130） |
| P3 7 路并发混合 | `_pr_scan/agent_bench.py` P3 | 多 agent workflow | 聚合 **702 t/s**（基线 478-497，**+40%**） |
| 150k 挤压 | freeze_test | 大 prefill 期间 decode 存活 | ITL p50 865ms 持续产出 |
| 31 并发风暴 | storm_with_sampler.py | 极限并发 + 大会话同进场 | floor 1054 MiB 零 OOM |

方法纪律：flush_cache 同起点、每维度对照组同任务形态、accept 必须同任务形态对照（§8.6）、生产时段不打长 decode 带载。

## 附录 B：13-17x 冷 prefill 差距归因（v1.0 结论 + chunk 更新）

冷 prefill 13-17x 的差距不是引擎本质差距：8x 来自 chunked prefill 税 + 1.6x 每税点固定开销差。注意 §3.6 定案后 chunk 2048 就是本档吞吐最优（1024 会砍 2/3），所以"换裸机消除 chunked 约束"的预期收益要按 2048 口径重估。

## 附录 C：文件与数据索引

- 配置：`/root/sglang_qwen38_27b_dflash2.conf.sh`（包内 conf/ 同步副本）
- A/B 数据存档：`/root/ab-results.md`（包内 scripts/ 同步副本）
- 监控六件套：`sglang-kv-monitor/{server.py, index.html, DESIGN.md, README.md, loadgen.py, gpu_mem.ps1}`
- 主日志：`/root/sglang-qwen38-dflash2/main-boot.log`（PREFILL-REQ 行在这里）
- **补丁**：包内 `patches/qwen38-0909-merged.patch`（**单一合并 diff，`git apply` 用**）；
  `patches/by-commit/` 38 个 format-patch（**仅供阅读追溯，勿 `git am`**）
- **模型**：`/root/LLM/Qwen3.8-27B-Uncensored-NVFP4-v2`（HF 公开仓库同名，含复现脚本 + 中英文 README）
- 旧 vLLM 栈（保留对照）：`/root/vllm-research/qwen38-pr-stack`
- 评测脚本集：`_pr_scan/`（agent_bench / storm_with_sampler / mm_probe / template A/B 等）
- 量化/智力/拒答/视觉评测存档：`iq-nvfp4-*.json`、`eval-refusal-v2.json`、`eval-vision-v2.json`、`l2-baseline-20260904.txt`

## 附录 D：术语表

| 术语 | 含义 |
|------|------|
| GDN | Gated Delta Net，Qwen3.8 的线性注意力组件，混合架构中的 recurrent 侧 |
| DFlash2 | 集成的投机解码方案，K=8 draft + 目标 verify |
| ReplaySSM | fold-every-commit 策略：verify 存原始输入，commit 时重放折叠 SSM |
| mixed chunk | prefill 与 decode 混批调度，大 prefill 期间 decode 持续产出 |
| HiCache / L2 | sglang 分层缓存，GPU L1 逐出的 KV 备份到 host DRAM |
| write_back | 逐出即备份策略，evict==backup 帧帧相等 |
| host_used 100% | host 池内部 LRU 自逐出忙碌运转，≠ 备份失败（告警看 drop>0 组合，§5.2） |
| **HRRN** | Highest Response Ratio Next：按 `等待token/未缓存token` 排序的调度策略（§3.4） |
| lpm | Longest Prefix Match 调度：前缀最长的请求优先 prefill（已被 HRRN 取代） |
| **PDI** | `--prefill-decode-interval`：每做完一个 prefill chunk 先连做 N 轮 decode（§3.4b） |
| **NVFP4** | NVIDIA 4-bit 浮点权重格式；Ada 上走 Marlin W4A16 反量化路径（§3.7） |
| **Marlin** | 为无 FP4 tensor core 的 GPU 提供的 W4A16 反量化 GEMM kernel |
| **BACKUP-ON-EVICT** | 驱逐 mamba 节点前同步 D→H 备份，回来时从 host 秒回（§2.3 0031） |
| radix 驻留 | 已算完留池等前缀复用的 KV，可逐出，占满无害 |
| mamba_branch | 分叉点存在 mamba 状态时 KV 本可命中的长度（断链诊断核心字段） |
| accept len | 投机解码平均每轮接受的草稿 token 数（K=8 满分 8） |
| verify 轮间隔 | 一轮 verify 打包输出多个已接受 token；= ITL（per-token 间隔）× accept len，**不是 ITL 直读值** |
| draft CUDA graph | 把 draft 模型折进 CUDA graph；点亮需 capture 时 avail>1.0G（本栈实测无收益维持 off） |
| pickle 类路径漂移 | 跨进程 pickle 只序列化类路径引用，两端代码版本错位即 ModuleNotFoundError（§9.4） |
| **lazy kernel** | 首次用到才做 `cuModuleLoadData` 的 Triton kernel；服务期加载可 OOM（§7.7） |
| **断层成本** | 上游 PR 与我们抢跑树之间的地基差异；比 PR 自身 diff 大小更能预测移植成本（§4.8） |
