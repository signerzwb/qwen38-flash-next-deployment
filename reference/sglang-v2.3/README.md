# sglang-qwen38-27b 栈复刻包（v2.3，2026-09-09）

配套文档：`sglang-qwen38-stack-guide-v2.3-20260909.md`（完整技术方案，先读它）。本包是文档的可执行落地件。
历史版本归档：v2.2 见 `sglang-qwen38-stack-guide-v2.0-20260831.md` + 本文件旧版；v1.0 见 `sglang-qwen38-stack-guide-v1.0-20260829.md`。

## 目录

```
sglang-qwen38-stack-guide-v2.3-20260909.md  # 技术方案全文（设计要点/PR 来源/测试数据/方法论）
conf/
  sglang_qwen38_27b_dflash2.conf.sh   # 全部配置，唯一调参入口（逐项注释）
scripts/
  boot-sgl-stack.sh            # 完整启动（encoder 先起 → 主服务）
  start-sglang-qwen38-dflash2.sh   # 只起主服务 :8778
  start-sglang-qwen38-encoder.sh   # 只起视觉 encoder :8779
  restart-main-sglang.sh       # 只重启主服务（注意 §9.4 pickle 雷；含 warmup 挂钩）
  sglang-warmup.py             # 28 个 lazy kernel 启动期预热（§7.7）
  ab-results.md                # 全部 A/B 测试数据存档（原始数字）
patches/
  qwen38-0909-merged.patch     # ★ 单一合并 diff，用 git apply 打（见下）
  by-commit/                   # 38 个 format-patch，仅供阅读追溯，勿 git am
monitor/                       # KV 监控面板 v2.2（零依赖）
  server.py  index.html  DESIGN.md  README.md  loadgen.py  gpu_mem.ps1
```

## 复刻步骤

1. **环境**：WSL2 (ubuntu2204) + 双卡（48G 主卡 GPU0 + 12G 副卡 GPU1）。
   单 48G 卡也能跑（视觉塔不分离），见指南 §10.1 档位表。

2. **拉模型**：HuggingFace **`Qwen3.8-27B-Uncensored-NVFP4`**（公开，21.1 GB 含视觉塔）。
   想自己量化：模型仓库 README 有三步脚本（quantize → merge vision → composite config）。
   想省事可退回 FP8 版 `orcarouter/Qwen3.8-27B-Uncensored-FP8`（代价：池 310k、decode 减半）。

3. **建源码树**（⚠️ **v2.3 修正了打补丁的方式**）：

   ```bash
   git clone https://github.com/sgl-project/sglang.git
   cd sglang
   git fetch origin 0da6a6685648a415818bfa2e44471cb884009f35   # merge-base，2026-08-31
   git checkout -b qwen38-integration 0da6a66856
   git apply /path/to/patches/qwen38-0909-merged.patch
   # 验证：git rev-parse HEAD^{tree} 应得 53bbd2f6d78d9021a1e0e7021a263fea3081b507
   ```

   > **为什么不用 `git am` 逐个打 `by-commit/`**：提交链里有 **3 个 merge commit**，
   > 且部分父提交来自**不在上游的 PR 分支**，线性重放必失败（实测第 1 个就 apply 失败）。
   > 合并 diff 已实测验证：从 merge-base 一次 apply，tree 与生产 HEAD **逐字节一致**。
   > `by-commit/` 保留供阅读追溯（38 个 patch 的主题清单见指南 §2.3）。

   补丁主题速览（完整 38 项见指南 §2.3）：

   | # | 内容 |
   |---|------|
   | 0001-0004 | 模型支持基座 / 视觉塔分离 / 量化 lm_head / retraction 状态修复 |
   | 0005 | HiCache WSL workarounds（**WSL 必装**） |
   | 0006-0009 | mamba 准入槽 / split 修复 / COW 预热 / **ReplaySSM（最大收益项）** |
   | 0010-0013 | 请求级 prefill 归因日志（4 连迭代） |
   | 0014-0019 | HiCache 三修复 + mmap 去重 / custom_mask 上限 / verify 循环展开 |
   | 0020-0026 | chunked 进度保留 / **#36911 GraphPoolPrecarve + Borrow** 及配套 |
   | 0027-0029 | **#36288 + #36933 mixed chunk**（decode 冻结根除）及笔误修复 |
   | 0030-0032 | mamba 溢出环三连抢跑 / **BACKUP-ON-EVICT** / #37837 cursor 复位 |
   | 0033 | DFlash sampling accept_len 钳位（fold OOB 修复） |
   | 0034-0036 | #35544 移植尝试 + 回滚（**净零**，见指南 §4.8 失败记录） |
   | 0037-0038 | #36267 GDN prefill 投影 / #37324 radix offset（上游已合并） |

4. **配置**：拷 `conf/sglang_qwen38_27b_dflash2.conf.sh` 到 `/root/`，改模型路径/端口/IP；
   - **WSL 用户**确认两个 `SGLANG_*HICACHE*` 环境变量未被注释；
   - 确认 `SGLANG_ENABLE_GRAPH_POOL_PRECARVE/BORROW=1` 在位；
   - 确认 `--mamba-max-states-per-path 6` 在位（9-03 根因修复配套）；
   - **NVFP4 用户**确认 `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` 在位（不设必崩，指南 §3.7）。

5. **监控先行**：`python monitor/server.py <sglang地址> 8917`（**在 Windows 宿主侧跑**，
   指南 §8.1——UNC 日志 + powershell 计数器决定了它必须在宿主侧），
   浏览器开 `http://<host>:8917`。先于一切优化部署，没有观测就没有优化（指南 §8）。

6. **启动**：`bash scripts/boot-sgl-stack.sh`（必须在宿主侧保活的会话里**前台**跑，指南 §7.3），
   冒烟两步走：纯文本 loadgen + **带图探测**（指南 §9.4——纯文本不触发 encoder pickle 路径）。
   `restart-main-sglang.sh`（9-02 起含编码器前置检查）会在主服务健康后自动跑
   `scripts/sglang-warmup.py`（28 个 lazy kernel 提前到启动期加载，防首个真实请求时
   cuModuleLoadData OOM；非致命，失败仅告警）。

7. **逐项验证**：对照 `scripts/ab-results.md` 的口径做自己的 A/B（golden prompt ×3 + loadgen）。
   Agent 场景基准见指南附录 A（agent_bench 三维度）。

## v2.3 相对 v2.2 的关键变化

| 项 | v2.2 | v2.3 |
|---|---|---|
| 模型 | Qwen3.8-27B-Uncensored-FP8（29G） | **NVFP4 混合精度（21.1G）** |
| KV 池 | 310k | **500k** |
| decode | 48-82 t/s | **100-125 t/s** |
| 调度 | lpm | **hrrn**（短请求 3×） |
| decode 抖动 | 有（ITL 抖到 581ms） | **PDI=3 消除** |
| 补丁方式 | 32 个 format-patch，`git am` | **单一合并 diff，`git apply`**（原方式实测不可用） |
| 监控 | v2.0 | **v2.2**（启动日志显存归因 + 编码器健康） |

## 关键回滚开关（指南 §9.2 速查）

| 场景 | 动作 |
|------|------|
| KV 池 OOM | conf.sh `MAX_TOTAL_TOKENS` 回 470000（或 310000 保守档） |
| HiCache 异常 | `HICACHE_SIZE=0` |
| DFlash2 异常 | `ENABLE_DFLASH=0 ENABLE_MTP=1` |
| mixed chunk 异常 | 去掉 `--enable-mixed-chunk`（decode 退回冻结形态但能跑） |
| HRRN 异常 | `SCHEDULE_POLICY=lpm` |
| decode 抖动 | `PREFILL_DECODE_INTERVAL=0` |
| 归因日志刷屏 | `SGLANG_PREFILL_LOG_MIN_TOKENS` 调大 |
| BACKUP-ON-EVICT 异常 | 回滚 0031 中 `evict_component` 内备份块（`git revert` 该 patch 会连带掉归因日志，手工摘除更稳）；日志判据 `MAMBA-EVICT-BACKUP-FAIL` 异常堆栈 |
| 溢出环异常 | conf.sh 加 `--mamba-overflow-size 0` |
| Gumbel 采样异常 | `SGLANG_OPT_USE_GUMBEL_SAMPLE=0` |
| NVFP4 启动崩 | 确认 `SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1` 在位 |

## 注意

- 上游合并 #37065/#36136 后建议 merge 重摘（指南 §2.4 盯防清单）。
- **换代码版本后 encoder + 主服务必须同时重启**，带图冒烟必须做（指南 §9.4，8-31 生产崩溃教训）。
- WSL 专有坑（VA 1TB 上限、chunked 2048、无 P2P）在裸机 Linux 上可省略对应 workaround（指南 §7）。
- conf.sh 内含三模式投机切换（DFlash/MTP/DSpark），A/B 时只改两个环境变量 + 重启。
- **Ada（SM89）上 NVFP4 的 prefill 会比 FP8 慢 ~25%**（无 FP4 tensor core，走 Marlin 反量化）；
  Blackwell 无此代价。这是物理限制，不是配置问题（指南 §3.7）。
