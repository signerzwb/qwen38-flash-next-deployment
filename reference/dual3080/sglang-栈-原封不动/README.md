# sglang-qwen38-27b 栈复刻包

配套文档：`sglang-qwen38-stack-guide.md`（完整技术方案，先读它）。本包是文档的可执行落地件。

## 目录

```
sglang-qwen38-stack-guide.md   # 技术方案全文（设计要点/PR 来源/测试数据/方法论）
conf/
  sglang_qwen38_27b_dflash2.conf.sh   # 全部配置，唯一调参入口（逐项注释）
scripts/
  boot-sgl-stack.sh            # 完整启动（encoder 先起 → 主服务）
  start-sglang-qwen38-dflash2.sh   # 只起主服务 :8778
  start-sglang-qwen38-encoder.sh   # 只起视觉 encoder :8779
  restart-main-sglang.sh       # 只重启主服务（改 conf.sh 后用）
  stop-sglang-qwen38-{dflash2,encoder}.sh
  ab-results.md                # 全部 A/B 测试数据存档（359 行原始数字）
patches/                       # 源码树 12 个栈专属提交（git format-patch 导出）
monitor/                       # KV 监控面板 v1.3.2（零依赖）
  server.py  index.html  DESIGN.md  README.md  loadgen.py  gpu_mem.ps1
```

## 复刻步骤

1. **环境**：WSL2 (ubuntu2204) + 双卡（48G 主卡 GPU0 + 12G 副卡 GPU1）。
   单 48G 卡也能跑（视觉塔不分离），见指南 §10.1 档位表。

2. **源码树**：克隆 sglang，切出集成分支，基线选一个上游主线的稳定点，
   然后按序 apply patches（每个 patch 就是本栈的一个提交，顺序即依赖序）：

   ```bash
   cd sglang-source
   git checkout -b qwen38-integration <基线commit>
   for p in /path/to/patches/*.patch; do git am "$p" || break; done
   # 某个失败时：git am --show-current-patch 看冲突，解决后 git am --continue
   # 0004 (HiCache WSL workarounds) 标题带 WIP：非 WSL 环境可先跳过，冲突概率最低先试
   ```

   patch 对应关系（详见指南 §2.2）：
   | # | 内容 |
   |---|------|
   | 0001 | 视觉塔分离（language-only 模式跳过 vision tower） |
   | 0002 | #35496 量化 lm_head |
   | 0003 | #35957 retraction 丢 recurrent state |
   | 0004 | HiCache WSL workarounds（两个环境变量开关，**WSL 必装**） |
   | 0005 | #36415 mamba 准入槽扣账 |
   | 0006 | #36696 mamba radix split 崩溃修复 |
   | 0007 | #36266 COW 内核预热 |
   | 0008 | **#36683 ReplaySSM**（手搓版，最大收益项） |
   | 0009-0012 | 请求级 prefill 归因日志（4 连迭代） |

3. **配置**：拷 `conf/sglang_qwen38_27b_dflash2.conf.sh` 到 `/root/`，
   改模型路径/端口/IP；**WSL 用户确认两个 `SGLANG_*HICACHE*` 环境变量未被注释**。

4. **监控先行**：`python monitor/server.py <sglang地址> 8917`，浏览器开 `http://<host>:8917`。
   先于一切优化部署，没有观测就没有优化（指南 §8）。

5. **启动**：`bash scripts/boot-sgl-stack.sh`（必须在宿主侧保活的会话里跑，指南 §7.3），
   用 `monitor/loadgen.py` 冒烟。

6. **逐项验证**：对照 `scripts/ab-results.md` 的口径做自己的 A/B（golden prompt ×3 + loadgen）。

## 关键回滚开关（指南 §9.2 速查）

| 场景 | 动作 |
|------|------|
| KV 池 OOM | conf.sh `MAX_TOTAL_TOKENS` 回 262000 |
| HiCache 异常 | `HICACHE_SIZE=0` |
| DFlash2 异常 | `ENABLE_DFLASH=0 ENABLE_MTP=1` |
| 归因日志刷屏 | `SGLANG_PREFILL_LOG_MIN_TOKENS` 调大 |

## 注意

- 本包 patch 基于 2026-08-29 时点的上游 sglang；上游合并 #36683/#33639 后建议 rebase 重摘（指南 §2.2 盯防清单）。
- 非裸机 Linux 专有坑（VA 1TB 上限、chunked 2048、无 P2P）在 WSL 上才存在，裸机可省略对应 workaround（指南 §7）。
- conf.sh 内含三模式投机切换（DFlash/MTP/DSpark），A/B 时只改两个环境变量 + restart-main。
