# sglang vs vLLM A/B — Qwen3.8-27B @ RTX4090(48G)+4070Ti(12G), WSL2
# 测试日期:2026-08-25  统一脚本:/root/ab_suite.py,统一端口 8778

## 最终结论:sglang 胜出,已固化为正式栈

### 正式 sglang 配置(conf.sh 已固化)
- 分支 qwen38-dflash2-integration + cherry-pick:
  - #35496 (1cf2b8c) DFlash2 selector 支持量化 lm_head
  - #35957 (54ec2c46) decode retraction 不再丢 recurrent state(手工合并)
- DFlash2 K=8 / draft window 2048 / fp8 draft KV
- max-running-requests=5,max-mamba-cache-size=20,bf16 SSM,cuda-graph bs≤5
- chunked prefill 2048,prefill-max-requests=1,page-size 64,radix LFU
- HiCache 关闭(HICACHE_SIZE=0;conf.sh 条件化,>0 即重开)
- mem-fraction-static 0.985 → KV 227,200 tokens + available 0.11G

### 关键实验:HiCache 是 prefill 衰减的唯一来源(64k 真冷 prefill)
| 配置 | 第1轮 | 第2轮 |
|---|---|---|
| HiCache 开 | 18.0s (3079 t/s) | 44.1s (1260 t/s) ← -59% |
| HiCache 关 | 16.8s (3313 t/s) | 16.8s (3298 t/s) ← 零衰减 |
机理:WSL 下为稳定性禁用了 MHA staged write-back 快速内核,
每次 radix 逐出走慢速 memcpy 回写,开销随缓存积累而放大。

## 统一口径最终数据(各取稳定轮)

| 指标 | vLLM(生产配置) | sglang(正式配置) | 胜者 |
|---|---|---|---|
| 单路 decode t/s | 46.5(基线)/32.9(复测) | **58-66** | sglang +30% |
| conc4 聚合 decode | 147-293(波动大) | 150-225 | 平手(vLLM 上限高但不稳)|
| 真冷 prefill 16k | 43.7s (318 t/s) | **3.3s (4263 t/s)** | sglang 13x |
| 真冷 prefill 32k | 119.6s (232 t/s) | **7.2s (3895 t/s)** | sglang 17x |
| 真冷 prefill 64k | 244.4s (227 t/s) | **16.8s (3300 t/s)** | sglang 15x |
| agent 冷启动(16k 头) | 41.8s | **3.3s** | sglang 13x |
| 缓存命中追问 TTFT | 1.0s(首次还 41.8s)| **0.15s** | sglang |
| 子代理×4 并发总墙钟 | 16.7s | **1.7s** | sglang 10x |
| 90k 长解码 t/s | 64.6 | 63.6-65.8 | 平手 |
| 90k 冷 prefill+解码总时 | 406s | **26s** | sglang 15x |
| Anthropic /v1/messages 原生 | 需代理 | **原生支持** | sglang |

## 附注
- vLLM 冷 prefill 慢是架构性的(GDN align 每 chunk 物化状态),
  此前"修复后 5.6s"实为缓存命中假象,nonce 全冷测试证实。
- sglang S1 的 conc1/conc2 偶发低值(27-34 t/s)是 WDDM 时钟爬坡,
  稳定后 58-66 t/s。
- DFlash accept len ~3.0-3.6(两引擎相近);sglang K=8 vs vLLM K=7。
- 上游还有 #35543/#35769(HiCache 修复)未取,因我们已关 HiCache;
  若未来 WSL 内核修复或换原生 Linux 可重评。

## 2026-08-25 晚间定案:切 B 方案(池子优先)
用户负载画像:主力 Claude Code 单路长会话(到 200k 触发 compact),多工具(codex/hermes)
轮流打同一 API,子代理偶发 → 并发让位给 KV 池。
- conf.sh 改回:MAX_RUNNING_REQUESTS=3 / MAX_MAMBA_CACHE_SIZE=12 / CUDA_GRAPH_MAX_BS_DECODE=3
- 实测:max_total_num_tokens=262144(官方标称对齐),available_gpu_mem=0.52G,hicache off
- 验证:decode 66.2 t/s、冷 prefill 4240 t/s、P4 长解码 65.2、子代理×4 总墙钟 1.9s——全部无回归
- 注意:262k 是零余量达标(池恰好等于上下文名义值,顶满输入时输出位置紧张);
  vLLM 的 340k 对 262k 有真实余量。质量上 >128k 本就进入退化区,此极限位日常用不满。
- boot-sgl-stack.sh 已重建为正式资产(encoder+主服务一键拉起)。

## 2026-08-25 深夜:278k 实验与回滚(最终定案 262k)
- 实验:MAX_TOTAL_TOKENS=278528 → 实际分配 270,976(+8.8k)
- 压测通过(3×148k 连续打满、无崩溃、性能零回归),但日志出现
  'free device mem: 0.00 GiB'——Triton 内核贴零懒加载,长期 OOM 风险高
- 判定:8.8k 收益不换稳定性 → 回滚 MAX_TOTAL_TOKENS=262144
- 回滚后回归验证全绿:decode 57.9-63.7、prefill 4266 t/s、P4 长解码 64.5、子代理×4 墙钟 1.85s
- 结论:262k 是本机'稳定优先'约束下的正确极限位;
  单请求 >270k 需求出现时临时用 MAX_TOTAL_TOKENS=278528 起专用实例。
- 运维备注:wsl.exe 会话退出会连带杀掉 setsid+nohup 后台启动,
  启动栈必须在宿主侧保活的会话里跑 boot-sgl-stack.sh(Claude Code 的 run_in_background 即可)。

## 2026-08-25 深夜:MTP(EAGLE)方案测试 vs DFlash2 生产
背景:27B 官方 checkpoint 内置 MTP 头(mtp.* 22 张量,FP8),conf.sh 加 ENABLE_MTP=1 分支
(EAGLE + draft-model-path=主模型 + 3/1/4 + mem-fraction 0.94 + 无 expandable_segments)。

### 关键坑:WSL GPU VA 空间耗尽(dxgkrnl 上限 ~1TB)
- 症状:启动后 /health 503,scheduler 死旋在 mamba alloc_group_end,
  strace 8000+ ioctl 全部 EOVERFLOW(-75),dmesg 刷 dxgkio_reserve_gpu_va failed。
- VA 对照:EAGLE scheduler 进程 1060.5 GB(触顶)vs DFlash2 736.9 GB(健康)。
- 根因:expandable_segments 按 2MB 粒度 reserve VA,EAGLE 的双 worker
  (target_verify/draft_extend CUDA graph + draft 池)多出 ~320GB,打穿 dxgkrnl 上限。
- 修复:MTP 模式下不设 expandable_segments(conf.sh 已按 ENABLE_MTP 条件化),即恢复健康。

### 性能对照(同硬件、同 262k 池、单路 512-token 连续生成、greedy)
- DFlash2(K=8):55.9 / 57.3 / 57.0 tok/s,accept len 2.75-3.65
- MTP(3/1/4):53.3 / 55.2 / 55.1 tok/s,accept len 2.77 / accept rate 0.59
- prefill 两者持平(MTP 冷 prefill 4229 t/s);P4 长解码 MTP 61.4 vs DFlash2 64.5-66
- 子代理并发:MTP 下 TTFT 明显劣化(sub2 ttft 11.5s vs DFlash2 0.7s)

### 结论:DFlash2 继续生产,MTP 不切换
- 单路 decode DFlash2 快 ~3-4%,长解码快 ~5%,子代理爆发快一个量级
- MTP 唯一理论优势(不依赖独立草稿 checkpoint)对已持有 DFlash2 权重的我们无意义
- 但 MTP 路径已在 conf.sh 可用(ENABLE_DFLASH=0 ENABLE_MTP=1 一键切换),
  且修复了 VA 耗尽的通用知识(未来任何加 draft worker 的实验都要注意)

## 2026-08-27 API 链路采样参数审计(agent→cc-switch→sub2api→CPA→sglang)
方法:sglang 开 --log-requests --log-requests-level 1,无参/带参探针从各跳打入,
比对到达

## 2026-08-27 DFlash2 草稿量化实测(三选一)
候选:tcclaviger-FP8(dynamic per-token, sglang原生quant_method=fp8) /
josch15366-FP8(ct per-channel) / gratex-W4A16-g128-GPTQ(ct pack-quantized)
基准:同 target(Uncensored-FP8)、同 conf、draft_ab.py 3任务×3rep 取中位数。

结果:
1. tcclaviger FP8:✅ 唯一可用的量化草稿。秒启无改。
   decode: task0 74.8 / task1 80.6 / task2 64.6 t/s(BF16 基线 75.9/90.0/62.9,
   平均约 -4%)。accept len 分布 2.95-3.67(基线同时段窗口难精确对齐,
   历史分布 2.77-5.12)。KV 池同为 262144(max-total-tokens 钉死),
   省出的 1.46G 全部落入显存余量(0.52→1.98GB,余量翻3.8倍——
   对 OOM 安全垫和未来扩池都是实打实的收益)。
2. josch15366 FP8:❌ 启动即崩。ct config 的 targets 只写了拆分名(q/k/v_proj),
   sglang 融合成 qkv_proj 后 find_matched_target 失配 → ValueError。
   (vLLM 同样加载不了,作者卡上已自认。)
3. gratex W4A16 GPTQ:❌ 能启动但草稿提议全废:accept len≈1.0(基线3+),
   decode 崩到 21-22 t/s。该 checkpoint 需要 apply_dense_kv_fix.py 预处理
   打包权重,直接用是坏的;另外其 JIT gptq_marlin 内核首编需 ~2分钟
   (cuda graph 捕获阶段等待会 watchdog 超时,手动预热后可过)。

判定:
- 单纯换 quantized draft 不值:-4% decode + accept len 不稳定,省的显存
  在 262k 钉死配置下只变成余量。**BF16 草稿继续生产。**
- 但 tcclaviger FP8 是有效备件:若未来需要把池推过 270k(+13k tokens)
  或跑 >200k 极限上下文,它可以让出 1.46G 显存实现;一行环境变量切换:
  DRAFT_MODEL_PATH=/root/LLM/Qwen3.8-27B-DFlash2-FP8-tcclaviger bash /root/start-sglang-qwen38-dflash2.sh
- grtex 目录里 scripts/apply_dense_kv_fix.py 未验证;若修复后接受率恢复,
  W4A16 省 2.4G(池可到 ~277k)仍值得二次评估。

## 2026-08-27 同权重智商对决:采样档位影响(sglang 上的同一 FP8 权重)
问题:用户问'vLLM 驱动 vs sglang 驱动,智商一样吗'。
背景:两引擎跑的是**同一个 checkpoint 文件**,逐位相同的权重;
唯一系统性差异是社区默认采样预设(vLLM 工具链惯用 T0.7/top_p0.8,
Qwen 官方 gen config 为 T1.0/top_p0.95/top_k20)。
方法:sglang 单引擎、同权重、三档采样(T1.0-genconfig / T0.7-vllmstyle /
T0-greedy-ref)× 20 题 × 2rep,机器判分。初始结果 14/13/14。
修正:复核发现两题答案键错误——
(a) 'exactly-one of 3或5 @1000' 真值是 401(我误写 271;三档全部答对401,全对!)
(b) '(7^3-10)/31' 除不尽=10.74(陷阱题设计失败,各档都给了诚实近似,不记分)
(c) 家族推理题(FAMILY=5)因 max_tokens 截断在思考阶段未输出标记,3 档全判 MISS
   属提取器假阴性。
修正后有效口径:数学4题+科学3题共14个判定点×2rep:
T1.0: 11.5/14 | T0.7: 12/14 | T0-ref: 12.5/14 —— 三档无显著差异。
结论:**vLLM 与 sglang 跑同一份权重的'智商'完全一致**(权重相同→能力相同);
若体感有差,来自:(1)采样预设不同导致的随机性差异(高温方差不稳);
(2)sub2api 强制覆盖参数(见上节审计);(3)草稿量化等性能侧差异不影响输出分布。

## 2026-08-27 复审(两仓库 PR 例行扫描,sglang 落后 446 / vLLM 落后 350)
盯防列表状态:#36267/#36266/#36041/#35788/#35985 全部仍 open 未合并,无需动作。

【重要发现】GDN packed decode 的 beta BF16 舍入(vLLM #53877 同源问题):
- vLLM #53877(draft)揭示:fused_recurrent packed decode 内核里
  beta_val = sigmoid(fp32).to(bf16).to(fp32) 的往返舍入,随 decode 步数
  在 recurrent state 里复利累积;1000 步后输出 rel L2 差 17 倍;
  用户症状为长生成卡死循环/答案不收尾(vLLM 插件侧 20/42→1/42)。
- 核对本地树:fused_recurrent.py:254 存在完全相同代码,且该内核正是
  我们的生产 decode 主路径(dispatcher 日志 TritonGDNKernel + packed_decode=True)。
- 关键约束:此前 #36014 类一致性要求 decode 与 verify 的 beta 逐位一致。
  我们的 verify 路径经 fused_gdn_gating(:41 同样把结果舍到 bf16 再存入
  fp32 缓冲)。单点修 decode 会重新引入不一致(mamba radix 复用污染)。
  正确修法是双点协同:同时去掉 :254 与 fused_gdn_gating:41 的下转型,
  全链路保持 fp32(beta)。(update 内核 :493 本就是纯 fp32,不动。)
- 状态:待用户批准后实施+数值验证(triton 双版本 1000 步 drift 对照 +
  accept len / 长生成稳定性回归)。ssm state 存储本身仍是 bf16(性能取舍保留)。

【本轮其他值得记的】
- sglang #36583(KV 预算被加载器临时引用挤占,GC 后再测余量)— 我们显式
  钉死 max-total-tokens,不受影响;纯知识储备。
- sglang #36615(只读预编译 JIT 缓存目录)— 可避免 W4A16 类实验首编撞
  watchdog 的坑。
- 新模型情报:Qwen3.8-Flash-Next(Qwen4 架构预览,176B-A6B MoE,GDN+QSA),
  day-0 已进主线,但硬件门槛 H200 起,与我们无关;标志 sglang 对该家族的
  支持仍在高强度演进。
- vLLM 侧 #53864(CuteDSL KKT 牛顿迭代第4轮)、#53970(DFlash token 预算 OOB)
  均为其特有路径,不移植。

## 2026-08-27 GDN beta 精度修复:已实施并上线 ✅
改动(2 行,fused_recurrent.py + fused_gdn_gating.py):beta 从 sigmoid 后
bf16 往返改为全程 fp32。Triton 内核 JIT 自动重编(启动自动完成)。

数值验证:
1. 1000 步递推漂移 A/B(vs 纯 fp32 参考实现,Qwen3.8 真实形状 H16/HV48/K128/V128):
   新内核 relL2 = 4.8e-07(存储噪声级) vs 旧行为 4.1e-04 —— 差 ~865 倍,
   与 vLLM #53877 报告量级一致。
2. verify/decode 一致性保持:gating 输出 beta 脱离 bf16 网格(0/384 值在网格上),
   两路径数学一致;残余 1.2e-07 为 triton 与 torch 的 sigmoid 实现差(<1 ulp),无状态累积。

生产回归(ab_suite,与修复前对照):
- decode conc1=67.3 / P4 长解码 65.3 / 冷 prefill 4244 t/s —— 全部零回归
- accept len 分布健康:tail 达 6.28/6.65/7.25(DFlash K=8 满载态)
- 长生成探针:1500 token 连续输出 101-104 tok/s,无退化

已进入生产(main-beta-fix.log 实例在线)。记忆中的待办已销。

## 2026-08-27 DSpark(RadixArk) vs DFlash2(z-lab) 生产对比
对象:RadixArk/Qwen3.8-27B-DSpark(SpecForge 训练,链式草稿+置信度门控动态深度,
block_size=7 → verify 窗口 8,与 DFlash K=8 等宽,公平对照)
同 target(Uncensored-FP8)、同 conf、同 ab_suite 口径。

结果(全部劣于 DFlash2):
- decode conc1:37.1 t/s vs DFlash ~67 t/s(-45%)
- conc2/3/4 agg:56.1/94.4/134.2(DFlash 同口径 100.6/166/203)
- P4 长解码:55.9 vs 65.3
- accept len:1.9-3.5 分布(中位~2.4),DFlash 常态 3.0+,tail 7+
- 模型卡自证口径(T=0.6 创意长文 1500 tok):49-52 t/s
归因:
1. DSpark 的置信度头在非 Qwen 官方 FP8 target 上失配——它是为
   Qwen/Qwen3.8-27B-FP8 训练的,我们是同底 finetune(Uncensored),
   草稿分布与目标分布的偏移对'逐深度提前放弃'的设计更敏感;
2. 每步 draft_decode graph(1.13s 捕获)在线运行成本更高,
   短任务 conc 测试里 TTFT 也变差;
3. DFlash 的整块扩散式提议对此负载(代码+中文创作混合)接受率更稳。
结论:DFlash2 继续生产。DSpark 保留为 conf.sh 一键切换
(ENABLE_DFLASH=0 ENABLE_DSPARK=1 DRAFT_MODEL_PATH=/root/LLM/Qwen3.8-27B-DSpark);
若未来换官方 Qwen FP8 target 且以英文为主,可重评。

## 2026-08-27 官方 FP8 target 上 DFlash2 vs DSpark(RadixArk)对决
靶子:/root/LLM/Qwen3.8-27B-FP8(官方,非 uncensored);同池配置(262k/3并发);
D Flash=K8(D=8)、DSpark=gamma7(verify 窗 8),窗口长度相同。
conf.sh 踩坑修复:DSPARK 分支的 draft path 原来错写成 
(BF16 DFlash 权重),已改为 ;DSpark 需要加
--speculative-draft-model-quantization unquant(草稿是 BF16 而 target 是 FP8,
默认继承 quantization 会拒载)。

| 指标 | DFlash2 (K=8) | DSpark (γ=7,D=8) |
|---|---|---|
| 单路 decode t0/t1/t2 | 68.1 / 87.1 / 65.7 | 49.5 / 62.3 / 43.6 |
| accept len(日志均值) | 3.16-3.75 | 1.82-2.58 |
| conc4 聚合 tok/s | 111.7 | 84.0 |
| cold prefill ~14k | 3.30s | 3.18s(非投机路径持平) |

结论:同 verify 窗口下 DFlash2 全面胜出——单路 +27~50%,并发聚合 +33%。
根因:DSpark 的 confidence-gated 深度在 greedy 单流下平均只走 ~2.3/8 窗,
而 DFlash 的全窗扩散验证能稳定吃满 3.4+;RadixArk 卡上的 H200 数据与其一致
(DSpark 强项是不确定性采样下的深度弹性),greedy agent 场景不是它的主场。
生产保持 DFlash2;DSPARK 一键切换保留(conf.sh ENABLE_DFLASH=0 ENABLE_DSPARK=1)。

## 2026-08-28 PR 例行扫描(sglang 落后 484 / vLLM 落后 379)
盯防列表:#36267/#36266/#36041/#35788/#35985/#36310/#36065/#35954/#36014
及 vLLM #53877/#53542/#53463/#53070/#52297/#52244 —— 全部仍 open,零合并。

### 本轮新发现(按价值排序)
1. **#36568 skip DP1 redundant all_gather(+11.45% decode TPS/user)**:
   本地树已有现成开关 SGLANG_SCHEDULER_SKIP_ALL_GATHER(默认 False)。
   TP1 单卡也走 all_gather_into_tensor+同步,设 1 即白捡。
   → 已实测,见下节。
2. **#36696 mamba radix _split_node child-key 漂移(page_size>1 树腐蚀)**:
   核对本地 mamba_radix_cache.py:1212/1230,确认**存在同款代码**
   (注册用 key[split_len:] 推导、删除用 node 自己的 key 重推,split_len<page_size
   时两表达式不一致)。我们 page_size=64 + mamba radix 长期运行,有踩雷风险。
   修法:注册处改用 new_node.key 自己重推(与删除处同源)。待实施。
3. **#36683 ReplaySSM spec-verify for DFlash/DSPARK on GDN(open)**:
   把 DFlash2 草稿中间态搬上固定 ring(D=0,mamba 比率回落到无spec值),
   正是我们的场景。合并后值得评估。
4. **#36722 GDN GQA head pairing HV=2H 修复(open)**:核对 27B
   linear_num_value_heads=48 / linear_num_key_heads=16 => HV=3H,不中招;仅记录。
5. vLLM #53970(DFlash token 预算 OOB)、#53929(adaptive DSpark for Qwen GDN)
   —— DFlash2 在 vLLM 侧仍在快速演进,与我们无关但证明方向正确。

### 实测:SGLANG_SCHEDULER_SKIP_ALL_GATHER=1(模仿 #36568)
   —— 用户要求暂停实测(GPU 在用),栈已停。待 GPU 空闲后:
   SGLANG_SCHEDULER_SKIP_ALL_GATHER=1 起服务,跑 ab_suite.py 对照 conc=1-4/预fill/P4。
   #36696 同款 child-key bug 的修复也在待办清单。

## 2026-08-29 PR 复扫(sglang 落后 509 / vLLM 落后 397)

### 好消息:#36568 已合并上游
但核对源码后确认**对我们无效**:该优化路径只在 mlp_sync(DP attention/EP gather)
激活时运行;我们 TP1+无 DP+a2a=none => require_mlp_sync=False,
maybe_prepare_mlp_sync_batch 根本不会进 all_gather。零收益,不 chase。
(早前'本地已有 env 开关可白捡'的判断作废——那条路径我们不执行。)

### 新合并值得记录
- sglang #36705(HiCache mmap 双重填充,-13% 分配时间):我们 HiCache 关,不适用。
- sglang #35944(scheduler metadata 异步 H2D 前 pin):通用的正确性加固,
  在 509 提交池里,未来 rebase 自然带上。
- vLLM #52789(mamba 前缀缓存 mid-prefill 内部 checkpoint,+9~25% TTFT):
  对照发现 **sglang 早已有等价机制**——
  (radix 缓存的 mamba state 存 int8 checkpoint 池,~2x 缓存容量),我们基线里就有。
  未开的原因:与 spec(DFLASH)的 track_interval 交互、显存代价需实测。
  列为**候选实验**(GPU 空闲时跑:开/关对照 cold prefill 与 accept len)。

### 仍未合并的盯防(状态刷新)
- #36696(mamba radix split child-key 漂移):复核了我们树里的删除路径
  (_delete_tombstone_leaf:1365-1367 用 node.key.child_key 反推),
  bug 的必要条件是 split_len < page_size;而我们的插入路径强制页对齐
  (insert :702-712 断言 page_aligned_len==len),match 返回页对齐,
  => split_len 恒为 64 的倍数,**不会触发**。降级为'仅记录'。
- #36770(mamba radix 满槽无槽可逐出时断言):HiCache 场景,我们不受影响。
- #36722(GDN HV=2H 头配对):27B 是 HV=3H(48/16),不中招。
- #36683(ReplaySSM spec-verify for DFlash/DSPARK):仍 open,合并后评估。
- 上游三个 CI 挂着的记忆池 PR(#36310/#36065/#36041 等)依旧原地踏步。

### 结论
本轮没有需要立即移植的补丁;唯一候选实验是开
 --enable-int8-mamba-checkpoint 做 A/B(需实测与 DFLASH 的兼容与收益)。

## 2026-08-29 int8-mamba-checkpoint A/B 结果:不采纳
配置:int8 臂 = --enable-int8-mamba-checkpoint --int8-mamba-ckpt-size 8
+ mem-fraction 0.96(默认 2x 槽在 WSL 显存贴零触发 device-not-ready 崩溃,
  8 槽+0.96 才稳;此为 WSL 特有约束)。
- 单路/并发/prefill/长解码:全部持平(59.3/108.4/140.8/235.9, P4 64.8),
  KV 池同为 242k(0.96 挤压)vs 262k。
- 单文档 warm 复用:两臂相同(84k 前缀 0.73-0.75s,~40x 加速)。
- 多文档 churn(4x84k 与 4x40k 两轮):两臂全部 0/4 命中——载入下一个文档
  会把上一个文档的前缀从树里连 KV 一起剪掉(full usage 仅 0.17 仍逐出),
  mamba 检查点容量才是约束,int8 快照未改变此行为。
结论:对'单会话+长前缀重入'收益为零(本来就命中),对'多文档并存'无效
(逐出机制不因此放宽),还占用 0.96 显存预算。不采纳,生产保持现状。

## 2026-08-29 重新审视:vLLM/sglang 13-17x prefill 差距是否正常
用户质疑:正常不该差这么大。复核结论——**直觉对,13x 需要拆解归因**:

【模型结构事实】Qwen3.8-27B = 64 层 hybrid:48 GDN(线性注意力)+ 16 full-attn
(每4层1个),dense FFN 17408,hidden 5120。GDN 状态 O(1)/token 是池子优先
部署的根本前提;full-attn 只有 16 层,KV/tok = 2*4*256*1B = 2KB(fp8)。

【vLLM 侧慢的拆解(源码复核 + 档案)】
1. engine-chunk 数量:vLLM 当时 max-num-batched-tokens=4096(且我们不敢开更大,
   8192 chunk 在 WSL 上触发 GDN FLA 内核 device-not-ready 崩溃——本次实测证实),
   16k 输入 = 4+ 次 engine chunk;sglang 2048/chunk 但**单次 FLA 调用内处理
   完整 seqlen**,chunk 边界开销不在 GDN 上。
2. mamba-cache-mode=align:vLLM 的 prefix-caching 兼容模式要求每个 engine
   chunk 物化(conv+ssm 中间状态写回),per-chunk 固定税 ×4 次;sglang 的
   mamba radix 在 FLA 内部以 chunk 粒度累积,无 engine 级物化税。
3. host sync 密度:vLLM gdn_attn build() 有大量 .item()/.tolist()(236-320行),
   WSL 下每次 sync ~ms 级;sglang metadata 全 GPU 侧。
=> 13-17x = 8x chunk 税 × ~1.6x 每税点差(物化+sync),不是引擎本质差距;
   在原生 Linux + 大 chunk 下差距会显著收窄(但 decode 仍 sglang 略优)。

【我们自己的部署再审视】
- sglang chunked-prefill 8192 实测崩溃(FLA recompute_w_u device-not-ready),
  2048 是 WSL 稳态;prefill 4262→4183 无改善空间,当前配置即最优。
- 结论维持:sglang 生产不动。差距的相当部分是 vLLM 部署参数与 WSL 环境税,
  非 sglang 独有魔法。

## 2026-08-29: HiCache 重新开启 — L2 回取收益实测(推翻 8-25 结论)
背景:多工具轮转(claude ~200k + codex ~100k)超出 262k L1,切换被迫全量重 prefill。
方法学复核:8-25 的"3300→1300 tok/s 衰减"实为 nonce 全冷测试只测到逐出回写税
(write_back 一次性债),L2 回取收益从未测过。

配置:HICACHE_SIZE=14 → 三池切分(实测比例):
- 主 KV host 池 302,848 tokens / 9.92 GB(> L1 262k,警告消失)
- mamba host 4.08 GB(逐出分支的 GDN state 可备份/恢复)
- draft host 3.10 GB(DFlash draft KV 的 L2 层)
- write_back + kernel io + page_first;hicache_attached=True

L2 回取实测(hicache_l2_verify.py,3×93k 文档填满 L1 逼逐出):
| 阶段 | 耗时 | 吞吐 |
|---|---|---|
| FILL0/1(冷) | 34.2s / 34.3s | 2720 / 2713 t/s |
| FILL2(触发逐出+回写) | 42.7s | 2183 t/s(-20%,一次性) |
| RESTORE0(从 L2 恢复) | **17.8s** | **5233 t/s** ← 快 47% |
| RESTORE1(从 L2 恢复) | **18.2s** | **5120 t/s** ← 快 47% |

机理:混合架构下 mamba 状态不可拆,逐出以整分支为粒度(mamba checkpoint 备份到
host),所以 RESTORE 是整分支 L2 回取而非部分命中。H2D 回取 ~5.2k t/s,
约为重算的 2 倍。

结论:开启定案。收益=超 L1 轮转场景恢复快 2x(200k 切回 ≈38s vs 重算 74s);
税=逐出时的一次性回写(-20% 吞吐,仅持续到回写完成);≤262k 单会话场景无税无益。
运维:重启主服务用 /root/restart-main-sglang.sh(内置健康轮询,宿主侧保活会话跑)。
