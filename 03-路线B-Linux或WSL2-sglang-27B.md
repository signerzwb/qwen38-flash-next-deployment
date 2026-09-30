# 03 路线 B：Linux / WSL2 + sglang + Qwen3.8-27B

适用：单卡 48G 的 Linux 裸机或 WSL2；双卡用同一套，只改 TP。
做完的结果：8778 端口一个 OpenAI 兼容接口，262144 上下文，DFlash 投机解码，3 路并发，带视觉。

本路线比 Strata 麻烦得多（要自己编译带补丁的 sglang），但换来的是：**内存要求低、支持并发、可 systemd 托管**。

---

## 一、前置

| 项 | 要求 |
|---|---|
| 系统 | Ubuntu 22.04 / 24.04。WSL2 用 Ubuntu-24.04 发行版 |
| 驱动 | Windows 侧装 NVIDIA 驱动 ≥580；WSL 里直接 `nvidia-smi` 能看到卡和正确显存 |
| 磁盘 | 至少 100GB（模型 35GB + 源码 + 编译缓存） |
| 内存 | ≥64GB。WSL2 建议在 `.wslconfig` 里放开到 96-120GB |
| Python | 3.12（用 venv，不要动系统 Python） |

WSL2 的内存上限配置，写在 `C:\Users\<你>\.wslconfig`：

```ini
[wsl2]
memory=120GB
swap=16GB
processors=16
```

改完 `wsl --shutdown` 再进，用 `free -g` 确认。

## 二、下载模型（约 35GB）

| 模型 | 仓库 | 大小 | 用途 |
|---|---|---:|---|
| Qwen3.8-27B-FP8 | `Qwen/Qwen3.8-27B-FP8` | 约 30GB | 主模型（fp8 量化，本地推理够用） |
| Qwen3.8-27B-DFlash2 | `Qwen/Qwen3.8-27B-DFlash2` | 约 3.8GB | 投机解码的草稿模型 |
| （可选）Uncensored NVFP4 | `piscesbody/Qwen3.8-27B-Uncensored-NVFP4` | 约 21GB | 更省显存、更快的替代主模型 |

魔搭下载（国内快）：

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3.8-27B-FP8      --local_dir /root/LLM/Qwen3.8-27B-FP8
modelscope download --model Qwen/Qwen3.8-27B-DFlash2  --local_dir /root/LLM/Qwen3.8-27B-DFlash2
```

魔搭上找不到同名仓库时，走 HF 镜像：

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.8-27B-FP8     --local-dir /root/LLM/Qwen3.8-27B-FP8
huggingface-cli download Qwen/Qwen3.8-27B-DFlash2 --local-dir /root/LLM/Qwen3.8-27B-DFlash2
```

**判断模型是不是 FP8**：看 `config.json` 里有量化配置（`quantization_config`），有就对了。
注意启动时**不要**再手动传 `--quantization fp8`，让 sglang 自己读配置。

目录约定（后面 conf 里的路径要和这里一致）：

```
/root/LLM/Qwen3.8-27B-FP8
/root/LLM/Qwen3.8-27B-DFlash2
```

## 三、建源码树 + 打补丁 + 建虚拟环境

这套栈必须用**打过补丁的 sglang**（官方主线跑不了 Qwen3.8 的混合线性注意力 + DFlash2 投机）。
补丁包就是 `sglang-qwen38-stack-package-v2.3` 里的 `patches/qwen38-0909-merged.patch`。

```bash
# 1) 拉源码（国内可以用 gitee 镜像）
git clone https://gitee.com/mirrors/sglang.git /root/sglang-source-v23
cd /root/sglang-source-v23

# 2) 切到补丁的基线提交
git fetch origin 0da6a6685648a415818bfa2e44471cb884009f35
git checkout -b qwen38-integration 0da6a66856

# 3) 一次性打合并补丁（不要用 by-commit 逐个打，会失败）
git apply /root/patches/qwen38-0909-merged.patch

# 4) 校验：官方包打完后 tree 值应为 53bbd2f6d78d9021a1e0e7021a263fea3081b507
git rev-parse 'HEAD^{tree}'
```

> 我们这台机器上的树还叠了后续的本地改动（FP8 profile、视觉、调参），所以 tree 值和上面不同。
> 新机器按官方包复刻即可，拿到 53bbd2f6 就说明补丁完整。

建虚拟环境并装依赖：

```bash
python3.12 -m venv /root/sglang-env-v23
/root/sglang-env-v23/bin/pip install -U pip wheel
cd /root/sglang-source-v23
/root/sglang-env-v23/bin/pip install -e .
```

我们这台机器上跑通的版本组合（可作为对照）：

| 包 | 版本 |
|---|---|
| Python | 3.12.3 |
| torch | 2.13.0 |
| flashinfer（含 flashinfer-python） | 0.6.17 |
| sglang_kernel | 0.4.6.post1 |
| sglang（本地源码树） | 0.0.0（editable） |

CUDA 运行库走 venv 里的 pip 包，启动脚本里要指过去：

```bash
cuda_home=/root/sglang-env-v23/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$cuda_home
export PATH=/root/sglang-env-v23/bin:$cuda_home/bin:$PATH
export LD_LIBRARY_PATH=$cuda_home/lib64:/usr/lib/wsl/lib
```

## 四、配置（照着改三处就能跑）

把 `config/` 里的四个文件拷到 `/root/`：

```
run-sglang-qwen38-v23.sh              # 启动入口（systemd 调它）
sglang-qwen38-v23-single48g.conf.sh   # 全部参数都在这里
run-sglang-qwen38-fp8.sh              # FP8 profile：覆盖模型路径/上下文/显存比例
sglang-qwen38.service / sglang-keepalive.service / fp8.conf   # systemd
```

**必须改的三处**：`MODEL_PATH`、`DRAFT_MODEL_PATH`、`CUDA_VISIBLE_DEVICES`。
其余保持默认即可。

### 关键参数逐项说明（48G 单卡 FP8 profile）

| 参数 | 值 | 为什么 |
|---|---|---|
| `--context-length` | 262144 | 原生长度上限，吃满 |
| `--kv-cache-dtype` | fp8_e4m3 | KV 用 FP8，262K 全程不到 1G，省显存 |
| `--mem-fraction-static` | 0.93 | 留 7% 余量，**不要上 0.95+**，投机解码的工作区会超 |
| `--tp-size` | 1 | 单卡。双卡改成 2 |
| `--max-running-requests` | 3 | 3 路并发。再高会挤占 KV |
| `--chunked-prefill-size` | 2048 | 长文分块预填，8192 会撑大激活峰值 |
| `--max-mamba-cache-size` | 12 | 12 个状态槽，对应 3 路并发 × 4 |
| `--schedule-policy` | lpm | 前缀缓存友好；短请求为主时可试 hrrn |
| `--prefill-decode-interval` | 3 | 消除 decode 抖动 |
| `--speculative-algorithm` | DFLASH | 投机解码，配 DFlash2 草稿模型 |
| `--speculative-num-draft-tokens` | 8 | 每步草稿 8 个 token |
| `--attention-backend` / `--mamba-backend` | flashinfer | FP8 KV 与线性注意力都走 FlashInfer |
| `--enable-multimodal` | 开 | 视觉。**不要加 `--language-model-only`**，那会关掉视觉 |
| `--reasoning-parser` | qwen3 | 把思考内容拆到 `reasoning_content` |
| `--tool-call-parser` | qwen3_coder | 工具调用解析 |
| `--enable-metrics` / `--enable-cache-report` | 开 | 能看到命中率与速度，排障必备 |

环境变量：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 必须，防碎片化 OOM
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ENABLE_GRAPH_POOL_PRECARVE=1
export SGLANG_ENABLE_GRAPH_POOL_BORROW=1
export SGLANG_DISABLE_SILU_FP4_QUANT_FUSION=1             # 只有 NVFP4 模型需要
```

### 显存不够时的调整顺序（从上往下试）

1. `CONTEXT_LENGTH` 262144 → 196608 → 131072，同时把 `MAX_TOTAL_TOKENS` 改成一样的值
2. `MEM_FRACTION_STATIC` 0.93 → 0.90
3. `MAX_MAMBA_CACHE_SIZE` 12 → 8
4. 关掉投机：`ENABLE_DFLASH=0`（速度掉，但显存立刻松）
5. 换更小的模型（NVFP4 21GB 替代 FP8 30GB）

## 五、用 systemd 托管（崩了自动拉起）

WSL2 想用 systemd，先在 `/etc/wsl.conf` 里打开：

```ini
[boot]
systemd=true
```

然后 `wsl --shutdown` 重进。裸机 Linux 本来就带 systemd，跳过这步。

主服务 `/etc/systemd/system/sglang-qwen38.service`：

```ini
[Unit]
Description=SGLang v2.3 Qwen3.8-27B multimodal server
After=network.target

[Service]
Type=simple
ExecStart=/root/run-sglang-qwen38-v23.sh
Restart=always
RestartSec=8
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:/root/sglang-qwen38-v23.log
StandardError=append:/root/sglang-qwen38-v23.log

[Install]
WantedBy=multi-user.target
```

`Restart=always` 是关键：sglang 崩了不会自愈，靠 systemd 拉起来。

**再套一层 SSE keep-alive 代理**（`sglang-keepalive.service`，监听 8779 转发到 8778）。
原因：uvicorn 默认 5 秒断开空闲长连接，客户端复用连接池时会报
`stream disconnected before completion` / `error sending request`。
代理和 `SGLANG_TIMEOUT_KEEP_ALIVE=300` 一起解决这个问题。

```bash
systemctl daemon-reload
systemctl enable --now sglang-qwen38.service sglang-keepalive.service
```

## 六、启动与验证

```bash
systemctl status sglang-qwen38.service
tail -f /root/sglang-qwen38-fp8.log
```

看到这两行说明起来了（第二行的数字必须等于你配的上下文长度）：

```
max_total_num_tokens=262144, ... context_len=262144, max_running_requests=3, available_gpu_mem=1.22 GB
The server is fired up and ready to roll!
```

验证三件事：

```bash
KEY='<你的APIKEY>'

# 1) 模型与上下文
curl -s http://127.0.0.1:8778/v1/models -H 'Authorization: Bearer <APIKEY>'

# 2) 不带 key 必须 401
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8778/v1/models

# 3) 普通对话（创作类关思考）
curl -s http://127.0.0.1:8778/v1/chat/completions -H 'Authorization: Bearer <APIKEY>' -H 'Content-Type: application/json' -d '{model:Qwen3.8-27B,messages:[{role:user,content:用一句话介绍你自己}],max_tokens:200,chat_template_kwargs:{enable_thinking:false}}'
```

启动耗时参考：加载权重约 30 秒，加 CUDA graph 捕获，从命令到 ready 约 1 分钟。
首次启动更慢（要编译 kernel），日志会有进度，耐心等。

## 七、让局域网和公网能访问（WSL2 特有的一步）

WSL2 的网络是 NAT 的，Windows 上访问不到 WSL 的端口，必须做端口转发。用管理员 PowerShell：

```powershell
# 先取 WSL 的 IP
wsl -d Ubuntu-24.04 -u root -- hostname -I

# 把 Windows 的 8778 转发到 WSL 里的 8779（那是 keep-alive 代理）
netsh interface portproxy add v4tov4 listenport=8778 listenaddress=0.0.0.0 connectport=8779 connectaddress=<上一步拿到的WSL_IP>
```

注意：**WSL 的 IP 每次重启都会变**，所以重启后要重新执行这条命令（可以先 delete 再 add）。
这就是为什么要套 keep-alive 代理：对外始终是 Windows 的 8778，不用改客户端。

再放行防火墙（管理员 PowerShell，给 TCP 8778 加入站放行规则）。

公网用 frp，和路线 A 完全一样，只是 `localPort = 8778`：

```toml
serverAddr = '<你的frp服务器IP>'
serverPort = 7000
auth.method = 'token'
auth.token = '<你的token>'

[[proxies]]
name = 'sglang-qwen38'
type = 'tcp'
localIP = '127.0.0.1'
localPort = 8778
remotePort = 30000
```

## 八、运维要点

**日志该看哪几行**：

| 关键字 | 含义 |
|---|---|
| `max_total_num_tokens=... context_len=...` | 启动后 KV 池与上下文长度，必须和配置一致 |
| `The server is fired up and ready to roll!` | 服务就绪 |
| `hit_device` / `hit_host` | 前缀缓存命中（device=显存，host=内存），越高越好 |
| `gen throughput` | 出字速度 |
| `MAMBA-AUDIT nodes=... states=...` | 线性注意力状态槽的心跳，并发跑久了看它有没有涨爆 |
| `OOM` / `KernelOOM` / `CUDA out of memory` | 显存不够，按第四节的顺序降参数 |

**三条容易被忽略的纪律**：

1. **切模型前先停掉另一个服务**。48G 卡上同时存在两个栈必定 OOM。
2. **创作类任务关思考**：请求体加 `chat_template_kwargs`（见第六节第 3 条），
   否则 3000 token 的预算会被思考吃光，正文是空的。
3. **`--language-model-only` 绝对不要加**：那是关掉视觉的开关，加了就没有图片能力。
