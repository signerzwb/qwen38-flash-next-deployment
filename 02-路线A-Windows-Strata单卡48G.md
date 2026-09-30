# 02 路线 A：Windows + Strata + Qwen3.8-Flash-Next（单卡 48G）

目标机器：Windows 10/11 + 单卡 48G（或双 3090 合计 48G）+ 128G 内存 + NVMe。
做完的结果：本机 8080 端口有一个 OpenAI 兼容接口，局域网和公网都能访问，262K 上下文，带视觉。

---

## 一、前置检查（做之前先跑这三条）

```powershell
nvidia-smi                                    # 驱动版本要 >= 580，显卡能看到，显存是空的
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory   # 内存 >= 64GB，推荐 128GB
Get-PSDrive C | Select-Object Used,Free        # C 盘或目标盘至少空出 150GB
```

关键：**开始前确认显存是干净的**。别的程序（另一个模型服务、Stable Diffusion、游戏）占着显存时启动，
引擎只在启动那一刻探测空闲显存，脏启动会让缓存池少一大截，甚至直接 OOM。

## 二、拿到项目

```powershell
cd D:\
git clone https://github.com/Niko1221/Strata.git
# 或者直接下压缩包解压：
# https://github.com/Niko1221/Strata/archive/refs/heads/main.zip
```

装好后的目录约定（很重要，模型和程序是分开的）：

```
D:\Strata\            <- 程序本体（升级时整个替换，不影响模型）
D:\Strata-data\       <- 模型与缓存，必须在同一个父目录下
    models\IQ3_S\     <- 权重文件
    packs\mtp\        <- 引擎运行时生成的打包文件
```

## 三、准备模型文件（国内网络看这段）

官方 `START-HERE.bat` 会直接从 huggingface.co 下载，国内常常很慢。推荐**手动用镜像下好再启动**，
安装脚本检测到文件已存在就会跳过下载。

需要的三个文件（仓库 `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`）：

| 文件 | 大小 | 放到哪 |
|---|---:|---|
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` | 51.0 GB | `D:\Strata-data\models\IQ3_S\` |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` | 26.8 GB | 同上 |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf`（视觉编码器） | 0.85 GB | `D:\Strata-data\models\`（注意：上一层） |

用镜像下载（把 `huggingface.co` 换成 `hf-mirror.com` 即可）：

```powershell
$base = 'https://hf-mirror.com/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main'
New-Item -ItemType Directory -Force -Path 'D:\Strata-data\models\IQ3_S' | Out-Null
curl.exe -L -C - -o 'D:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf' $base/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
curl.exe -L -C - -o 'D:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf' $base/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
curl.exe -L -C - -o 'D:\Strata-data\models\mmproj-Qwen3.8-Flash-Next-BF16.gguf' $base/mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

`-C -` 是断点续传，断了就重新跑同一条命令。文件大建议用 aria2：

```
aria2c -c -x 16 -s 16 -k 1M --dir=D:\Strata-data\models\IQ3_S -o <文件名> <同样的 URL>
```

内存只有 64GB 的话，把 IQ3_S 换成 IQ2_XS，上下文上限会被自动压到 128K。

## 四、安装与首次启动

双击 `START-HERE.bat`，回答四个问题：

1. 模型与大小：**Qwen3.8-Flash-Next + IQ3_S**（文件已手动放好时会被直接识别）
2. 上下文：**262144**
3. 图片（视觉）：**开**
4. 实验性速度投影：**关**

之后它会自己建 Python 环境、下载预编译引擎、打包模型，然后启动服务。
日志里出现这一行就是好了：

```
ready: http://127.0.0.1:8080/v1  (context 262144 tokens, images on, API key required)
```

**首次启动 1-3 分钟电脑会非常卡**，因为要把 50GB 专家权重读进内存并锁页，属正常现象，不要关窗口。
以后再启动，直接双击 `START-HERE.bat`，不会再下载任何东西。

## 五、后台常驻启动（不依赖 bat 窗口）

bat 窗口关掉服务就停了。要让它后台常驻，用下面的命令拉起来；把它包成 .ps1，
再用「任务计划程序」在登录时触发，就实现开机自启：

```powershell
Start-Process -FilePath 'D:\Strata\.venv\Scripts\python.exe' `
  -ArgumentList 'D:\Strata\serve\server.py','--engine','strata','--config','D:\Strata\strata-iq3_s.json','--port','8080' `
  -WorkingDirectory 'D:\Strata' -WindowStyle Hidden `
  -RedirectStandardOutput 'D:\Strata\strata-serve.log' `
  -RedirectStandardError  'D:\Strata\strata-serve.err.log'
```

要停下来，在任务管理器里找到 `python.exe`、`strata.exe`、`strata-vision.exe` 这三个程序并结束它们即可；
重启电脑当然也可以。

## 六、验证

```powershell
# 1) 模型在线 + 上下文长度（期望 model id 正确、n_ctx = 262144）
curl.exe -s http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer <APIKEY>'

# 2) 不带 key 必须被拒（期望 401）

# 3) 普通对话（创作类记得关思考）
curl.exe -s http://127.0.0.1:8080/v1/chat/completions -H 'Authorization: Bearer <APIKEY>' -H 'Content-Type: application/json' -d '{model:qwen3.8-flash-next-iq3_s,messages:[{role:user,content:用一句话介绍你自己}],max_tokens:200,chat_template_kwargs:{enable_thinking:false}}'
```

更完整的验收（长文找数字、测 TTFT 与出字速度）用本包 `scripts/ab_run.py`，口径见 `07` 号文档。

## 七、必须知道的两个运行参数

**1. 关思考（对创作类任务收益最大）**

请求体里加这一项：

```json
{chat_template_kwargs: {enable_thinking: false}}
```

写 600 字小说，同一台机器：关思考 8.2s 出稿；不关思考 26.4s 还在数字数、正文是空的。
需要推理的数学题、多跳问答再打开。

**2. 长对话别让它把上下文吃满**

262K 是上限不是舒适区，超过 128K 后速度会下降。客户端侧把自动压缩阈值调小，
例如涨到 185K 就触发压缩，避免单会话跑到 250K 再出问题。

## 八、局域网与公网

局域网：直接访问 `http://<本机内网IP>:8080/v1`。连不上通常是 Windows 防火墙拦了 ——
用管理员身份的 PowerShell 给 TCP 8080 加一条入站放行规则（Strata 的启动日志末尾会把整条命令直接打印出来，
照着复制即可），并确认当前网络在 Windows 里被设成「专用网络」。

公网：用 frp 把本机 8080 映射到有公网 IP 的服务器上。frpc.toml 示例：

```toml
serverAddr = '<你的frp服务器IP>'
serverPort = 7000
auth.method = 'token'
auth.token = '<你的token>'

[[proxies]]
name = 'strata-8080'
type = 'tcp'
localIP = '127.0.0.1'
localPort = 8080
remotePort = 30000
```

之后客户端用 `http://<frp服务器IP>:30000/v1`。**公网必须带 API key。**

## 九、日志与排错入口

| 现象 | 看哪里 |
|---|---|
| 起不来 / 显存不够 | `strata-serve.log` 里的 `avail mem` 行，启动时可用显存要 ≥ 40GB |
| 加载很久 | `[strata] still starting (Ns)` 属正常，第一次要 1-3 分钟 |
| 回答卡住不返回 | 先看是不是思考太长把 max_tokens 烧完了（把 max_tokens 提到 6000+ 或关思考） |
| 长文变慢 | 上下文超过 128K 之后的正常退化 |

更多见 `08-排错-FAQ.md`。
