# 05：参数、验收与排错

## 关键参数

| 参数 | 初始选择或含义 |
|---|---|
| `--family qwen` | 原版 Qwen3.8-Flash-Next |
| `--model IQ3_S` / `IQ2_XS` | GGUF 量化档，结合内存和速度选择 |
| `--context 131072` / `262144` | 安装器上下文配置；以实际配置和服务返回值为准 |
| `--kv int8` | KV 精度，与权重量化是不同选项 |
| `--vision gpu` | 图片能力及视觉编码器所在设备 |
| `--gpu 0` / `--gpus 0,1` | 单卡 / 双卡，采用 `nvidia-smi` 编号 |
| `--layer-split auto` | 双卡自动按层分割 |
| `--data-dir` | 数据目录，复用 models、packs、mtp |
| `--gguf-dir` | 已有两个模型分片所在目录 |
| `--experimental-speed-projection off` | 初始关闭实验性投影 |
| `--no-start` | 安装准备完成后不启动；仍然会下载、安装和生成数据 |

核对版本生成的引擎参数默认包含 `--expert-cache auto`、`--prefill auto`、`--spec 4`、`--spec-min-p 0.5` 和 `--mtp <实际路径>`。视觉开启时还包含 `--vision` 和显存预留；KV streaming 条件见 04 文档。

优先保留安装器生成值。没有目标机器实测时不要随意写死专家缓存大小、分层位置或速度指标。

## 接口验收

建议使用本仓库 `scripts/verify.py`。脚本只在你主动执行时请求已经运行的服务，不会安装或启动模型。

先在自己的终端环境设 `STRATA_API_KEY`，然后运行：

```powershell
python scripts/verify.py --endpoint http://127.0.0.1:8080 --model qwen3.8-flash-next-iq3_s --expected-context 131072
```

如果实际部署选了 IQ2_XS，模型名相应为 `qwen3.8-flash-next-iq2_xs`。服务端 `/v1/models` 返回值应与请求匹配。

脚本检查模型名、上下文、无 Key 拒绝以及普通对话，任何失败返回非零状态。默认只做短请求，不自动发满上下文压力测试。

长文验收需要明确添加选项，例如：

```powershell
python scripts/verify.py --endpoint http://127.0.0.1:8080 --model qwen3.8-flash-next-iq3_s --expected-context 262144 --long-words 20000
```

`--long-words` 是生成文档的英文单词数量，不是精确 token 数。记录接口的实际 `prompt_tokens`。第一次请求标为首次请求，不能未经核实就称为冷缓存；重复请求用来观察缓存复用。

从小规模逐步增加文档，满档验收必须实际接近所配 token 长度，并正确回答长文问题。仅在 `/v1/models` 看到 262144 不等于已经完成 256K 实测。

## 合法请求体

保存为 `request.json`，再用 `curl --data-binary @request.json` 提交；不要在 JSON 中省略双引号。

```json
{
  "model": "qwen3.8-flash-next-iq3_s",
  "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
  "max_tokens": 200,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

API 入口是 `/v1/chat/completions`。带图片时还要人工用已知内容的图片核对视觉答案，确认 `mmproj`、视觉程序和视觉配置都已启用；仅短文本验收脚本不能证明视觉通过。

## 测速记录

- TTFT 从发出请求前起表，到第一个非空 `content` 到达；不要用空 role chunk 当第一个输出。
- 长文分别记录首次与重复请求的 TTFT、实际输入输出 token 数及缓存状态。
- 解码速度需要足够长的输出；找一个金额只有几个 token，不适合据此判断稳定 tok/s。
- 同时记 GPU 显存、系统内存、swap/pagefile、模型量化、上下文、MTP、KV streaming、驱动与引擎版本。
- 对目标机器进行真实测试后再报告速度；本次整理没有产生新的测速结果。

## 常见问题

| 现象 | 核对内容 |
|---|---|
| 找不到 GGUF | 量化子目录、两个分片的文件名、`--gguf-dir` 和完整校验 |
| 视觉失败 | `mmproj` 实际位置、JSON 的 `vision` 字段、视觉引擎是否启动 |
| 卡在首次准备 | 下载 / pack / MTP 日志是否有进度；不重复启动同一任务 |
| 下载 MTP 失败 | 原 checkpoint 地址与 HTTP Range；`HF_ENDPOINT` 不保证覆盖硬编码直链 |
| 262144 被降回 131072 | 安装器实际内存判断和上下文门槛 |
| 显存不足 | 已有 GPU 任务、空闲显存、视觉占用、KV 与每张卡固定开销 |
| 系统持续换页或卡顿 | 量化档和内存余量，必要时降低上下文或量化档 |
| 只用了单卡 | JSON 的 `gpu` 是否数组，日志是否输出 layer split |
| 空答案 | `finish_reason`、思考是否耗尽输出预算，按任务关闭思考或增加预算 |
| API 访问失败 | host、端口、模型名、密钥以及实际防火墙规则 |

停止与资源释放方式见 03 / 04 文档，避免误停其他模型或绘图任务。
