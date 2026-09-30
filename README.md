# Qwen3.8 Flash-Next 部署资料备份

本仓库保存 `Qwen3.8 flash next-部署文档包-20260930.zip` 中的部署说明、配置、验收脚本和参考补丁，方便以后从 GitHub 获取并继续尝试部署。

**本次仅整理上传资料，没有安装环境、下载模型、启动服务或使用 GPU。这里的配置是待核实模板，不是本次已经部署通过的结果。**

## 获取资料

```bash
git clone https://github.com/signerzwb/qwen38-flash-next-deployment.git
cd qwen38-flash-next-deployment
```

也可以使用 GitHub 的 **Code → Download ZIP**。需要保持原部署包结构时，下载 [去除密钥后的完整部署包](downloads/qwen38-deployment-package-20260930-sanitized.zip)。

## 从哪里开始

| 内容 | 文件 |
|---|---|
| 原包总览 | [00：先读我](00-README-先读我.md) |
| 硬件和模型选型 | [01：选型与硬件要求](01-选型与硬件要求.md) |
| Windows、Strata、Flash-Next | [02：路线 A](02-路线A-Windows-Strata单卡48G.md) |
| Linux / WSL2、SGLang、27B | [03：路线 B](03-路线B-Linux或WSL2-sglang-27B.md) |
| 双 RTX 3090 | [04：双 3090](04-双3090%2848G%29方案.md) |
| 双 RTX 3080 20GB | [05：双 3080](05-双3080%2840G%29方案.md) |
| 模型与国内镜像 | [06：下载](06-模型下载与国内镜像.md) |
| 原文记录的成绩及测试口径 | [07：实测报告](07-实测报告-本机数据与验收口径.md) |
| 排错 | [08：FAQ](08-排错-FAQ.md) |
| 复现前需要修正的问题 | [KNOWN-ISSUES.md](KNOWN-ISSUES.md) |
| 配置与测试脚本 | [config](config/) / [scripts](scripts/) |
| SGLang v2.3 原包展开后的补丁和说明 | [reference/sglang-v2.3](reference/sglang-v2.3/) |
| 双 3080 原包展开后的配置和说明 | [reference/dual3080](reference/dual3080/) |

## 两条路线分别需要什么

**部署 Qwen3.8 Flash-Next 本身时，先看路线 A。**路线 B 的主模型是 Qwen3.8-27B，作为另一条路线保留，不要混成同一个模型。

| 项目 | Strata + Flash-Next | SGLang + 27B |
|---|---|---|
| 程序来源 | [Niko1221/Strata](https://github.com/Niko1221/Strata) | [sgl-project/sglang](https://github.com/sgl-project/sglang)，加本包 v2.3 补丁 |
| 主模型来源（原文提供） | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) | [Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)；NVFP4 方案见 06 |
| 辅助模型 | 同仓库的 `mmproj` 视觉文件；MTP / pack 路径按 Strata 的安装流程生成并核对 | [Qwen/Qwen3.8-27B-DFlash2](https://huggingface.co/Qwen/Qwen3.8-27B-DFlash2) |
| 环境 | Windows 或 Linux、NVIDIA 驱动、Strata 对应平台引擎及它的 Python 环境 | Ubuntu / WSL2、Python 3.12 venv、PyTorch、FlashInfer、SGLang kernel、补丁源码 |
| 关键参数 | GGUF 量化档、GPU 选择、262144 上下文、int8 KV、`kv-resident`、MTP、视觉 | TP、KV 精度、上下文、显存比例、Mamba 状态槽、DFlash、HiCache |

下载链接和版本来自原文，上传时没有实际下载或验证远端仓库；以后部署前应先核实项目、模型文件及版本仍可取得。模型权重、引擎二进制和完整虚拟环境没有放入本仓库。

## 给以后接手部署的人 / Codex

可直接把下面这段连同本仓库链接交给以后接手的人：

> 请先阅读 README、KNOWN-ISSUES 和路线 A 文档，再根据目标机器的系统、GPU、显存、内存和正在运行的服务制定 Qwen3.8 Flash-Next 部署方案。确认模型仓库、Strata 源码及引擎版本、量化档和上下文限制。按实际目录生成配置，使用自己的 API Key。原包配置存在路径和参数错误，请核实后修正。测试数据是原文记录，不能冒充这台目标机器已经通过的测试。当前目标是部署 Flash-Next；除非我另选路线，不要把它替换成 SGLang 的 27B 模型。开始安装、下载或启动服务须以我当时的部署请求为准。

## 来源和密钥处理

- 保留原文及配置的参考结构，没有把原文数字改成新的实测结果。
- 写死的 API Key 已从顶层文件和嵌套参考包中替换为 `REPLACE_WITH_YOUR_API_KEY`；这只是占位符，后续部署必须改成自己的密钥。
- 原下载 ZIP 保留在本机，未原样上传到公开仓库。
- [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json) 记录来源文件名、原包和脱敏包的 SHA256，以及发生替换的文件位置，不记录密钥值。
- 本仓库仅保管用户提供的资料；涉及的第三方源码、补丁、模型和引擎以各上游项目的许可为准。
