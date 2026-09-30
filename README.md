# Qwen3.8 Flash-Next 部署说明

本仓库只保存 **Qwen3.8-Flash-Next + Strata** 的部署资料：模型下载、Windows / Linux 环境、单卡与双卡参数、启动停止、视觉和接口验收。

资料来自 2026-09-30 的部署文档包，并在 2026-10-01 根据 Strata 上游源码重新核对。只更新了文档和示例，没有在本机或远端安装、启动或运行模型。

## 按顺序阅读

| 步骤 | 文档 |
|---|---|
| 选择量化与上下文 | [01：硬件与模型](docs/01-hardware.md) |
| 下载程序、GGUF、视觉文件和 MTP | [02：下载与环境](docs/02-download-and-environment.md) |
| 在 Windows 安装、启动和停止 | [03：Windows 部署](docs/03-windows-deployment.md) |
| Linux 和双 GPU | [04：Linux 与双卡](docs/04-linux-and-multi-gpu.md) |
| 参数、验收及排错 | [05：参数与验收](docs/05-parameters-validation.md) |
| 核对版本和信息来源 | [来源与版本](docs/SOURCES.md) |

可参考 [IQ3_S 配置示例](config/strata-iq3_s.example.json) 和 [接口验收脚本](scripts/verify.py)。配置应优先由上游安装器按目标机器生成，示例中的路径和 API Key 需要自行填写。

## 获取资料

```bash
git clone https://github.com/signerzwb/qwen38-flash-next-deployment.git
```

或使用 GitHub 的 **Code → Download ZIP**。只需本仓库资料的独立压缩包：[Flash-Next 部署资料 ZIP](downloads/qwen38-flash-next-deployment-docs.zip)。

模型权重、运行引擎和虚拟环境不随资料上传；下载入口在 02 文档。

## 给以后接手部署的人 / Codex

> 请按这个仓库部署 Qwen3.8-Flash-Next，使用 Strata 和原模型的 GSQ-RCO GGUF。先读 README 和 docs，检查目标机器的系统、GPU、显存、内存、磁盘以及现有服务，再核实程序版本、模型文件和 MTP 下载来源。按实际路径生成配置，选择量化档、上下文、视觉及单卡/双卡。保留自己的 API Key，验证模型信息、普通对话、视觉和实际上下文，再记录真实速度及资源占用。资料只说明部署方法，不能当作这台目标机器已经部署或通过测试的证据。

本仓库没有有效 API Key。上游程序、模型和引擎的使用许可以各自项目为准。
