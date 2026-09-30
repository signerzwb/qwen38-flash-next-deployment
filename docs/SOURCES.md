# 信息来源和版本

核对日期：2026-10-01（Asia/Shanghai）。

本说明依据用户提供的 `Qwen3.8 flash next-部署文档包-20260930.zip` 中 Flash-Next 相关内容重新整理；没有将整份原包再次发布到当前目录。

同时通过 GitHub API 只读核对了 Strata 的以下源码与上游文档：

- 仓库：[Niko1221/Strata](https://github.com/Niko1221/Strata)
- 源码快照：[`d6708a4aae15b4860000d54c8af9e84d684bce09`](https://github.com/Niko1221/Strata/tree/d6708a4aae15b4860000d54c8af9e84d684bce09)
- 安装参数、模型路径、依赖和内存判断：[setup.py](https://github.com/Niko1221/Strata/blob/d6708a4aae15b4860000d54c8af9e84d684bce09/setup.py)
- Windows 入口：[START-HERE.bat](https://github.com/Niko1221/Strata/blob/d6708a4aae15b4860000d54c8af9e84d684bce09/START-HERE.bat)
- Linux 入口：[setup.sh](https://github.com/Niko1221/Strata/blob/d6708a4aae15b4860000d54c8af9e84d684bce09/setup.sh)
- 双 GPU：[MULTI_GPU.md](https://github.com/Niko1221/Strata/blob/d6708a4aae15b4860000d54c8af9e84d684bce09/docs/MULTI_GPU.md)
- MTP 的源仓库与 Range 获取方式：[mtp_fetch.py](https://github.com/Niko1221/Strata/blob/d6708a4aae15b4860000d54c8af9e84d684bce09/tools/mtp_fetch.py)
- 核对时最新引擎发布：[v0.1.29](https://github.com/Niko1221/Strata/releases/tag/v0.1.29)，列表仅包含 Windows 资产。

相较原包，修正了：模型直链的量化子目录、程序与视觉文件路径、动态内存门槛、双卡按层配置、Linux 引擎获取方式及 WSL2 KV streaming 区别。

模型仓库与 MTP 来源由核对版本源码提供。本次未实际下载模型；元数据查询受连接限制，具体文件大小与 SHA256 应在未来下载时核对。

没有运行上游安装器、没有下载模型或引擎、没有操作 GPU，也没有生成新的目标机器实测。当前配置与验收脚本只进行了静态检查。
