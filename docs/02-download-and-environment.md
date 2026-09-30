# 02：下载与环境

## 程序和引擎

- Strata 程序：[Niko1221/Strata](https://github.com/Niko1221/Strata)。
- 源码 ZIP：[下载 main](https://github.com/Niko1221/Strata/archive/refs/heads/main.zip)。要复刻本说明核对的版本，用 `docs/SOURCES.md` 中的固定提交。
- 引擎发布页：[Releases](https://github.com/Niko1221/Strata/releases)。

核对时最新 release 为 `v0.1.29`，其列表仅有 Windows 资产 `strata-windows-x64.zip`。上游 Linux 安装器会查找对应预编译资产，若没有可用构建，会走编译路径。因此 Linux 部署要准备编译工具，不能把 Windows ZIP 当作 Linux 引擎。

通常先获取完整 Strata 程序，再让它的安装器安装适配引擎。单独拿到引擎 ZIP 不代表 Python 服务、pack 工具和数据都齐全。

## 模型文件

原模型 GGUF 仓库：

[ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)

IQ3_S 需要两个 GGUF 分片，以及开启视觉时的 `mmproj`：

```text
IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf
IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf
mmproj-Qwen3.8-Flash-Next-BF16.gguf
```

**远端分片 URL 包含量化子目录 `IQ3_S/`**；原包示例漏了这一级。换量化时同时修改远端目录与文件名。文件大小、SHA256 和当前目录以仓库元数据为准；本次没有实际下载权重或核实所有文件元数据。

Windows 约定：

```text
D:\Strata\                         程序
D:\Strata-data\models\IQ3_S\      两个主分片
D:\Strata-data\models\            mmproj 位于这一层
D:\Strata-data\packs\iq3_s\       安装器生成的 pack
D:\Strata-data\mtp\rt\            安装器准备的 MTP 运行数据
```

## 国内镜像和断点续传

可以先用 HF 镜像手动下载到上述目录，再给安装器传 `--gguf-dir`。以下命令是供未来部署时使用的示例，执行会下载模型：

```powershell
$flashQuant = 'IQ3_S'
$flashBase = 'https://hf-mirror.com/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main'
$flashDst = "D:\Strata-data\models\$flashQuant"
New-Item -ItemType Directory -Force -Path $flashDst | Out-Null
foreach ($flashPart in 1,2) {
    $flashFile = 'Qwen3.8-Flash-Next-GSQ-RCO-{0}-{1:D5}-of-00002.gguf' -f $flashQuant,$flashPart
    curl.exe -fL -C - -o "$flashDst\$flashFile" "$flashBase/$flashQuant/$flashFile"
    if ($LASTEXITCODE -ne 0) { throw "下载未完成：$flashFile，保留文件以便续传" }
}
curl.exe -fL -C - -o 'D:\Strata-data\models\mmproj-Qwen3.8-Flash-Next-BF16.gguf' "$flashBase/mmproj-Qwen3.8-Flash-Next-BF16.gguf"
```

`-C -` 从已有文件长度续传；镜像不可用时再核对官方地址。Linux 可用相同 URL 和 `curl -fL -C -`，路径换成 Linux 数据目录。下载后用仓库提供的 SHA256 核对，不能只凭存在同名文件判断完整。

本仓库没有假定所有安装器网络请求都会读取 `HF_ENDPOINT`。核对源码中的 GGUF 和 MTP 工具使用自己的直链，镜像需要分别核实。

## MTP 与首次准备

GGUF 分片之外，Strata 会获取 MTP 数据、构建 pack，再生成运行配置。核对版本的 `tools/mtp_fetch.py` 从 [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) 的 BF16 checkpoint 读取索引，并通过 HTTP Range 获取需要的张量；不是要求下载整个 BF16 checkpoint。

只把三个 GGUF 文件放好，不代表 MTP 与 pack 已准备完毕。若 MTP 源站不通，需要核对 `tools/mtp_fetch.py` 的源地址及镜像的 Range 支持，或复用经过校验的既有 `mtp/` 和 `packs/`。不要下载已有的大文件，也不要用另一套草稿模型替代这些数据。

## Python 与 CUDA 依赖

Strata 的启动入口创建项目内 `.venv`。核对版本安装器列出的 Python 包包括 `numpy`、`jinja2`、`regex`、`pyyaml`、`tqdm`、`requests`、`cmake`、`ninja`、`pillow`、`psutil`；CUDA 运行库为 `nvidia-cublas==13.0.2.14` 与 `nvidia-cuda-runtime==13.0.96`。

优先交给匹配版本的安装器处理。国内 pip 镜像是否有指定 NVIDIA 包应先核对；缺少包不能静默换成不同 CUDA 大版本。源码编译的 CUDA Toolkit 和预编译引擎所需的运行库也应分别确认。
