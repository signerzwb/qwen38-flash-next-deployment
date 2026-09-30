# 04：Linux 与双 GPU

## Linux 安装入口

原生 Ubuntu / Linux 使用同一 Strata 项目和 GGUF、MTP 数据。Windows 引擎不能复制过去直接运行；需要 Linux 引擎或在 Linux 编译。

```bash
git clone https://github.com/Niko1221/Strata.git "$HOME/Strata"
cd "$HOME/Strata"
git checkout d6708a4aae15b4860000d54c8af9e84d684bce09
chmod +x setup.sh
./setup.sh --setup --family qwen --model IQ3_S --context 131072 --kv int8 --vision gpu --gpus 0,1 --data-dir "$HOME/Strata-data" --port 8080 --host 127.0.0.1 --experimental-speed-projection off
```

单卡用 `--gpu 0`。选择其他量化时修改 `--model`，手动下载文件时加入对应 `--gguf-dir`。仅安装不启动时添加 `--no-start`。

`setup.sh` 会创建 `.venv`；缺少 Python/venv 时尝试通过发行版包管理器安装。核对时最新 release 没有 Linux 预编译资产，安装器可能提示准备 C++ 编译工具和 CUDA Toolkit；应阅读日志中的具体缺失项。`--build` 可明确要求源码编译。

## 双卡参数

```text
--gpus 0,1
--layer-split auto
```

编号使用 `nvidia-smi` 的编号。也可用 `--gpus all`，但只有两张目标卡时显式写 `0,1` 更清楚。安装器会记住选择。

双卡配置 JSON 对应字段：

```json
{
  "gpu": [0, 1],
  "layer_split": "auto"
}
```

这是展示两个字段的片段，不是完整服务配置。完整配置由安装器生成。

Strata 按连续层区间分配到每张卡，自动分割根据空闲显存选择位置。日志应明确出现 layer split 和两张卡；“检测到两张卡”并不等于服务已经同时使用它们。

这一方式不要求 NVLink，也不能照其他引擎的 tensor-parallel 参数来设置。双卡速度需要按实际硬件、量化、上下文与输出类型测试。

## KV streaming 与 WSL2

核对版本中，Windows / 原生 Linux 在上下文至少 65536、内存足够且 KV 类型适配时，安装器会考虑加入 `--kv-resident 32768`，把完整 KV 放在主机内存，GPU 常驻一部分。

**WSL2 在这份源码中会关闭这一 KV streaming 路径**，理由是驱动的主机页锁定限制。不能把 Windows / 原生 Linux 的同一份 `--kv-resident` 配置直接当作 WSL2 已验证方案。

`k8v4` KV 也不使用这条 streaming 路径。让安装器按环境生成，再核对实际参数和日志。

## 启动与停止

准备完成后用生成的 `run-iq3_s.sh` 或 `./setup.sh` 启动。前台服务在对应终端按 Ctrl+C 停止，再核对 GPU 进程与显存释放情况。需要 systemd 托管时，应以实际用户目录、Python 环境和配置为基础另建服务，不复制 Windows 路径。
