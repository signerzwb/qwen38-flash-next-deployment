# 03：Windows 部署

本文中的安装和启动命令供未来实际部署使用；执行 `START-HERE.bat` 会安装所需组件并启动模型，除非明确传 `--no-start`。

## 检查机器

```powershell
nvidia-smi
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
Get-PSDrive D | Select-Object Used,Free
```

确认驱动、两卡编号、空闲显存、系统内存与磁盘。先处理已有模型或绘图服务的资源冲突；不要仅因为看到配置文档就结束正在运行的其他任务。

## 获取程序

```powershell
Set-Location D:\
git clone https://github.com/Niko1221/Strata.git
Set-Location D:\Strata
```

复刻本说明的源码快照时可在这个新 checkout 中执行：

```powershell
git checkout d6708a4aae15b4860000d54c8af9e84d684bce09
```

也可使用新版，但应重新核对参数和 release，并记录所用提交和引擎版本。

## 选择参数并安装

交互方式：双击 `START-HERE.bat`，选择原模型、量化档、上下文、视觉和 GPU。实验性速度投影先关闭。

明确指定单卡、IQ3_S、128K 的 PowerShell 示例：

```powershell
.\START-HERE.bat --setup --family qwen --model IQ3_S --context 131072 --kv int8 --vision gpu --gpu 0 --data-dir D:\Strata-data --port 8080 --host 127.0.0.1 --experimental-speed-projection off
```

双卡将 `--gpu 0` 换成 `--gpus 0,1`。内存与实际分配都允许满档时，把 `--context` 换成 `262144`；内存较小时先考虑 IQ2_XS。

已经手动下载 GGUF 时添加：

```text
--gguf-dir D:\Strata-data\models\IQ3_S
```

只安装并生成配置、暂不启动：在同一条安装命令最后添加 `--no-start`。这个选项仍会创建环境、下载缺失文件和准备模型，不是只读检查。

初次准备包含依赖安装、引擎获取、模型校验、pack 和 MTP 处理，耗时取决于下载与硬盘。不要反复重启已经有进度的准备过程。

## 配置、后续启动和停止

安装器生成的文件名示例：

```text
D:\Strata\strata-iq3_s.json
D:\Strata\run-iq3_s.bat
```

以实际生成的配置和路径为准；本仓库的 `config/strata-iq3_s.example.json` 仅用于对照参数，不应替代首次 pack / MTP 准备。

以后可运行生成的 `run-iq3_s.bat` 或 `START-HERE.bat`。停止前台服务可在对应窗口按 Ctrl+C，或按上游说明关闭该服务窗口；之后用 `nvidia-smi` 确认进程和显存已释放。不要按名字批量结束所有 `python.exe`，其他任务可能也使用 Python。

若需要后台运行，在实际部署完成后再为这个服务配置独立进程管理及停止入口。本说明没有添加开机自启。

## 局域网访问与 API Key

在配置中设 `host: "0.0.0.0"`，并为 `api_key` 填写自己生成的随机密钥。上游也接受 `--host` / `--api-key` 安装参数，但命令行密钥可能被写入终端历史，使用时注意存放方式。

本机访问 `http://127.0.0.1:8080/v1`，其他机器访问 `http://目标机器IP:8080/v1`。若防火墙拦截，按实际网络访问需求设置规则；不要为部署随意修改整机网络配置。

确认带 Key 的模型请求成功、无 Key 请求被拒，然后进行普通对话和视觉验收。公网访问方式另按具体使用需求配置。
