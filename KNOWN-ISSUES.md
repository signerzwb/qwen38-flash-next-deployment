# 复现前需要核实的问题

这份清单来自对 2026-09-30 部署文档包的静态阅读。本次没有运行包内脚本或开展部署，也没有把以下模板改成已经在新机器通过验证的配置。

## 1. 双卡参数与模板不一致

`04` / `05` 要求双卡使用 `TP_SIZE=2`，但 `config/sglang-qwen38-v23-single48g.conf.sh` 写死了 `CUDA_VISIBLE_DEVICES="0"` 和 `--tp-size 1`，没有读取 `TP_SIZE`。部署双卡前必须实际修改配置生成的启动参数，不能只增加一个变量。

这个配置同样写死 KV 精度与 DFlash 参数，没有实现文档中的 `KV_CACHE_DTYPE` / `ENABLE_DFLASH` 开关。需要在脚本中真正接入变量或调整参数数组。

## 2. SGLang profile 与 systemd 安装位置不明确

默认主服务启动 NVFP4 profile，其默认上下文为 380000、显存比例 0.95、HiCache 24。FP8 launcher 的默认值另设为 262144 / 0.93 / HiCache 0。

`config/fp8.conf` 是 systemd drop-in，不能只复制到 `/root/` 就期望它生效。应放到主服务的 drop-in 目录（如 `/etc/systemd/system/sglang-qwen38.service.d/fp8.conf`），或者为 FP8 建独立服务。启动前核对实际 `ExecStart`，避免错误使用 NVFP4 默认档。

## 3. 补丁校验命令读错对象

`git apply` 修改工作区后，`git rev-parse 'HEAD^{tree}'` 仍然读取基线提交的树，不会包含新补丁。

核对合并补丁的目标树时可在独立源码 checkout 中 `git apply --index <补丁>`，随后用 `git write-tree` 比较目标树。若已用 `git apply` 修改工作区，应先确认只有补丁改动，再将相关改动加入索引后比较。不要在含有无关改动的仓库中一概 `git add -A`。原文目标树为 `53bbd2f6d78d9021a1e0e7021a263fea3081b507`，以后还须核实它与实际补丁、基线一致。

## 4. Windows 路径与视觉文件位置冲突

路线 A 文档采用 `D:\Strata` / `D:\Strata-data`，但 `config/strata-iq3_s.json` 使用 `C:\`。

文档将视觉 `mmproj` 放在 `Strata-data/models/`；JSON 指向 `Strata-data/models/IQ3_S/`。实际部署时统一所有路径，并核对当前 Strata 安装程序生成的目录。JSON 中记录的是单卡 GPU 0；双卡选择也需要依据当前 Strata 版本重新生成和确认。

`02` 中关于 IQ2_XS 在 64GB 内存下被压回 128K 的一句话，与 `01` / `05` 给出的 IQ3 系列 90GB 门槛表述冲突。应根据所使用版本的源码和启动日志核实，而非混用两段结论。

## 5. keep-alive 代理文件缺失

`sglang-keepalive.service` 引用了 `/root/sse-keepalive-proxy.py`，本包和两个嵌套参考包都没有这个文件。需要从已确认来源补齐，或先采用不依赖它的入口配置，不能直接启用这个单元。

HTTP 空闲连接 keep-alive 与 SSE 流中的心跳是两件事；代理是否必要及相关超时行为应按实际客户端和服务版本验证。

## 6. 接口示例与验收脚本

- `02` / `03` 的部分 `curl -d` 请求使用未加双引号的属性名与字符串，不是合法 JSON。建议先保存合法 JSON 到请求文件，再提交。
- `scripts/verify.py` 没有将模型 ID、上下文、无 Key 校验和答案 FAIL 全部转换成非零退出码；不能只看脚本退出成功就判断验收通过。
- 流式计时起点位于 `urlopen` 返回之后，TTFT 少计了建立连接及收到 HTTP 响应头前的耗时。以后测速应在请求发出之前起表，并以第一个非空内容 token 为终点。
- 长文长度通过随机单词近似生成，不是精确 tokenizer 计数；实际 `prompt_tokens` 应记录。第一次请求也不保证真正冷缓存，应先核实缓存状态。
- `ab_run.py` 保存响应但没有逐题自动判分，也没有关思考选项。原文“13 题都过”与部分题正文为空的记录应由原始响应复核。

## 7. 硬件与实测口径

文档中的单卡 48GB 数据是其记录的 RTX 4090D 48GB 测试台，不能当成本机双 3090 或双 3080 的成绩。不同模型、量化、投机解码、功率限制、缓存状态和输出题型的 tok/s 也不能直接比较。

`04` 使用 Strata / IQ2_XS 的速度量级推测 SGLang / FP8 表现，没有同配置测试支撑。`05` 的 HiCache 关闭配置也不能直接用于复刻先前选择的 HiCache 12GB/rank 方案。

关于 KV 每 token 占用、262K 增量和“能够装下”的预算只是原文估算，应按确切模型配置、KV 层数、精度、TP 分片方式和实际分配日志重新计算。双卡显存需要分片才能使用，并不等同于一张大显存卡。

## 8. 依赖与下载

原包给出了源码基线、补丁和一组版本，但没有完整依赖锁文件和引擎校验清单。以后部署时先确认上游仓库与模型存在、所选 release 可下载、驱动与 CUDA 相容，再锁定版本并记录文件校验值。不要把文档中的 `pip install -e .` 当作完整依赖复刻保证。

## 9. 配置密钥

包中原有的固定密钥已替换为 `REPLACE_WITH_YOUR_API_KEY`，包括嵌套 ZIP 内的文件。模板仍可能以该占位符为默认值；启用服务之前必须设置自己的密钥，不要使用共享占位符作为有效鉴权凭据。
