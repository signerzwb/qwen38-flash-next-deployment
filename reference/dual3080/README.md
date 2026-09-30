# 双 RTX 3080 · Qwen3.8-27B int4 · sglang 部署建议（压缩包索引）

本压缩包是把「单卡 4090 + FP8」的 sglang 经验，外推到「双 3080 + int4」的部署建议。**先读 `双3080-int4部署建议.md`。**

## 目录

| 路径 | 内容 |
|---|---|
| `双3080-int4部署建议.md` | **核心方案**（差异/三步走/工程坑/预期/待确认） |
| `sglang-栈-原封不动/` | 桌面 stack 包原样：源码补丁 0001~0012、启动/停止脚本、conf、监控面板、完整技术文档 `sglang-qwen38-stack-guide.md` |
| `双卡int4推荐配置/` | 针对双 3080 + int4 的参数：`sglang_qwen38_27b_int4_double.conf.sh` + 启动脚本 |
| `测速脚本/` | 验证用 Python 脚本：`vcheck.py`（检查模型/聊天）、`bm8780.py`（流式测速/含 TTFT）、`chk.py`、`coldp.py`、`coldreal.py`、`bm27.py`、`bm.py` |

## 快速上手（对，跳到这）

1. 先打开 `双3080-int4部署建议.md`，看完「差异」和「三步走」。
2. `sglang-栈-原封不动/` → 按里面 `README.md` 装 sglang 源码 + 补丁。
3. `双卡int4推荐配置/` → 把 `sglang_qwen38_27b_int4_double.conf.sh` 里的 `MODEL_PATH`、`CUDA_VISIBLE_DEVICES` 改成你机器实际的。
4. 启动：`bash start-sglang-qwen38-int4-double.sh`（后台加 nohup + 日志）。
5. 验证：`python 测速脚本/vcheck.py`，然后 `python 测速脚本/bm8780.py 8778`。
6. 看日志里 `hit_device`（前缀缓存命中）和 `gen throughput`（出字速度）。

> 这台机器上可能有三个值来自 4090：`TP_SIZE`、`CONTEXT_LENGTH`、`ENABLE_DFLASH` 都是**从保守档起步**，稳了再逐项加大，别一上来全套照搬。