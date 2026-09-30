# 08 排错 FAQ

按「症状 → 原因 → 处理」写，都是我们实际踩过的。

## 启动阶段

**1. 启动直接 OOM / 提示显存不足**

- 原因 A：别的程序占着显存（另一个模型服务、绘图工具、游戏）。引擎**只在启动那一刻探测一次空闲显存**，
  脏启动会让缓存池少一大截。
  处理：先确认 `nvidia-smi` 里显存是空的，再启动。切换模型时也要先把上一个服务停干净。
- 原因 B：参数太激进。处理：按 `03` 文档第四节的顺序降 —— 先降 `CONTEXT_LENGTH`，
  再降 `MEM_FRACTION_STATIC`，再降 mamba 槽，最后关投机。

**2. 加载很久，以为卡死了**

正常。Strata 首次要把 50GB 专家权重读进内存并锁页，日志会打印 `[strata] still starting (Ns)`，
1-3 分钟都算正常；sglang 首次要编译 kernel，也会慢。**耐心等，不要反复重启。**

**3. sglang 启动报错，说找不到某个参数或 kernel**

补丁没打全。回到 `03` 文档第三节，确认：

```bash
cd /root/sglang-source-v23 && git rev-parse 'HEAD^{tree}'
# 官方包打完之后应该是 53bbd2f6d78d9021a1e0e7021a263fea3081b507
```

注意**不要**用 `git am` 逐个打 `by-commit/`，那批提交里有 merge commit，线性重放必然失败，要用合并 diff。

**4. 运行中（不是启动时）突然 CUDA OOM**

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 没设。这些模型有临时工作区分配，
不开这个开关会碎片化，跑一段时间后 OOM。

## 接口与网络

**5. 客户端报 `stream disconnected before completion` 或 `error sending request`**

uvicorn 默认 5 秒断开空闲长连接，客户端复用连接池时就报这个。两处一起改：

- 服务端设 `SGLANG_TIMEOUT_KEEP_ALIVE=300`
- 前面套一层 SSE keep-alive 代理（本包 `config/sglang-keepalive.service`，
  监听 8779 转发到 8778，Windows 侧再把 8778 转发到 8779）

**6. WSL2 重启后，局域网/公网突然不通了**

WSL 的 IP 每次重启都会变，而 Windows 上的端口转发记录还是旧 IP。
重新执行一次端口转发（先 delete 再 add），或者干脆用 keep-alive 代理统一入口。

**7. 局域网连不上**

- Windows 防火墙没放行端口（用管理员 PowerShell 加入站规则）
- 当前网络被设成「公用网络」，改成「专用网络」
- 客户端写的是 `127.0.0.1`，跨机器要用内网 IP

**8. 不带 key 也能调**

那说明 key 没生效。sglang 要显式传 `--api-key`，Strata 默认要求 key。
部署完必须做一次「不带 key 返回 401」的检查。

## 使用阶段

**9. 问它写小说，结果什么都不返回 / 一直转圈**

思考把 max_tokens 吃光了。请求体加：

```json
{chat_template_kwargs: {enable_thinking: false}}
```

或者把 max_tokens 提到 6000 以上。数学、多跳这种需要推理的题再把思考打开。

**10. 输出格式不对（多了换行 / 多了前后缀）**

27B FP8 的正文常带两个前导换行，程序化解析前先 `strip()`。
需要严格格式时优先用 Strata，或者把提示词写成「只输出…不要任何其他内容」并用关思考模式。

**11. 长对话用着用着就卡**

上下文涨到 200K 以上后速度会明显下降。客户端侧把自动压缩阈值调小（例如 185K 触发压缩），
不要让单会话一路涨到 250K。

**12. 图片识别不了**

- 启动参数里**不要**加 `--language-model-only`（那是关视觉的开关）
- Strata 侧确认 `mmproj-Qwen3.8-Flash-Next-BF16.gguf` 放在了 `Strata-data/models/`（不是 `models/IQ3_S/`）
- sglang 侧确认 conf 里有 `--enable-multimodal`

**13. 改了模型名，客户端连不上**

服务端改 `--served-model-name` 后，客户端请求体里的 `model` 字段也要改成一样的名字。

**14. 两个服务都想开**

不行。48G 卡上两套栈的显存加起来会爆。切换顺序：先停一个（确认 `nvidia-smi` 显存归零），再起另一个。

## 一句话应急清单

1. 慢 → 看是不是在思考（关思考）。
2. 空 → 看 finish_reason 是不是 length（加 max_tokens）。
3. 崩 → 看 `nvidia-smi` 显存和日志里的 OOM 行（降上下文/降显存比例）。
4. 连不上 → 看防火墙、端口转发、key。
