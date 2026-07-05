# 04｜服务端视角：vLLM V1 架构与请求生命周期

这篇从后端服务视角看 vLLM V1 架构。

vLLM 不是简单的“一个 Python 函数调用模型”。在线服务时，它更像一个多进程推理服务系统。

## 1. 两类入口

vLLM 主要有两类入口：

```text
离线推理入口：LLM class
在线服务入口：vllm serve
```

### 1.1 LLM class

`LLM` 类适合离线推理。代码直接调用：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["hello"], SamplingParams(max_tokens=32))
```

它适合：

- 本地实验。
- 批处理。
- benchmark。
- 学习引擎行为。

### 1.2 vLLM serve

在线服务入口是：

```bash
vllm serve <model>
```

它会启动 HTTP server，对外提供 OpenAI-compatible API。

适合：

- 业务系统接入。
- Web/API 服务。
- 流式输出。
- 多用户并发。

## 2. V1 多进程架构

vLLM V1 使用多进程架构，把不同职责拆开。

核心进程包括：

```text
API Server Process
Engine Core Process
GPU Worker Process
DP Coordinator Process（使用 data parallel 时）
```

可以先用这个图理解：

```text
Client
  |
  v
API Server Process
  |
  |  ZMQ / internal RPC
  v
Engine Core Process
  |
  |  dispatch execution
  v
GPU Worker Process(es)
  |
  v
GPU model execution
```

## 3. API Server Process

API Server Process 负责和外部世界打交道。

职责：

- 接收 HTTP 请求。
- 兼容 OpenAI API 协议。
- 做输入处理，例如 tokenization、多模态数据加载。
- 把请求送到 Engine Core。
- 把生成结果流式返回给客户端。

后端视角下，它类似：

```text
网关 + 协议适配层 + 输入预处理 + streaming response writer
```

注意：API Server 本身不直接执行模型 forward。真正调度和执行在后面的 Engine Core 和 GPU Worker。

## 4. Engine Core Process

Engine Core 是 vLLM 的调度核心。

它负责：

- 管理请求状态。
- 运行 scheduler。
- 管理 KV Cache。
- 决定每一轮哪些请求进入 batch。
- 协调 GPU Worker 执行。

可以把它理解成：

```text
推理服务的大脑
```

它关心的问题包括：

```text
当前有哪些 waiting/running 请求？
哪些请求处于 prefill？哪些处于 decode？
这一轮 token budget 怎么分？
KV Cache block 是否够？
请求是否需要抢占？
哪些请求已经完成，可以释放资源？
```

这和游戏服务器 tick 调度很像，只是这里调度的是 GPU forward 和 KV Cache block。

## 5. GPU Worker Process

GPU Worker 负责实际模型执行。

职责：

- 持有模型权重。
- 准备输入 tensor。
- 执行 forward。
- 调用 attention backend / CUDA kernel。
- 返回 logits 或采样结果。

如果使用 tensor parallel，可能会有多个 GPU Worker 协同执行同一个模型。

例如：

```bash
vllm serve <model> --tensor-parallel-size 4
```

典型单机 4 卡 tensor parallel 可以理解为：

```text
1 API Server
1 Engine Core
4 GPU Workers
```

## 6. Data Parallel 下的进程数量

如果使用 data parallel：

```bash
vllm serve <model> --tensor-parallel-size 2 --data-parallel-size 4
```

可以理解为有 4 组 engine core，每组内部用 2 张 GPU 做 tensor parallel。

整体会出现：

```text
多个 API Server
多个 Engine Core
多个 GPU Worker
可能还有 DP Coordinator
```

这时候要重点关注 CPU 资源，因为进程和线程数会上升。

## 7. 一次请求的生命周期

下面用在线服务为例。

### 7.1 接收请求

客户端请求：

```text
POST /v1/chat/completions
```

API Server 收到请求，解析 JSON，处理 messages、sampling params、stream 参数等。

### 7.2 输入处理

API Server 做：

```text
chat template
  -> tokenizer
  -> prompt token ids
```

如果是多模态模型，还可能加载图片、音频、视频等输入。

### 7.3 进入等待队列

请求被送到 Engine Core，进入 waiting/running 队列。

此时请求还没有真正占用完整的 decode 资源。

### 7.4 Prefill

Prefill 阶段会把 prompt token 全部处理一遍，生成第一批 KV Cache。

如果 prompt 很长，prefill 会很重，TTFT 主要受它影响。

```text
prompt tokens -> model forward -> KV Cache -> first token/logits
```

### 7.5 Decode

Decode 阶段通常一次生成一个或少量 token。

每一轮 decode 都会：

```text
读取历史 KV Cache
计算当前 token
采样下一个 token
追加新的 KV Cache
返回 token 给客户端
```

### 7.6 Streaming 输出

如果客户端开启 streaming，API Server 会持续把 token 以 SSE/chunk 形式返回。

后端要注意：streaming 请求生命周期更长，会持续占用连接和推理资源。

### 7.7 请求结束和资源释放

请求达到 stop 条件后：

- 从 running 队列移除。
- 释放 KV Cache block。
- 关闭或结束 stream。
- 返回 final response。

## 8. Prefill 和 Decode 的差异

这是理解 vLLM 调度的关键。

| 阶段 | 输入规模 | 主要影响指标 | 特点 |
|---|---:|---|---|
| Prefill | prompt 长度 | TTFT | 计算密集，长 prompt 很重 |
| Decode | 每轮新 token | TPOT / 吞吐 | 逐 token 生成，强依赖 KV Cache |

TTFT：Time To First Token，首 token 延迟。

TPOT：Time Per Output Token，每个输出 token 时间。

长 prompt 场景通常 prefill 重；长输出场景通常 decode 重。

## 9. Scheduler 为什么复杂？

因为它要同时处理：

- 新请求不断到来。
- 老请求不断生成 token。
- 有些请求 prompt 很长。
- 有些请求输出很长。
- KV Cache block 有限。
- GPU batch token budget 有限。
- streaming 要及时返回。

一个简化版调度循环可以理解为：

```python
while True:
    finished = collect_finished_requests()
    free_kv_blocks(finished)

    new_requests = recv_new_requests()
    waiting_queue.push(new_requests)

    batch = scheduler.select(
        waiting_queue=waiting,
        running_queue=running,
        kv_cache_budget=free_blocks,
        token_budget=max_num_batched_tokens,
    )

    outputs = gpu_workers.execute(batch)
    update_request_states(outputs)
    stream_tokens_to_clients(outputs)
```

真实 vLLM 更复杂，但主线就是：

```text
队列 -> 调度 -> 执行 -> 更新状态 -> 输出 -> 释放资源
```

## 10. 看源码的建议顺序

不要从全部文件开始扫。

建议按职责边界读：

```text
entrypoints：请求从哪里进入
engine/core：调度循环在哪里
scheduler：一轮 batch 如何选择
kv_cache / block manager：KV block 如何分配释放
worker / model_runner：如何准备输入并执行 forward
attention backend：底层 attention 如何读 KV Cache
```

读的时候始终带着一个问题：

```text
一个请求从 HTTP 进来，到 token 流式返回，中间经过哪些对象？
```

## 11. 本文小结

vLLM V1 架构可以从三个核心职责理解：

```text
API Server：处理协议和输入输出
Engine Core：调度请求和管理 KV Cache
GPU Worker：执行模型 forward
```

一次请求的关键阶段是：

```text
HTTP request
  -> tokenize
  -> waiting queue
  -> prefill
  -> decode loop
  -> stream token
  -> finish and free KV cache
```

理解这条链路后，再看 Scheduler、Block Manager、Model Runner、Attention Backend 就不会迷路。

## 参考资料

- vLLM Architecture Overview：https://docs.vllm.ai/en/latest/design/arch_overview/
- vLLM OpenAI-Compatible Server：https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- vLLM GitHub：https://github.com/vllm-project/vllm
