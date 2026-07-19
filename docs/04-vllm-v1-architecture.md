# 04｜服务端视角：vLLM V1 架构与请求生命周期

> 版本基线：vLLM v0.25.0。本文讨论的是 **V1 Engine 架构**；它与 Worker 内部可选择的 Model Runner V1/V2 不是同一个“V1/V2”。

读完这一篇，目标不是记住所有类名，而是能回答：

```text
请求在哪个进程被处理？谁维护请求和 KV 账本？谁持有 GPU tensors？
一个 token 怎样从 HTTP 请求走到模型，再回到 SSE？
```

## 1. 两类入口不是同一种前端拓扑

### 1.1 离线 `LLM`

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["hello"], SamplingParams(max_tokens=32))
```

`LLM` 是同一 Python 应用内的离线入口。V1 `LLMEngine` 创建 InputProcessor、OutputProcessor 和 EngineCoreClient；默认离线路径可以直接在本进程驱动 Engine Core，而不需要 HTTP/ZMQ 前端隔离。

### 1.2 在线 `vllm serve`

```bash
vllm serve <model>
```

在线入口创建 API Server 和 `AsyncLLM`。`AsyncLLM` 通过异步多进程 client 启动或连接 Engine Core，前端与 Core 之间使用 ZMQ 传递可序列化的请求和输出。

所以“离线和在线共用引擎能力”不等于“二者的进程图完全相同”。

## 2. 先分清职责，再讨论进程

核心职责可以画成：

```text
Client
  |
  v
API Server / Renderer / AsyncLLM
  |  EngineCoreRequest, ZMQ
  v
Engine Core: Scheduler + request state + KV block bookkeeping
  |  SchedulerOutput
  v
Executor -> Worker -> Model Runner -> Attention Backend -> GPU
  |
  v
EngineCoreOutputs -> OutputProcessor -> JSON / SSE
```

### 2.1 API Server / AsyncLLM

负责：

- 接收 OpenAI-compatible HTTP 请求；
- 渲染 chat template、tokenization 和多模态输入处理；
- 把输入转换成 `EngineCoreRequest`；
- 为每个请求建立输出 collector；
- detokenize、检查 stop string，返回 JSON 或 SSE。

模型 forward 不在 API Server 中执行。

### 2.2 Engine Core

负责：

- 保存内部 `Request`；
- 运行 Scheduler；
- 维护 waiting/running 队列；
- 维护请求到 KV block IDs 的账本；
- 调用 Executor 执行一次调度结果；
- 用模型输出更新请求状态并生成 `EngineCoreOutputs`。

要特别区分：Engine Core 管的是 KV block 的逻辑分配；真正的 GPU KV tensors 在 Worker 一侧。

### 2.3 Executor

Executor 抽象“怎样调用一个或多个 Worker”：

- `UniProcExecutor`：单 world-size 默认路径，Worker 在 Engine Core 进程内。
- `MultiprocExecutor`：本地多进程执行，为 GPU ranks 创建 Worker 子进程。
- Ray / external launcher：用于相应的分布式场景。

Scheduler 不需要把每个 backend 的 IPC 细节写进自身逻辑。

### 2.4 Worker 与 Model Runner

Worker 负责设备初始化、分布式环境、模型加载、显存 profiling、KV Cache tensors 和模型执行。Model Runner 负责把 `SchedulerOutput` 合并到设备侧 batch，准备 token IDs、positions、block table/slot mapping、attention metadata 并执行 forward/sample。

v0.25.0 会根据模型与功能兼容性选择 Model Runner V2，或回退到 Model Runner V1；也可用 `VLLM_USE_V2_MODEL_RUNNER` 显式控制。这个选择发生在 Worker 内部，不改变 API → Engine Core → Scheduler 的主干。

## 3. “一个 Worker 一块 GPU”为什么不是完整进程结论？

官方 Architecture Overview 用“每 GPU 一个 Worker process”描述多 GPU部署，这是理解职责和容量的好近似；但看 v0.25.0 源码时还要加上 Executor 细节。

### 3.1 在线单 GPU，TP=1、DP=1

`ParallelConfig` 在 `world_size == 1` 时默认选择 `uni`，`UniProcExecutor` 直接构造 Worker wrapper。因此常见主干是：

```text
1 API Server process
1 Engine Core process（内部同时运行单 GPU Worker）
```

不要再额外虚构一个必然存在的 GPU Worker OS 进程。实际部署还可能有 tokenizer/media 等辅助线程或进程，不在这张核心图中。

### 3.2 单机 TP=4

默认本地多 GPU backend 是 `mp`：

```text
1 API Server
1 Engine Core
4 GPU Worker subprocesses
```

这时官方给出的总数是 6 个核心进程。

### 3.3 TP=2、DP=4

每个 DP rank 有自己的 Engine Core 和一个含 2 个 TP ranks 的模型副本。v0.25.0 官方默认 API Server 数会随 DP size 扩展，并增加 DP Coordinator：

```text
4 API Servers
4 Engine Cores
8 GPU Workers
1 DP Coordinator
= 17 个核心进程
```

如果显式设置 API Server count、使用 external load balancer、Ray 或跨节点配置，拓扑会变化。进程数应从实际参数推导，不能把某一张示意图当作固定常数。

## 4. 在线请求的完整正向链路

下面以 `/v1/chat/completions` 为例。

### 4.1 协议与输入渲染

API Server 校验请求；Renderer 应用 chat template，把 messages 变成模型输入。`AsyncLLM.add_request()` 再调用 `InputProcessor.process_inputs()`：

- 校验模型输入和 sampling params；
- 得到 prompt token IDs；
- 处理 EOS、generation config、LoRA/多模态等元数据；
- 构造 `EngineCoreRequest`。

### 4.2 前端登记输出状态

`AsyncLLM` 在当前 API 进程的 `OutputProcessor` 中登记请求，并创建逐请求 `RequestOutputCollector`。之后通过 `add_request_async()` 把 ADD 消息发往 Core。

这样输出回程时可以按 request ID 放回正确的异步生成器。

### 4.3 ZMQ 进入 Engine Core

Engine Core 的输入线程反序列化消息，`preprocess_add_request()` 把它变成内部 `Request`，并准备 prefix hashes。主循环收到 ADD 后调用 Scheduler `add_request()`，请求进入 `waiting`。

此时尚未按最大上下文为请求预留完整 KV Cache。

### 4.4 Scheduler 组成一轮工作

Scheduler 先处理运行中请求，再尝试准入等待请求。它受这些约束：

```text
token budget            <= max_num_batched_tokens / max_num_scheduled_tokens
scheduled sequences     <= max_num_seqs
sequence length         <= max_model_len
allocated KV blocks     <= 当前可分配容量
其他约束                = LoRA、多模态、结构化输出、KV connector 等
```

新请求会先查 Automatic Prefix Caching 命中，再调用 `KVCacheManager.allocate_slots()`。如果容量不够，新请求继续等待；运行中请求也可能被 preempt 并在后面 recompute。

### 4.5 Executor / Worker 执行

`SchedulerOutput` 包含新/续跑请求、每个请求本轮 token 数、block IDs、完成/抢占的 request IDs 等。Executor 把它交给 Worker；Model Runner 更新 batch 和 block table，准备输入并执行模型。

Worker 返回采样 token IDs 等 `ModelRunnerOutput`，不负责把它们变成最终 HTTP 文本。

### 4.6 Scheduler 更新状态

`Scheduler.update_from_output()`：

- 把采样 token 加到请求；
- 检查长度/EOS 等模型侧停止条件；
- 更新 `num_computed_tokens` 和请求状态；
- 释放已完成请求的 KV blocks；
- 生成面向前端的 `EngineCoreOutputs`。

### 4.7 输出回程

`AsyncLLM._run_output_handler()` 从 Engine Core 拉取输出，`OutputProcessor` detokenize、处理 stop strings、生成 `RequestOutput` 并推入逐请求 collector。OpenAI serving 层再编码成 JSON 或 SSE chunk。

如果客户端断开，`AsyncLLM.generate()` 被取消后会 abort 请求，前端和 Core 都要清理状态。

## 5. V1 没有两个永久隔离的“prefill/decode 引擎”

初学时说“请求先 prefill、再 decode”是正确的计算阶段描述；但不要据此想象 Scheduler 中必然存在两个完全独立的执行引擎。

V1 请求的关键量是：

```text
num_computed_tokens：已经经过模型计算的 token 数
当前 token 数：prompt + 已接受输出 + 可能的 speculative tokens
```

- 差距很大：表现为 prefill，可被 chunk。
- 差距通常为 1：表现为标准 decode 一步。
- speculative decoding：差距可以大于 1。

统一的 token 预算让同一轮可以混排 decode 和 chunked prefill。v0.25.0 在支持时默认启用 chunked prefill，并优先安排 decode，再用剩余 budget 调度 prefill。

## 6. 同步逻辑与默认异步调度

为了理解主线，可以先看同步 `EngineCore.step()`：

```text
Scheduler.schedule()
  -> Executor.execute_model()
  -> future.result()
  -> Scheduler.update_from_output()
```

但 v0.25.0 在配置兼容时默认启用 async scheduling。此时 `step_with_batch_queue()` 会让多个 batch in flight，把 CPU scheduling 和 GPU execution 跨批次重叠。

所以源码图中“schedule → execute → update”表示一个 batch 的逻辑顺序，不代表默认运行时必须在单个串行函数调用内完成全部阶段。调试同步基线时可显式使用 `--no-async-scheduling`。

## 7. Prefill、Decode 与输出时序

| 阶段 | 本轮通常计算什么 | 主要读取/写入 | 直接相关指标 |
|---|---|---|---|
| Prefill | prompt 中未缓存的 token，可 chunk | 写入 prompt KV | TTFT、prefill tokens/s |
| Decode | 标准路径中每序列一个待计算 token | 读历史 KV，写新 token KV | ITL/TPOT、output tokens/s |
| Output | detokenize、stop 检查、编码响应 | CPU 请求状态 | E2E、streaming 开销 |

刚采样出的 token 已经可以返回文本，但它本身的 KV 尚未写入；下一轮将其作为输入后才写入 KV 并采样后续 token。

## 8. 反向清理链路同样重要

请求可能因 EOS、长度、stop string、客户端取消或错误结束。清理涉及两边：

```text
API/AsyncLLM：关闭 collector、停止 streaming、移除 OutputProcessor 状态
Engine Core：从队列/registry 移除请求、释放 KV block 所有权、通知 Worker 清理状态
```

只看正向生成而忽略 abort，是理解在线资源泄漏和断连问题的常见盲点。

## 9. 建议的源码阅读顺序

按一次在线请求追踪：

1. `vllm/entrypoints/openai/api_server.py`：server 怎样创建与启动。
2. `vllm/v1/engine/async_llm.py`：`generate()`、`add_request()`、输出 handler。
3. `vllm/v1/engine/input_processor.py`：`EngineCoreRequest` 怎样形成。
4. `vllm/v1/engine/core_client.py`：ZMQ ADD/ABORT/输出通信。
5. `vllm/v1/engine/core.py`：输入线程、busy loop、同步/异步 step。
6. `vllm/v1/core/sched/scheduler.py`：队列、预算、prefill/decode 统一调度。
7. `vllm/v1/core/kv_cache_manager.py`：prefix hit 和 slots。
8. `vllm/v1/executor/`：uni/mp/Ray 怎样调用 Worker。
9. `vllm/v1/worker/gpu_worker.py`：设备初始化和 Model Runner 选择。
10. `vllm/v1/worker/gpu/model_runner.py` 或 `gpu_model_runner.py`：V2/V1 runner。

## 10. 本篇自检题

1. 在线单 GPU 默认为什么不一定有独立 GPU Worker OS 进程？
2. API Server 和 Engine Core 各自保存哪一部分请求状态？
3. 为什么 prefill/decode 可以由同一个 token 差值模型统一调度？
4. 默认 async scheduling 下，“schedule → execute → update”为什么仍然成立？
5. stop string 在前端触发时，为什么还要向 Engine Core abort？
6. V1 Engine 与 Model Runner V1/V2 有什么区别？

## 参考资料

- [vLLM v0.25.0 Architecture Overview](https://docs.vllm.ai/en/v0.25.0/design/arch_overview/)
- [vLLM v0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/)
- [vLLM v0.25.0 OpenAI-Compatible Server](https://docs.vllm.ai/en/v0.25.0/serving/online_serving/openai_compatible_server/)
- [vLLM v0.25.0 Model Runner V2 设计](https://docs.vllm.ai/en/v0.25.0/design/model_runner_v2/)
