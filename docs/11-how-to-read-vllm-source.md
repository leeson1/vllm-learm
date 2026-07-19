# 11｜如何读 vLLM v0.25.0 源码：追踪一条请求

> 版本基线：本机 `/Users/leeson/codes/vllm` 的 v0.25.0 tag，commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。源码行号会随 commit 变化，因此笔记必须同时记录 commit、文件和 symbol。

## 1. 先确定阅读场景

第一次不要同时追多卡、MoE、spec decode、KV connector 和异步执行。建议使用：

```text
decoder-only generation
单 GPU，TP=1，DP=1
Model Runner V1
同步调度
无 speculative decoding
本地 KV Cache
```

启动基线：

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --no-async-scheduling \
  --enforce-eager \
  --generation-config vllm
```

这不是 v0.25.0 默认性能配置，而是减少并发时间线、Runner 和 CUDA Graph 分支的教学配置。主链路读通后必须再对照默认 V2/async/O2 路径。

## 2. 要回答的八个问题

每打开一个文件，只围绕：

1. HTTP 请求在哪里变成模型输入？
2. chat template 和 tokenization 在哪里发生？
3. API 进程怎样把请求送到 Engine Core？
4. waiting/running 状态保存在哪里？
5. 一轮怎样分配 token budget 与 KV blocks？
6. Worker 怎样得到 input IDs、positions、block table？
7. 新 K/V 怎样写入、历史 K/V 怎样读取？
8. token 怎样 detokenize、stop、stream 并最终释放资源？

如果一段代码不能帮助回答这些问题，第一次可以跳过。

先用这张图建立进程和回程方向：

```text
进程 A：API Server

  HTTP / Chat Request
          │
          ▼
  Router → Serving → AsyncLLM/InputProcessor
          │ EngineCoreRequest
          │ ZMQ ADD
══════════╪══════════════════════════════════════════════
          ▼
进程 B：Engine Core

  input socket → Request → Scheduler → Executor
                                      │
                                      ▼
                          GPU Worker / Model Runner
                                      │
                         model output / token IDs
                                      ▼
  Scheduler update ← EngineCoreOutputs
          │
          │ ZMQ output
══════════╪══════════════════════════════════════════════
          ▼
进程 A：OutputProcessor → detokenize/stop → JSON 或 SSE

注：单 GPU UniProcExecutor 下，Worker 可与 Engine Core 同进程；
    多卡 mp 通常会再启动独立 GPU Worker 进程。
```

## 3. v0.25.0 在线 Chat 请求主链路

### 3.1 HTTP 路由

```text
vllm/entrypoints/openai/chat_completion/api_router.py
  -> create_chat_completion()
```

这里接收 OpenAI Chat Completions 请求，取得 serving handler，并返回 JSON 或 SSE `StreamingResponse`。不要从巨大的 `api_server.py` 开始搜索所有路由；v0.25.0 已把各 API 拆到各自的 router/serving 模块。

### 3.2 Chat 渲染和参数转换

```text
vllm/entrypoints/openai/chat_completion/serving.py
  -> _create_chat_completion()
```

关注：

- messages 怎样应用 chat template；
- token IDs 怎样形成；
- OpenAI 字段怎样转成 `SamplingParams`；
- request ID 怎样生成；
- `engine_client.generate()` 怎样被调用。

### 3.3 AsyncLLM 与 InputProcessor

```text
vllm/v1/engine/async_llm.py
  -> generate()
  -> add_request()

vllm/v1/engine/input_processor.py
  -> process_inputs()
```

`InputProcessor` 产生可发送给 Engine Core 的 `EngineCoreRequest`。`AsyncLLM.generate()` 同时维护逐请求输出收集器；客户端取消或生成器结束时还要走 abort/清理路径。

### 3.4 API 进程到 Engine Core

```text
vllm/v1/engine/core_client.py
  -> add_request_async()

vllm/v1/engine/core.py
  -> process_input_sockets()
```

在线多进程拓扑中，请求通过 ZMQ 消息进入 Engine Core。Core 将 `EngineCoreRequest` 转成内部 `Request`，再加入 Scheduler。

### 3.5 Scheduler

```text
vllm/v1/core/sched/scheduler.py
  -> add_request()
  -> schedule()
  -> update_from_output()
  -> finish_requests()
```

第一次只跟这些状态：

```text
request.status
request.num_tokens
request.num_tokens_with_spec
request.num_computed_tokens
waiting / running
token_budget
num_scheduled_tokens
req_to_new_blocks
```

理解 `num_tokens_with_spec - num_computed_tokens` 后，prefill、decode 和 chunked prefill 会落到同一模型里。

### 3.6 KV Cache Manager

```text
vllm/v1/core/kv_cache_manager.py
  -> get_computed_blocks()
  -> allocate_slots()
  -> cache_blocks()
  -> free()
```

再下钻：

```text
block_pool.py
single_type_kv_cache_manager.py
kv_cache_utils.py
```

观察 hash、ref count、free queue、touch 和 eviction，不要套用旧 Block Manager/COW 叙述。

### 3.7 Engine Core 和 Executor

同步基线：

```text
vllm/v1/engine/core.py:EngineCore.step()

schedule
  -> model_executor.execute_model
  -> scheduler.update_from_output
```

默认 async scheduling 则看 `step_with_batch_queue()`：它让多个 batch in flight，但单个 batch 的逻辑依赖没有改变。

### 3.8 Worker 与 Model Runner

```text
vllm/v1/worker/gpu_worker.py
vllm/v1/worker/gpu_model_runner.py       # V1
vllm/v1/worker/gpu/model_runner.py       # V2
```

同步教学基线先看 V1 的 input preparation、`BlockTable`、slot mapping 和 sampler。之后去掉 `VLLM_USE_V2_MODEL_RUNNER=0`，根据日志确认默认是否选择 V2，再比较 V2 persistent batch/staged writes。

### 3.9 Attention 与输出回程

```text
vllm/v1/attention/backend.py
vllm/v1/attention/backends/<selected_backend>.py

vllm/v1/engine/output_processor.py
vllm/v1/engine/async_llm.py:_run_output_handler()
```

把写路径 `slot_mapping`、读路径 `block table` 与输出侧 detokenize/stop 分开追踪。stop string 在 API/OutputProcessor 侧结束时，Engine Core 仍必须收到 abort/finish 才能释放 KV ownership。

## 4. 推荐阅读顺序

```text
1. chat_completion/api_router.py
2. chat_completion/serving.py
3. async_llm.py + input_processor.py
4. core_client.py + core.py
5. scheduler.py
6. kv_cache_manager.py + block_pool.py
7. gpu_worker.py + 一个 Model Runner
8. block_table.py + 当前 attention backend
9. output_processor.py
10. 默认 V2/async 路径差异
```

不要一开始从 model definitions、所有 engine args 或单个 CUDA kernel 开始。

## 5. 用运行轨迹代替静态抄代码

### 实验 A：一个请求、三个输出 token

```text
prompt: "hello"
temperature: 0
max_tokens: 3
stream: true
```

每轮记录：

- request status；
- `num_tokens` 与 `num_computed_tokens`；
- scheduled token 数；
- block IDs；
- sampler token；
- 是否 finished/free。

### 实验 B：两个请求 continuous batching

```text
A: input 128, output 128
B: input 4096, output 16，稍晚到达
```

观察 B 进入后，A decode 与 B prefill chunk 如何共享 token budget。

### 实验 C：默认路径对照

移除教学参数，记录：

- V1 还是 V2 Model Runner；
- async scheduling 是否启用；
- attention backend；
- optimization/CUDA Graph 模式；
- Nsight 时间线上 CPU/GPU 是否重叠。

## 6. 日志应该怎样加？

只在小模型、小请求、同步基线中加结构化日志：

```text
step_id
request_id
status_before/after
num_tokens / num_computed_tokens
num_scheduled_tokens
new_block_ids
sampled_token_ids
finish_reason
```

不要记录完整 prompt，也不要在正式性能压测时保留逐 token Python 日志。日志改变 CPU 开销和时序，调试结果不能直接当性能结果。

## 7. 源码笔记模板

```text
版本/commit：
运行配置：V1/V2、sync/async、backend、TP/DP
入口 symbol：
输入：
持有的状态：
输出：
上游/下游：
失败/清理路径：
本次观察事实：
源码事实：
仍待验证：
```

区分“运行观察”和“由源码推断”能避免把某个模型/配置的现象写成框架不变量。

## 8. 验收标准

最终交付：

1. 一张 HTTP → Scheduler → GPU → SSE 流程图；
2. 一份单请求逐 step trace；
3. 一份双请求混排 trace；
4. V1 sync 与默认 V2/async 的差异表；
5. 一个取消请求后 KV 被释放的反向链路说明。

如果只能背类名，仍不算读通；能够用 request state 和 token/KV ownership 解释每一步，才算完成。

## 参考资料

- [vLLM v0.25.0 Architecture Overview](https://docs.vllm.ai/en/v0.25.0/design/arch_overview/)
- [vLLM v0.25.0 Model Runner V2 Design](https://docs.vllm.ai/en/v0.25.0/design/model_runner_v2/)
- [vLLM v0.25.0 Prefix Caching Design](https://docs.vllm.ai/en/v0.25.0/design/prefix_caching/)
- [vLLM v0.25.0 CUDA Graphs](https://docs.vllm.ai/en/v0.25.0/design/cuda_graphs/)
