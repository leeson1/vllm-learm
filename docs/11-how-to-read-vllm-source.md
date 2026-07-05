# 11｜如何读 vLLM 源码：从请求生命周期切进去

第三阶段开始，目标从“理解模块”转向“形成岗位能力”。

读 vLLM 源码不要一上来就全局搜索 CUDA kernel，也不要从模型结构开始。更建议从一条请求生命周期切进去：

```text
HTTP 请求
  -> Engine 接收
  -> Scheduler 调度
  -> KV Cache 分配
  -> Model Runner 执行 forward
  -> Sampler 生成 token
  -> Output Processor 返回流式结果
```

这条链路读通，比零散看文件更有效。

## 1. 先明确读源码的目标

读源码不是为了记住每个类名，而是为了回答几个工程问题：

1. 请求从哪里进入？
2. 请求状态在哪里保存？
3. 一轮调度如何决定跑哪些请求？
4. KV Cache block 如何分配和释放？
5. forward 前准备了哪些 tensor 和元数据？
6. logits 如何变成 next token？
7. 流式输出如何返回给客户端？
8. 发生 OOM、抢占、取消请求时如何处理？

只要你能沿着这些问题读源码，就不会迷路。

## 2. 不建议的读法

### 2.1 从 CUDA kernel 开始

不建议一上来读：

```text
csrc/
attention kernel
paged attention kernel
custom op
```

因为你还不知道这些 kernel 的输入从哪里来，也不知道 block table、slot mapping、seq lens 的业务含义。

先读 kernel 容易陷入：

```text
每一行都像懂了，但不知道整体干什么。
```

### 2.2 从所有配置参数开始

vLLM 参数非常多。如果一开始就背参数，会很碎。

更好的方式是先按资源分类：

```text
显存：gpu_memory_utilization、max_model_len、kv_cache_dtype
调度：max_num_batched_tokens、max_num_seqs、chunked prefill
并行：tensor_parallel_size、pipeline_parallel_size、data_parallel_size
优化：prefix caching、speculative decoding、quantization
```

参数要和模块一起看。

### 2.3 从模型结构开始

模型结构当然重要，但 vLLM 的核心竞争力不是“实现了 Llama”，而是：

```text
如何把很多请求高效调度到 GPU 上服务。
```

所以源码阅读优先级应该是 serving runtime，而不是模型定义本身。

## 3. 推荐阅读顺序

推荐按下面顺序：

```text
1. OpenAI-compatible API 入口
2. Engine / request 生命周期
3. Scheduler
4. KV Cache Manager / Block Manager
5. Model Runner
6. Sampler / Output Processor
7. Attention Backend
8. 分布式、多卡、量化、spec decode
```

这和前两阶段文章顺序一致。

## 4. 第一步：找到请求入口

先看 OpenAI-compatible server。

你要搞清楚：

```text
/v1/chat/completions
/v1/completions
```

这些请求进入服务后，会被转换成什么内部对象。

重点关注：

1. HTTP 请求参数如何解析。
2. SamplingParams 如何构造。
3. prompt 如何 tokenize。
4. 请求如何提交给 engine。
5. streaming response 如何逐步返回。

不要纠结 FastAPI 细节，重点是 API 层如何进入推理引擎。

## 5. 第二步：读 Engine

Engine 是连接 API 层和执行层的核心。

你要找的问题：

```text
请求如何 add？
每一轮 step 在哪里触发？
输出如何被取走？
请求取消如何处理？
```

可以把 Engine 理解为主循环管理器：

```text
while has_requests:
    scheduler.schedule()
    executor.execute_model()
    process_outputs()
```

真实实现会更复杂，尤其 V1 多进程架构下，Engine 和 Worker 之间会有通信。但你先抓住这个抽象就够了。

## 6. 第三步：读 Scheduler

读 Scheduler 时只问 5 个问题：

1. waiting 队列在哪里？
2. running 队列在哪里？
3. token budget 如何计算和扣减？
4. prefill/decode 如何混排？
5. KV block 不够时如何处理？

建议边读边画状态机：

```text
WAITING -> RUNNING -> FINISHED
             |
             v
          PREEMPTED
```

不要试图一次看懂所有功能分支。先把普通文本生成请求跑通，再看 prefix cache、spec decode、LoRA、多模态。

## 7. 第四步：读 KV Cache Manager

读 KV Cache 时，重点不是 tensor 形状，而是资源生命周期。

你要追踪：

```text
初始化时分了多少 block？
请求 prefill 时申请了多少 block？
decode 追加 token 时什么时候需要新 block？
请求结束时在哪里释放？
共享 block 的 ref count 如何维护？
```

可以自己写一个小表：

| 请求 | 逻辑 block | 物理 block | ref count |
|---|---|---|---|
| A | 0 | 101 | 1 |
| A | 1 | 88 | 1 |
| B | 0 | 101 | 2 |

这样你会更容易理解 prefix caching 和 Copy-on-Write。

## 8. 第五步：读 Model Runner

Model Runner 的核心问题是：

```text
调度结果如何变成模型输入？
```

重点看：

1. input_ids 如何拼接。
2. positions 如何生成。
3. block_tables 如何传入。
4. slot_mapping 如何生成。
5. seq_lens 如何传给 attention backend。
6. logits 如何进入 sampler。

读的时候建议画一张数据流图：

```text
SchedulerOutput
  -> InputBatch
  -> ModelInput
  -> forward
  -> logits
  -> SamplerOutput
```

## 9. 第六步：读 Sampler 和输出处理

很多人会忽略 Sampler，但线上服务里它很重要。

因为不同请求可能有不同采样参数：

```text
temperature
top_p
top_k
presence_penalty
frequency_penalty
max_tokens
stop tokens
```

你要看：

1. logits 如何按请求切分。
2. 每个请求如何应用自己的 SamplingParams。
3. stop 条件在哪里判断。
4. streaming token 如何返回。
5. detokenize 在哪里做。

## 10. 第七步：最后再看 Attention Backend / CUDA

读到这里，你再看 attention backend，会清楚很多。

你已经知道：

```text
block table 是什么
slot mapping 是什么
seq_lens 是什么
prefill 和 decode 为什么不同
```

再看 kernel 时，就不是盲读。

你可以重点关注：

1. prefill attention 是否走 FlashAttention 类 backend。
2. decode attention 如何读取 paged KV cache。
3. block table 如何传入 kernel。
4. CUDA Graph 捕获和 replay 的边界在哪里。

## 11. 推荐使用的调试方法

### 11.1 加日志

不要只靠静态阅读。

可以在这些地方加日志：

```text
add request
schedule start/end
allocated blocks
execute_model start/end
sampler output
request finished
```

日志不要太细，否则高并发下会影响性能。调试时只跑小模型、小并发。

### 11.2 单请求跑通

先只跑一个请求：

```text
prompt: hello
max_tokens: 5
stream: true
```

把每一轮 step 打出来。

你会看到：

```text
prefill once
then decode token by token
```

### 11.3 双请求观察 continuous batching

再跑两个请求：

```text
A: 短 prompt，短输出
B: 长 prompt，长输出
```

观察它们如何进入 waiting/running，如何混排。

### 11.4 长 prompt 观察 chunked prefill

构造一个长 prompt，看 prefill 是否被拆成 chunk。

重点观察：

```text
每轮 token budget 如何分配
长 prefill 是否阻塞 decode
```

## 12. 建议做的源码阅读笔记

每读一个模块，写 4 个部分：

```text
模块职责：它负责什么？
核心数据结构：它维护哪些状态？
输入输出：上游给它什么，它给下游什么？
关键问题：如果出 bug，可能表现为什么？
```

例如 Scheduler：

```text
职责：选择下一轮执行请求
数据结构：waiting/running、token budget、scheduled outputs
输入：新请求、运行中请求、资源状态
输出：本轮要执行的 batch
问题：TTFT 高、decode 卡顿、抢占频繁、GPU 利用率低
```

## 13. 最小源码阅读任务

如果你想形成简历项目，可以做一个最小任务：

```text
目标：解释一次请求从 HTTP 到 token streaming 的完整链路。

产物：
1. 一张流程图
2. 一篇源码阅读笔记
3. 一个带日志的本地 demo
4. 一组不同参数下的现象对比
```

这个任务比“我看过 vLLM 源码”有说服力得多。

## 14. 本文小结

读 vLLM 源码，最重要的是顺序。

推荐主线：

```text
请求入口
  -> Engine
  -> Scheduler
  -> KV Cache Manager
  -> Model Runner
  -> Sampler / Output
  -> Attention Backend / CUDA
```

不要从最难的 kernel 开始，也不要死记配置参数。

读源码的目标是建立一张运行时地图：

```text
请求状态怎么流动，GPU 资源怎么分配，KV Cache 怎么管理，输出 token 怎么返回。
```

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM 架构设计文档：https://docs.vllm.ai/en/latest/design/architecture.html
- vLLM GitHub：https://github.com/vllm-project/vllm
