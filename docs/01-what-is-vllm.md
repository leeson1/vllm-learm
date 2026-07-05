# 01｜vLLM 是什么：从 LLM 推理服务的瓶颈说起

## 1. 先给结论

vLLM 是一个面向大语言模型推理和在线服务的高吞吐推理引擎。

它解决的核心问题不是“怎么训练模型”，而是：

```text
模型已经训练好了，权重也有了，如何用有限 GPU 显存支撑更多并发请求，并且尽量降低延迟、提高吞吐？
```

如果从后端工程视角看，vLLM 更像是一个“LLM 推理服务运行时”：

- 对外提供 Python 离线推理接口和 OpenAI-compatible HTTP API。
- 对内负责请求调度、batch 合并、KV Cache 管理、GPU worker 执行、流式输出。
- 在 GPU 显存、吞吐、延迟之间做工程权衡。

官方对 vLLM 的定位是：高吞吐、内存高效的 LLM 推理与服务引擎。它的核心技术之一是 PagedAttention，用类似操作系统分页的方式管理 Transformer 推理过程中的 KV Cache。

## 2. LLM 推理和普通后端请求有什么不同？

普通后端请求一般是：

```text
request -> 业务逻辑 -> DB/Redis/RPC -> response
```

LLM 推理请求更像是：

```text
request prompt
  -> prefill：把 prompt 全部算一遍，生成第一批 KV Cache
  -> decode：一次生成一个 token，每生成一个 token 都要复用历史 KV Cache
  -> stream response
```

它有几个典型特点：

### 2.1 请求耗时长

一次聊天请求可能持续几百毫秒到几十秒。服务端不是算一次就结束，而是不断生成 token。

### 2.2 请求长度差异大

有的用户 prompt 只有几十 token，有的有几千、几万 token；有的输出几十 token，有的输出几千 token。

这会导致 batch 内部非常不均匀：短请求结束了，长请求还在跑。

### 2.3 显存瓶颈不只来自模型权重

很多人第一反应是：模型大，所以显存主要被权重吃掉。

但在线推理时，另一个关键显存消耗是 KV Cache。并发越高、上下文越长、输出越长，KV Cache 越大。

可以粗略理解为：

```text
总显存 ≈ 模型权重 + KV Cache + 临时激活/工作区 + 框架开销
```

模型权重相对固定，KV Cache 则随着在线请求动态增长和释放。vLLM 的核心价值正是把这部分动态内存管理好。

## 3. 为什么不能简单地“一个请求一个 batch”？

GPU 擅长大规模并行。如果每个请求单独执行，GPU 利用率会很低。

所以推理服务一般会把多个请求合成 batch：

```text
request A
request B   -> batch -> GPU forward
request C
```

问题是，LLM decode 阶段是逐 token 生成的。每个请求的剩余长度都不同，传统静态 batch 很容易产生浪费：

- A 已经生成完了，B/C 还没结束。
- 新请求 D 来了，但当前 batch 还没结束。
- 某些请求很长，占住显存和 batch slot。

因此现代 LLM serving 会使用 continuous batching，也就是持续地把新请求插入执行队列，把结束请求移出去，让 GPU 尽可能保持高利用率。

vLLM 在工程上围绕这个目标做了大量工作：调度、KV Cache block 管理、请求状态维护、流式输出等。

## 4. vLLM 的几个关键词

### 4.1 PagedAttention

PagedAttention 是 vLLM 最经典的设计。它把每个请求的 KV Cache 拆成固定大小的 block，然后用 block table 建立逻辑块到物理块的映射。

这和操作系统虚拟内存很像：逻辑上连续，不要求物理上连续。

好处是：

- 减少显存碎片。
- 按需分配 KV Cache。
- 支持不同请求之间共享前缀 KV Cache。
- 更容易支持 beam search、parallel sampling 这类共享历史上下文的解码方式。

### 4.2 KV Cache

Transformer 生成 token 时，需要关注历史 token。为了避免每次都重复计算历史 token 的 Key/Value，推理引擎会缓存它们，这就是 KV Cache。

KV Cache 是推理服务里的“核心资产”。它决定了：

- 能同时服务多少请求。
- 最大上下文能开多大。
- 长文本场景下 TTFT 和吞吐表现。

### 4.3 Scheduler

Scheduler 决定这一轮 GPU forward 要跑哪些请求、每个请求跑多少 token、是否要抢占、是否要等待 KV block。

后端开发可以把它类比成：

```text
游戏服务器里的 tick 调度器 + 资源分配器 + 队列管理器
```

只是它调度的资源不是玩家行为，而是 GPU 算力、KV Cache block、batch token budget。

### 4.4 OpenAI-Compatible Server

vLLM 可以启动一个兼容 OpenAI API 形式的服务。这样业务系统可以像调用 OpenAI 一样调用本地或私有部署的模型。

常见启动方式：

```bash
vllm serve <model>
```

然后通过 `/v1/chat/completions`、`/v1/completions` 等接口访问。

## 5. 后端开发应该怎么理解 vLLM？

不要只把 vLLM 当成“Python AI 框架”。它更像一个高性能服务端系统。

可以按下面几层理解：

```text
API 层：HTTP / OpenAI-compatible / streaming
调度层：请求队列、continuous batching、prefill/decode 调度
内存层：KV Cache、block table、prefix cache
执行层：worker、model runner、attention backend、CUDA kernel
部署层：多卡、多机、监控、压测、限流、故障恢复
```

这对 C++/Go 后端开发是有迁移价值的：

- 调度思想：类似游戏服 tick、任务队列、协程调度。
- 内存思想：类似对象池、slab allocator、分页管理。
- 并发思想：类似多线程 worker、生产消费队列、流式响应。
- 性能思想：吞吐、尾延迟、资源利用率之间的权衡。

## 6. 第一阶段不要急着学什么？

不建议一开始就钻这些内容：

- CUDA kernel 细节。
- FlashAttention 数学推导。
- Tensor Parallel 通信实现。
- MoE 专用优化。
- 复杂量化算法。

更合理的顺序是：

```text
跑通服务
  -> 理解一次请求的生命周期
  -> 理解 prefill/decode
  -> 理解 KV Cache
  -> 理解 PagedAttention
  -> 理解 scheduler
  -> 再看 CUDA / 多卡 / kernel
```

## 7. 本文小结

vLLM 的本质是一个 LLM 推理服务引擎。它的核心价值不是“让模型更聪明”，而是“让模型服务得更高效”。

它重点解决三个问题：

1. GPU 怎么更忙：continuous batching、调度优化。
2. 显存怎么更省：PagedAttention、KV Cache block 管理。
3. 服务怎么更好接入：OpenAI-compatible API、流式输出、部署工具。

对想转 AI 推理/HPC 的后端开发来说，vLLM 是非常合适的切入点：它既有系统工程，又有 GPU 计算，还能逐步接触 CUDA、内存管理、分布式推理。

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM GitHub：https://github.com/vllm-project/vllm
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
