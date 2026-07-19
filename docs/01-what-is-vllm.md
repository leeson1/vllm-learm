# 01｜vLLM 是什么：从 LLM 推理服务的瓶颈说起

> 版本基线：vLLM v0.25.0。本文先建立系统地图；涉及当前实现的结论，可在末尾列出的源码入口中核对。

## 1. 先给结论

vLLM 是面向大语言模型推理与在线服务的引擎。模型权重已经训练完成后，它负责把请求变成高效的模型执行，并在 GPU 显存、吞吐和延迟之间做资源调度。

从后端工程视角看，它更像“LLM 推理服务运行时”，而不只是一个 Python 模型库：

- 对外提供离线 Python 接口和 OpenAI-compatible HTTP API。
- 对内负责输入处理、continuous batching、KV Cache 管理、模型执行和流式输出。
- 支持张量并行、流水线并行、数据并行等扩展方式。

它不负责训练模型，也不会提高模型本身的知识或推理能力。

## 2. 一次生成请求为什么特殊？

普通后端请求常被抽象成一次执行：

```text
request -> 业务逻辑 -> DB / RPC -> response
```

自回归生成请求则包含两个主要计算阶段：

```text
prompt
  -> prefill：处理输入 token，写入这些 token 的 KV Cache，并得到首个采样位置的 logits
  -> decode：反复处理新 token，读取历史 KV Cache，采样后续 token
  -> output / stream
```

### 2.1 Prefill

Prefill 处理 prompt 中尚未命中的 token。输入 token 多时，它通常能形成较大的矩阵计算；用户感知的 TTFT（Time To First Token）包含排队、输入处理和这部分模型执行等时间。

如果启用了 Automatic Prefix Caching，完全相同且已缓存的完整前缀 block 可以跳过重复计算；未命中的后缀仍要 prefill。

### 2.2 Decode

标准自回归生成通常每轮为每个序列采样一个新 token。下一轮再把这个 token 送入模型、写入它对应的 KV，并采样再下一个 token。

因此一个请求会跨越很多次调度迭代，而不是一次 forward 后立即结束。Decode 的用户体验常用 ITL（Inter-Token Latency）或 TPOT（Time Per Output Token）衡量。

### 2.3 请求之间高度不均匀

不同请求的 prompt 长度、输出长度、到达时间都不同：

- 短请求已经完成时，长请求可能仍在 decode。
- 新请求到达时，已有请求可能只生成了一部分。
- 长 prompt 会占用更多 prefill token budget。
- 活跃 token 越多，KV Cache 通常占用越高。

这使固定不变的静态 batch 很难持续高效。

## 3. 三类核心瓶颈

### 3.1 模型权重与执行工作区

模型权重通常是显存中的大头之一，并且在服务运行期间相对固定。模型执行还需要激活、通信 buffer、编译或 CUDA Graph 相关内存等空间。

### 3.2 动态增长的 KV Cache

Attention 在生成新 token 时需要历史 Key/Value。缓存历史 K/V 可以避免每轮重复计算整个上下文，但缓存本身要占显存。

对于普通 decoder-only attention，一个未分片模型中每个 token 的 KV 大小可粗略写成：

```text
2 × 层数 × KV heads × head_dim × 每元素字节数
```

其中 `2` 分别代表 K 和 V。总占用还要乘以当前缓存的 token 数。GQA/MQA 的 KV head 数较少，MLA、滑动窗口或混合层模型的布局又不同，所以不能只用“参数量”推断 KV Cache 大小。

### 3.3 调度与服务层开销

GPU 之外还有：

- chat template、tokenization、detokenization；
- HTTP、SSE streaming、进程间通信；
- Scheduler 的准入、抢占和 block 分配；
- 多卡 collective communication。

短请求或小模型下，这些 CPU/网络开销更容易显现。

可以把显存近似看成：

```text
模型权重 + KV Cache 物理池 + 激活/工作区 + 编译/CUDA Graph/框架开销
```

这是容量地图，不是可以用一条通用公式精确计算的显存账单。

## 4. 为什么需要 continuous batching？

如果每个请求单独执行，小 batch 往往无法充分利用 GPU。静态 batch 又要等待整组请求一起结束，已经完成的位置会浪费。

Continuous batching 的核心是按迭代重新决定本轮工作：完成的请求退出，新请求在满足 token、序列数和 KV Cache 约束时加入，运行中的请求继续推进。

在 vLLM V1 中，可以先记住两个预算：

```text
max_num_batched_tokens：单次迭代最多调度多少 token
max_num_seqs：单次迭代最多处理多少序列
```

实际调度还要考虑 KV block、模型长度、结构化输出、多模态输入、KV connector 等约束。Continuous batching 不是简单把“所有等待请求”拼进一个 tensor。

## 5. 五个关键词

### 5.1 Paged KV Cache / PagedAttention

vLLM 把请求的 KV Cache 组织成固定 token 容量的 block。请求在逻辑上拥有连续 token，物理 KV block 不需要连续；block table 保存逻辑位置到物理 block ID 的关系。

这带来两个直接收益：

- 调度器以 block 为单位动态分配和回收请求的 KV 容量；
- 固定大小 block 降低了“为最大序列预留整段连续空间”造成的浪费和外部碎片。

要区分两个动作：GPU 上的 KV Cache tensor pool 通常在初始化时建立；请求运行时分配的是池中 block 的所有权和映射，不是每生成一个 token 都调用一次 CUDA 内存分配。

### 5.2 Automatic Prefix Caching（APC）

APC 会复用已有请求计算过的相同前缀 KV。在 v0.25.0 的 V1 实现中，缓存键由父 block hash、当前 block token 和必要的额外信息组成，并且只缓存完整 block。

APC 主要减少重复前缀的 prefill 计算。它不会减少新输出 token 的 decode 计算，也不会帮助只有“中间内容相同、前缀不同”的请求。

### 5.3 Scheduler

Scheduler 决定一次迭代处理哪些请求、每个请求处理多少 token，以及能否分配足够的 KV slots。V1 的主队列是 `waiting` 和 `running`；prefill 与 decode 的推进统一体现在“已经计算的 token 数”和“当前需要计算的 token 数”之间的差距上。

### 5.4 Executor / Worker / Model Runner

Engine Core 把 `SchedulerOutput` 交给 Executor。Worker 持有模型和设备侧 KV Cache，Model Runner 把 block IDs、token IDs、positions 等变成模型 forward 所需的张量和 attention metadata。

“Worker”是职责名，不一定始终等于独立 OS 进程：v0.25.0 单 GPU 默认使用 `UniProcExecutor`，Worker 与 Engine Core 在同一个进程内；多 GPU `mp` 后端才会创建独立 Worker 进程。

### 5.5 OpenAI-Compatible Server

在线入口是：

```bash
vllm serve <model>
```

v0.25.0 支持 `/v1/completions`、`/v1/chat/completions` 等多种 API。Chat Completions 要求模型 tokenizer 提供 chat template，或启动时显式指定模板。

## 6. 从请求生命周期建立全局地图

第一阶段先记住这条主线：

```text
HTTP / LLM.generate
  -> 输入处理与 token IDs
  -> Engine Core 接收请求
  -> Scheduler 查 prefix hit、分配 KV block、组成 continuous batch
  -> Executor / Worker / Model Runner 执行模型
  -> Scheduler 更新请求状态
  -> detokenize / JSON 或 SSE 输出
  -> 请求结束，释放请求持有的 KV blocks
```

不要把“API、调度、内存、执行”混成一层。后面读源码时，职责边界比文件数量更重要。

## 7. PagedAttention 解决什么，不解决什么？

它直接服务于 paged KV 布局和 attention 访问，并帮助改善 KV 内存利用率；与调度器结合后，可以支撑动态请求长度和 continuous batching。

它不直接解决：

- 模型质量与幻觉；
- tokenizer、HTTP 或网络延迟；
- 权重显存本身；
- 多机通信成本；
- 业务侧限流、租户隔离和 SLA。

还要注意，v0.25.0 官方的 Paged Attention 设计页明确标注为历史文档，不能把其中某个旧 CUDA kernel 当成所有当前 attention backend 的唯一实现。

## 8. 第一篇的自检题

读完后应该能回答：

1. 为什么 decode 请求要跨多次调度迭代？
2. 模型权重和 KV Cache 的生命周期有什么不同？
3. PagedAttention 中“逻辑连续、物理不连续”分别指什么？
4. 为什么 prefix caching 主要改善 TTFT，而不是 decode ITL？
5. `max_num_batched_tokens` 和 KV Cache 容量为什么是两种不同预算？

## 9. 源码核对入口

以下路径均对应本机 v0.25.0 源码：

- `vllm/entrypoints/llm.py`：离线 `LLM` 入口。
- `vllm/v1/engine/async_llm.py`：在线异步请求与输出回程。
- `vllm/v1/engine/core.py`：Engine Core 调度/执行主循环。
- `vllm/v1/core/sched/scheduler.py`：`waiting`、`running`、token budget 与调度。
- `vllm/v1/core/kv_cache_manager.py`：prefix hit、slot 分配和释放。
- `vllm/v1/executor/uniproc_executor.py`：单进程 Executor。
- `vllm/v1/executor/multiproc_executor.py`：多进程 GPU Workers。

## 参考资料

- [vLLM v0.25.0 Quickstart](https://docs.vllm.ai/en/v0.25.0/getting_started/quickstart/)
- [vLLM v0.25.0 Architecture Overview](https://docs.vllm.ai/en/v0.25.0/design/arch_overview/)
- [vLLM v0.25.0 Automatic Prefix Caching](https://docs.vllm.ai/en/v0.25.0/design/prefix_caching/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [Paged Attention 历史设计页](https://docs.vllm.ai/en/v0.25.0/design/paged_attention/)
