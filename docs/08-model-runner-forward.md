# 08｜Model Runner：一次 forward 前后到底发生了什么

前两篇讲了 Scheduler 和 KV block。这一篇进入执行层：Model Runner。

一句话概括：

```text
Model Runner 负责把 Scheduler 选出来的一批请求，整理成模型可以执行的输入，在 GPU 上完成 forward，并把 logits / hidden states / KV Cache 更新等结果交回上层。
```

如果 Scheduler 是“决定跑什么”，Block Manager 是“保证资源够不够”，Model Runner 就是“真正把这一轮跑起来”。

## 1. 不要把 forward 理解得太简单

很多人刚开始会以为 forward 就是：

```python
outputs = model(input_ids)
```

但在 vLLM 这种推理引擎里，一次 forward 前后要做很多工程准备。

因为输入不是一个普通 tensor，而是一批状态不同的请求：

```text
request A：decode，当前只要生成 1 个 token
request B：decode，当前只要生成 1 个 token
request C：prefill，当前要处理 512 个 prompt token
request D：chunked prefill，当前处理第 2 个 chunk
```

Model Runner 需要把这些请求整理成一个 GPU 友好的 batch。

## 2. 一次推理 step 的大致链路

可以先看一条简化链路：

```text
API / Engine
  -> Scheduler 生成 schedule
  -> KV Cache Manager 分配 block
  -> Model Runner 准备输入
  -> Attention Backend 读取/写入 KV Cache
  -> Model forward
  -> Sampler 采样下一个 token
  -> Output Processor 处理输出
  -> 请求状态更新，进入下一轮
```

Model Runner 位于中间执行层。

它既要懂模型输入，又要懂 vLLM 的调度元数据。

## 3. forward 前：准备哪些输入？

一次 forward 前，通常要准备这些东西。

### 3.1 token ids

也就是本轮要送进模型的 token。

prefill 请求可能有很多 token：

```text
[101, 2054, 2003, ...]
```

decode 请求通常只有一个 token：

```text
[上一次生成的 token]
```

Scheduler 决定本轮每个请求处理多少 token，Model Runner 负责把它们拼成 batch 输入。

### 3.2 positions

Transformer 需要知道每个 token 的位置。

```text
request A 已经有 128 个历史 token，本轮 decode token 的 position = 128
request B 已经有 2048 个历史 token，本轮 position = 2048
```

position 影响 RoPE / positional embedding。

如果 position 错了，模型输出就会错。

### 3.3 slot mapping

slot mapping 可以粗略理解为：

```text
本轮每个 token 的 KV 应该写到哪个物理位置。
```

因为 vLLM 的 KV Cache 是分页管理的，逻辑 token 位置和物理显存位置不是简单连续关系。

所以 Model Runner 要把调度结果转换成 attention backend 能理解的映射信息。

### 3.4 block tables

block table 描述每个请求的逻辑 block 到物理 block 映射。

Attention kernel 读取历史 KV 时，需要知道：

```text
这个请求的第 0 个逻辑 block 在哪个 physical block？
第 1 个逻辑 block 在哪里？
第 2 个逻辑 block 在哪里？
```

block table 是 PagedAttention 能工作的关键元数据之一。

### 3.5 sequence lengths

Attention 需要知道每个序列当前长度。

```text
request A seq_len = 129
request B seq_len = 2049
request C seq_len = 512
```

不同请求长度不一样，kernel 需要用这些信息做正确的 causal attention。

## 4. prefill 和 decode 在 Model Runner 里有什么不同？

### 4.1 prefill 路径

prefill 输入一段 prompt token。

```text
input: prompt tokens
output: prompt 对应的 KV Cache + logits
```

它的重点是：

1. 一次处理多个 token。
2. 批量写入 KV Cache。
3. 通常决定首 token 延迟。

### 4.2 decode 路径

decode 输入当前 token。

```text
input: last generated token
output: next token logits + 追加一个 KV
```

它的重点是：

1. 每个请求每轮通常一个 token。
2. 需要读取完整历史 KV。
3. 反复执行很多轮。

### 4.3 混合 batch

vLLM 的一轮 batch 可能同时包含 prefill 和 decode。

所以 Model Runner 不能只处理一种固定形态，而要根据调度元数据组织输入。

```text
batch = [
  decode A: 1 token,
  decode B: 1 token,
  prefill C: 256 tokens,
  prefill D: 512 tokens,
]
```

这就是为什么 vLLM 的执行层比普通模型 demo 复杂很多。

## 5. forward 中：模型到底做什么？

从 Transformer 视角看，forward 大致经过：

```text
input_ids
  -> embedding
  -> N 层 Transformer block
       -> attention
       -> MLP
  -> lm_head
  -> logits
```

vLLM 的特殊点主要在 attention。

attention 需要：

1. 把本轮 token 的 K/V 写入 KV Cache。
2. 根据 block table 读取历史 K/V。
3. 执行 causal attention。
4. 输出 hidden states。

所以 attention backend 是 Model Runner 调用链里非常关键的一层。

## 6. forward 后：logits 还不能直接返回

模型 forward 后得到的是 logits。

```text
logits: 每个 token 对整个词表的分数
```

但用户要的是文本 token。

还需要 sampler：

```text
logits
  -> temperature
  -> top_p / top_k
  -> repetition penalty
  -> sampling / greedy
  -> next token id
```

不同请求可能有不同 SamplingParams：

```text
request A: temperature = 0.7, top_p = 0.9
request B: temperature = 0.0, greedy
request C: max_tokens = 1024
```

所以采样也是请求级别的。

## 7. 输出之后：请求状态如何更新？

每轮生成后，要更新请求状态：

```text
append new token
update seq len
check stop tokens
check max_tokens
check eos
update streaming output
```

如果请求结束：

```text
release KV blocks
remove from running
return final output
```

如果没结束：

```text
继续留在 running，等待下一轮调度
```

因此一次 forward 不是结束点，只是一轮循环的一部分。

## 8. CUDA Graph 在这里起什么作用？

CUDA Graph 可以减少 CPU launch overhead。

普通执行大致是：

```text
Python / PyTorch 每次 forward 都发起一堆 CUDA kernel launch
```

如果每轮 decode 形态比较固定，就可以捕获成 CUDA Graph：

```text
capture 一段固定执行图
后续 replay，减少 CPU 调度开销
```

但 CUDA Graph 对输入形状比较敏感。

vLLM 需要在性能收益和动态 batch 灵活性之间做权衡。

可以这样理解：

```text
动态调度越灵活，执行图越难固定。
执行图越固定，CUDA Graph 越容易复用。
```

## 9. Model Runner 和 Worker 的关系

在多 GPU / 多进程架构中，vLLM 会有 worker 进程。

Worker 更像执行单元：

```text
Worker
  -> 持有模型权重
  -> 持有或访问 KV Cache
  -> 调用 Model Runner 执行 forward
```

Model Runner 则更关注单次 forward 的准备和执行。

你可以粗略理解为：

```text
Worker：进程级执行容器
Model Runner：模型级执行逻辑
Attention Backend：核心算子/内核实现
```

## 10. 从后端角度看 Model Runner

后端开发可以把 Model Runner 类比为一个“协议适配 + 执行器”。

上游 Scheduler 给的是调度结果：

```text
哪些请求、本轮多少 token、对应哪些 block
```

下游模型需要的是 tensor：

```text
input_ids
positions
block_tables
slot_mapping
seq_lens
```

Model Runner 做的事情就是：

```text
业务状态 -> 计算图输入
计算图输出 -> 业务状态更新
```

这和游戏服务器里把玩家状态打包成战斗逻辑输入，再把战斗结果写回玩家状态很像。

## 11. 常见性能瓶颈在哪里？

### 11.1 CPU 侧准备太慢

如果 batch 构造、元数据准备、进程通信太慢，GPU 会等 CPU。

表现可能是：

```text
GPU utilization 不高，但请求延迟高
```

### 11.2 batch 形态不稳定

动态 batch 长度变化大，会影响 kernel 选择、CUDA Graph 复用和整体吞吐。

### 11.3 attention 读 KV 成本高

长上下文 decode 时，每个新 token 都要读大量历史 KV。

这会让 decode 更容易受 memory bandwidth 影响。

### 11.4 采样和输出处理变成瓶颈

高并发流式输出时，detokenizer、HTTP streaming、日志和 metrics 也可能成为瓶颈。

不要只盯 GPU kernel。

## 12. 读源码时看什么？

建议围绕一次 forward 的输入输出读：

1. Scheduler 输出了什么调度结构？
2. Model Runner 如何构造 input_ids 和 positions？
3. block tables 和 slot mapping 从哪里来？
4. attention backend 如何拿到这些元数据？
5. logits 如何进入 sampler？
6. next token 如何写回 request state？

关键词可以先搜：

```text
model_runner
execute_model
prepare_input
input_batch
sampling_metadata
logits
sampler
```

## 13. 本文小结

Model Runner 是 vLLM 执行链路里的关键转换层。

你需要记住：

1. Scheduler 决定本轮跑哪些请求。
2. Block Manager 决定 KV block 如何映射。
3. Model Runner 把调度结果整理成模型输入。
4. Attention Backend 根据 block table 读写 KV Cache。
5. Sampler 把 logits 转成 next token。
6. 请求状态更新后进入下一轮调度。

不要把 vLLM 的 forward 理解成简单的 `model(input_ids)`。

更准确的理解是：

```text
一次 forward 是调度、内存映射、模型执行、采样、状态更新共同组成的一轮服务端循环。
```

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM 架构设计文档：https://docs.vllm.ai/en/latest/design/architecture.html
- vLLM CUDA Graphs 设计文档：https://docs.vllm.ai/en/latest/design/cuda_graphs.html
- vLLM GitHub：https://github.com/vllm-project/vllm
