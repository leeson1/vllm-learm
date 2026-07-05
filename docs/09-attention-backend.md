# 09｜Attention Backend：FlashAttention、PagedAttention kernel 与 CUDA Graph

这一篇进入更靠近 GPU 的部分：Attention Backend。

一句话概括：

```text
Attention Backend 负责选择和执行具体的 attention 实现，让模型在不同硬件、不同 batch 形态、不同 KV Cache 布局下尽量高效地完成注意力计算。
```

如果 Model Runner 是“组织模型输入”，那 Attention Backend 就是“真正把 attention 算快”。

## 1. 为什么 attention 是推理核心？

Transformer 的一层大致包含：

```text
Attention
MLP
Norm
Residual
```

LLM 推理时，attention 特别关键，因为它和上下文长度强相关。

生成第 N 个 token 时，模型需要关注前面已经存在的 token：

```text
当前 token -> query
历史 token -> key/value from KV Cache
attention -> 当前 hidden state
```

上下文越长，decode 阶段读取历史 KV 的成本越高。

所以 LLM Serving 的很多优化都围绕 attention 展开。

## 2. prefill attention 和 decode attention 不一样

### 2.1 prefill

prefill 一次处理 prompt 中的一批 token。

```text
输入长度：可能是几百、几千、几万 token
计算形态：大块矩阵计算
主要目标：尽快完成首轮上下文处理，降低 TTFT
```

prefill 阶段常见优化是使用高效 attention 算法，例如 FlashAttention 类实现。

### 2.2 decode

decode 每轮通常每个请求只生成一个 token。

```text
输入：当前 token
历史：完整 KV Cache
计算形态：小 query + 大量历史 KV 读取
主要目标：降低 TPOT，提高输出 token 吞吐
```

decode 阶段最关键的是高效读取 KV Cache。

PagedAttention 主要就是为这种分页 KV Cache 布局服务。

## 3. FlashAttention 解决什么问题？

普通 attention 如果直接算，会产生很大的中间矩阵：

```text
QK^T -> attention scores -> softmax -> AV
```

当序列很长时，中间 attention matrix 很大，显存访问成本高。

FlashAttention 的核心思想可以粗略理解为：

```text
通过分块计算和融合，减少 HBM 读写，避免显式存储完整 attention matrix。
```

重点不是“数学变了”，而是“计算组织方式变了”。

它让 attention 更接近 GPU 友好的执行方式：

1. 分块加载 Q/K/V。
2. 在 SRAM / shared memory 中做更多计算。
3. 减少对 GPU 全局显存的反复读写。
4. 提升长序列 prefill 的效率。

## 4. PagedAttention kernel 解决什么问题？

vLLM 的 KV Cache 不是为每个请求分配连续大块显存，而是分页成 block。

所以 attention kernel 不能简单假设：

```text
request A 的 KV Cache 在物理显存上连续排列
```

它需要通过 block table 找到真实物理 block。

简化理解：

```text
for each request:
    for each logical block in history:
        physical_block = block_table[logical_block]
        read K/V from physical_block
        compute attention
```

PagedAttention kernel 的价值是：

```text
在 KV Cache 物理不连续的前提下，仍然高效完成 attention。
```

这就是 vLLM 内存管理和 attention kernel 必须配合的原因。

## 5. block table 为什么会进入 kernel 路径？

因为逻辑 token 到物理显存的映射不是固定连续的。

例如：

```text
request A logical blocks:
  0 -> physical block 102
  1 -> physical block 7
  2 -> physical block 88
```

attention 读取历史 KV 时必须知道这些映射。

所以 block table 不是普通业务元数据，而是执行路径上的关键数据。

这会带来一个工程权衡：

```text
分页管理提升了显存利用率，但 kernel 需要支持间接寻址。
```

这也是为什么 vLLM 的高性能不只来自调度，还来自内存布局和 kernel 实现协同。

## 6. Attention Backend 是什么？

Attention Backend 可以理解为 attention 实现的适配层。

不同场景可能使用不同 backend：

```text
prefill: 适合 FlashAttention 类实现

decode: 适合 PagedAttention / paged KV cache kernel

不同硬件: NVIDIA / AMD / CPU / TPU 可能有不同实现

不同模型: MLA、GQA、MHA、滑动窗口、量化 KV 也会影响选择
```

Backend 层存在的意义是：

```text
上层模型不应该到处写硬件和 kernel 分支，而是通过统一接口调用合适的 attention 实现。
```

## 7. GQA / MQA 对 attention 有什么影响？

现代 LLM 常见：

```text
MHA：Multi-Head Attention
MQA：Multi-Query Attention
GQA：Grouped-Query Attention
```

区别可以粗略理解为：

```text
MHA：每个 query head 有自己的 key/value head
MQA：多个 query head 共享一组 key/value
GQA：一组 query head 共享一组 key/value
```

GQA/MQA 的好处是减少 KV Cache 体积。

因为 KV head 数变少了：

```text
KV Cache 大小 ∝ num_kv_heads
```

这会直接影响：

1. 显存占用。
2. KV 读取带宽。
3. attention kernel 的输入布局。
4. 最大可服务并发。

所以 attention backend 必须理解模型的 head 结构。

## 8. CUDA Graph 在 Attention Backend 周围的作用

LLM decode 阶段会执行非常多轮。

如果每一轮都从 CPU 发起大量 kernel launch，会产生明显 overhead。

CUDA Graph 的思路是：

```text
先捕获一段固定形态的 CUDA 执行图
后续重复 replay
```

它的收益主要是降低 CPU launch overhead。

但是 CUDA Graph 要求执行形态相对稳定。

vLLM 的难点在于：

```text
Scheduler 需要动态 batch，CUDA Graph 喜欢静态形状。
```

所以 vLLM 需要在动态调度和静态图复用之间做工程折中。

## 9. 为什么 attention backend 会影响调参？

一些参数会改变 batch 形态和 KV 形态，从而影响 backend 效率。

### 9.1 `--max-num-batched-tokens`

它影响 prefill batch token 数。

如果 token budget 变大，prefill 可能更容易形成大计算块，但单轮耗时也会变长。

### 9.2 `--max-num-seqs`

它影响 decode 阶段并发序列数。

序列数太小，decode batch 小，GPU 可能吃不满。

序列数太大，KV Cache 压力和调度开销上升。

### 9.3 `--block-size`

block size 会影响 KV Cache 分页粒度。

它既影响显存碎片，也影响 block table 长度和 kernel 间接访问形态。

### 9.4 `--dtype` / `--kv-cache-dtype`

精度影响显存占用和带宽压力。

例如 KV Cache 量化可以降低显存和带宽压力，但可能引入精度和 kernel 支持问题。

## 10. 从 CUDA 视角看 attention 优化

如果你后面要往 CUDA/HPC 深挖，可以重点关注这些点：

1. global memory 读写次数。
2. shared memory / register 使用。
3. warp-level 并行组织。
4. memory coalescing。
5. tensor core 使用。
6. kernel fusion。
7. launch overhead。
8. 不同 batch shape 下的 occupancy。

但第二阶段不要急着写 kernel。

更好的顺序是：

```text
先理解 KV Cache 布局
  -> 再理解 block table 如何传给 kernel
  -> 再理解 prefill/decode attention 差异
  -> 最后再看具体 CUDA kernel
```

## 11. 读源码时看什么？

建议先围绕接口读，而不是一头扎进 `.cu` 文件。

先看这些问题：

1. vLLM 如何选择 attention backend？
2. Model Runner 传给 attention 的元数据有哪些？
3. block table 和 slot mapping 在哪里使用？
4. prefill 和 decode 是否走不同 kernel？
5. CUDA Graph 捕获的边界在哪里？
6. 不同 dtype / kv cache dtype 如何影响 backend？

关键词：

```text
attention_backend
paged_attention
flash_attention
block_tables
slot_mapping
kv_cache
cuda_graph
```

## 12. 和后端开发的类比

Attention Backend 有点像网络库里的 IO backend：

```text
业务层：我要发包
IO backend：epoll / kqueue / io_uring / IOCP
```

业务层不应该到处关心底层细节。

vLLM 里也类似：

```text
模型层：我要做 attention
Attention Backend：选择具体 kernel 和执行策略
```

区别是这里的 backend 面向 GPU kernel，而不是 OS IO。

## 13. 本文小结

Attention Backend 是 vLLM 性能链路里最靠近 GPU 的核心层之一。

你需要记住：

1. prefill 和 decode 的 attention 形态不同。
2. FlashAttention 主要通过减少显存读写提升 attention 效率。
3. PagedAttention kernel 让分页 KV Cache 布局可以高效参与 attention。
4. block table 是执行路径上的关键元数据。
5. CUDA Graph 可以减少 launch overhead，但和动态 batch 存在权衡。
6. 想往 HPC/CUDA 深挖，attention backend 是非常值得研究的入口。

到这里，vLLM 第二阶段的主链路已经串起来了：

```text
Scheduler 决定跑谁
  -> Block Manager 管 KV block
  -> Model Runner 组织 forward
  -> Attention Backend 执行核心 attention
```

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM Attention Backend Feature Support：https://docs.vllm.ai/en/latest/design/attention_backend.html
- vLLM Paged Attention 设计文档：https://docs.vllm.ai/en/latest/design/paged_attention.html
- vLLM CUDA Graphs 设计文档：https://docs.vllm.ai/en/latest/design/cuda_graphs.html
- FlashAttention 论文：https://arxiv.org/abs/2205.14135
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
