# 03｜核心原理：PagedAttention 与 KV Cache 管理

这一篇讲 vLLM 最核心的概念：KV Cache 和 PagedAttention。

如果只记一句话：

```text
vLLM 通过 PagedAttention 把每个请求的 KV Cache 拆成固定大小的 block，用类似操作系统分页的方式管理 GPU 显存，从而减少碎片，提高并发。
```

## 1. 为什么会有 KV Cache？

LLM 生成文本是自回归的：

```text
已有 token -> 生成下一个 token -> 再把新 token 加入上下文 -> 继续生成
```

例如：

```text
输入：我 喜欢
输出第 1 个 token：写
上下文变成：我 喜欢 写
输出第 2 个 token：代码
上下文变成：我 喜欢 写 代码
```

Transformer 每一层 attention 都需要用到历史 token 的 Key 和 Value。

如果每生成一个 token 都重新计算全部历史 token 的 Key/Value，代价非常高。因此推理时会把历史 token 的 K/V 缓存下来，这就是 KV Cache。

## 2. KV Cache 为什么会成为瓶颈？

模型权重是固定的，但 KV Cache 是动态的。

它和下面几个因素相关：

```text
KV Cache 大小 ∝ 层数 × KV head 数 × head_dim × token 数 × 并发请求数 × dtype 大小
```

这意味着：

- 模型越大，KV Cache 越大。
- 上下文越长，KV Cache 越大。
- 并发越高，KV Cache 越大。
- 输出越长，decode 阶段 KV Cache 会继续增长。

在线服务里，很多时候不是模型权重放不下，而是并发起来后 KV Cache 顶不住。

## 3. 传统 KV Cache 管理的问题

一种朴素做法是：给每个请求预留一段连续显存。

比如一个请求最多支持 8192 token，就提前分配足够大的空间。

问题很明显：

### 3.1 预留浪费

用户可能只输入了 200 token，输出 100 token，但你为了支持最大长度，可能预留了 8192 token 的空间。

### 3.2 内部碎片

请求 A 用了 300 token，但分配了 8192 token，剩下的大量空间不可用。

### 3.3 外部碎片

GPU 显存里有很多空洞，但没有足够大的连续空间给新请求。

### 3.4 共享困难

多个请求有相同前缀时，理论上可以复用前缀 KV Cache。但如果每个请求都是一整段连续缓存，共享和写时复制会比较麻烦。

## 4. PagedAttention 的核心思想

PagedAttention 借鉴操作系统虚拟内存分页。

操作系统里：

```text
虚拟地址连续
物理内存可以不连续
页表负责虚拟页 -> 物理页映射
```

vLLM 里可以类比为：

```text
请求的 token 序列逻辑上连续
KV Cache 物理 block 可以不连续
block table 负责 logical block -> physical block 映射
```

一个请求的 KV Cache 不再需要是一整段连续显存，而是由多个固定大小 block 组成。

示意：

```text
Request A logical blocks:
[A0] [A1] [A2]

Block table:
A0 -> physical block 7
A1 -> physical block 2
A2 -> physical block 9

GPU physical KV blocks:
[0] [1] [2:A1] [3] [4] [5] [6] [7:A0] [8] [9:A2]
```

逻辑上 A0/A1/A2 连续，但物理上可以散落在不同位置。

## 5. 为什么这样能省显存？

### 5.1 按需分配

请求刚进来时，只需要为已有 token 分配 block。生成过程中不够了再分配新 block。

### 5.2 减少碎片

固定大小 block 比大块连续空间更容易复用。

这类似内存池/slab allocator：统一规格的小块比变长大块更容易管理。

### 5.3 支持共享

如果两个请求有相同前缀，它们可以指向同一批 physical block。

例如：

```text
Request A: 你是一个游戏后端专家，请解释 Redis 排行榜
Request B: 你是一个游戏后端专家，请解释 vLLM KV Cache

共同前缀：你是一个游戏后端专家，请解释
```

共同前缀对应的 KV block 可以共享。后续不同部分再分配各自 block。

### 5.4 支持写时复制

如果共享 block 后面需要修改，可以 copy-on-write。

这和操作系统 fork 后共享物理页、写入时复制的思想类似。

## 6. PagedAttention 对 attention 计算有什么影响？

传统 attention 计算时，通常假设 K/V 在连续内存里。

PagedAttention 下，K/V 被拆成 block，而且物理地址不连续。因此 attention kernel 需要根据 block table 找到每个 logical block 对应的 physical block。

也就是说，PagedAttention 不只是一个内存管理策略，它还要求 attention kernel 能理解这种分页布局。

这也是为什么 vLLM 不只是 Python 调度代码，还包含底层 kernel 和 attention backend。

## 7. 和后端开发经验怎么类比？

### 7.1 类比对象池

游戏服务器里经常会做对象池：

```text
频繁创建/销毁对象 -> 对象池复用 -> 减少分配开销和碎片
```

vLLM 的 KV block 管理也类似：

```text
频繁增长/释放 KV Cache -> block pool 复用 -> 减少显存碎片
```

### 7.2 类比分页内存

操作系统分页：

```text
虚拟页 -> 物理页
```

vLLM：

```text
logical KV block -> physical KV block
```

### 7.3 类比资源调度

请求不是只消耗 CPU，它还消耗 KV block。

Scheduler 每一轮不仅要问：

```text
哪些请求该跑？
```

还要问：

```text
有没有足够 KV block？
这个请求能不能继续生成？
是否需要抢占或等待？
```

## 8. Prefix Caching 和 PagedAttention 的关系

Automatic Prefix Caching 的目标是：如果新请求和已有请求共享前缀，就直接复用已有前缀的 KV Cache，跳过共享部分的重复计算。

PagedAttention 的 block 化设计让这种共享更自然，因为共享单位可以是 KV block。

不过要注意：Prefix Caching 不是万能的。它主要适合“前缀完全相同”的场景，比如：

- 固定 system prompt。
- 固定 few-shot 示例。
- RAG 模板前半部分相同。
- 多轮对话里部分上下文重复。

如果相同内容不在前缀，普通 prefix caching 的收益就会下降。

## 9. PagedAttention 解决什么，不解决什么？

### 9.1 它解决

- KV Cache 显存碎片。
- 请求间前缀共享。
- 长上下文/高并发下的显存利用率。
- 动态 batch 场景下的缓存管理。

### 9.2 它不直接解决

- 模型本身质量。
- 网络延迟。
- tokenizer 性能。
- 业务限流。
- 多机通信成本。
- 所有 attention 计算量问题。

PagedAttention 是内存管理和 kernel 执行层面的优化，不是模型算法本身的能力提升。

## 10. 学源码时重点看什么？

读源码时建议围绕问题看：

```text
1. 一个请求进来后，什么时候申请 KV block？
2. 每个请求的 block table 存在哪里？
3. 请求生成 token 后，KV block 如何追加？
4. 请求结束后，KV block 如何释放？
5. 多个请求共享前缀时，引用计数如何维护？
6. attention kernel 如何根据 block table 读 K/V？
```

不要一开始就从 kernel 开始看。先把资源生命周期搞清楚。

## 11. 本文小结

KV Cache 是 LLM 推理服务的关键资源。它随着请求长度和并发动态变化，管理不好就会造成显存浪费和并发下降。

PagedAttention 的核心是：

```text
把 KV Cache 拆成固定大小 block，逻辑连续，物理可不连续，用 block table 做映射。
```

它的价值是：

- 近似按需分配。
- 减少显存碎片。
- 提高 batch 并发能力。
- 支持 prefix sharing 和 copy-on-write。

这正是 vLLM 高吞吐、高显存利用率的基础。

## 参考资料

- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- vLLM Paged Attention 设计文档：https://docs.vllm.ai/en/latest/design/paged_attention/
- vLLM Automatic Prefix Caching：https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
