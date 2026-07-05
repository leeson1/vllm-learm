# 07｜Block Manager：KV block 分配、回收、共享与抢占

上一篇讲 Scheduler，这一篇讲它背后的核心资源：KV Cache block。

如果把 Scheduler 类比成“调度器”，那 Block Manager / KV Cache Manager 就像“显存资源管理器”。

一句话概括：

```text
Block Manager 负责把请求需要的 KV Cache 映射到 GPU 显存中的固定大小 block，并处理分配、释放、复用、共享和抢占。
```

## 1. 先理解 KV Cache 为什么重要

Transformer 自回归生成时，每生成一个 token 都要关注历史 token。

如果每一步都重新计算所有历史 token 的 Key/Value，成本会非常高。

所以推理引擎会把历史 token 的 Key/Value 缓存下来：

```text
prompt token -> 计算 K/V -> 写入 KV Cache
new token    -> 只计算新 token 的 K/V，然后读取历史 KV 做 attention
```

这就是 KV Cache。

它的特点是：

1. 随着序列长度增长。
2. 随着并发请求数量增长。
3. 占用 GPU 显存。
4. 生命周期和请求强绑定。

在线推理时，KV Cache 往往比你想象中更关键。

## 2. 如果不用 block，会有什么问题？

最朴素的做法是：每个请求预分配一大块连续显存。

例如最大上下文是 8192：

```text
request A -> 预留 8192 token 的 KV 空间
request B -> 预留 8192 token 的 KV 空间
request C -> 预留 8192 token 的 KV 空间
```

但真实请求可能是：

```text
A 实际只用了 200 token
B 实际用了 600 token
C 实际用了 7000 token
```

这样会产生巨大浪费。

类似游戏服务器里给每个玩家都预分配一个超大背包：

```text
最大 10000 个道具格
每个玩家登录时都直接占满 10000 格内存
```

显然不合理。

另一个问题是外部碎片：

```text
显存中有很多空洞，但没有一段足够大的连续空间给新请求。
```

PagedAttention 的核心价值就是避免这种连续大块分配的问题。

## 3. vLLM 的 block 思想

vLLM 把 KV Cache 拆成固定大小 block。

例如 block size = 16 tokens：

```text
逻辑 token 序列：
0 1 2 ... 15 | 16 17 ... 31 | 32 ... 47

逻辑 block：
block 0       | block 1       | block 2
```

每个请求看到的是逻辑上连续的 token，但物理显存可以不连续：

```text
request A logical blocks:
  logical 0 -> physical block 100
  logical 1 -> physical block 37
  logical 2 -> physical block 203
```

这和操作系统分页非常像：

```text
虚拟地址连续，不要求物理页连续。
```

vLLM 通过 block table 建立逻辑 block 到物理 block 的映射。

## 4. Block Manager 管哪些数据？

可以先用一个简化模型理解：

```text
FreeBlockList：当前空闲物理 block
BlockTable：每个请求的逻辑 block -> 物理 block 映射
RefCount：物理 block 被多少请求引用
ComputedFlag：某个 block 是否已经计算完成，可用于 prefix cache
```

真实实现会有更多细节，但主干就是这些。

## 5. 分配：什么时候需要新 block？

请求进入 prefill 时，需要根据 prompt 长度分配 KV block。

例如：

```text
block_size = 16
prompt_len = 40
需要 block 数 = ceil(40 / 16) = 3
```

decode 阶段也可能需要新 block。

例如当前请求已经生成到第 16、32、48 个 token 的边界时，需要追加一个新 block。

简化伪代码：

```cpp
int NeedBlocks(int num_tokens, int block_size) {
    return (num_tokens + block_size - 1) / block_size;
}

bool Allocate(Request& req, int new_tokens) {
    int need = CalcAdditionalBlocks(req, new_tokens);
    if (free_blocks.size() < need) {
        return false;
    }

    for (int i = 0; i < need; ++i) {
        auto block = free_blocks.pop();
        req.block_table.push_back(block);
    }
    return true;
}
```

这个伪代码不是 vLLM 源码，只是帮助你建立模型。

## 6. 回收：请求结束后发生什么？

当请求完成、取消或超时时，它占用的 KV block 可以释放。

```text
request finished
  -> 遍历 block table
  -> ref count--
  -> 如果 ref count == 0，放回 free list
```

这里要注意共享场景。

如果某个 block 被 prefix cache 或多个请求共享，不能直接释放物理 block，只能减少引用计数。

```text
physical block 100 ref_count = 3
释放 request A -> ref_count = 2，不能回收
释放 request B -> ref_count = 1，不能回收
释放 request C -> ref_count = 0，可以回收
```

## 7. 共享：为什么多个请求可以共用 KV？

很多请求有共同前缀：

```text
system prompt: 你是一个专业助手...
工具说明: xxx
few-shot 示例: xxx
用户问题: ...
```

如果前缀完全一样，就没必要重复计算前缀 KV。

vLLM 可以让多个请求共享已经计算好的前缀 block：

```text
request A: [shared block 1][shared block 2][private block A]
request B: [shared block 1][shared block 2][private block B]
```

这样可以节省：

1. GPU 计算：不用重复 prefill 前缀。
2. GPU 显存：共享 KV block。
3. TTFT：命中前缀缓存时首 token 更快。

## 8. Copy-on-Write 是什么？

共享 block 有一个问题：如果某个请求要继续往 block 里写数据怎么办？

如果 block 被多个请求引用，直接写会影响其它请求。

所以需要 Copy-on-Write：

```text
如果 block ref_count > 1，并且当前请求要修改它：
  1. 分配一个新的物理 block
  2. 拷贝旧 block 内容
  3. 当前请求指向新 block
  4. 旧 block ref_count--
```

这和操作系统 fork 后的写时复制很像。

后端类比：

```text
多个玩家共享一份静态配置，没有问题。
某个玩家要修改自己的副本时，必须 copy 一份私有数据。
```

## 9. 抢占：KV block 不够怎么办？

高并发时，KV block 可能不够。

这时 Scheduler 和 Block Manager 要协作处理。

常见思路包括：

1. 让新请求等待。
2. 暂停某些运行中的请求。
3. 释放某些请求的 KV block，后面重新计算。
4. 根据策略选择牺牲哪个请求。

这就是 preemption。

可以理解为：

```text
当前显存资源不够，需要把某些请求从 running 状态踢回等待或重算状态。
```

它会带来代价：

- 被抢占请求延迟上升。
- 如果 KV 被释放，后面可能需要 recompute。
- 系统吞吐和尾延迟都会受影响。

所以抢占不是优化手段，而是资源不足时的保护机制。

## 10. Block Manager 和 Scheduler 的关系

Scheduler 想调度一个请求时，必须问资源管理器：

```text
这轮要处理这些 token，需要多少 KV block？
现在够不够？
```

如果够：

```text
分配 block -> 加入本轮 batch -> Model Runner 执行
```

如果不够：

```text
等待 / 抢占 / 降低本轮调度量
```

所以 Scheduler 不是只按队列顺序调度，它必须受 KV Cache 容量约束。

## 11. block size 有什么影响？

block size 是 KV Cache 管理的关键粒度。

如果 block size 太大：

```text
内部碎片增加。
例如只多生成 1 个 token，也可能占用一个大 block。
```

如果 block size 太小：

```text
block table 更长，元数据更多，attention kernel 访问映射也更复杂。
```

所以 block size 是一个工程折中：

```text
显存利用率 vs 元数据/调度/kernel 复杂度
```

## 12. 读源码时看什么？

读 Block Manager / KV Cache Manager 建议围绕这些问题：

1. 物理 block 池在哪里初始化？
2. 每个请求的 block table 如何维护？
3. prefill 和 decode 分别什么时候追加 block？
4. prefix cache 命中后如何复用已有 block？
5. 请求结束、取消、抢占时如何释放？
6. ref count 在哪里增加和减少？

关键词可以先搜：

```text
kv_cache_manager
block_manager
block_table
allocate
free
prefix_cache
ref_cnt
preempt
```

## 13. 和游戏后端内存池的类比

你可以把 KV block 看成对象池里的固定大小对象。

```text
对象池：
  预先分配 N 个对象，业务按需申请/释放。

KV block 池：
  预先占用一部分 GPU 显存切成 N 个 block，请求按需申请/释放。
```

区别是：

1. KV block 在 GPU 显存上。
2. KV block 会被 attention kernel 直接读取。
3. block table 会进入模型执行路径。
4. block 分配策略会直接影响吞吐和尾延迟。

所以它既是内存管理问题，也是推理性能问题。

## 14. 本文小结

Block Manager 是 vLLM 能高效服务长上下文和高并发请求的关键。

你需要记住：

1. KV Cache 是在线推理的核心显存资源。
2. vLLM 用固定大小 block 管理 KV Cache。
3. block table 让逻辑 token 连续、物理显存不连续成为可能。
4. ref count 和 Copy-on-Write 支持共享和安全修改。
5. KV block 不够时会影响 Scheduler，甚至触发抢占。

如果 Scheduler 解决的是“下一轮跑谁”，那 Block Manager 解决的是：

```text
这些请求有没有足够显存资源可以跑？
```

## 参考资料

- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- vLLM 官方文档：https://docs.vllm.ai/
- vLLM Paged Attention 设计文档：https://docs.vllm.ai/en/latest/design/paged_attention.html
- vLLM Automatic Prefix Caching：https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
