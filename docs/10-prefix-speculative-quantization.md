# 10｜Prefix Caching、Speculative Decoding 与 Quantization

第二阶段最后一篇，先把 vLLM 里几个常见高级能力串起来：

```text
Prefix Caching
Speculative Decoding
Quantization
```

它们解决的问题不同，但目标都很一致：

```text
在尽量不牺牲效果的前提下，降低重复计算、提升吞吐、降低显存占用或降低延迟。
```

## 1. 先建立总览

可以先用一句话区分：

```text
Prefix Caching：复用已经算过的前缀 KV，减少重复 prefill。

Speculative Decoding：用更便宜的方式先猜多个 token，再让大模型验证，减少大模型 decode 次数。

Quantization：用更低精度表示权重或 KV Cache，降低显存和带宽压力。
```

它们分别对应三类优化：

```text
少算：Prefix Caching
快算：Speculative Decoding
省资源：Quantization
```

## 2. Prefix Caching 解决什么问题？

很多线上请求有相同前缀。

例如：

```text
system prompt: 你是一个专业的代码助手...
工具说明: 你可以调用以下工具...
few-shot examples: 示例 1、示例 2、示例 3...
用户问题: xxx
```

如果每个请求都重新计算固定前缀，会浪费 prefill 计算。

Prefix Caching 的思路是：

```text
相同前缀已经算过 KV Cache，就直接复用。
```

简化示意：

```text
request A:
  [system prompt][tools][用户问题 A]

request B:
  [system prompt][tools][用户问题 B]

共享部分：
  [system prompt][tools]
```

如果共享前缀命中，就可以减少 prefill token 数。

## 3. Prefix Caching 为什么依赖 KV block？

vLLM 的 KV Cache 是 block 化管理的。

Prefix Caching 通常不是按字符或字符串复用，而是按 token/block 复用已经计算完成的 KV。

可以理解为：

```text
prefix tokens -> hash / match -> 找到已计算 KV blocks -> 引用这些 blocks
```

这就要求：

1. 前缀 token 必须一致。
2. 对应 KV block 已经计算完成。
3. block 可以被安全共享。
4. ref count 正确维护。

所以 Prefix Caching 不是单独功能，它依赖前面讲过的 Block Manager。

## 4. Prefix Caching 适合什么场景？

适合：

```text
固定 system prompt
固定工具说明
固定 few-shot 模板
RAG 模板前缀一致
Agent 工作流上下文大量重复
批量请求共享长前缀
```

不适合：

```text
每个请求前缀都完全不同
prompt 很短
用户输入变化发生在最前面
缓存命中率很低
```

Prefix Caching 的收益高度依赖命中率。

如果命中率低，它不仅收益有限，还会带来缓存管理开销。

## 5. Prefix Caching 的后端类比

它很像缓存系统：

```text
Redis cache：
  key 命中 -> 少查 DB
  key 未命中 -> 走完整计算，再写缓存

Prefix cache：
  prefix 命中 -> 少做 prefill
  prefix 未命中 -> 正常 prefill，再记录可复用 KV
```

也像游戏服务器里的配置/地图资源共享：

```text
多个玩家共享同一份地图静态数据。
玩家自己的动态状态另算。
```

区别是 Prefix Cache 缓的是 GPU KV Cache，不是普通 CPU 对象。

## 6. Speculative Decoding 解决什么问题？

LLM decode 阶段是一 token 一 token 生成。

```text
大模型生成 token 1
大模型生成 token 2
大模型生成 token 3
...
```

每一步都要跑大模型，成本很高。

Speculative Decoding 的思路是：

```text
先用较便宜的方法猜多个 token，再让大模型一次性验证。
```

常见形式是 draft model + target model：

```text
draft model：快速猜 token A B C D

target model：验证 A B C D 是否可接受
```

如果猜得准，大模型一次 forward 可以确认多个 token，从而减少 decode 轮数。

## 7. Speculative Decoding 的简化流程

简化流程：

```text
1. draft model 生成 k 个候选 token
2. target model 对这些 token 做验证
3. 接受前面连续正确的一段
4. 如果遇到不接受的 token，从 target model 分布重新采样
5. 进入下一轮
```

例如：

```text
draft 猜：A B C D

target 验证：A 接受，B 接受，C 拒绝

最终接受：A B
然后根据 target 分布采样一个替代 C 的 token
```

核心收益来自：

```text
一次 target model forward 尽量产出多个有效 token。
```

## 8. Speculative Decoding 的收益条件

它不是必然加速。

收益取决于：

1. draft model 是否足够快。
2. draft token 接受率是否足够高。
3. target model 验证开销是否划算。
4. batch 形态是否适合。
5. 系统是否有额外显存放 draft model。

如果 draft 很慢，或者接受率很低，可能反而变慢。

可以用公式直觉理解：

```text
收益 ≈ 多接受 token 带来的 target decode 轮数减少
     - draft model 额外开销
     - 验证和调度开销
```

## 9. Speculative Decoding 的工程代价

工程上它会增加复杂度：

1. 需要管理 draft model。
2. 需要额外显存。
3. Scheduler 要支持特殊解码流程。
4. KV Cache 状态更复杂。
5. 输出 token 接受/拒绝会影响请求状态更新。

所以它不是“打开一定更快”的开关。

适合先在固定 workload 上压测，再决定是否用于线上。

## 10. Quantization 解决什么问题？

Quantization 即量化。

它的核心思想是：

```text
用更低 bit 表示模型权重、激活或 KV Cache，从而降低显存占用和内存带宽压力。
```

常见：

```text
FP16 / BF16
FP8
INT8
INT4
GPTQ
AWQ
bitsandbytes
KV Cache quantization
```

不同量化方式优化对象不同。

## 11. 权重量化

权重量化主要减少模型权重显存。

例如把 FP16 权重压到 INT8 或 INT4。

收益：

1. 模型更容易放进单卡。
2. 可以给 KV Cache 留更多显存。
3. 某些硬件和 kernel 下吞吐可能提升。

代价：

1. 可能有精度损失。
2. kernel 支持依赖硬件和后端。
3. 某些量化格式加载和部署更复杂。

## 12. KV Cache 量化

KV Cache 量化针对的是在线推理中动态增长的 KV。

为什么重要？

```text
KV Cache 大小 ∝ batch_size × sequence_length × layers × kv_heads × head_dim × dtype_size
```

如果把 KV Cache 从 FP16 降到更低精度，显存和带宽压力会下降。

收益：

1. 支持更高并发。
2. 支持更长上下文。
3. decode 读取 KV 的带宽压力降低。

代价：

1. 可能影响输出质量。
2. 需要 attention backend 支持。
3. 不同模型对 KV 精度敏感度不同。

## 13. 量化不是只看“显存变小”

量化可能带来三种结果：

```text
显存下降，速度提升
显存下降，速度差不多
显存下降，速度反而下降
```

为什么会反而下降？

1. 反量化开销大。
2. kernel 不够优化。
3. 硬件不擅长某种低精度。
4. batch 太小，收益被 overhead 抵消。

所以量化必须压测。

不能只看模型文件大小。

## 14. 三者分别影响哪些指标？

| 能力 | 主要优化 | 主要影响指标 | 风险 |
|---|---|---|---|
| Prefix Caching | 减少重复 prefill | TTFT、prefill 吞吐 | 命中率低则收益小 |
| Speculative Decoding | 减少 target decode 轮数 | TPOT、输出吞吐 | draft 开销、接受率不足 |
| Quantization | 降低显存/带宽压力 | 并发、吞吐、部署成本 | 精度损失、kernel 支持 |

## 15. 如何选择优先级？

如果你的业务是固定 system prompt + 工具调用：

```text
优先看 Prefix Caching
```

如果你的业务 decode 很长，且有合适 draft model：

```text
尝试 Speculative Decoding
```

如果你的模型太大、显存紧张、上下文长：

```text
优先看 Quantization / KV Cache dtype
```

如果你刚开始学习：

```text
先理解 Prefix Caching
再理解 Quantization
最后再研究 Speculative Decoding
```

因为 Prefix Caching 和前面学的 KV block 关系最直接。

## 16. 一个后端视角的决策框架

不要问：

```text
这个功能快不快？
```

要问：

```text
我的 workload 是否匹配它？
```

可以按下面检查：

```text
1. prompt 是否大量重复？
   是 -> Prefix Caching 可能有收益

2. 输出是否很长？
   是 -> Speculative Decoding 可能有收益

3. 显存是否限制并发或上下文？
   是 -> Quantization / KV Cache 量化可能有收益

4. 当前瓶颈是 TTFT、TPOT、吞吐还是 OOM？
   不同瓶颈用不同手段
```

## 17. 读源码时看什么？

### 17.1 Prefix Caching

关键词：

```text
prefix_cache
automatic_prefix_caching
hash
computed_block
block_table
ref_count
```

关注：

1. prefix 如何 hash。
2. block 如何判定可复用。
3. ref count 如何维护。
4. cache miss 后如何写入。

### 17.2 Speculative Decoding

关键词：

```text
spec_decode
speculative
draft_model
target_model
acceptance
proposal
```

关注：

1. draft token 如何产生。
2. target 如何验证。
3. 接受/拒绝如何更新输出。
4. KV Cache 如何保持一致。

### 17.3 Quantization

关键词：

```text
quantization
awq
gptq
bitsandbytes
fp8
kv_cache_dtype
```

关注：

1. 权重加载时如何选择量化方法。
2. kernel 是否支持对应 dtype。
3. KV Cache dtype 如何影响 attention backend。
4. 精度和性能如何压测。

## 18. 本文小结

第二阶段最后，你要把这三个能力放到 vLLM 主链路里理解：

```text
Prefix Caching：和 KV Cache / Block Manager 强相关，减少重复 prefill。

Speculative Decoding：和 Scheduler / Model Runner / Sampler 强相关，减少 target model decode 轮数。

Quantization：和模型加载 / Attention Backend / KV Cache 强相关，降低显存和带宽压力。
```

它们不是孤立功能，而是分别插在 vLLM 的不同层：

```text
API / Engine
  -> Scheduler
  -> KV Cache Manager / Prefix Cache
  -> Model Runner
  -> Attention Backend / Quantized Kernel
  -> Sampler / Spec Decode
  -> Output
```

到这里，第二阶段的关键模块已经完成：

1. Scheduler：调度。
2. Block Manager：KV 资源管理。
3. Model Runner：模型执行入口。
4. Attention Backend：核心 attention 实现。
5. Prefix / Spec Decode / Quantization：高级优化能力。

下一阶段就可以进入岗位导向：源码阅读、压测、瓶颈定位、框架对比，以及从 C++ 游戏后端迁移到 AI 推理岗位的路线。

## 参考资料

- vLLM Automatic Prefix Caching：https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
- vLLM Speculative Decoding：https://docs.vllm.ai/en/latest/features/spec_decode.html
- vLLM Quantization：https://docs.vllm.ai/en/latest/features/quantization/
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- Speculative Decoding 论文：https://arxiv.org/abs/2211.17192
