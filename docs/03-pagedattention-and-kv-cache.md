# 03｜核心原理：PagedAttention 与 KV Cache 管理

> 版本基线：vLLM v0.25.0，重点讲 V1 的 paged KV Cache 与 Automatic Prefix Caching。官方 Paged Attention kernel 页面已标为历史文档，本文只把它用于概念背景，不把它当作当前所有 backend 的唯一实现。

如果只记一句话：

```text
vLLM 启动时建立 KV Cache 物理 block 池；调度时按请求分配 block ID，
用 block table 把逻辑 token 位置映射到物理 KV slots。
```

## 1. 为什么需要 KV Cache？

Decoder-only LLM 自回归生成：

```text
prompt -> token T1 -> token T2 -> token T3 -> ...
```

每层 self-attention 都需要历史 token 的 Key 和 Value。如果生成每个 token 时都重新计算完整历史，重复工作会随序列增长。推理引擎因此保存已经计算过的 K/V，后续迭代读取缓存，只计算新增 token 对应的模型状态。

一个容易忽略的时序是：本轮采样出的 token 还没有作为模型输入执行，它对应的 KV 要到下一轮 forward 才写入。可以把标准路径理解为：

```text
已计算 prompt KV -> 采样 T1
输入 T1 并写入 T1 的 KV -> 采样 T2
输入 T2 并写入 T2 的 KV -> 采样 T3
```

这也是 V1 Scheduler 用 `num_computed_tokens` 追赶当前 token 数来统一 prefill 和 decode 的基础。

## 2. KV Cache 到底有多大？

对普通 decoder-only attention，先忽略 TP、对齐和特殊 attention，每个请求每个 token 的 KV 字节数可粗略估算为：

```text
KV bytes/token
  = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

- `2`：一份 Key 和一份 Value。
- `num_kv_heads`：不是 query head 数；GQA/MQA 会显著减少 KV heads。
- `bytes_per_element`：由 KV Cache dtype 决定，不一定和量化权重格式相同。

活动请求的粗略总量再乘以它们当前缓存的 token 数。

这个公式不适合直接套到所有模型：

- TP 下 KV heads 可能被分片或复制，每卡大小不总是简单除以 TP 数。
- MLA 使用不同的潜在表示。
- 滑动窗口、local attention、Mamba/attention 混合模型有不同 cache spec。
- block 对齐、元数据和 backend workspace 也会产生额外开销。

因此公式用于建立量级直觉；实际容量以 vLLM 启动日志中的 `GPU KV cache size` 和 `Maximum concurrency` 为准。

## 3. 连续大块分配有什么问题？

PagedAttention 论文讨论的朴素基线，是按请求最大长度预留连续 KV 空间。假设请求最多 8192 token，但实际只使用 300 token，会产生大量预留浪费；变长请求反复进入和退出还会造成外部碎片。

这里要保持表述边界：这是解释 paged 设计动机的对照模型，不代表所有非 vLLM 引擎今天都仍采用同一种朴素实现。

分页后仍有内部浪费：一个序列的最后一个未填满 block 会留下空 slot。但浪费被限制在 block 粒度，而不是整个最大序列长度。

## 4. 两层“分配”不要混淆

### 4.1 启动阶段：建立物理 KV Cache

Worker 加载模型后会做内存 profiling，依据可用预算计算 KV Cache 配置，并在 GPU 上创建 KV Cache tensors。`gpu_memory_utilization` 或显式 `kv_cache_memory_bytes` 会影响这一步。

这是一块长期存在的设备内存池。

### 4.2 调度阶段：分配 block 所有权

请求进入 Engine Core 时，Scheduler 还没有立即为它占满最大上下文。下一次调度会：

1. 查找可复用的完整前缀 blocks。
2. 计算本轮要处理多少新 token。
3. 向 `KVCacheManager.allocate_slots()` 申请所需 block IDs。
4. 把 block IDs 发送给 Worker/Model Runner。

所以“按需分配”指从已建立的 pool 中取得 block，并更新引用计数和请求映射；不是每个 token 都触发 `cudaMalloc`。

## 5. Logical block、physical block 与 slot

设 block size 为 4 个 token，一个请求有 10 个需要缓存的 token：

```text
logical blocks:  L0=[t0..t3]  L1=[t4..t7]  L2=[t8..t9]
block table:     L0 -> P7     L1 -> P2     L2 -> P9
```

物理上：

```text
P0  P1  P2:L1  P3  P4  P5  P6  P7:L0  P8  P9:L2
```

逻辑序列连续，但 physical block IDs 可以离散。对某个 token position，Model Runner 根据 block table 和 block size 得到：

```text
logical_block_index = position // block_size
offset_in_block      = position % block_size
physical_block_id    = block_table[logical_block_index]
slot                 = physical_block_id * block_size + offset_in_block
```

真实 v0.25.0 还要处理多个 KV cache groups、混合 attention、speculative slots 等情况；上式只是标准单组 attention 的核心映射。

## 6. 为什么分页能改善容量利用率？

### 6.1 请求按增长分配

请求只持有当前计算和必要 lookahead 所需的 blocks。随着序列增长再获得新 block；结束或抢占后释放所有权。

### 6.2 固定规格便于池化复用

请求不再要求一段足以容纳最大上下文的连续物理区域。任何可回收的合适 block ID 都能重新分配，显著降低外部碎片问题。

### 6.3 block table 解耦逻辑顺序与物理地址

Scheduler 管理的是请求到 block IDs 的账本；Worker 持有真正的 KV tensors。Attention backend 通过 block table/slot mapping 访问正确位置，不要求一个请求的 K/V 物理连续。

## 7. V1 Automatic Prefix Caching 怎么工作？

APC 与 paged layout 相关，但不是同一个概念：

- Paged KV Cache 解决 block 化存储与寻址。
- APC 决定哪些已经计算的完整 blocks 可以跨请求复用。

v0.25.0 V1 使用链式 hash。一个 block 的身份包含：

```text
parent block hash
+ current block token IDs
+ extra hashes（例如 LoRA、multi-modal input、cache_salt）
```

父 hash 使“相同 block token、不同历史前缀”不会被当成同一份 KV。官方实现只缓存完整 block，因为部分 block 还没有稳定的完整 token 内容。

新请求到达时，`get_computed_blocks()` 查找最长完整 block 前缀。即使整个 prompt 都命中，v0.25.0 仍会保留最后一个 token 重新计算以得到 logits；受 block 对齐限制，实际重算有时会覆盖一个完整 block。

### 7.1 ref count、free queue 与 cache mapping

`KVCacheBlock` 记录 `block_id`、`block_hash` 和 `ref_cnt`。一个 block 可以同时被多个请求引用。

请求结束时：

- 请求到 blocks 的映射被移除；
- blocks 的引用计数下降；
- 引用为 0 的 block 回到 free queue；
- 已缓存的完整 block hash 映射可以继续保留，直到该 block 真正被重新分配时按 LRU 规则驱逐。

因此“free”不等于立即擦除所有 prefix cache 元数据。一个 `ref_cnt=0` 的缓存 block 既可被相同前缀再次命中，也可在容量不足时被驱逐并复用。

### 7.2 为什么这里不需要泛化成写时复制？

对 APC，共享的是已经计算完成的前缀 full blocks。新请求的不同后缀会分配新 blocks，不会回头修改共享前缀，因此不能简单描述成“共享后写入就 copy-on-write”。

PagedAttention 原论文还讨论 parallel sampling/beam search 的共享，这属于更广的历史设计背景；阅读 v0.25.0 V1 APC 时，应以 hash、ref count、free queue 和 eviction 这条实现链为准。

## 8. 一次请求的 block 生命周期

### 8.1 请求到达

`EngineCoreRequest` 转成内部 `Request`，进入 Scheduler `waiting` 队列。此时不会按 `max_model_len` 预占整条序列的 blocks。

### 8.2 第一次调度

Scheduler 查 prefix hit、计算新 token 数、调用 `allocate_slots()`。容量不足时，新请求可以继续等待；运行中请求也可能因 KV Cache 不足而被抢占并在之后重算。

### 8.3 Worker 执行

Model Runner 把请求 block IDs 写入 persistent batch 的 block table，并为本轮 token 生成 slot mapping。模型 forward 将新 K/V 写入对应 slots，attention backend 从 paged KV 中读取历史。

### 8.4 继续 decode

新采样 token 加入请求，但要在下一轮作为输入，才产生自己的 KV。序列跨过 block 边界时，Scheduler 再申请 block。

### 8.5 完成或 abort

Scheduler 释放请求持有的 blocks。可缓存 full blocks 的数据可能留在 pool 中等待复用；未缓存或被驱逐的 blocks 可直接承担新请求。

## 9. APC 的收益边界

适合：

- 固定 system prompt；
- 相同 few-shot 示例；
- 多个问题共享完全相同的长文档前缀；
- 多轮请求重复提交同一段历史前缀。

不适合或收益有限：

- 相同文本不在前缀；
- 前缀很短，省下的计算小于管理开销；
- 大量唯一 prompt，没有重复；
- 主要瓶颈是长输出 decode。

APC 不改变模型输出，但多租户场景要考虑缓存侧信道。v0.25.0 支持 request `cache_salt`，只有相同 salt 的请求才能复用对应前缀，可用于划分信任域。

## 10. PagedAttention 解决什么，不解决什么？

它帮助解决：

- 变长请求 KV Cache 的 block 化管理；
- 逻辑连续、物理离散的 KV 寻址；
- continuous batching 下的动态分配与回收；
- APC 所需的 block 级共享基础。

它不直接解决：

- 模型权重显存；
- 所有 attention 计算量；
- tokenizer、网络与业务排队；
- 多卡通信；
- 错误的容量配置或无限制的外部流量。

## 11. 本篇自检题

1. 为什么 KV Cache 公式使用 `num_kv_heads` 而不是固定使用 attention heads？
2. “KV Cache pool 已预分配”和“请求 blocks 按需分配”为什么不矛盾？
3. block table 如何把 token position 映射到 slot？
4. APC 为什么只命中完整 block，为什么 hash 中要包含 parent hash？
5. 请求结束后，`ref_cnt=0` 的缓存 block 为什么还可能保留 hash？

## 12. 源码核对入口

- `vllm/config/cache.py`：`CacheConfig`、默认 block size、KV dtype 与显存预算。
- `vllm/v1/core/kv_cache_utils.py`：`KVCacheBlock`。
- `vllm/v1/core/block_pool.py`：block pool、free queue、hash mapping 与 eviction。
- `vllm/v1/core/kv_cache_manager.py`：`get_computed_blocks()`、`allocate_slots()`、`free()`。
- `vllm/v1/core/single_type_kv_cache_manager.py`：按 cache spec 的 block 管理。
- `vllm/v1/core/sched/scheduler.py`：Scheduler 如何查询缓存并申请 slots。
- `vllm/v1/worker/gpu_model_runner.py`：标准 V1 Model Runner 的 block table 与 slot mapping。
- `vllm/v1/attention/`：当前各 attention backend 对 paged KV 的接入。

## 参考资料

- [vLLM v0.25.0 Automatic Prefix Caching 设计](https://docs.vllm.ai/en/v0.25.0/design/prefix_caching/)
- [vLLM v0.25.0 Paged Attention 历史设计页](https://docs.vllm.ai/en/v0.25.0/design/paged_attention/)
- [vLLM v0.25.0 Engine Arguments](https://docs.vllm.ai/en/v0.25.0/configuration/engine_args/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
