# 07｜KV Cache Manager：v0.25.0 的 block 分配、缓存与抢占

> 版本基线：vLLM v0.25.0，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。本文只把旧版 PagedAttention/Block Manager 设计作为历史背景，主线以 V1 `KVCacheManager`、`BlockPool` 和 `SingleTypeKVCacheManager` 的实际实现为准。

## 1. 先修正术语

在 v0.25.0 V1 中，核心类不是一个统一的 `BlockManager`，而是：

```text
KVCacheManager
  -> KVCacheCoordinator
      -> SingleTypeKVCacheManager（按 cache spec 管请求 blocks）
  -> BlockPool（block 对象、free queue、hash mapping）
```

Scheduler 通过 `KVCacheManager` 查询已计算前缀、申请 slots、缓存完整 blocks 和释放请求。不同 attention 类型、混合 KV Cache、KV connector 等情况会通过 coordinator 和具体 manager 扩展。

整体关系如下：

```text
CPU 控制面

  Scheduler
      │ get_computed_blocks / allocate_slots / free
      ▼
  KVCacheManager
      ├── BlockPool ─────────────── free queue / hash / ref_cnt
      └── KVCacheCoordinator
              └── SingleTypeKVCacheManager ── request -> block IDs
                                                     │
                                                     │ block IDs
                                                     ▼
Worker 执行面                                Block Table / Slot Mapping
                                                     │
                                                     ▼
GPU 数据面                                     预分配 KV Tensor Pool
```

## 2. 两层“分配”必须分开

### 2.1 启动阶段：建立真实 KV tensor pool

Worker 在模型加载和显存 profiling 后确定 KV Cache 配置，并在 GPU 上创建真实 KV tensors。它不是每个请求到来后才调用一次大块 `cudaMalloc`。

### 2.2 请求阶段：分配 CPU 侧 block ownership

请求执行期间，Scheduler 分配的是 block IDs/slots：

```text
request logical block 0 -> physical block 91
request logical block 1 -> physical block 7
request logical block 2 -> physical block 143
```

Worker 随后把 block table 和写入位置转换为 attention backend 使用的元数据。

因此以下两句话可以同时成立：

- KV tensor pool 在启动时预分配。
- 请求需要的 KV blocks 在运行时按增长分配。

## 3. v0.25.0 的核心数据结构

### 3.1 `KVCacheBlock`

`vllm/v1/core/kv_cache_utils.py` 中的 block 记录：

- 不变的 `block_id`；
- 完整 block 可拥有的 `block_hash`；
- 当前引用数 `ref_cnt`；
- free queue 的前后指针。

它是 CPU 侧账本对象，不等于某次请求新申请的一段 GPU 内存。

### 3.2 `BlockPool`

`BlockPool` 维护：

```text
所有 KVCacheBlock 对象
free block queue
block_hash -> block IDs 的映射
```

free queue 同时承担两种角色：

- `ref_cnt == 0` 的 block 可以重新分配；
- 如果它仍保留完整 block 的 hash，也可以在被驱逐前再次命中 prefix cache。

所以“free”不等于立即清空缓存内容。

### 3.3 请求到 blocks 的映射

具体的 single-type manager 保存每个 request 的 block 列表。Scheduler 看到的是请求逻辑顺序；Worker/attention backend 通过 block table 找到物理页。

## 4. 新请求如何获得 KV slots？

主线是：

```text
1. 对 prompt 的完整 token blocks 计算链式 hash
2. get_computed_blocks() 查最长可复用前缀
3. allocate_slots() 计算还需多少 blocks
4. touch 命中的 blocks：增加 ref_cnt，必要时移出 free queue
5. 从 free queue 取新 blocks
6. 返回本轮新增 block IDs 给 Scheduler/Worker
```

如果空间不足，`allocate_slots()` 返回 `None`，由 Scheduler 决定等待或抢占。

即使 prompt 全部命中，也不能凭 KV 直接得到“下一个 token”的 logits。v0.25.0 会至少保留最后一个 token 重新计算；因为 cache 只按完整 block 命中，实际重算可能覆盖最后一个完整 block。

## 5. Automatic Prefix Caching

v0.25.0 使用链式 block hash。身份包含：

```text
parent block hash
+ current block token IDs
+ extra keys（LoRA、multi-modal hash、cache_salt 等）
```

只缓存 full blocks。partial block 的 token 内容仍会增长，不进入稳定的可复用 hash 边界。

请求结束时：

```text
移除 request -> blocks 映射
按反向顺序降低 blocks 的 ref_cnt
ref_cnt 变为 0 的 block 回到 free queue
完整 block 的 hash 可继续保留
真正重新分配该 block 时才执行 eviction/覆盖
```

一个 full block 的典型生命周期是：

```text
 free, no hash
      │ allocate
      ▼
 active, ref_cnt=1 ── block 写满 ──> active + cached hash
      │ request finish                         │ prefix hit
      ▼                                        ▼
 free queue, ref_cnt=0, hash 保留 <────── shared, ref_cnt>0
      │
      ├── 再次命中：touch，移出 free queue
      └── 容量需要：evict hash，重新分配并覆盖
```

反向释放使请求尾部、复用概率通常更低的 blocks 更早进入可驱逐位置。

## 6. 为什么 APC 主线不是 Copy-on-Write？

v0.25.0 V1 APC 共享的是已经计算完成的 full prefix blocks。新请求拥有不同后缀时，会为后缀分配新 blocks，而不是回头修改共享前缀。

```text
request A: [shared full block 0][new block A]
request B: [shared full block 0][new block B]
```

因此不应把当前 APC 实现描述为：

```text
ref_cnt > 1 时修改共享页 -> 拷贝旧页 -> 写新页
```

PagedAttention 原论文讨论过 parallel sampling/beam search 的共享与写时复制，这是重要历史设计，但不是解释 v0.25.0 V1 APC 的准确主线。当前应重点理解 hash、ref count、free queue、touch、eviction 和 append-only request block table。

## 7. 分配到重复 hash 的 block 怎么办？

并发请求可能各自计算出内容相同的完整 block。V1 的请求 block table 是 append-only：已经给某请求追加的新 block 不会为了去重而替换成另一物理 block。因此同一 hash 可能暂时对应多个 block IDs。

这不是 COW。它是并发计算与 append-only block table 下允许的重复缓存；请求释放后，冗余 block 会自然回到 pool。

## 8. KV 不足与 preemption

运行中请求继续增长时，需要新的 KV block。如果申请失败，Scheduler 可能抢占一个 running 请求：

```text
释放被抢占请求的 KV ownership
num_computed_tokens = 0
状态改为 PREEMPTED
放回 waiting queue
恢复后 recompute
```

这说明 preemption 是容量压力信号，而不是免费的调度优化。需要同时观察 `vllm:kv_cache_usage_perc`、`vllm:num_preemptions`、waiting queue 和尾延迟。

## 9. 不要把所有模型都套进标准公式

标准 decoder-only full attention 的基本容量公式很有用：

```text
bytes/token/GPU
  = 2(K,V)
  × local_attention_layers
  × local_kv_heads
  × head_dim
  × dtype_bytes
```

但 v0.25.0 还支持 MLA、sliding-window attention、Mamba、hybrid KV cache、不同 page size、KV quantization 和 KV connectors。真实容量应以启动时解析出的 KV cache specs 和日志为准。

## 10. 最小实验

构造三次请求：

```text
R1 = 4096 token 固定前缀 + 问题 A
R2 = 同一前缀 + 问题 B
R3 = 只改前缀中间一个 token + 问题 C
```

记录：

- 每次 `prompt_tokens_cached`；
- R1 完成前后 KV Cache usage；
- R2/R3 的 TTFT；
- block size 对可命中 token 数的影响；
- 使用不同 `cache_salt` 后是否还能复用。

验收目标是能解释“请求结束后 block 为什么既 free 又 cached”。

## 11. 源码核对入口

- `vllm/v1/core/kv_cache_utils.py`：`KVCacheBlock` 与 block hash。
- `vllm/v1/core/block_pool.py`：free queue、touch、allocate、free、evict。
- `vllm/v1/core/kv_cache_manager.py`：命中查询、slots 分配和释放入口。
- `vllm/v1/core/kv_cache_coordinator.py`：不同 cache manager 的协调层。
- `vllm/v1/core/single_type_kv_cache_manager.py`：请求 block table、cache full blocks。
- `vllm/v1/core/sched/scheduler.py`：KV 申请失败后的等待和抢占。
- `vllm/v1/kv_cache_interface.py`：KV cache spec 与 page 字节计算。
- `vllm/v1/worker/block_table.py`：Worker 侧 block table 与 slot mapping。

## 12. 自检题

1. GPU KV tensor pool 与 CPU `BlockPool` 分别保存什么？
2. 为什么 `ref_cnt=0` 的 cached block 仍可命中？
3. APC 为什么只复用完整 blocks？
4. 为什么 v0.25.0 APC 不需要把共享后缀解释成 COW？
5. preemption 后为何把 `num_computed_tokens` 重置为 0？

## 参考资料

- [vLLM v0.25.0 Prefix Caching Design](https://docs.vllm.ai/en/v0.25.0/design/prefix_caching/)
- [vLLM v0.25.0 Automatic Prefix Caching](https://docs.vllm.ai/en/v0.25.0/features/automatic_prefix_caching/)
- [vLLM v0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
