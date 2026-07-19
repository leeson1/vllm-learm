# 06｜Scheduler：v0.25.0 的 continuous batching 与 chunked prefill

> 版本基线：vLLM v0.25.0，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。本文讨论 V1 Engine 的默认 Scheduler；具体默认值仍会根据模型、设备和 usage context 解析，实验时必须保存最终启动日志。

## 1. 先给结论

v0.25.0 的 Scheduler 不是“prefill 队列加 decode 队列”，而是一个统一的、受三类预算约束的 token 调度器：

```text
请求状态：waiting / running
计算预算：由 max_num_batched_tokens 解析出的有效 token budget
序列预算：max_num_seqs
显存预算：KV Cache 可分配 blocks
```

对每个请求，Scheduler 关心的核心差值是：

```text
本轮仍需计算的 token
  = request.num_tokens_with_spec
  - request.num_computed_tokens
```

这个差值可以表示首次 prefill、chunked prefill、普通 decode，也可以包含 speculative tokens。prefill/decode 是不同执行形态，但不是两套永久隔离的引擎。

## 2. 一轮 `schedule()` 做什么？

`vllm/v1/core/sched/scheduler.py:Scheduler.schedule()` 的主线可以压缩成：

```text
1. 建立本轮 token budget 和 encoder budget
2. 先尝试调度 running 请求
3. 为请求计算 num_new_tokens，并受 token budget 限制
4. 申请本轮需要的 KV slots
5. slots 不足时，必要情况下抢占较低优先级请求
6. 再尝试从 waiting 队列准入新请求
7. 对新请求查询 prefix cache 命中并申请 blocks
8. 生成 SchedulerOutput 交给 Model Executor
```

从请求状态看，一轮调度可以画成：

```text
  新请求 ──> WAITING ── KV 足够、预算允许 ──> RUNNING
                ▲                              │
                │                              ▼
                │                     schedule → execute → update
                │                              │
                │                 ┌────────────┴───────────┐
                │                 │                        │
                │              未完成                    已完成
                │                 │                        │
                │                 └──> RUNNING          FINISHED
                │
                └── PREEMPTED <── KV 不足，释放 ownership ── RUNNING

  每一轮的三道门：token budget → sequence budget → KV block budget
```

默认 FCFS 策略下，running 请求通常是正在 decode 或继续 chunked prefill 的请求。先处理 running 请求会自然保护正在生成的流，但源码仍以统一的 token 差值和预算组织它们。

`--scheduling-policy` 还支持 priority。使用 priority 时必须同时考虑优先级和到达时间，不能把它理解成业务层完整的租户公平性方案。

## 3. 三种预算不要混淆

### 3.1 Token budget

`--max-num-batched-tokens` 是用户可见的批次 token 上限。Scheduler 实际使用内部 `max_num_scheduled_tokens` 作为本轮 budget；它通常等于前者，但在 async/speculative 等配置下可能为输出 placeholders 预留空间而更小。它约束的是本轮计算形状，不是 KV Cache 总容量，也不是 HTTP 总并发。

```bash
vllm serve <model> --max-num-batched-tokens 8192
```

调大后，一轮可能容纳更多 prefill token 或更多请求，吞吐可能上升；但更重的混合 batch 也可能影响 decode ITL。结论必须通过 workload 验证。

### 3.2 Sequence budget

`--max-num-seqs` 限制一轮中最多有多少运行序列：

```bash
vllm serve <model> --max-num-seqs 128
```

它不是业务侧的连接数限制。超过本轮运行容量的请求仍可能留在 Engine waiting queue，API 层也可能已经接收更多连接。

### 3.3 KV block budget

Scheduler 在 `allocate_slots()` 返回 `None` 时知道 KV 空间不足。对 waiting 请求，通常是不准入；对已经 running 的请求，可能需要抢占其他 running 请求后重试。

所以即使 token budget 还有剩余，也不代表请求一定能执行：

```text
有计算预算 + 没有 KV blocks = 仍然不能 schedule
```

## 4. v0.25.0 的 chunked prefill

V1 中，只要模型和功能组合支持，chunked prefill 默认启用。`EngineArgs._set_default_chunked_prefill_and_prefix_caching_args()` 会根据模型能力解析最终配置，不能只背 `SchedulerConfig` 类里的字段默认值。

开启后，长 prompt 可以被拆到多轮 token budget 中：

```text
prompt 12000 tokens, max_num_batched_tokens 4096

step 1: prefill 4096
step 2: decode requests + next prefill chunk
step 3: decode requests + remaining prefill
```

它的主要价值是让长 prefill 不必独占一整轮，同时允许 compute-heavy prefill 与 memory-heavy decode 混排。

可显式控制：

```bash
vllm serve <model> --enable-chunked-prefill
vllm serve <model> --no-enable-chunked-prefill
```

调优方向不是固定单调关系：

- 较小 token budget 往往更保护 decode ITL，但可能增加长 prompt 的完成轮数。
- 较大 token budget 往往改善 prefill 效率和 TTFT，但可能形成更重的混合 batch。
- 关闭 chunked prefill 时，`max_num_batched_tokens` 必须足以容纳允许的长请求；不要沿用开启时的小预算配置。

## 5. 抢占在 v0.25.0 中意味着什么？

当 running 请求需要新 KV slots、但 block 不足时，默认 Scheduler 可以抢占请求：

```text
running -> PREEMPTED -> waiting
```

`_preempt_request()` 会释放该请求持有的 KV blocks，把 `num_computed_tokens` 重置为 0，并把请求放回 waiting queue。恢复时需要重新计算，因此 v0.25.0 V1 的核心代价是 recomputation，而不是免费的暂停/恢复。

频繁 preemption 通常说明容量或准入配置失衡。优先检查：

- KV Cache 总容量和 `gpu_memory_utilization`；
- `max_num_seqs` 是否过大；
- 输入/输出长度是否失控；
- 外层 arrival rate 是否已经超过服务能力；
- TP/PP 改变后权重和通信成本是否值得。

## 6. 默认 async scheduling 怎样改变观察方式？

v0.25.0 在配置兼容时默认启用 async scheduling。Engine Core 会通过 batch queue 让多个 batch in flight，重叠 CPU scheduling/input preparation 与 GPU execution。

逻辑依赖仍是：

```text
schedule(batch N)
  -> execute(batch N)
  -> update(batch N)
```

但时间线上可能看到：

```text
GPU execute N
与 CPU schedule / prepare N+1 重叠
```

为了第一次读源码，可以先用同步基线：

```bash
vllm serve <model> --no-async-scheduling
```

理解后再去掉该参数，比较默认路径。不要把同步教学路径误认为 v0.25.0 的默认运行时。

## 7. 最小实验

准备两个请求：

```text
A: prompt 128，output 256，先到达
B: prompt 8192，output 16，稍后到达
```

分别使用两组 `max_num_batched_tokens`，其他条件不变，记录：

- A 的 ITL P50/P99；
- B 的 TTFT；
- 每轮 scheduled tokens；
- waiting/running 数量；
- KV Cache 使用率与 preemption；
- GPU 时间线是否出现空洞。

验收不是“哪组更快”，而是能根据每轮 token 组成解释 A、B 指标为什么变化。

## 8. 源码核对入口

- `vllm/v1/core/sched/scheduler.py`：`schedule()`、`_preempt_request()`、waiting/running。
- `vllm/v1/core/sched/output.py`：`SchedulerOutput`。
- `vllm/v1/core/sched/async_scheduler.py`：异步调度下的状态更新。
- `vllm/v1/core/kv_cache_manager.py`：`get_computed_blocks()`、`allocate_slots()`。
- `vllm/config/scheduler.py`：Scheduler 配置结构和约束。
- `vllm/engine/arg_utils.py`：模型/设备相关默认值的最终解析。
- `vllm/config/vllm.py`：有效 scheduled-token budget、async scheduling 的兼容性与默认开启逻辑。
- `vllm/v1/engine/core.py`：同步 step 与 batch queue 路径。

## 9. 自检题

1. 为什么 `max_num_batched_tokens` 有余量时，请求仍可能无法运行？
2. chunked prefill 为什么可能改善 decode ITL？
3. v0.25.0 为什么不能简单描述成两个独立的 prefill/decode 队列？
4. preemption 后为什么需要 recompute？
5. async scheduling 改变了哪个时间关系，又没有改变哪个逻辑依赖？

## 参考资料

- [vLLM v0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/)
- [vLLM v0.25.0 Engine Arguments](https://docs.vllm.ai/en/v0.25.0/configuration/engine_args/)
- [vLLM v0.25.0 Architecture Overview](https://docs.vllm.ai/en/v0.25.0/design/arch_overview/)
