# 05｜工程调优地图：吞吐、延迟、显存与常见参数

> 版本基线：vLLM v0.25.0。性能结论必须绑定模型、硬件、输入/输出长度分布和到达率；本文给的是实验地图，不是万能参数表。

调优的基本方法是：

```text
固定 workload -> 记录基线 -> 找到资源瓶颈 -> 一次改一个变量 -> 比较完整指标
```

## 1. 先把指标定义清楚

### 1.1 TTFT（Time To First Token）

从请求到达服务端到第一个输出 token 可用的时间。它通常包含：

```text
排队 + 输入处理 + 未缓存前缀的 prefill + 首 token 采样 + 输出交付
```

所以“TTFT 主要受 prefill 影响”只是无排队时的近似。高负载下，queue time 可能更重要。

### 1.2 ITL（Inter-Token Latency）

相邻输出 token 到达之间的延迟。它能展示 decode 过程中是否抖动，尤其适合 streaming 体验。

### 1.3 TPOT（Time Per Output Token）

请求级 decode 平均 token 时间。v0.25.0 的服务端统计口径是：

```text
TPOT = (最后一个 token 时间 - 第一个 token 时间) / (输出 token 数 - 1)
```

因此不把 prefill 阶段产生的首 token 计入分母；只有一个输出 token 时记录为 0。TPOT 是请求内平均值，ITL 分布能保留更多尾延迟信息；跨系统对比时仍应注明具体公式。

### 1.4 E2E latency

请求从到达到完成的总时间。它同时受 TTFT、输出长度、decode 速度和排队影响。

### 1.5 Throughput

至少区分：

```text
requests/s
input tokens/s
output tokens/s
total tokens/s
```

两个系统的 total tokens/s 相同，也可能一个偏 prefill、一个偏 decode；只给单一 tokens/s 没法判断是否适合业务。

### 1.6 Tail latency 与 goodput

线上应看 P95/P99 TTFT、ITL、E2E。还可以定义满足 SLA 的 goodput，例如“TTFT < 1 s 且 ITL < 50 ms 的 requests/s”。平均值不能描述超时和长尾。

## 2. 三类资源与两个队列

### 2.1 GPU 显存

主要包括：

```text
模型权重
KV Cache pool
激活 / workspace / 通信 buffer
torch.compile / CUDA Graph / 框架开销
```

权重决定模型能否部署；KV Cache 容量决定能同时保留多少活动 token。vLLM 启动日志会打印 `GPU KV cache size` 和按 `max_model_len` 估算的 `Maximum concurrency`，这比只看 `nvidia-smi` 更有解释力。

### 2.2 GPU 计算与显存带宽

Prefill 通常更容易形成 compute-bound 的大矩阵；标准 decode 每序列每轮 token 少，要读取权重和历史 KV，常更依赖 batch、带宽与 kernel 效率。这是常见规律，不是所有模型/backend 的绝对分类。

### 2.3 CPU、内存与网络

API Server、Renderer、tokenizer、detokenizer、多模态加载、ZMQ、SSE、metrics 都消耗 CPU。v0.25.0 的 DP 默认会增加 API Server 与 Engine Core 数，CPU 和主存必须随进程拓扑一起规划。

### 2.4 外部队列与 Engine waiting queue

业务网关的限流/排队与 vLLM Scheduler 的 `waiting` 不是一回事。无限把请求送入 vLLM，只会把压力变成 queue time 和尾延迟。生产系统通常需要在外层按并发或 token budget 做背压。

## 3. v0.25.0 的版本敏感默认值

开始实验前先知道这些默认行为：

- `gpu_memory_utilization` 默认 `0.92`。
- 对支持的模型，prefix caching 默认启用；不要把 `--enable-prefix-caching` 当成每次都必须添加的优化开关。
- V1 在模型支持时默认启用 chunked prefill。
- 配置兼容时，async scheduling 默认启用；可用 `--no-async-scheduling` 建立同步调试基线。
- optimization level 默认 `-O2`。
- `max_num_batched_tokens` 与 `max_num_seqs` 的最终默认值会随使用入口、设备内存/型号和 world size 解析，不要抄一个常数覆盖所有机器。

应从启动日志或实际 `VllmConfig` 确认最终值。版本升级后重新核对，不能沿用本文默认值。

## 4. 容量参数：它们各自在控制什么？

### 4.1 `--gpu-memory-utilization`

```bash
vllm serve <model> --gpu-memory-utilization 0.92
```

它限制当前实例为 model executor 使用的 GPU 显存比例。未设置 `kv_cache_memory_bytes` 时，vLLM profiling 后用该预算推导 KV Cache 大小。

常见方向：

- 提高：通常能留下更多 KV Cache，减少容量型等待或 preemption。
- 降低：给同卡其他进程留空间，但 KV 容量下降。

它不会让多个同卡实例自动协调；每个实例看到的是自己的比例。值越高也不是无条件更好，必须给实际共存进程和峰值分配留出安全空间。

### 4.2 `--kv-cache-memory-bytes`

```bash
vllm serve <model> --kv-cache-memory-bytes 8G
```

显式指定每张 GPU 的 KV Cache 字节数，并覆盖 `gpu_memory_utilization` 对 KV Cache 容量的推导。适合做精确容量实验，但错误估算可能导致启动失败或浪费。

### 4.3 `--max-model-len`

```bash
vllm serve <model> --max-model-len 8192
```

它限制 prompt + generation 的最大序列长度。它不是每请求预留量，也不直接决定整个 KV pool 大小。

降低它的主要作用：

- 拒绝业务不需要的超长请求；
- 确保至少能容纳目标序列；
- 改变“按最大长度估算的并发数”口径。

对真实并发，应该用业务长度分布计算活动 token，而不是只用 `KV cache tokens / max_model_len`。

### 4.4 `--kv-cache-dtype`

KV Cache 可以独立选择 dtype。低精度 KV 能增加可缓存 token 数，但支持范围、精度影响和 kernel 性能依赖硬件/backend/模型。权重量化不会自动把 KV Cache 一起量化。

## 5. 调度参数：吞吐和延迟如何交换？

### 5.1 `--max-num-batched-tokens`

它限制单次迭代最多处理的 token 数。v0.25.0 V1 chunked prefill 会先安排 decode，再把剩余 token budget 用于 prefill。

官方给出的常见方向：

- 较小：prefill 对 decode 的干扰减少，ITL 往往更好。
- 较大：一次能推进更多 prefill，TTFT 和吞吐通常更有机会改善。

这不是单调保证。过大的迭代会拉长单轮执行时间；模型、GPU 和到达率不同，拐点也不同。官方对“大 GPU 上的小模型吞吐”给过 `>8192` 的建议，但不能直接当作所有环境的最优值。

### 5.2 `--max-num-seqs`

它是单次迭代可处理序列数上限：

- 提高上限允许更大的 decode batch，但只有到达率与 KV 容量足够时才会用到。
- 降低上限可减少同轮并发和 KV 压力，官方也把它列为频繁 preemption 时的缓解手段。

设置更大不会立即为所有序列预留 KV，也不是业务层最大连接数。

### 5.3 Chunked prefill

支持时默认开启。它把长 prefill 切块，与 decode 混排，避免一个超长 prompt 独占整轮。重点观察：

- P99 ITL 是否因长 prompt 到达而恶化；
- 长 prompt TTFT 是否可接受；
- 不同 `max_num_batched_tokens` 下 GPU 利用率和吞吐。

### 5.4 Async scheduling

它用多个 in-flight batches 重叠 CPU scheduling 与 GPU execution，通常改善 GPU 空隙。为了读源码或定位时序问题，可关闭建立同步基线；不能因为同步路径更容易理解，就把它误写成 v0.25.0 默认性能路径。

## 6. Prefix caching：先测命中率，再谈收益

对支持的模型，v0.25.0 默认启用 APC。调优重点不是机械加 flag，而是验证 workload 是否有重复完整前缀。

观察：

```text
vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:prompt_tokens_cached
vllm:request_prefill_kv_computed_tokens
```

APC 只减少命中前缀的 prefill 计算：

- 固定长 system prompt/文档前缀：TTFT 和 input throughput 可能明显改善。
- 唯一 prompt：命中率低，收益有限。
- 长输出：decode 仍要逐步执行，ITL 不会因 APC 自动改善。

做 A/B 时可用 `--no-enable-prefix-caching` 关闭，并保持请求顺序、前缀内容和 cache warm-up 一致。

## 7. 多 GPU：TP、PP 与 DP 的目标不同

### 7.1 Tensor Parallel（TP）

将每层参数切到多 GPU。优先用于：

- 模型单卡放不下；
- 分摊每卡权重后需要更多 KV 空间。

代价是每层 collective communication。无 NVLink 时，PCIe 通信可能抵消收益；官方对某些不均匀或无 NVLink 场景建议评估 PP。

### 7.2 Pipeline Parallel（PP）

按层切分模型。适合模型跨节点、TP 已到高效上限，或模型结构更适合按层分割的情况。它会引入流水线调度与 latency trade-off。

### 7.3 Data Parallel（DP）

复制完整模型副本，让不同副本处理不同请求。模型已经能在一个 GPU/一组 GPU 上放下、目标是扩总吞吐时，DP 比单纯增大 TP 更符合扩容语义。

记住：

```text
TP / PP：让一个模型副本跨更多卡
DP：复制模型副本以并行服务不同请求
```

## 8. 权重 dtype、权重量化与 KV 量化

三者不要混为一个开关：

| 手段 | 主要改变 | 不自动保证 |
|---|---|---|
| `--dtype` | 权重/计算 dtype 选择 | KV 一定同格式、延迟一定下降 |
| `--quantization` | 权重表示与对应 kernels | KV Cache 一定变小 |
| `--kv-cache-dtype` | KV Cache 每元素字节数 | 精度无损、所有 backend 更快 |

量化收益依赖 checkpoint、硬件指令、kernel 和 batch shape。只报告显存下降而不报告 TTFT/ITL/吞吐和输出质量，不算完整结论。

## 9. Preemption 是容量告警，不是免费调度

KV Cache 不足时，V1 可以 preempt 请求释放 blocks。v0.25.0 默认模式是 `RECOMPUTE`，不是 `SWAP`：请求之后重新计算被释放的 KV。

这保证系统能继续推进，但增加计算与 E2E latency。监控：

```text
vllm:num_preemptions
vllm:kv_cache_usage_perc
vllm:num_requests_waiting
vllm:request_queue_time_seconds
```

频繁 preemption 时，官方建议方向包括：增加可用 KV 预算，降低 `max_num_seqs` 或 `max_num_batched_tokens`，或通过 TP/PP 分摊权重以释放每卡 KV 空间。每个方向都有吞吐、延迟或通信代价，必须复测。

## 10. 四类 workload 的观察重点

| Workload | 首要指标 | 容易暴露的瓶颈 | 优先实验 |
|---|---|---|---|
| 短输入 + 短输出 | E2E、requests/s、P99 | HTTP/tokenizer/scheduler 开销 | 并发、DP、CPU |
| 长输入 + 短输出 | TTFT、input tokens/s | prefill、queue、APC 命中 | prefix、chunked prefill、token budget |
| 短输入 + 长输出 | ITL/TPOT、output tokens/s | decode、KV 增长 | seq 上限、KV 容量、并行策略 |
| 长输入 + 长输出 | 全部尾延迟、preemption | 显存与排队共同饱和 | 容量隔离、限流、扩容 |

平均输入/输出长度不够。至少保留 P50/P95/P99 或真实分布；一个少量超长请求就可能改变调度行为。

## 11. 一组可复现的基线实验

### 11.1 启动服务

本地隔离环境可先不加 API key，避免 benchmark header 成为额外变量：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --generation-config vllm
```

记录完整启动日志、GPU 型号、驱动、vLLM 版本与最终解析配置。

### 11.2 固定长度和到达率

```bash
vllm bench serve \
  --backend openai \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --ignore-eos \
  --request-rate 4 \
  --num-prompts 100 \
  --save-result \
  --result-dir benchmark-results
```

先 warm up，再正式采样；小样本只用于验证流程，不用于稳定的 P99 结论。

### 11.3 一次只改一个变量

建议最小矩阵：

```text
baseline
max_num_batched_tokens: 2048 / 8192 / 16384
max_num_seqs:           64 / 256
prefix caching:         on / off（另做固定共享前缀 workload）
arrival rate:           1 / 4 / 16 / inf
```

不同参数启动出来的是不同 server run。每轮保存 command、日志和原始 JSON，不要只抄一行 tokens/s。

## 12. 最小观测面

v0.25.0 `/metrics` 至少关注：

```text
延迟：vllm:time_to_first_token_seconds
      vllm:inter_token_latency_seconds
      vllm:e2e_request_latency_seconds
      vllm:request_queue_time_seconds

队列：vllm:num_requests_running
      vllm:num_requests_waiting
      vllm:num_requests_waiting_by_reason

容量：vllm:kv_cache_usage_perc
      vllm:num_preemptions

吞吐：vllm:prompt_tokens
      vllm:generation_tokens

缓存：vllm:prefix_cache_queries
      vllm:prefix_cache_hits
      vllm:prompt_tokens_cached
```

再配合 GPU utilization/memory、CPU utilization、网络和错误率。Prometheus histogram 的 P95/P99 需要用 bucket 做 `histogram_quantile`，不能读取一个不存在的“p99 字段”。

## 13. 生产落地的三个边界

### 13.1 外层必须背压

按并发请求、预估输入 token、最大输出 token 或租户配额限流。vLLM 的 `max_num_seqs` 不是完整业务限流器。

### 13.2 性能日志不要记录敏感 prompt

记录长度、时间、finish reason、模型和必要采样参数即可。需要记录内容时必须经过数据治理，不能为了排障默认落盘全部用户输入。

### 13.3 不同 SLA 可以拆池

短请求低延迟、长上下文、批处理和不同模型不一定适合混在同一 Scheduler。拆池会损失部分共享与聚合机会，但能隔离尾延迟和容量风险。

## 14. 第一阶段总验收

完成前 5 篇后，应能交付一份小型实验报告，包含：

1. 固定版本、模型、GPU、启动命令和 workload。
2. TTFT、ITL/TPOT、E2E、input/output throughput 的平均与尾部指标。
3. KV Cache 使用率、waiting、preemption、GPU/CPU 利用率。
4. 一个单变量参数实验及对结果的资源解释。
5. 明确区分观测事实、源码事实与尚未验证的推测。

## 15. 源码核对入口

- `vllm/config/cache.py`：KV Cache 容量、dtype、prefix caching 默认配置。
- `vllm/config/scheduler.py`：token/seq budget、chunked prefill、async scheduling。
- `vllm/config/vllm.py`：async scheduling 与 Model Runner 的兼容性解析。
- `vllm/engine/arg_utils.py`：设备/入口相关的最终 batch 默认值。
- `vllm/v1/core/sched/scheduler.py`：预算实际如何消费、何时 preempt。
- `vllm/v1/metrics/`：Prometheus 指标定义与记录。
- `vllm/benchmarks/serve.py`：`vllm bench serve` 参数与统计口径。

## 参考资料

- [vLLM v0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/)
- [vLLM v0.25.0 Engine Arguments](https://docs.vllm.ai/en/v0.25.0/configuration/engine_args/)
- [vLLM v0.25.0 Production Metrics](https://docs.vllm.ai/en/v0.25.0/usage/metrics/)
- [vLLM v0.25.0 Parallelism and Scaling](https://docs.vllm.ai/en/v0.25.0/serving/parallelism_scaling/)
- [vLLM v0.25.0 Benchmarking](https://docs.vllm.ai/en/v0.25.0/benchmarking/)
