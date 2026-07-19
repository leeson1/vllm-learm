# 12｜如何压测 vLLM v0.25.0：从吞吐到 goodput

> 版本基线：vLLM v0.25.0 的 `vllm bench serve`，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。压测客户端必须与被测 server 使用同一版本或显式记录差异。

## 1. 压测首先是一份问题定义

开始前必须写清楚：

```text
目标：峰值吞吐、在线延迟、容量、回归，还是功能对比？
模型：checkpoint/revision、dtype、quantization、max_model_len
硬件：GPU/CPU/内存/驱动/CUDA/互联
Server：完整命令和最终解析日志
Workload：输入/输出长度分布、arrival rate、burstiness、并发上限
采样：temperature、ignore_eos、seed 等
SLO：TTFT/TPOT/E2E 的目标和统计口径
```

没有 workload 和 SLO，单独的 tokens/s 没有可迁移结论。

一轮可信的性能实验是闭环，不是一次命令：

```text
┌──────────────┐    requests     ┌──────────────┐
│ Workload 生成 │ ──────────────> │ vLLM Server  │
└──────┬───────┘                  └──────┬───────┘
       │ 客户端计时                         │ /metrics / GPU trace
       └────────────────┬──────────────────┘
                        ▼
                 原始 JSON + 指标快照
                        │
                        ▼
                 曲线 / 容量拐点 / 异常
                        │
                        ▼
                     瓶颈假设
                        │
                        ▼
                 只修改一个变量
                        │
                        └──────────> 回到 Workload
```

## 2. v0.25.0 的核心指标

### TTFT

请求开始到第一个输出 token。包含客户端/网络、API 输入处理、排队、prefill 和第一次输出交付。

### TPOT

通常按首 token 之后的生成耗时除以后续输出 token 数。它是请求级平均，不等于每个 token 间隔。

### ITL

相邻流式 token 到达间隔。P99 ITL 能暴露单个请求在 continuous batching、长 prefill 干扰或服务侧同步下的卡顿。

### E2E latency

从请求开始到完成。受输入、输出、排队和所有服务层开销共同影响。

### Throughput

至少分开记录：

- request throughput；
- input token throughput；
- output token throughput；
- total token throughput。

### Goodput

满足指定 TTFT/TPOT/E2E SLO 的完成请求数/秒。v0.25.0 `vllm bench serve --goodput` 可以按 SLO 计算，比“峰值吞吐但 P99 已不可用”更接近线上能力。

## 3. 两种流量模型

### 3.1 Open-loop arrival rate

`--request-rate` 控制请求到达率。有限值默认使用指数到达间隔，即 Poisson 流量；`--burstiness` 改变 Gamma 分布形状：

```text
burstiness < 1：更突发
burstiness = 1：Poisson
burstiness > 1：更均匀
```

当 server 跟不上时，waiting queue 会增长，能真实暴露排队和 tail latency。

### 3.2 最大吞吐/并发限制

`--request-rate inf` 会尽快发请求，可配合 `--max-concurrency` 测固定并发或最大吞吐。它不等于真实线上 arrival process。

报告必须说明使用哪一种，不能把固定并发和固定 QPS 的结果直接比较。

## 4. 四类基础 workload

| 类型 | 示例长度 | 首要指标 | 容易暴露 |
|---|---:|---|---|
| 短输入短输出 | 128 / 32 | E2E、req/s | API、scheduler、launch overhead |
| 长输入短输出 | 4096 / 32 | TTFT、input tok/s | prefill、chunking、APC |
| 短输入长输出 | 128 / 512 | TPOT/ITL、output tok/s | decode、KV 读取 |
| 长输入长输出 | 4096 / 512 | 全部尾延迟、容量 | KV、preemption、排队 |

随机定长适合控制变量，但不能代表真实流量。完成基础矩阵后，再加入真实长度分布、共享前缀和 burst trace。

## 5. 可复现的基线命令

启动服务：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --generation-config vllm
```

压测：

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
  --num-prompts 1000 \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,90,95,99 \
  --save-result \
  --save-detailed \
  --result-dir benchmark-results
```

`--ignore-eos` 用于固定输出长度实验，可能产生不自然文本；质量评测或真实业务回放不应机械开启。

先用几十请求确认流程，再用足够样本正式统计。100 个请求的 P99 实际只由极少数样本决定，不能当稳定容量结论。

## 6. 找到容量拐点

固定 workload，逐级增加 arrival rate：

```text
1 -> 2 -> 4 -> 8 -> 16 -> 32 req/s
```

每档观察：

- throughput 是否继续近似线性增长；
- TTFT/queue time 是否开始陡升；
- TPOT/ITL 是否恶化；
- waiting 是否持续累积；
- KV usage 和 preemption；
- 错误率和客户端实际发出速率。

典型饱和信号：吞吐趋于平台，而排队和 P99 快速上升。生产容量通常要低于这个拐点并留出 burst/故障余量。

## 7. 参数实验一次只改一个变量

推荐矩阵：

```text
max_num_batched_tokens: 2048 / 8192 / 16384
max_num_seqs:           64 / 128 / 256
prefix caching:         on / off（只用 shared-prefix workload）
KV dtype:               auto / fp8（同时做质量验证）
runner/scheduling:       教学 V1 sync / 默认路径
```

每个 server run 保存：

- 完整启动命令；
- vLLM commit/version；
- 启动日志与最终配置；
- 原始 benchmark JSON；
- `/metrics` 快照；
- GPU/CPU 监控；
- workload seed 和数据摘要。

否则“配置 A 比 B 快”无法复现。

## 8. 专门测试 Prefix Caching

v0.25.0 提供 synthetic prefix repetition dataset：

```bash
vllm bench serve \
  --backend openai \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name prefix_repetition \
  --num-prompts 1000 \
  --prefix-repetition-prefix-len 2048 \
  --prefix-repetition-suffix-len 128 \
  --prefix-repetition-num-prefixes 5 \
  --prefix-repetition-output-len 32 \
  --request-rate 4 \
  --save-result
```

APC on/off 各跑一轮，并增加一个长度相同的 random dataset 作为负对照。记录 `prompt_tokens_cached`、prefix queries/hits、TTFT 和 input throughput。

## 9. 服务端观测面

`/metrics` 至少采集：

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:num_requests_waiting_by_reason
vllm:kv_cache_usage_perc
vllm:num_preemptions
vllm:iteration_tokens_total

vllm:time_to_first_token_seconds
vllm:inter_token_latency_seconds
vllm:request_time_per_output_token_seconds
vllm:e2e_request_latency_seconds
vllm:request_queue_time_seconds
vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds

vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:prompt_tokens_cached
```

Prometheus Counter 导出时可能带 `_total` 后缀；应以 `/metrics` 实际暴露名称和 HELP/TYPE 为准。Histogram 的 P95/P99 要从 buckets 计算，不存在一个通用的直接 `p99` 字段。

## 10. Profiler 与 benchmark 要分开

PyTorch Profiler、逐 token 日志和详细 shape/stack 收集都会改变性能。先用普通指标定位，再用少量请求做 profiler 验证；不要把 profiling run 的延迟当基线。

v0.25.0 可用：

```bash
vllm serve <model> \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"./profile"}'

vllm bench serve ... --profile --num-prompts 2
```

GPU 时间线优先使用 Nsight Systems；只有确认某个 kernel 是热点后，再用 Nsight Compute 深挖。

## 11. 压测报告模板

```text
1. 问题与 SLO
2. 硬件、软件和完整命令
3. Workload/arrival process/样本数
4. 控制变量和实验矩阵
5. TTFT/TPOT/ITL/E2E/throughput/goodput
6. queue/KV/preemption/GPU/CPU 证据
7. 拐点和异常请求分析
8. 因果解释与尚未验证的假设
9. 推荐配置及安全余量
10. 原始结果路径和复现命令
```

## 12. 验收标准

完成以下产物：

1. 四类基础 workload 的原始 JSON；
2. arrival rate 增长时吞吐与 P99 曲线；
3. 至少一个单变量参数实验；
4. 一个 shared-prefix 正/负对照；
5. 一次“客户端瓶颈不是 server 瓶颈”的排除证据；
6. 一页明确区分事实、推断和下一步验证的结论。

## 13. 源码核对入口

- `vllm/benchmarks/serve.py`：CLI、arrival process、指标和结果 JSON。
- `vllm/benchmarks/datasets/`：random、prefix repetition、ShareGPT 等 workload。
- `vllm/benchmarks/lib/endpoint_request_func.py`：不同 endpoint 的请求计时。
- `vllm/v1/metrics/loggers.py`：v0.25.0 指标定义。
- `vllm/v1/metrics/prometheus.py`：Prometheus 导出。

## 参考资料

- [vLLM v0.25.0 Benchmark CLI](https://docs.vllm.ai/en/v0.25.0/benchmarking/cli/)
- [vLLM v0.25.0 Benchmarking Overview](https://docs.vllm.ai/en/v0.25.0/benchmarking/)
- [vLLM v0.25.0 Production Metrics](https://docs.vllm.ai/en/v0.25.0/usage/metrics/)
- [vLLM v0.25.0 Profiling](https://docs.vllm.ai/en/v0.25.0/contributing/profiling/)
