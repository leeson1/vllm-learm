# 05｜工程调优地图：吞吐、延迟、显存与常见参数

这篇先不做完整压测，只建立 vLLM 性能调优地图。

学习 vLLM 时，很容易看到一堆参数：

```text
--gpu-memory-utilization
--max-model-len
--max-num-batched-tokens
--max-num-seqs
--tensor-parallel-size
--enable-prefix-caching
--dtype
--quantization
```

不要死记参数。先理解它们分别影响什么资源。

## 1. LLM 推理服务的核心指标

### 1.1 TTFT

TTFT：Time To First Token。

意思是：客户端发出请求后，多久收到第一个 token。

它主要受 prefill 阶段影响。

长 prompt、复杂 chat template、tokenizer 慢、排队时间长，都会导致 TTFT 变高。

### 1.2 TPOT

TPOT：Time Per Output Token。

意思是：生成阶段每个 token 平均耗时。

它主要受 decode 阶段影响。

### 1.3 Throughput

吞吐可以有多种口径：

```text
requests/s
input tokens/s
output tokens/s
total tokens/s
```

LLM serving 最常用的是 token 级吞吐。

### 1.4 Tail Latency

尾延迟很重要。例如 P99 TTFT、P99 E2E latency。

平均值好看不代表线上体验好。长 prompt、长输出、排队、抢占都可能让尾延迟恶化。

## 2. 三个基本资源

vLLM 调优本质上是在调三类资源：

```text
GPU 显存
GPU 算力
CPU / 网络 / Python 服务层资源
```

### 2.1 GPU 显存

显存主要被这些东西占用：

```text
模型权重
KV Cache
临时激活 / workspace
CUDA Graph / 编译缓存 / 框架开销
```

其中 KV Cache 和请求并发、上下文长度强相关。

### 2.2 GPU 算力

prefill 通常更像大矩阵计算，decode 通常更容易受 memory bandwidth、batch size、KV 读取影响。

### 2.3 CPU 资源

不要忽视 CPU。

CPU 可能负责：

- HTTP 请求处理。
- tokenizer。
- detokenizer。
- 多模态数据加载。
- 进程间通信。
- streaming response。
- metrics/logging。

vLLM V1 是多进程架构，多卡和 data parallel 下 CPU 资源需求会上升。

## 3. 参数一：`--gpu-memory-utilization`

示例：

```bash
vllm serve <model> --gpu-memory-utilization 0.9
```

它控制 vLLM 可以使用多少比例 GPU 显存。

直觉：

```text
值越大 -> KV Cache 空间越多 -> 可能支持更高并发/更长上下文
值越小 -> 更保守 -> OOM 风险更低，但容量下降
```

不要盲目设成 1.0。线上要给系统、驱动、临时 buffer 留余量。

## 4. 参数二：`--max-model-len`

示例：

```bash
vllm serve <model> --max-model-len 8192
```

它控制最大上下文长度。

上下文越长，单请求最坏情况下需要的 KV Cache 越多。

如果业务只需要 4K，就不要默认开到 32K 或 128K。

后端视角：这类似“配置最大背包容量”。容量越大，单个玩家理论上能占用的资源越高，整体并发越容易下降。

## 5. 参数三：`--max-num-batched-tokens`

它限制一次调度 batch 中的 token budget。

可以粗略理解为：

```text
每一轮 GPU forward 最多处理多少 token
```

影响：

- 值大：可能提高吞吐，尤其 prefill-heavy 场景。
- 值小：可能降低单轮耗时，改善部分延迟，但吞吐可能下降。

这个参数和 workload 强相关，不能脱离压测空谈。

## 6. 参数四：`--max-num-seqs`

它限制同时参与调度的序列数量。

可以理解为最大并发序列数上限。

影响：

- 值大：并发潜力更高，但 KV Cache 压力更大。
- 值小：更保守，尾延迟可能更稳定，但吞吐上限下降。

## 7. 参数五：`--tensor-parallel-size`

示例：

```bash
vllm serve <model> --tensor-parallel-size 2
```

Tensor Parallel 是把一个模型拆到多张 GPU 上。

适用场景：

- 单卡放不下模型。
- 希望更高吞吐。
- 模型较大，需要多卡协同。

代价：

- GPU 间通信增加。
- 部署复杂度增加。
- 小模型不一定收益明显。

如果两张 GPU 没有 NVLink，只靠 PCIe，TP 通信可能成为瓶颈。是否值得必须压测。

## 8. 参数六：`--enable-prefix-caching`

Automatic Prefix Caching 适合大量请求共享相同前缀的场景。

示例场景：

```text
固定 system prompt
固定 few-shot examples
RAG 模板前缀相同
Agent 工作流中大量重复上下文
```

它能降低共享前缀的重复 prefill 计算。

但如果请求之间没有公共前缀，收益就不明显。

## 9. 参数七：dtype / quantization

### 9.1 dtype

常见：

```bash
--dtype auto
--dtype float16
--dtype bfloat16
```

精度影响：

- 显存占用。
- 算子性能。
- 硬件兼容性。
- 数值稳定性。

### 9.2 quantization

量化可以降低权重显存，比如 INT8、INT4、FP8 等。

但要注意：

```text
权重量化 ≠ KV Cache 一定变小
显存下降 ≠ 延迟一定下降
```

量化可能引入额外 kernel、反量化开销，收益取决于模型、硬件、backend。

## 10. 不同 workload 下优先看什么？

### 10.1 短 prompt + 短输出

常见于分类、简单问答。

关注：

- 请求调度开销。
- HTTP/tokenizer 开销。
- batch 是否足够大。
- P99 latency。

### 10.2 长 prompt + 短输出

常见于 RAG、长文总结。

关注：

- TTFT。
- prefill 吞吐。
- prefix caching。
- chunked prefill。
- max-num-batched-tokens。

### 10.3 短 prompt + 长输出

常见于写作、代码生成。

关注：

- decode TPOT。
- KV Cache 增长。
- streaming 稳定性。
- max-num-seqs。

### 10.4 长 prompt + 长输出

最重场景。

关注：

- 显存容量。
- KV Cache。
- 尾延迟。
- 抢占策略。
- 多卡/多机部署。

## 11. 压测时不要只看一个指标

至少要同时看：

```text
TTFT avg / p95 / p99
TPOT avg / p95 / p99
output tokens/s
request/s
GPU util
GPU memory
CPU util
排队时间
错误率 / OOM / timeout
```

如果只看 tokens/s，可能会把尾延迟调爆。

如果只看 P99，又可能把吞吐调得太保守。

## 12. 一个简单调优流程

建议流程：

```text
1. 固定模型、硬件、业务输入输出分布
2. 先跑默认参数，记录基线
3. 确定瓶颈：显存、GPU 算力、CPU、网络、排队
4. 一次只改一个参数
5. 记录 TTFT / TPOT / throughput / OOM
6. 找到吞吐和尾延迟的平衡点
```

不要一次改一堆参数，否则不知道哪个参数起作用。

## 13. 后端工程落地建议

### 13.1 服务前面一定要有限流

LLM 请求成本高，不能像普通接口一样无限排队。

建议：

- 按用户限流。
- 按 token budget 限流。
- 按并发数限流。
- 对超长 prompt 做拒绝或降级。

### 13.2 记录 token 级指标

至少记录：

```text
input_tokens
output_tokens
TTFT
E2E latency
finish_reason
model
sampling params
```

否则后面无法分析性能和成本。

### 13.3 区分不同模型池

不同模型、不同上下文长度、不同 SLA 的请求不一定适合混在同一个服务里。

可以拆成：

```text
低延迟小模型池
高吞吐批处理池
长上下文模型池
代码模型池
```

### 13.4 不要只靠平均值报警

建议看 P95/P99、OOM、队列长度、GPU memory watermark。

## 14. 本文小结

vLLM 调优的主线是：

```text
显存决定容量
调度决定吞吐和尾延迟
prefill 影响 TTFT
decode 影响 TPOT
workload 决定参数方向
```

常见参数可以按资源归类：

```text
显存容量：--gpu-memory-utilization, --max-model-len, quantization
调度容量：--max-num-batched-tokens, --max-num-seqs
多卡扩展：--tensor-parallel-size, --data-parallel-size
重复前缀：--enable-prefix-caching
```

学习阶段先建立这张地图，后续再逐个深入 Scheduler、Block Manager、Attention Backend。

## 参考资料

- vLLM Optimization and Tuning：https://docs.vllm.ai/en/latest/configuration/optimization/
- vLLM Engine Arguments：https://docs.vllm.ai/en/latest/configuration/engine_args/
- vLLM Metrics：https://docs.vllm.ai/en/latest/usage/metrics/
- vLLM Benchmarking：https://docs.vllm.ai/en/latest/benchmarking/
