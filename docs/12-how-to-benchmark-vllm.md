# 12｜如何压测 vLLM：不要只看 tokens/s

学习 vLLM 到第三阶段，必须开始做压测。

因为很多推理优化不能靠感觉判断：

```text
max_num_batched_tokens 调大到底有没有收益？
prefix caching 是否真的命中？
量化后是更快还是只省显存？
TP=2 在没有 NVLink 的机器上是否划算？
```

这些都必须靠压测回答。

## 1. 压测前先明确问题

不要一上来就跑 benchmark。

先明确你想回答什么问题：

```text
问题 1：这台机器最多能支撑多少并发？
问题 2：P99 TTFT 是否满足业务要求？
问题 3：长 prompt 场景是否会拖慢流式输出？
问题 4：开启 prefix caching 是否有收益？
问题 5：量化后吞吐和质量如何变化？
问题 6：多卡 tensor parallel 是否比单卡更快？
```

不同问题需要不同压测方法。

## 2. 先理解几个指标

### 2.1 TTFT

TTFT：Time To First Token。

```text
客户端发出请求 -> 收到第一个 token
```

它主要受：

1. 排队时间。
2. tokenizer。
3. prefill。
4. 调度策略。
5. prefix cache 命中率。

影响。

### 2.2 TPOT

TPOT：Time Per Output Token。

```text
生成阶段每个 token 的平均间隔
```

它主要受 decode 阶段影响。

长上下文、高并发、KV Cache 读取、attention backend 都会影响 TPOT。

### 2.3 E2E Latency

端到端延迟：

```text
客户端发请求 -> 完整响应结束
```

它受输入长度、输出长度、排队、prefill、decode、网络、detokenize 全部影响。

### 2.4 Throughput

吞吐不要只看 requests/s。

LLM Serving 更常用：

```text
input tokens/s
output tokens/s
total tokens/s
```

因为两个请求的 token 数可能差十倍。

### 2.5 Tail Latency

线上尤其要看：

```text
P50 / P90 / P95 / P99 TTFT
P50 / P90 / P95 / P99 TPOT
P50 / P90 / P95 / P99 E2E latency
```

平均值好看，不代表线上体验好。

## 3. 压测 workload 怎么设计？

压测必须贴近业务。

不要只用一个固定 prompt。

### 3.1 短输入短输出

适合测试基础 overhead：

```text
input: 50 tokens
output: 50 tokens
```

观察：

```text
HTTP / tokenizer / scheduler overhead
```

### 3.2 长输入短输出

适合测试 prefill：

```text
input: 4000 tokens
output: 100 tokens
```

观察：

```text
TTFT
chunked prefill
prefix caching
KV block 分配
```

### 3.3 短输入长输出

适合测试 decode：

```text
input: 100 tokens
output: 2000 tokens
```

观察：

```text
TPOT
output tokens/s
streaming 稳定性
```

### 3.4 长输入长输出

适合测试极限压力：

```text
input: 8000 tokens
output: 2000 tokens
```

观察：

```text
显存占用
抢占
OOM 风险
尾延迟
```

### 3.5 共享前缀 workload

用于测试 prefix caching：

```text
固定 system prompt + 固定工具说明 + 不同用户问题
```

观察：

```text
cache hit rate
TTFT 变化
prefill tokens/s 变化
```

## 4. 并发模式怎么选？

### 4.1 固定并发

例如：

```text
并发 1、2、4、8、16、32、64、128
```

观察系统从低负载到饱和的曲线。

### 4.2 固定 QPS

例如：

```text
每秒 1、2、5、10、20 个请求
```

更接近线上流量。

重点看排队是否开始积压。

### 4.3 burst 流量

例如短时间打入大量请求：

```text
10 秒内突然进入 500 个请求
```

观察：

```text
waiting queue
TTFT P99
抢占
服务是否稳定
```

## 5. 推荐压测步骤

### 5.1 单请求基线

先跑单请求，确认服务正常。

记录：

```text
TTFT
TPOT
E2E
显存占用
GPU utilization
```

### 5.2 并发阶梯压测

逐步增加并发：

```text
1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64
```

每档固定跑一段时间。

观察：

```text
吞吐是否继续上升
TTFT 是否开始陡增
TPOT 是否明显变差
显存是否接近上限
```

### 5.3 找到饱和点

饱和点通常表现为：

```text
吞吐不再明显上升
但延迟快速恶化
```

这个点非常关键。

线上通常不会把系统压到饱和点，而是留出安全余量。

### 5.4 参数对比实验

一次只改一个参数。

例如：

```text
max_num_batched_tokens = 4096 / 8192 / 16384
```

不要同时改多个参数，否则无法判断原因。

## 6. vLLM 常见压测维度

### 6.1 `--max-num-batched-tokens`

关注：

```text
prefill 吞吐
TTFT
单轮调度耗时
```

长 prompt 场景尤其重要。

### 6.2 `--max-num-seqs`

关注：

```text
并发能力
KV Cache 占用
TPOT
抢占频率
```

### 6.3 `--gpu-memory-utilization`

关注：

```text
可用 KV block 数量
OOM 风险
最大并发
```

不要盲目拉满。线上要给临时 buffer、驱动、监控等留余量。

### 6.4 `--enable-prefix-caching`

关注：

```text
cache hit rate
TTFT
prefill tokens/s
显存占用变化
```

没有共享前缀的 workload 下，它的收益会很小。

### 6.5 `--tensor-parallel-size`

关注：

```text
单卡是否放得下模型
多卡通信开销
吞吐是否提升
延迟是否恶化
```

没有 NVLink 时，TP 通信可能成为瓶颈，必须实际压测。

## 7. 监控应该看什么？

### 7.1 GPU

用 `nvidia-smi` 只能看粗略情况。

至少关注：

```text
GPU utilization
显存占用
功耗
温度
显存带宽相关指标
```

更深入可以用 Nsight Systems / Nsight Compute。

### 7.2 CPU

不要忽略 CPU。

关注：

```text
CPU utilization
tokenizer/detokenizer 开销
HTTP server 开销
进程间通信
日志/metrics 开销
```

如果 GPU 利用率低但延迟高，可能是 CPU 侧瓶颈。

### 7.3 服务指标

建议至少记录：

```text
request count
running requests
waiting requests
input tokens/s
output tokens/s
TTFT histogram
TPOT histogram
E2E latency histogram
preemption count
cache hit rate
OOM / error count
```

## 8. 压测结果怎么分析？

### 8.1 GPU 不满，延迟高

可能原因：

1. CPU tokenizer 慢。
2. 请求构造/调度 overhead 高。
3. batch 太小。
4. 网络或客户端压测工具瓶颈。
5. 日志太多。

### 8.2 GPU 满，吞吐上不去

可能原因：

1. 模型计算已饱和。
2. attention 受显存带宽限制。
3. batch shape 不适合。
4. KV Cache 读取成本高。

### 8.3 TTFT 高

可能原因：

1. waiting queue 积压。
2. prefill 太重。
3. 长 prompt 阻塞。
4. chunked prefill 参数不合适。
5. prefix cache 未命中。

### 8.4 TPOT 高

可能原因：

1. decode batch 太小或太大。
2. 长上下文导致 KV 读取重。
3. attention backend 性能不足。
4. CPU launch overhead 或 CUDA Graph 未有效复用。

### 8.5 显存很快打满

可能原因：

1. max_model_len 太大。
2. max_num_seqs 太大。
3. 输出太长。
4. KV Cache dtype 太大。
5. prefix cache 占用未释放。

## 9. 压测报告应该怎么写？

一份有价值的压测报告应该包含：

```text
1. 硬件环境：GPU、CPU、内存、驱动、CUDA、网络
2. 软件版本：vLLM 版本、PyTorch、模型、dtype、量化方式
3. 启动参数：完整 vllm serve 命令
4. workload：输入/输出长度分布、并发/QPS、请求数
5. 指标：TTFT、TPOT、吞吐、显存、GPU/CPU 利用率
6. 结果曲线：并发增加时吞吐和延迟如何变化
7. 结论：瓶颈在哪里，建议参数是什么
```

不要只贴一张 tokens/s 截图。

## 10. 最小实践任务

建议做一个小项目：

```text
目标：对同一个模型做 4 组压测。

组 1：短输入短输出
组 2：长输入短输出
组 3：短输入长输出
组 4：共享前缀请求
```

每组记录：

```text
TTFT P50/P95/P99
TPOT P50/P95/P99
output tokens/s
GPU 显存
GPU utilization
```

然后写一篇分析文章。

这比单纯说“我会 vLLM 调参”更有含金量。

## 11. 本文小结

压测 vLLM 的核心不是跑出一个漂亮数字，而是建立因果关系：

```text
参数变化 -> 调度变化 -> KV Cache 变化 -> GPU/CPU 变化 -> TTFT/TPOT/吞吐变化
```

你需要重点记住：

1. 不同 workload 会得到完全不同的结论。
2. TTFT、TPOT、吞吐、尾延迟要一起看。
3. 一次只改一个参数。
4. 找到饱和点比追求峰值更重要。
5. 线上配置要留安全余量。

## 参考资料

- vLLM Benchmark 文档：https://docs.vllm.ai/en/latest/contributing/benchmarks.html
- vLLM Metrics 文档：https://docs.vllm.ai/en/latest/usage/metrics.html
- vLLM Engine Arguments：https://docs.vllm.ai/en/latest/configuration/engine_args.html
- NVIDIA Nsight Systems：https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute：https://developer.nvidia.com/nsight-compute
