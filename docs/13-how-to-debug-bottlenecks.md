# 13｜如何定位 GPU / CPU / 网络瓶颈

压测之后，下一步是定位瓶颈。

很多人看到 vLLM 慢，第一反应是：

```text
是不是 GPU 不行？
是不是参数没调好？
是不是模型太大？
```

但线上推理服务的瓶颈可能出现在很多位置：

```text
客户端压测工具
网络
HTTP server
tokenizer / detokenizer
Scheduler
KV Cache
GPU kernel
多进程通信
流式输出
日志和 metrics
```

这篇建立一套排障方法。

## 1. 先从现象分类

不要直接猜原因。

先把问题归类。

### 1.1 TTFT 高

表现：

```text
请求发出后，很久才收到第一个 token。
```

可能原因：

1. 请求在 waiting queue 中排队。
2. prompt 太长，prefill 重。
3. 长 prompt 阻塞 decode。
4. tokenizer 慢。
5. prefix cache 没命中。
6. GPU 已经饱和。

### 1.2 TPOT 高

表现：

```text
第一个 token 出来了，但后续 token 间隔很长。
```

可能原因：

1. decode batch 形态不好。
2. 长上下文导致 KV Cache 读取重。
3. attention backend 受显存带宽限制。
4. CUDA Graph 没有有效复用。
5. CPU launch overhead 高。
6. streaming 输出处理慢。

### 1.3 吞吐低

表现：

```text
GPU 看起来没满，但 tokens/s 上不去。
```

可能原因：

1. batch 太小。
2. 客户端压测工具打不满。
3. CPU 成为瓶颈。
4. tokenizer / detokenizer 慢。
5. 网络连接数不足。
6. 日志或 metrics 过重。

### 1.4 显存 OOM

表现：

```text
请求量上来后 OOM，或者服务启动时就 OOM。
```

可能原因：

1. 模型权重太大。
2. max_model_len 太大。
3. max_num_seqs 太大。
4. gpu_memory_utilization 太激进。
5. KV Cache 占用过高。
6. 量化 / dtype 配置不合理。

## 2. 建立分层排查模型

建议按下面顺序排查：

```text
客户端
  -> 网络
  -> API / HTTP server
  -> CPU preprocessing
  -> Scheduler / queue
  -> KV Cache / memory
  -> GPU execution
  -> output streaming
```

不要一开始就用 Nsight。

先用低成本指标缩小范围，再上重型 profiler。

## 3. 第一步：确认压测工具没有瓶颈

很多“服务端瓶颈”其实是客户端打不满。

检查：

```text
压测机 CPU 是否打满？
压测机网络是否打满？
连接数是否足够？
请求是否真的并发发出？
是否同步等待导致压测串行化？
```

建议：

1. 压测机和服务机分开。
2. 客户端记录每个请求的时间线。
3. 服务端也记录请求到达时间。
4. 对比两边统计是否一致。

## 4. 第二步：看 GPU 是否忙

用 `nvidia-smi` 可以先粗看：

```bash
nvidia-smi dmon
nvidia-smi pmon
```

观察：

```text
GPU utilization
显存占用
功耗
温度
是否降频
```

几种典型情况：

### 4.1 GPU 利用率低，延迟高

通常说明瓶颈可能在 GPU 之前：

```text
CPU / tokenizer / 调度 / 网络 / batch 太小
```

### 4.2 GPU 利用率高，延迟也高

可能是 GPU 已经饱和：

```text
模型计算重
attention 读 KV 重
并发过高
上下文太长
```

### 4.3 显存接近上限

说明 KV Cache 或模型权重压力大。

需要看：

```text
max_model_len
max_num_seqs
gpu_memory_utilization
kv_cache_dtype
quantization
```

## 5. 第三步：看 CPU 是否成为瓶颈

LLM Serving 不是纯 GPU 服务。

CPU 可能负责：

```text
HTTP 请求处理
tokenizer
detokenizer
Sampling 后处理
进程间通信
日志
metrics
streaming response
```

检查：

```bash
top
htop
pidstat -p <pid> 1
mpstat -P ALL 1
```

如果 CPU 单核打满，要特别注意 Python 侧逻辑或 tokenizer。

常见现象：

```text
GPU utilization 不高
CPU 某几个核心很高
TTFT/TPOT 都不稳定
```

这时不要继续盲目调 GPU 参数。

## 6. 第四步：看请求队列

如果 vLLM metrics 可用，重点看：

```text
waiting requests
running requests
request queue time
preemption count
```

如果 waiting requests 持续增长：

```text
输入流量 > 服务处理能力
```

这时 TTFT 一定会变差。

解决方向：

1. 降低输入 QPS。
2. 增加副本。
3. 调整 max_num_batched_tokens / max_num_seqs。
4. 限制 max_model_len / max_tokens。
5. 针对长请求做限流或隔离。

## 7. 第五步：看 KV Cache 状态

KV Cache 是 vLLM 的核心资源。

重点关注：

```text
KV block 使用率
剩余 block 数
抢占次数
prefix cache 命中率
OOM / allocation failure
```

典型问题：

### 7.1 KV block 不够

表现：

```text
并发上来后抢占频繁
TTFT 和 TPOT 都变差
显存占用高
```

解决方向：

1. 降低 max_num_seqs。
2. 降低 max_model_len。
3. 限制 max_tokens。
4. 使用更低精度 KV Cache。
5. 使用更小模型或量化模型。

### 7.2 prefix cache 命中率低

如果开启 prefix caching 但命中率低，说明 workload 不匹配。

检查：

```text
system prompt 是否真的一致？
模板中是否包含动态字段？
时间戳、request id 是否放在前缀中？
用户输入是否太早出现？
```

## 8. 第六步：区分 prefill 瓶颈和 decode 瓶颈

这是 LLM Serving 排障最重要的分类之一。

### 8.1 prefill 瓶颈

表现：

```text
TTFT 高
长 prompt 请求影响明显
input tokens/s 成为关键指标
```

解决方向：

1. chunked prefill。
2. prefix caching。
3. 限制 prompt 长度。
4. 优化 tokenizer。
5. 调整 max_num_batched_tokens。

### 8.2 decode 瓶颈

表现：

```text
TPOT 高
output tokens/s 上不去
长输出请求拖慢整体
```

解决方向：

1. 调整 max_num_seqs。
2. 使用更适合的 attention backend。
3. 开启 CUDA Graph。
4. 尝试 speculative decoding。
5. 限制 max_tokens。

## 9. 第七步：使用 profiler

当普通指标无法解释问题时，再上 profiler。

### 9.1 Nsight Systems

适合看全链路时间线：

```text
CPU 线程
CUDA kernel launch
GPU kernel 执行
CPU/GPU 空洞
同步点
```

重点看：

```text
GPU 是否有长时间空闲？
CPU 是否在 kernel launch 前卡住？
是否存在频繁同步？
kernel 间隔是否很大？
```

### 9.2 Nsight Compute

适合深入单个 kernel：

```text
occupancy
memory bandwidth
shared memory
warp stall
tensor core 使用
```

这个阶段更偏 CUDA/HPC。

如果你还没读懂 attention backend，不建议太早深入 Nsight Compute。

## 10. 常见瓶颈模式

### 10.1 CPU 喂不饱 GPU

现象：

```text
GPU utilization 低
CPU 单核/少数核心高
请求延迟高
```

排查：

```text
tokenizer 是否慢？
日志是否太多？
metrics 是否过重？
Python 侧是否有串行逻辑？
```

### 10.2 GPU 算力饱和

现象：

```text
GPU utilization 高
功耗高
吞吐到平台期
```

排查：

```text
模型是否太大？
batch 是否已经合理？
是否需要更多 GPU / 更小模型 / 量化？
```

### 10.3 显存限制并发

现象：

```text
显存高
KV block 不够
抢占增加
OOM
```

排查：

```text
max_model_len 是否过大？
max_num_seqs 是否过大？
max_tokens 是否无限制？
KV dtype 是否可降低？
```

### 10.4 网络或 streaming 成为瓶颈

现象：

```text
服务端生成快，但客户端接收慢
大量长连接
HTTP streaming 占用 CPU
```

排查：

```text
客户端读取是否及时？
反向代理是否缓冲？
网络带宽是否足够？
连接数是否过高？
```

## 11. 一个推荐排障 checklist

```text
1. 确认压测工具没瓶颈
2. 看 TTFT、TPOT、吞吐、P99
3. 看 GPU utilization 和显存
4. 看 CPU utilization 和单核热点
5. 看 waiting/running queue
6. 看 KV Cache 使用率和抢占
7. 区分 prefill 还是 decode 瓶颈
8. 一次只改一个参数验证
9. 必要时用 Nsight Systems
10. 最后才深入单个 CUDA kernel
```

## 12. 面试中怎么表达这类能力？

不要说：

```text
我会调 vLLM 参数。
```

更好的表达是：

```text
我会根据 TTFT、TPOT、tokens/s、GPU 利用率、KV Cache 使用率和队列长度判断瓶颈位于 prefill、decode、CPU 服务层还是显存资源层，并通过控制变量压测验证参数调整是否有效。
```

这更像工程能力。

## 13. 本文小结

定位 vLLM 瓶颈的核心是分层和分类。

你需要记住：

1. TTFT 高通常先看排队和 prefill。
2. TPOT 高通常先看 decode、KV 读取和 attention backend。
3. GPU 不满不代表 GPU 慢，可能是 CPU 或客户端喂不饱。
4. 显存问题通常和 KV Cache、max_model_len、max_num_seqs 强相关。
5. profiler 是最后验证工具，不是第一步。

形成这套排障思路后，你就不只是“会用 vLLM”，而是开始具备 LLM Serving 工程能力。

## 参考资料

- vLLM Metrics 文档：https://docs.vllm.ai/en/latest/usage/metrics.html
- vLLM Benchmark 文档：https://docs.vllm.ai/en/latest/contributing/benchmarks.html
- vLLM Engine Arguments：https://docs.vllm.ai/en/latest/configuration/engine_args.html
- NVIDIA Nsight Systems：https://developer.nvidia.com/nsight-systems
- NVIDIA Nsight Compute：https://developer.nvidia.com/nsight-compute
