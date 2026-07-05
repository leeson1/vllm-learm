# 15｜从 C++ 游戏后端转 AI 推理：能力迁移路线

这一篇是第三阶段的收尾，也是这个仓库目前最贴近个人路线的一篇。

背景假设：

```text
你有 C++ / Go 后端经验
做过游戏服务器
理解业务状态、并发、资源管理、性能问题
想转 AI 推理 / LLM Serving / HPC / CUDA 方向
```

先给结论：

```text
从游戏后端转 AI 推理是可行的，但不能只学 Python 调 API。
真正有迁移价值的是：服务端系统能力 + 性能分析能力 + GPU/CUDA 基础 + LLM Serving 工程理解。
```

## 1. 你已有的能力不是废的

游戏后端经验和 AI 推理不是完全断层。

很多底层能力可以迁移。

### 1.1 状态管理

游戏服务器每天都在处理状态：

```text
玩家状态
背包状态
活动状态
战斗状态
排行榜状态
```

LLM Serving 也有状态：

```text
request state
sequence state
KV Cache state
scheduler state
worker state
streaming output state
```

你需要把“玩家状态机”的经验迁移成“推理请求生命周期”的理解。

### 1.2 调度能力

游戏服务器有 tick、定时器、消息队列、异步任务。

vLLM 有 Scheduler：

```text
waiting queue
running queue
prefill/decode
continuous batching
token budget
KV block budget
```

这不是陌生领域，只是资源从“玩家请求 / 业务任务”变成了“GPU 算力 / KV Cache / token budget”。

### 1.3 资源管理

游戏后端会考虑：

```text
对象池
内存池
连接池
限流
背压
热点玩家
活动峰值
```

vLLM 会考虑：

```text
KV Cache block pool
GPU memory utilization
max_num_seqs
max_model_len
preemption
prefix cache
```

本质都是有限资源下的并发管理。

### 1.4 性能排障

游戏服常见排障：

```text
CPU 飙高
消息堆积
DB 慢查询
Redis 热 key
网络抖动
内存泄漏
```

推理服务排障：

```text
TTFT 高
TPOT 高
GPU 不满
显存 OOM
KV block 不够
CPU tokenizer 慢
streaming 堵塞
```

排障方法论是可迁移的。

## 2. 你缺的能力是什么？

### 2.1 LLM 推理基础

需要理解：

```text
Transformer 推理
prefill / decode
KV Cache
attention
sampling
context length
```

不要求一开始会训练模型，但必须懂推理链路。

### 2.2 GPU 基础

需要理解：

```text
GPU 并行模型
CUDA kernel
global memory / shared memory / register
memory bandwidth
warp / block / grid
tensor core
kernel launch overhead
```

目标不是马上写复杂 kernel，而是能看懂性能瓶颈。

### 2.3 LLM Serving 工程

需要理解：

```text
OpenAI-compatible API
continuous batching
PagedAttention
KV Cache 管理
prefix caching
speculative decoding
quantization
多卡并行
metrics / benchmark
```

这是 vLLM 仓库的核心学习目标。

### 2.4 Python / PyTorch 生态

即使你主语言是 C++，也绕不开：

```text
Python
PyTorch
HuggingFace Transformers
tokenizer
模型权重格式
```

但不要把重点放成“学 Python Web”。

Python 只是进入 AI 推理生态的工具。

## 3. 一条务实学习路线

建议分 5 个阶段。

### 阶段 1：会用 vLLM

目标：

```text
能启动服务
能调用 OpenAI API
能跑离线推理
能调整基础参数
```

任务：

1. 本地跑一个小模型。
2. 用 curl 调 `/v1/chat/completions`。
3. 跑 streaming。
4. 改 `max_tokens`、`temperature`、`max_model_len`。
5. 记录 TTFT、TPOT。

产物：

```text
一篇《vLLM quickstart + 参数解释》文章
```

### 阶段 2：理解主链路

目标：

```text
能解释请求从 HTTP 到 token 输出的全过程。
```

任务：

1. 读 Engine / Scheduler。
2. 理解 prefill 和 decode。
3. 理解 KV Cache block。
4. 画一张请求生命周期图。
5. 写源码阅读笔记。

产物：

```text
一张架构图 + 一篇源码阅读笔记
```

### 阶段 3：会压测和定位瓶颈

目标：

```text
能用数据说明系统瓶颈在哪里。
```

任务：

1. 设计短输入短输出 workload。
2. 设计长输入短输出 workload。
3. 设计短输入长输出 workload。
4. 测不同并发。
5. 调整 `max_num_batched_tokens`、`max_num_seqs`。
6. 记录 GPU/CPU/显存/TTFT/TPOT。

产物：

```text
一份压测报告
```

这是转岗最有价值的项目产物之一。

### 阶段 4：补 CUDA / HPC 基础

目标：

```text
能理解 GPU kernel 为什么快或慢。
```

任务：

1. 写 vector add。
2. 写 matrix transpose。
3. 写 naive matmul。
4. 优化 shared memory matmul。
5. 用 Nsight Systems 看 kernel launch。
6. 用 Nsight Compute 看 memory bandwidth。

产物：

```text
一个 CUDA demo 仓库 + 每个 demo 的性能分析笔记
```

不要急着写 FlashAttention。

先把 CUDA memory hierarchy 和 profiling 搞明白。

### 阶段 5：深入 attention / TensorRT-LLM / SGLang

目标：

```text
从会用 vLLM 过渡到理解推理框架底层优化。
```

任务：

1. 阅读 PagedAttention kernel 设计。
2. 理解 FlashAttention。
3. 对比 vLLM / TensorRT-LLM / SGLang。
4. 尝试 TensorRT-LLM 跑同一个模型。
5. 尝试 SGLang 的结构化输出和 prefix 复用场景。

产物：

```text
一篇三框架对比压测文章
```

## 4. 推荐项目组合

如果你想找 AI 推理岗位，建议做 3 个项目。

### 项目 1：vLLM 学习文章仓库

就是当前这个仓库。

目标：

```text
证明你系统学习过 LLM Serving 主链路。
```

重点不是文章多，而是结构清楚。

### 项目 2：vLLM 压测与瓶颈分析

做一个独立项目或当前仓库的 benchmark 目录。

内容：

```text
benchmark scripts
workload config
result csv
charts
analysis report
```

这比单纯源码笔记更接近岗位。

### 项目 3：CUDA 基础优化 demo

内容：

```text
vector add
reduce
scan
transpose
matmul
softmax
attention toy implementation
```

每个 demo 都要有：

```text
baseline
optimized version
profiling result
性能解释
```

## 5. 简历上怎么写？

不要写：

```text
学习过 vLLM，了解 AI 推理。
```

可以写成：

```text
系统学习 vLLM LLM Serving 架构，梳理请求从 OpenAI-compatible API 到 Scheduler、KV Cache Manager、Model Runner、Attention Backend、Sampler 的完整生命周期；能够解释 continuous batching、PagedAttention、chunked prefill、prefix caching 对 TTFT、TPOT、吞吐和显存占用的影响。
```

如果做了压测，可以写：

```text
基于 vLLM 搭建 LLM 推理压测环境，设计短输入短输出、长输入短输出、短输入长输出、共享前缀等 workload，对比 max_num_batched_tokens、max_num_seqs、max_model_len、prefix caching 等参数对 TTFT、TPOT、output tokens/s、GPU 利用率和 KV Cache 使用率的影响，并输出瓶颈分析报告。
```

如果做了 CUDA demo，可以写：

```text
实现 CUDA 基础算子优化 demo，包括 reduce、transpose、matmul、softmax，使用 Nsight Systems / Nsight Compute 分析 kernel launch、memory bandwidth、occupancy 和 shared memory 使用情况。
```

这类表达比“熟悉 CUDA”更可信。

## 6. 面试准备重点

### 6.1 高频问题

你需要能回答：

1. LLM 推理为什么分 prefill 和 decode？
2. KV Cache 是什么，为什么占显存？
3. PagedAttention 解决什么问题？
4. continuous batching 和普通 batching 区别是什么？
5. TTFT 和 TPOT 分别受什么影响？
6. max_num_batched_tokens 和 max_num_seqs 怎么影响性能？
7. 为什么 GPU 利用率低但延迟高？
8. prefix caching 什么时候有用？
9. speculative decoding 为什么可能加速，也为什么可能不加速？
10. vLLM 和 TensorRT-LLM / SGLang 区别是什么？

### 6.2 不要装懂的问题

如果不会 CUDA kernel，不要硬说自己会。

更好的表达是：

```text
我目前已经能理解 vLLM serving 主链路、KV Cache 管理和压测指标，正在补 CUDA memory hierarchy 和 profiling，目标是进一步理解 attention backend 的 kernel 优化。
```

这比泛泛说“熟悉底层优化”更真实。

## 7. 时间规划

如果每天能投入 1.5～2 小时，可以这样排：

```text
第 1～2 周：跑通 vLLM + API + 基础参数
第 3～4 周：读请求生命周期 + Scheduler + KV Cache
第 5～6 周：做压测脚本和报告
第 7～10 周：CUDA 基础 demo
第 11～12 周：框架对比和简历项目整理
```

如果工作很忙，可以拉长到 3～4 个月。

关键是要产出，而不是只看资料。

## 8. 从游戏后端迁移时要避免什么？

### 8.1 只停留在业务后端

只会 API 接入、部署、nginx、容器，不足以体现 AI 推理能力。

必须深入到：

```text
prefill/decode
KV Cache
Scheduler
GPU 指标
压测分析
```

### 8.2 只学模型算法

你不是要转算法岗，不需要一开始去卷训练、论文、数学推导。

你更适合切入：

```text
推理系统
性能工程
服务化
GPU 资源管理
```

### 8.3 只看文章不做实验

转岗最需要证据。

证据来自：

```text
代码
压测数据
图表
问题分析
源码笔记
```

## 9. 最小可行转岗项目

建议你最终形成这个结构：

```text
vllm-learn/
  docs/
    01-15 系列文章
  benchmarks/
    run_benchmark.py
    workloads/
      short_input_short_output.json
      long_input_short_output.json
      short_input_long_output.json
      shared_prefix.json
    results/
      result.csv
      charts/
    reports/
      benchmark-report.md
  cuda-demos/
    vector_add/
    reduce/
    matmul/
    softmax/
```

当前仓库已经完成了文章主线，下一步可以补 benchmark 和 cuda-demos。

## 10. 本文小结

从 C++ 游戏后端转 AI 推理，最合理的路线不是直接跳到 CUDA kernel，也不是只学模型算法。

更稳的路径是：

```text
vLLM 使用
  -> LLM Serving 主链路
  -> 压测和瓶颈定位
  -> CUDA/HPC 基础
  -> TensorRT-LLM / SGLang / attention backend
```

你已有的后端经验可以迁移到：

```text
调度
状态管理
资源管理
并发
性能排障
服务化
```

真正要补的是：

```text
LLM 推理机制
GPU 执行模型
KV Cache / attention
推理框架压测和优化
```

这个方向值得做，但要用项目产出来证明，而不是只停留在“我想转 AI”。

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM GitHub：https://github.com/vllm-project/vllm
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- NVIDIA CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA Nsight Systems：https://developer.nvidia.com/nsight-systems
- NVIDIA TensorRT-LLM：https://nvidia.github.io/TensorRT-LLM/
- SGLang：https://docs.sglang.ai/
