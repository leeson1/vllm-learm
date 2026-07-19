# 14｜以 vLLM v0.25.0 为基线比较 TensorRT-LLM 与 SGLang

> 比较基线：vLLM 固定 v0.25.0、commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。TensorRT-LLM 与 SGLang 必须在实际评测时另行固定版本/commit、依赖和启动配置；本文比较工程侧重点，不宣称跨版本的绝对性能排名。

## 1. 先给结论

三个项目都能服务 LLM，请不要用一句“谁更快”代替选型：

```text
vLLM v0.25.0
  通用模型服务与研究/生产功能面
  V1 Engine、continuous batching、paged KV、丰富 endpoint/后端

TensorRT-LLM
  NVIDIA 平台上的模型与 kernel/通信优化栈
  更强调硬件特化、低精度和多 GPU 性能工程

SGLang
  LLM serving/runtime 与复杂生成程序
  强调前缀复用、结构化生成和应用执行形态
```

它们的能力范围持续重叠。选型必须绑定模型、硬件、API、workload、SLO 和团队维护成本。

可以先用“主要关注层次”建立位置感，边界并不绝对：

```text
更靠近应用

  复杂生成程序 / structured output / Agent runtime
      └──────────────────────────── SGLang 重点之一

  OpenAI API / 通用 Serving / 调度 / KV 管理
      └──────────── vLLM v0.25.0 的典型学习主线

  模型执行 / 低精度 / kernel / 多 GPU 通信
      └──────────────────── TensorRT-LLM 重点之一

  NVIDIA GPU / CUDA / Tensor Core / NVLink

更靠近硬件

三者都可能跨越多层；图表示学习和选型切入点，不是功能边界。
```

## 2. vLLM v0.25.0 的具体位置

### 服务面

- OpenAI-compatible API 与离线 `LLM`；
- Chat/Completions/Responses、pooling 等多类入口；
- `vllm bench serve` 和 Prometheus metrics；
- TP、PP、DP、EP 等并行方式；
- 多种 attention backend、quantization 和 speculative 方法。

### 执行面

- V1 Engine 统一请求状态、Scheduler 和 KV 管理；
- 支持时默认 chunked prefill 与 prefix caching；
- 兼容配置下默认 async scheduling；
- 多数受支持的非 MoE 生成模型默认 Model Runner V2；
- 默认 optimization level O2，CUDA Graph 模式随 backend 能力解析；
- paged KV 通过 block table/slot metadata 接入不同 attention backend。

### 工程代价

- 功能组合多，配置兼容矩阵复杂；
- V1/V2 Runner、sync/async、backend/graph 会形成不同路径；
- Python/PyTorch 服务和控制面仍需要 CPU 性能工程；
- 版本演进快，源码笔记和默认值必须固定 tag。

## 3. TensorRT-LLM 应怎样理解？

TensorRT-LLM 更贴近 NVIDIA 推理优化生态，重点通常包括：

- NVIDIA GPU 特化 kernels 和低精度路径；
- tensor/pipeline/expert parallel 与通信优化；
- 模型转换、构建/编译或运行时配置；
- KV Cache、in-flight batching、speculative decoding 等 serving 能力；
- 与 Triton Inference Server、NVIDIA 部署栈的整合。

它不等于“任何 NVIDIA GPU 上自动比 vLLM 快”。实际结果受到支持矩阵、模型结构、精度、构建参数、GPU 架构、batch 和上下文长度影响。

对学习者的价值是：看到更硬件特化的 kernel、量化、engine/runtime 和多卡优化怎样组织。代价是环境、构建、模型支持和版本耦合通常更需要工程投入。

## 4. SGLang 应怎样理解？

SGLang 同时面向高性能 serving 和复杂 LLM 程序/runtime。学习时重点关注：

- 前缀树/RadixAttention 类复用；
- continuous batching 与请求调度；
- structured output/constraint decoding；
- 多轮、RAG、Agent、工具调用中的共享执行结构；
- 多模型/多模态与分布式 serving。

它不只是“Agent 框架”，也不能因为有前缀复用就直接断言某类 workload 必然优于 vLLM APC。两者缓存粒度、调度、backend 和请求结构需要用同一输入分布实测。

## 5. 不应该直接比较的配置

以下比较没有意义：

```text
不同模型 revision
不同量化格式或质量
不同 max context / max output
一个 ignore_eos、另一个遇 EOS 停止
一个启用 prefix cache、另一个没有共享前缀
一个 TP=2、另一个单卡
一个测固定并发、另一个测固定 QPS
一个预热完成、另一个包含 compile/build
客户端和 token 统计口径不同
```

框架 benchmark 最难的不是跑命令，而是保证语义和资源配置等价。

## 6. 统一比较矩阵

### 6.1 固定环境

```text
GPU 型号/数量/功耗与时钟策略
CPU/NUMA/内存/互联
driver/CUDA/container
框架 commit 和依赖
模型 checkpoint/revision
dtype/quantization/KV dtype
```

### 6.2 固定 workload

至少覆盖：

| Workload | 目的 |
|---|---|
| 128 in / 32 out | 服务与 launch overhead |
| 4096 in / 32 out | prefill、chunking、prefix |
| 128 in / 512 out | decode/KV/ITL |
| 4096 in / 512 out | 容量、preemption、尾延迟 |
| 共享 4K 前缀 | cache reuse |
| burst trace | queue 与 goodput |

### 6.3 固定输出语义

- temperature/top-p/top-k/seed；
- EOS 与最大输出长度；
- chat template；
- stop 条件；
- structured output/grammar；
- tokenizer 和 token 计数。

### 6.4 同时报告

- TTFT、TPOT/ITL、E2E；
- request/input/output throughput；
- goodput；
- 峰值显存与最大稳定并发；
- 启动/构建时间；
- 错误率、质量和功能缺口；
- 部署包大小、运维和升级成本。

## 7. 按目标选择起点

### 学习通用推理服务

优先 vLLM v0.25.0：可以从 HTTP、Scheduler、KV Cache 一路追到 Model Runner、attention backend 和 CUDA Graph，且本仓库已经固定源码。

### 深入 NVIDIA 性能栈

完成 vLLM 的可复现基线后，再用相同模型/workload 对照 TensorRT-LLM。重点研究 kernel/quantization/通信/构建差异，而不是只跑一个吞吐数字。

### 复杂生成程序和前缀结构

当业务确实包含共享 system prompt、树状对话、RAG、structured output 或 Agent flow，再评估 SGLang。用真实请求结构，不要只用随机 prompts。

## 8. 对推理工程师最有价值的比较方式

不要写“框架功能表”作为最终项目。做一个可解释案例：

```text
同一 7B/8B 模型
同一 GPU
同一五类 workload
同一精度和输出语义

找出：
1. 容量拐点在哪里
2. TTFT/TPOT 分别由什么决定
3. 前缀复用何时生效
4. GPU 时间线和主要 kernel 有何不同
5. 为达到同一 SLO，哪个配置成本更低
```

最终结论可以是“没有统一赢家”。能解释 trade-off 才体现推理工程能力。

## 9. 推荐学习顺序

结合 C++/Go、CUDA 基础和 vLLM 部署经验：

```text
1. vLLM v0.25.0 请求/KV/Scheduler 主链路
2. vLLM benchmark + Nsight，形成可信基线
3. vLLM attention backend 或 paged KV microbenchmark
4. TensorRT-LLM：同模型、同 GPU、同 workload
5. SGLang：加入 shared-prefix/structured workload
```

没有完成第 2 步时，直接进入三框架比较很容易变成安装记录。

## 10. 验收标准

一份合格的选型报告应该：

1. 固定三个框架的版本/commit；
2. 证明模型、精度和请求语义等价；
3. 同时报告 latency、throughput、goodput、capacity 和质量；
4. 保存完整命令和原始数据；
5. 用 profiler/metrics 解释至少一个差异；
6. 把“测得事实”与“推测原因”分开；
7. 给出针对业务 workload 的选择，而不是普遍排名。

## 11. vLLM v0.25.0 源码核对入口

- `vllm/v1/engine/`：请求与 Engine Core。
- `vllm/v1/core/sched/`：调度。
- `vllm/v1/core/kv_cache_manager.py`：KV ownership/APC。
- `vllm/v1/worker/`：Worker 与 V1/V2 Model Runner。
- `vllm/v1/attention/backends/`：backend 实现。
- `vllm/model_executor/layers/quantization/`：量化。
- `vllm/benchmarks/serve.py`：在线基准。

## 参考资料

- [vLLM v0.25.0 文档](https://docs.vllm.ai/en/v0.25.0/)
- [vLLM v0.25.0 Parallelism and Scaling](https://docs.vllm.ai/en/v0.25.0/serving/parallelism_scaling/)
- [vLLM v0.25.0 Attention Backends](https://docs.vllm.ai/en/v0.25.0/design/attention_backends/)
- [NVIDIA TensorRT-LLM 文档](https://nvidia.github.io/TensorRT-LLM/)
- [SGLang 文档](https://docs.sglang.ai/)
