# vLLM v0.25.0 学习笔记

这个仓库用于系统学习 vLLM：从“怎么跑起来”，逐步过渡到“为什么快”“内部怎么调度”“KV Cache 怎么管理”“如何面向工程部署和调优”。

> 适合背景：有后端/系统开发经验，想往 AI 推理、HPC、CUDA、LLM Serving 方向转的人。

## 版本与证据约定

本仓库固定以 **vLLM v0.25.0** 为基线。vLLM 迭代很快，不固定版本时，默认值、进程拓扑和源码路径很容易互相串版。

- 官方行为以 [v0.25.0 文档](https://docs.vllm.ai/en/v0.25.0/) 为准。
- 实现细节以本机 `/Users/leeson/codes/vllm` 的 `v0.25.0` tag 为准。
- 每篇文章末尾都给出“源码核对入口”；源码结论尽量落到具体文件和符号，而不是只链接仓库首页。
- 论文或历史设计会明确标为“历史背景”。例如官方 Paged Attention 设计页已经注明它不再描述当前全部实现，不能直接把旧 kernel 文档当成 v0.25.0 的完整调用链。
- 文中的性能方向是待压测的假设，不当作对所有模型、硬件和 workload 都成立的保证。

## 学习路线

### 第一阶段：先建立全局地图

1. [vLLM 是什么：从 LLM 推理服务的瓶颈说起](docs/01-what-is-vllm.md)
2. [先跑起来：离线推理与 OpenAI-Compatible Server](docs/02-quickstart-offline-and-server.md)
3. [核心原理：PagedAttention 与 KV Cache 管理](docs/03-pagedattention-and-kv-cache.md)
4. [服务端视角：vLLM V1 架构与请求生命周期](docs/04-vllm-v1-architecture.md)
5. [工程调优地图：吞吐、延迟、显存与常见参数](docs/05-performance-tuning-map.md)

完成第一阶段后，应该能独立做到：

- 用固定版本跑通 `LLM.generate` 和 OpenAI-compatible server。
- 解释 prefill、decode、KV Cache、continuous batching 和 PagedAttention 的关系。
- 画出 API Server → Engine Core → Scheduler → Executor/Worker → 输出回程的主链路。
- 区分“启动时建立 KV Cache 物理池”和“调度时按请求分配 block 所有权”。
- 用 TTFT、ITL/TPOT、E2E latency、吞吐、KV Cache 使用率和 preemption 评价一次压测。
- 说明 `gpu-memory-utilization`、`max-model-len`、`max-num-batched-tokens`、`max-num-seqs`、TP/DP 各自在约束什么。

配套的 [v0.25.0 源码交互导览](vllm-v0.25-source-guide/index.html) 可在读完第 4 篇后使用；它用于定位代码，不替代前 5 篇的概念主线。

### 第二阶段：深入关键模块

6. [Scheduler：continuous batching、prefill/decode 混排、chunked prefill](docs/06-scheduler-continuous-batching.md)
7. [Block Manager：KV block 分配、回收、共享与抢占](docs/07-block-manager-kv-cache.md)
8. [Model Runner：一次 forward 前后到底发生了什么](docs/08-model-runner-forward.md)
9. [Attention Backend：FlashAttention、PagedAttention kernel 与 CUDA Graph](docs/09-attention-backend.md)
10. [Prefix Caching、Speculative Decoding 与 Quantization](docs/10-prefix-speculative-quantization.md)

### 第三阶段：面向岗位能力

11. [如何读 vLLM 源码：从请求生命周期切进去](docs/11-how-to-read-vllm-source.md)
12. [如何压测 vLLM：不要只看 tokens/s](docs/12-how-to-benchmark-vllm.md)
13. [如何定位 GPU / CPU / 网络瓶颈](docs/13-how-to-debug-bottlenecks.md)
14. [vLLM vs TensorRT-LLM vs SGLang](docs/14-vllm-vs-tensorrt-llm-vs-sglang.md)
15. [从 C++ 游戏后端转 AI 推理：能力迁移路线](docs/15-cpp-game-backend-to-ai-inference.md)

## 下一步实践建议

文章主线完成后，建议继续补实践目录：

```text
benchmarks/
  workloads/
  results/
  reports/

cuda-demos/
  vector_add/
  reduce/
  matmul/
  softmax/
```

也就是说，下一阶段不要继续只写文章，而要开始沉淀：压测脚本、结果数据、图表、CUDA demo 和性能分析报告。

## 主要资料来源

- vLLM v0.25.0 官方文档：https://docs.vllm.ai/en/v0.25.0/
- vLLM v0.25.0 源码：https://github.com/vllm-project/vllm/tree/v0.25.0
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
- NVIDIA TensorRT-LLM：https://nvidia.github.io/TensorRT-LLM/
- SGLang：https://docs.sglang.ai/

## 阅读建议

不要一开始就死磕 CUDA kernel。更推荐顺序是：

```text
会用 vLLM
  -> 理解 LLM 推理瓶颈
  -> 理解 KV Cache 为什么是核心资源
  -> 理解 Scheduler / Block Manager / Worker 的职责边界
  -> 再进入 CUDA kernel / attention backend / 多机多卡
```

对后端开发来说，vLLM 最值得先掌握的不是“模型结构”，而是：调度、内存管理、并发、资源隔离、服务化、可观测性。
