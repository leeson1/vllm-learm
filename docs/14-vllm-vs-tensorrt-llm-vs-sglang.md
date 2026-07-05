# 14｜vLLM vs TensorRT-LLM vs SGLang

这一篇做推理框架对比。

注意：这不是选边站。vLLM、TensorRT-LLM、SGLang 都是优秀的 LLM Serving / Inference 方向项目，只是设计重心不同。

更合理的问题不是：

```text
哪个框架最强？
```

而是：

```text
我的业务场景、硬件环境、团队能力和优化目标，适合哪个框架？
```

## 1. 先给一个粗略定位

| 框架 | 粗略定位 | 更适合关注 |
|---|---|---|
| vLLM | 通用 LLM Serving 引擎 | 易用性、PagedAttention、OpenAI API、生态、快速部署 |
| TensorRT-LLM | NVIDIA GPU 上的高性能 LLM 推理优化栈 | 极致性能、TensorRT engine、NVIDIA 硬件优化、多卡部署 |
| SGLang | 面向结构化生成和复杂 LLM 程序的 serving/runtime | RadixAttention、结构化输出、多轮/Agent/RAG 复杂流程 |

这只是入门定位，真实选择必须压测。

## 2. vLLM 的特点

vLLM 的核心标签：

```text
PagedAttention
continuous batching
OpenAI-compatible API
易部署
高吞吐 serving
```

它最适合用来学习 LLM Serving 的系统工程，因为模块边界非常典型：

```text
API
Scheduler
KV Cache Manager
Model Runner
Attention Backend
Worker
```

对后端开发来说，vLLM 的学习价值非常高。

## 3. vLLM 的优势

### 3.1 上手快

启动一个 OpenAI-compatible server 比较直接：

```bash
vllm serve <model>
```

业务可以用 OpenAI API 形式接入。

### 3.2 Serving 抽象清晰

vLLM 的调度、KV Cache、模型执行、输出处理都比较适合学习。

如果你的目标是转 AI 推理岗位，vLLM 是很好的切入口。

### 3.3 PagedAttention 代表性强

PagedAttention 是 vLLM 最经典的设计。

它把 KV Cache 拆成 block，用类似分页的方式降低显存碎片，支持更高并发和更灵活的内存管理。

### 3.4 生态活跃

vLLM 支持 OpenAI-compatible API、多种模型、量化、prefix caching、speculative decoding、多卡等能力。

它的工程生态对学习和原型验证都比较友好。

## 4. vLLM 的局限

### 4.1 极致性能未必总是最优

vLLM 很通用，但在特定 NVIDIA 硬件、特定模型、特定 batch 形态下，TensorRT-LLM 这类深度优化方案可能更强。

### 4.2 Python 服务栈仍在关键路径中

虽然核心计算在 GPU，但服务端调度、请求处理、tokenizer、输出等仍涉及 CPU 和 Python 侧工程。

高压场景下 CPU 也可能成为瓶颈。

### 4.3 功能多导致配置复杂

模型、量化、并行、attention backend、prefix/spec decode 都有很多参数。

真正线上用好仍然需要压测和调优。

## 5. TensorRT-LLM 的特点

TensorRT-LLM 是 NVIDIA 面向大模型推理的优化工具链。

它的核心标签：

```text
TensorRT engine
NVIDIA GPU 深度优化
高性能 kernel
FP8 / INT8 / INT4 等量化
多 GPU / 多节点优化
```

如果你追求 NVIDIA GPU 上的极限性能，TensorRT-LLM 是绕不开的。

## 6. TensorRT-LLM 的优势

### 6.1 更贴近 NVIDIA 硬件优化

TensorRT-LLM 可以利用 NVIDIA 生态中的优化能力，包括 TensorRT engine、专用 kernel、低精度计算、多 GPU 通信优化等。

### 6.2 适合性能压榨

如果业务模型固定、场景稳定、团队有足够工程能力，TensorRT-LLM 可能带来更好的性能上限。

### 6.3 部署大模型、多卡优化能力强

在多 GPU、多节点、FP8、量化、并行策略等方面，TensorRT-LLM 有很强的 NVIDIA 体系支持。

## 7. TensorRT-LLM 的局限

### 7.1 上手成本更高

相比 vLLM，TensorRT-LLM 更偏底层优化栈。

你需要理解：

```text
engine build
模型转换
plugin
quantization
并行策略
NVIDIA 软件栈版本兼容
```

### 7.2 灵活性可能弱于通用 serving 框架

当模型频繁变化、业务需要快速接入新模型时，TensorRT-LLM 的构建和调试成本可能更高。

### 7.3 强依赖 NVIDIA 生态

如果你要跨 AMD、TPU、CPU 等硬件，TensorRT-LLM 不是通用路线。

## 8. SGLang 的特点

SGLang 的核心标签：

```text
structured generation
RadixAttention
复杂 LLM 程序
多轮对话 / Agent / RAG
OpenAI-compatible serving
```

它不只是推理 backend，也强调如何表达和执行复杂的 LLM 程序。

例如：

```text
多次 generate
分支控制
并行请求
结构化输出
JSON / constrained decoding
```

这些场景下，SGLang 的设计重心和 vLLM 不完全一样。

## 9. SGLang 的优势

### 9.1 对复杂 LLM 程序友好

如果业务不是简单的一问一答，而是：

```text
RAG 多阶段流程
Agent 工具调用
结构化 JSON 输出
多轮状态复用
复杂 prompt 程序
```

SGLang 的前端表达和 runtime 优化更有吸引力。

### 9.2 RadixAttention 强调前缀复用

SGLang 的 RadixAttention 关注 KV Cache 复用，适合大量共享前缀、多轮、多分支场景。

### 9.3 结构化输出能力突出

复杂业务经常需要稳定 JSON、schema、约束解码。

SGLang 在这类场景的定位更明确。

## 10. SGLang 的局限

### 10.1 学习目标更偏复杂应用 runtime

如果你的第一目标是理解基础 LLM Serving 系统，vLLM 的主链路更直观。

SGLang 的学习价值很高，但它的重点更偏“复杂 LLM 程序如何高效执行”。

### 10.2 生态和部署要结合业务评估

是否适合生产，要看模型支持、硬件、团队使用经验、已有业务接入方式和压测结果。

## 11. 三者对比维度

| 维度 | vLLM | TensorRT-LLM | SGLang |
|---|---|---|---|
| 上手难度 | 较低 | 较高 | 中等 |
| 学习 serving 主链路 | 很适合 | 偏底层优化 | 适合复杂 runtime |
| NVIDIA 极致优化 | 较强 | 很强 | 较强 |
| 跨硬件通用性 | 较好 | 弱，主要 NVIDIA | 较好 |
| OpenAI API 接入 | 支持 | 可通过服务层支持 | 支持 |
| KV Cache 优化 | PagedAttention | Paged KV cache 等 | RadixAttention |
| 结构化生成 | 支持相关能力 | 不是主要定位 | 强定位 |
| 适合新手切入 | 很适合 | 不建议第一站 | 可作为第二站 |

## 12. 应该怎么选？

### 12.1 学习 AI 推理，优先 vLLM

如果你的目标是转 AI 推理/HPC：

```text
第一站：vLLM
第二站：CUDA / attention backend / TensorRT-LLM
第三站：SGLang / 复杂 serving runtime
```

原因：

1. vLLM 的服务端结构清晰。
2. PagedAttention 是经典设计。
3. 上手和压测成本低。
4. 很适合从后端视角迁移。

### 12.2 追求 NVIDIA 极致性能，看 TensorRT-LLM

如果业务特点是：

```text
模型固定
硬件固定为 NVIDIA
吞吐/延迟目标极高
团队有 CUDA/TensorRT 经验
```

TensorRT-LLM 值得深入。

### 12.3 复杂 Agent / RAG / 结构化输出，看 SGLang

如果业务特点是：

```text
多轮程序
共享前缀多
结构化输出强
复杂控制流
```

SGLang 值得重点评估。

## 13. 面试中如何表达三者差异？

可以这样说：

```text
vLLM 更像通用高吞吐 LLM Serving 引擎，核心是 PagedAttention、continuous batching 和 KV Cache 管理。

TensorRT-LLM 更贴近 NVIDIA GPU 极致优化，通过 TensorRT engine、低精度 kernel、多卡优化来压榨性能，上手和工程成本更高。

SGLang 更强调复杂 LLM 程序和结构化生成场景，通过 runtime 级 KV 复用和约束解码优化多轮、RAG、Agent 类 workload。
```

这比简单说“哪个更快”更专业。

## 14. 学习路线建议

结合你的后端背景，建议顺序：

```text
1. vLLM：建立 LLM Serving 主链路
2. 压测：掌握 TTFT / TPOT / tokens/s / KV Cache 指标
3. CUDA 基础：理解 kernel、memory hierarchy、profiling
4. TensorRT-LLM：理解 NVIDIA 优化栈
5. SGLang：理解复杂 LLM 程序和结构化生成 runtime
```

不要一开始就跳 TensorRT-LLM。

否则很容易陷入环境、编译、engine build、版本兼容，而没有建立 serving 主线。

## 15. 本文小结

三者不是互斥关系，而是学习和业务选择的不同侧重。

你需要记住：

1. vLLM：最适合建立 LLM Serving 系统认知。
2. TensorRT-LLM：适合 NVIDIA GPU 上追求更极致性能。
3. SGLang：适合复杂 LLM 程序、结构化输出、前缀复用场景。
4. 真正选型必须用自己的 workload 压测。
5. 对转岗来说，先 vLLM，再 CUDA/TensorRT-LLM，是更稳的路线。

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM GitHub：https://github.com/vllm-project/vllm
- NVIDIA TensorRT-LLM 文档：https://nvidia.github.io/TensorRT-LLM/
- NVIDIA TensorRT-LLM GitHub：https://github.com/NVIDIA/TensorRT-LLM
- SGLang 官方文档：https://docs.sglang.ai/
- SGLang GitHub：https://github.com/sgl-project/sglang
- SGLang 论文：https://arxiv.org/abs/2312.07104
