# 08｜Model Runner：v0.25.0 一次 forward 前后发生什么

> 版本基线：vLLM v0.25.0，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。v0.25.0 同时存在 Model Runner V1 与 V2；本文先说明共同职责，再明确两条实现路径，避免把 V1 数据结构当成所有默认配置的唯一实现。

## 1. Model Runner 的职责

Scheduler 输出的是“本轮跑哪些请求、每个请求跑多少 token、使用哪些 KV blocks”。模型不能直接消费这份高层调度结果。

Model Runner 负责把它变成设备可执行的批次：

```text
SchedulerOutput
  -> 更新持久请求状态
  -> 组织 input_ids / positions
  -> 更新 block table 与 attention metadata
  -> 选择/填充采样参数
  -> model forward
  -> logits / sampler
  -> ModelRunnerOutput
```

所以 vLLM 的一次 forward 不只是 `model(input_ids)`，而是服务状态到 tensor 状态的转换层。

一次执行的主数据流：

```text
SchedulerOutput
      │ requests / scheduled tokens / block IDs
      ▼
┌──────────────────── Model Runner ────────────────────┐
│ 更新 persistent request state                       │
│      ▼                                               │
│ input_ids + positions + sampling tensors             │
│      │                 block table / slot mapping ────┼──┐
└──────┼───────────────────────────────────────────────┘  │
       ▼                                                  ▼
  Model Layers ─────────── read/write ─────────────> GPU KV Pool
       │
       ▼
    logits ──> sampler ──> ModelRunnerOutput
                              │ new token IDs
                              ▼
                  Scheduler.update_from_output()
```

## 2. v0.25.0 有两套 GPU Model Runner

### 2.1 Model Runner V1

主文件：

```text
vllm/v1/worker/gpu_model_runner.py
```

它使用 V1 persistent batch、`CachedRequestState`、Worker 侧 `BlockTable` 等结构。仓库的交互源码导览固定 `VLLM_USE_V2_MODEL_RUNNER=0`，主要展示这一条路径。

### 2.2 Model Runner V2

主目录：

```text
vllm/v1/worker/gpu/
  model_runner.py
  input_batch.py
```

v0.25.0 对多数受支持的非 MoE 生成模型默认选择 V2；不支持的模型或功能会回退 V1。最终选择由 `VllmConfig.use_v2_model_runner` 及兼容性检查决定，而不是只看文件名。

V2 的关键变化包括：

- 每个活跃请求获得稳定的 persistent-state row；
- persistent state 与每步真正的模型输入解耦；
- 通过 staged writes 增量更新大 tensor；
- 从设计上以 async-first、避免 CPU/GPU 同步为目标。

不能把 V1 的 `CachedRequestState`、行交换和输入准备细节直接套到 V2。

## 3. 两条路径的共同输入

无论 V1/V2，Model Runner 都需要回答以下问题：

### 3.1 本轮有哪些 token？

Scheduler 为每个请求提供 `num_scheduled_tokens`。Runner 根据请求已有 token 状态取出本轮需要计算的 token IDs，并拼成连续批次。

```text
request A: positions 128..128
request B: positions 512..767

flattened input:
[A decode token][B prefill chunk 256 tokens]
```

因此同一批次可以同时包含单 token decode 和多 token prefill chunk。

### 3.2 每个 token 的 position 是什么？

position 表示 token 在其请求序列中的逻辑位置，不是它在本轮扁平 batch 中的下标。位置相关编码和 attention mask 都依赖它。

### 3.3 历史 KV 在哪里？

Scheduler 分配 block IDs；Runner 把它们写入每请求的 block table。attention backend 通过 block table 把逻辑历史位置映射到物理 KV pages。

### 3.4 新 K/V 写到哪里？

对标准 paged attention 路径，可以把写地址理解为：

```text
slot = physical_block_id * block_size + offset_in_block
```

V1 的 `BlockTable.compute_slot_mapping()` 直接体现这条关系。V2 的状态组织不同，但仍必须向 attention backend 提供等价的页表/写入元数据。MLA、hybrid KV、KV connector 等路径会扩展这一基本模型。

## 4. Prefill 与 decode 的 forward 形状

### Prefill / chunked prefill

- 单请求本轮可能包含多个 token；
- 需要为这些 token 批量产生 K/V；
- 首次请求通常没有历史 prefix hit，命中 APC 时只计算未缓存后缀；
- 主要影响 TTFT 和 input tokens/s。

### Decode

- 普通非投机路径通常每个请求计算一个新输入 token；
- 会读取该请求的大量历史 K/V；
- 新生成 token 的 KV 在下一轮把它作为输入时写入；
- 主要影响 ITL/TPOT 和 output tokens/s。

Speculative decoding 会让“decode 每请求严格一个 token”的描述失效：验证轮可能一次处理 `1 + num_speculative_tokens` 个位置。

## 5. Model forward、logits 与 sampler

Runner 在建立 forward context 后执行模型。每层 attention 读写 KV，最终 hidden states 进入 logits 计算和 sampler。

采样不是全局固定行为。每个请求可能有不同的：

- temperature、top-p、top-k、min-p；
- repetition/presence/frequency penalty；
- logprobs；
- seed；
- structured output 等约束。

Runner 必须把请求级参数映射到正确的 batch row。输出至少要携带新 token IDs，以及 Scheduler/输出处理所需的附加信息。

Engine Core 随后调用 Scheduler 的更新逻辑，推进 `num_computed_tokens`、处理 stop/finish，并释放完成请求的 KV ownership。

## 6. Persistent batch 为什么重要？

连续两轮的请求集合通常高度相似：大部分请求仍在 decode，只有少数加入或结束。如果每轮都用 Python 从零构造大型 block tables 和采样 tensor，CPU 准备开销可能限制 GPU。

V1/V2 都利用 persistent state 做增量更新，但设计不同：

- V1 把 persistent tensors 更直接地当作模型/采样输入，状态维护复杂。
- V2 为请求分配稳定 row，再从 persistent state gather 每步输入，减少重排和异步竞争。

学习时要先理解“为何持久化”，再分别看两套具体实现。

## 7. Async scheduling 对 Runner 的要求

默认 async scheduling 会让 CPU 准备 N+1 时 GPU 仍在执行 N。这要求：

- 避免无意的 device synchronization；
- 正确管理 pinned buffer 生命周期；
- 防止 CPU 覆盖 GPU 尚未读取完的状态；
- 正确处理 preemption/finish 与 in-flight batch 的状态关系。

这也是 Model Runner V2 明确采用 async-first 设计的原因之一。

## 8. CUDA Graph 在什么位置？

Runner 会根据 compilation config、attention backend 能力和当前 batch descriptor 选择 eager、piecewise graph 或 full graph 路径。CUDA Graph 的作用是减少重复形状下的 Python/CUDA launch overhead，不会消除模型计算和 KV 读写本身。

v0.25.0 默认 optimization level 是 O2，但实际 CUDA Graph 模式可能因 backend/模型能力降级。排障时可用 `--enforce-eager` 建立无 CUDA Graph 基线。

## 9. 推荐的双路径实验

### 9.1 同步 V1 教学基线

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve <model> \
  --no-async-scheduling \
  --enforce-eager
```

用单请求、`max_tokens=3`，记录每轮：

- scheduled token 数；
- input IDs 与 positions 的形状；
- block table 新增项；
- sampler 返回 token；
- `num_computed_tokens` 更新。

### 9.2 默认 v0.25.0 路径

去掉三个教学限制重新启动。根据日志确认是否选择 V2、async scheduling 和何种 compilation/CUDA Graph 模式，再用 Nsight Systems 比较 CPU/GPU 重叠。

验收目标不是只跑通，而是能指出两条路径“职责相同、数据结构和时间线不同”的位置。

## 10. 源码核对入口

- `vllm/v1/worker/gpu_worker.py`：Worker 初始化、runner 选择和执行入口。
- `vllm/v1/worker/gpu_model_runner.py`：Model Runner V1。
- `vllm/v1/worker/gpu/model_runner.py`：Model Runner V2。
- `vllm/v1/worker/gpu/input_batch.py`：V2 persistent input state。
- `vllm/v1/worker/block_table.py`：V1 block table 与 slot mapping。
- `vllm/v1/attention/backend.py`：通用 attention metadata。
- `vllm/v1/engine/core.py`：Executor 调用与输出回收。
- `vllm/config/vllm.py`：V2 runner、async scheduling、优化级别解析。
- `vllm/v1/cudagraph_dispatcher.py`：CUDA Graph runtime dispatch。

## 11. 自检题

1. SchedulerOutput 为什么不能直接传给 `model(input_ids)`？
2. logical position 与 flattened batch index 有什么区别？
3. block table 和 slot mapping 分别服务于哪类访问？
4. V1/V2 persistent batch 的共同目标和主要差异是什么？
5. 为什么 async scheduling 容易引入 buffer race？

## 参考资料

- [vLLM v0.25.0 Model Runner V2 Design](https://docs.vllm.ai/en/v0.25.0/design/model_runner_v2/)
- [vLLM v0.25.0 Architecture Overview](https://docs.vllm.ai/en/v0.25.0/design/arch_overview/)
- [vLLM v0.25.0 CUDA Graphs](https://docs.vllm.ai/en/v0.25.0/design/cuda_graphs/)
- [vLLM v0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/)
