# 09｜Attention Backend：v0.25.0 的分页 KV、kernel 与 CUDA Graph

> 版本基线：vLLM v0.25.0，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。本文不把历史 PagedAttention CUDA kernel 当成当前唯一实现；backend 选择和能力必须以启动日志、`AttentionBackendEnum` 与 v0.25.0 feature table 为准。

## 1. Attention Backend 解决什么？

模型层表达的是 attention 语义：

```text
Q, K, V
  -> 写入本轮 K/V
  -> 读取当前请求的历史 K/V
  -> causal / sliding-window / MLA 等 attention
  -> output
```

Attention Backend 把这些语义映射到具体硬件、数据布局和 kernel。它要同时适配：

- CUDA/ROCm/XPU 等平台；
- MHA/GQA/MLA、full/sliding-window/hybrid attention；
- prefill、mixed batch、uniform decode、speculative verification；
- 不同 KV dtype 和 block size；
- CUDA Graph 能力；
- prefix cache、KV connector 等功能组合。

所以“vLLM 使用 PagedAttention kernel”过于笼统。v0.25.0 有 backend registry、统一接口和多种实现。

## 2. FlashAttention 与 PagedAttention 不是同一维度

### FlashAttention

主要描述 attention 计算的 IO-aware 实现：通过 tiling 和融合减少 HBM 往返，避免显式物化完整 attention matrix。

### Paged KV / PagedAttention

主要描述历史 KV 的分页存储与寻址：请求的逻辑 token 连续，但物理 KV pages 可以离散，由 block table 映射。

在 vLLM 中二者可以组合：某个 FlashAttention backend 同时接受 paged KV 的 block table，并通过 slot mapping 把新 K/V scatter 到预分配 cache。

不要把它们理解成二选一：

```text
FlashAttention：如何高效计算
Paged KV：历史 K/V 如何布局与定位
```

把读写路径放在一张图里：

```text
当前 token ──> Q, K, V
              │  │  │
              │  └──┴─ + slot mapping ──> KV cache update ──┐
              │              （写路径）                       │
              │                                              ▼
              │                         ┌── 预分配 GPU KV Pool ──┐
              │                         │ P7 │ P2 │ P11 │ ...    │
              │                         └─────────┬──────────────┘
              │                                   │ 历史 K/V
              │  + block table                    │
              │  （逻辑 block -> 物理 page）       │
              ▼                                   ▼
                     Attention Backend
                              │
                              ▼
                       attention output

  写新 K/V 看 slot mapping；读历史 K/V 看 block table。
```

## 3. v0.25.0 的 backend 选择

配置入口是 `--attention-backend`，对应 `vllm/v1/attention/backends/registry.py` 中的枚举与注册表：

```bash
vllm serve <model> --attention-backend FLASH_ATTN
```

通常应先让 vLLM 自动选择。显式指定 backend 之前要确认：

- 当前 GPU 架构和已安装库支持；
- 模型 head size、dtype、attention 类型支持；
- KV Cache dtype 支持；
- prefix caching、spec decode、sliding window 等功能支持；
- CUDA Graph 支持级别；
- 选择没有触发回退或启动失败。

不要根据 backend 名字猜性能。相同 backend 在不同模型、batch 形状、上下文长度和硬件上可能走不同 kernel。

## 4. 写路径：新 K/V 如何进入分页缓存？

以标准 paged KV 路径为例，Runner 为本轮 token 提供 slot mapping：

```text
slot = physical_block_id * block_size + offset
```

每层产生新 K/V 后，backend 的 KV cache update 路径按 slot 把它们 scatter 到该层预分配的 KV tensor。`vllm/v1/attention/ops/paged_attn.py` 和不同 backend 的 `do_kv_cache_update()` 展示了这层接口。

slot mapping 服务的是“本轮写哪里”，不能替代 block table。

## 5. 读路径：历史 K/V 如何参与 attention？

block table 保存：

```text
request row + logical block index -> physical block ID
```

backend 结合 block table、sequence length、query start location 等元数据读取历史 pages。逻辑序列 `[0..N)` 不要求对应连续 physical blocks。

因此：

```text
slot mapping：新 K/V 写路径
block table：历史 K/V 读路径
```

具体 backend 可能对 block table 做转换、建立额外 workspace，或针对 prefill/decode 使用不同 kernel；不要假设所有实现都完全共享一种 tensor 形状。

## 6. Prefill、mixed 与 decode

### Prefill

- query token 多；
- attention 计算规模大；
- 常更偏 compute-heavy；
- 关注 TTFT、input tokens/s 和大矩阵 kernel 效率。

### Uniform decode

- 普通路径通常每请求一个 query token；
- 需要扫描越来越长的历史 KV；
- 常更偏 memory-bandwidth/launch-overhead；
- 关注 ITL/TPOT 和 output tokens/s。

### Mixed batch

chunked prefill 与 decode 可以同轮执行。backend 必须根据 metadata 正确处理非均匀 query lengths；这也是 CUDA Graph 能力不能只用 batch size 描述的原因。

## 7. v0.25.0 的 CUDA Graph 模式

CUDA Graph 通过 capture/replay 降低重复执行形状下的 CPU 和 kernel launch 开销。它不减少 GEMM/attention 的数学工作，也不增加 KV 容量。

v0.25.0 的 `CUDAGraphMode` 包含：

- `NONE`：不使用 CUDA Graph；
- `PIECEWISE`：attention 等不兼容部分留在图外；
- `FULL`：完整图；
- `FULL_DECODE_ONLY`：只为 uniform decode 使用 full graph；
- `FULL_AND_PIECEWISE`：decode full graph，其他 batch 使用 piecewise。

V1 默认 optimization level 是 O2，配置可采用 `FULL_AND_PIECEWISE`，但最终会根据模型和 attention backend 的 `AttentionCGSupport` 自动调整。不能仅凭“默认 O2”断言所有请求都 replay full CUDA Graph。

调试基线：

```bash
vllm serve <model> --enforce-eager
```

精确控制示例：

```bash
vllm serve <model> \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## 8. CUDA Graph 为什么需要 batch descriptor？

v0.25.0 使用 `BatchDescriptor` 描述可复用执行形状，核心信息包括 token 数、请求数和 batch 是否 uniform。dispatcher 根据当前 batch 和已经捕获的 keys，在 FULL、PIECEWISE 与 NONE 之间选择。

动态请求仍然能使用 CUDA Graph，是因为系统只对支持的形状做 capture/replay，并把输入写入固定或可复用 buffer；无法匹配时可以回退 eager/piecewise。

代价包括：

- 启动 warmup/capture 时间；
- graph 与 workspace 显存；
- 多种 batch shape 的捕获集合；
- backend/功能不兼容时的降级；
- 调试时间线更复杂。

## 9. 性能判断顺序

不要看到 TPOT 高就直接换 backend。建议依次检查：

1. workload 的输入/输出长度与 arrival rate；
2. waiting queue、KV usage 和 preemption；
3. 实际选择的 backend、dtype、runner 和 graph mode；
4. Nsight Systems 中 CPU/GPU 空洞与同步点；
5. 主要 kernel 占时及调用形状；
6. 最后才用 Nsight Compute 分析带宽、occupancy、warp stalls。

## 10. 最小对照实验

固定模型、GPU 和 workload，运行：

```text
A. 自动 backend + 默认 O2
B. 自动 backend + --enforce-eager
C. 一个确认兼容的显式 backend + 默认 O2
```

分两类 workload：

- `input=128, output=512`：观察 decode/launch/KV 读取；
- `input=4096, output=32`：观察 prefill 和 mixed batch。

记录 TTFT、TPOT、吞吐、启动时间、峰值显存和 Nsight 时间线。验收目标是指出变化来自 backend、graph、batch shape 还是其他瓶颈，而不是只给最快配置。

## 11. 源码核对入口

- `vllm/v1/attention/backend.py`：backend 接口、metadata、KV cache update、CG support。
- `vllm/v1/attention/backends/registry.py`：backend 注册与枚举。
- `vllm/v1/attention/backends/flash_attn.py`：FlashAttention backend 示例。
- `vllm/v1/attention/ops/paged_attn.py`：paged KV 写入操作入口。
- `vllm/v1/attention/ops/`：Triton/平台相关 attention 与 cache ops。
- `vllm/v1/worker/block_table.py`：V1 block table 与 slot mapping。
- `vllm/v1/cudagraph_dispatcher.py`：CUDA Graph runtime dispatch。
- `vllm/config/attention.py`：attention 配置。
- `vllm/config/compilation.py`：编译与 CUDA Graph 模式。
- `vllm/config/vllm.py`：optimization level 默认值和兼容性处理。

## 12. 自检题

1. FlashAttention 与 paged KV 分别解决什么问题？
2. slot mapping 和 block table 为什么缺一不可？
3. mixed batch 为什么比 uniform decode 更难复用 full CUDA Graph？
4. `--enforce-eager` 能帮助区分哪类问题？
5. 为什么不能说“vLLM v0.25.0 所有 attention 都走同一个 PagedAttention kernel”？

## 参考资料

- [vLLM v0.25.0 Attention Backends](https://docs.vllm.ai/en/v0.25.0/design/attention_backends/)
- [vLLM v0.25.0 CUDA Graphs](https://docs.vllm.ai/en/v0.25.0/design/cuda_graphs/)
- [vLLM v0.25.0 Optimization Levels](https://docs.vllm.ai/en/v0.25.0/design/optimization_levels/)
- [vLLM v0.25.0 Paged Attention 历史设计页](https://docs.vllm.ai/en/v0.25.0/design/paged_attention/)
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135)
- [PagedAttention 论文](https://arxiv.org/abs/2309.06180)
