# 15｜从 C++/Go 后端转 AI 推理：基于 vLLM v0.25.0 的实践路线

> 技术基线：vLLM v0.25.0、commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。目标不是“看完 vLLM 文档”，而是形成能够追源码、做实验、定位瓶颈和分析 GPU 路径的推理工程能力。

## 1. 已有能力怎样迁移？

### C++/Go 后端经验

可以直接迁移到：

- 请求状态机、生命周期与取消；
- queue、backpressure、priority 和 fairness；
- 内存池、对象 ownership、引用计数；
- 并发、线程/进程、RPC 与序列化；
- 指标、日志、压测、故障复现；
- NUMA、网络和多实例部署。

对应 vLLM v0.25.0：

```text
后端请求状态       -> Request / waiting / running / finished
服务 tick/调度     -> Scheduler.schedule()
对象池/页管理      -> KVCacheManager / BlockPool
工作线程/进程      -> API Server / Engine Core / Executor / Worker
流式协议           -> AsyncLLM / OutputProcessor / SSE
性能火焰图         -> Nsight Systems / torch profiler / metrics
```

### CUDA 基础

如果已经会 kernel、memory hierarchy 和 shared memory，不需要花数周重复 vector add。下一步应直接连接推理数据路径：

- KV Cache scatter/gather；
- block table/slot mapping；
- reduction/softmax；
- GEMM/attention 的 arithmetic intensity；
- CUDA Graph 与 launch overhead；
- NCCL/TP 通信时间线。

## 2. 真正需要补的能力

### LLM 推理语义

必须理解：

```text
tokenization/chat template
decoder-only forward
prefill / decode / chunked prefill
MHA / GQA / MLA
KV Cache 容量与生命周期
sampling / stop / structured output
quantization / speculative decoding
```

不需要先成为训练算法工程师，但必须知道模型结构怎样决定显存、计算形状和 kernel。

### PyTorch/Python 生态

Python 是 vLLM 控制面、模型加载、tensor glue、测试和分析工具。目标是能：

- 看懂 tensor shape/dtype/device；
- 写最小复现和 reference implementation；
- 跑 pytest、profiler 和 benchmark；
- 在 Python 与 C++/CUDA 边界定位开销。

不必把学习重点变成 Python Web 开发。

### 性能实验方法

推理工程岗位最看重的不是“知道 PagedAttention”，而是：

```text
定义 workload/SLO
  -> 建立基线
  -> 找容量拐点
  -> 提出瓶颈假设
  -> 用单变量实验/trace 证伪
  -> 修改配置或代码
  -> 回归验证质量与其他 SLO
```

## 3. 不再按 15 篇顺序被动阅读

采用实验循环：

```text
预测 -> 运行 -> 记录 -> 读源码 -> 修改一个变量 -> 解释
```

每个主题必须有可验收产物：代码、原始数据、图表、trace、测试或报告。单独的摘要文章不是完成标准。

整条路线只保留一条主干：

```text
┌────────────┐   ┌────────────┐   ┌──────────────┐
│ 服务基线    │ → │ Token / KV │ → │ 请求与 Scheduler│
└────────────┘   └────────────┘   └──────┬───────┘
                                         ▼
┌────────────┐   ┌────────────┐   ┌──────────────┐
│ 结项讲解    │ ← │ 高级功能一个 │ ← │ 压测与瓶颈定位 │
└──────▲─────┘   └────────────┘   └──────┬───────┘
       │                                  ▼
       └────────────────────────── C++/CUDA paged-KV

每一箭头都要留下：可运行代码 + 原始数据 + 一段自己的解释。
```

## 4. 第一阶段：固定 v0.25.0 服务基线

时间：2～3 天。

任务：

1. 固定模型 revision、GPU、driver、vLLM commit；
2. 跑离线 `LLM.generate`；
3. 跑 OpenAI-compatible server、非流式和 SSE；
4. 用 `vllm bench serve` 跑 512/128 random workload；
5. 保存 `/metrics`、启动日志和原始 JSON。

产物：

```text
labs/00-baseline/
  environment.md
  serve-command.txt
  benchmark-command.txt
  raw-result.json
  baseline-report.md
```

验收：能说明 TTFT、TPOT、ITL、E2E、input/output throughput 的实际统计口径。

## 5. 第二阶段：Token 与 KV Cache

时间：1 周。

任务：

1. 用小型 PyTorch reference 对比 no-cache 与 KV-cache decode；
2. 用 Go 或 C++ 写 KV 容量计算器；
3. 对 MHA/GQA 模型分别计算 bytes/token；
4. 与 vLLM v0.25.0 启动日志中的 KV 容量估算核对；
5. 用 shared-prefix 请求验证 APC。

产物：

```text
labs/01-kv-and-decode/
  attention_reference.py
  kv_capacity.go  # 或 C++
  tests/
  apc-result.json
  notes.md
```

验收：能解释 pool 预分配、request block ownership、block table、slot mapping 和 `ref_cnt=0 but cached`。

## 6. 第三阶段：请求生命周期与 Scheduler

时间：2 周。

先使用教学基线：

```bash
VLLM_USE_V2_MODEL_RUNNER=0 \
vllm serve <model> \
  --no-async-scheduling \
  --enforce-eager
```

任务：

1. 追踪 `max_tokens=3` 的单请求；
2. 记录每轮 request state、computed tokens、scheduled tokens、blocks；
3. 运行一长一短两个请求，观察 continuous batching；
4. 触发一次 preemption，观察释放与 recompute；
5. 恢复默认配置，对照 V2/async/O2 时间线；
6. 用 Go 写简化 Scheduler simulator。

Simulator 至少实现：

```text
arrival time
waiting/running
token budget
max sequences
chunked prefill
KV block allocation
preemption
```

验收：模拟器结果能与一个实际 vLLM trace 对照，并说明哪些框架细节被简化。

## 7. 第四阶段：压测与瓶颈定位

时间：2 周。

任务矩阵：

```text
128 / 32
4096 / 32
128 / 512
4096 / 512
shared 4096 prefix / 32
```

每组递增 arrival rate，找吞吐平台和 P99 拐点。然后只选择一个变量：

- `max_num_batched_tokens`；
- `max_num_seqs`；
- APC on/off；
- KV dtype；
- V1 sync 与默认路径。

Go 可以用于实现一个 SSE 客户端，记录：

```text
request start
first SSE delta
每个 token/chunk 到达时间
finish/error
```

正式性能结论仍要与 v0.25.0 `vllm bench serve` 的口径交叉验证。

产物：原始 JSON、曲线、Nsight Systems trace 和一份因果报告。

## 8. 第五阶段：CUDA 推理数据路径

时间：2 周。

不要一开始实现完整 FlashAttention。做一个与 vLLM 数据路径对应的 microbenchmark：

```text
输入：logical positions + block table + K/V vectors
任务：计算 slot mapping，并 scatter/gather paged KV
对照：连续布局 vs 离散 pages
```

逐步验证：

1. CPU reference 正确性；
2. naive CUDA kernel；
3. coalescing/vectorized load-store；
4. 不同 head dim、block size、batch/context；
5. Nsight Systems launch 时间；
6. Nsight Compute memory throughput/occupancy；
7. 与 vLLM `reshape_and_cache` 类 kernel 的职责对照。

产物：C++/CUDA 代码、单元测试、benchmark 表和 profiler 截图/报告。

## 9. 第六阶段：只选一个高级功能

时间：1～2 周。

优先级：

```text
1. Automatic Prefix Caching
2. KV FP8 或一个权重量化格式
3. 一种 speculative decoding proposer
4. TP=1 vs TP=2
```

不要同时改 APC、量化、spec decode 和 backend。每个实验都要有 baseline、质量约束、资源变化和 SLO 结果。

## 10. 结项项目结构

```text
vllm-inference-lab/
  README.md
  notes/
    request-lifecycle.md
    scheduler.md
    kv-cache.md
  labs/
    00-baseline/
    01-kv-and-decode/
    02-request-trace/
    03-scheduler-sim/
    04-serving-benchmark/
  cmd/
    streambench/           # Go
  cuda/
    paged_kv/              # C++/CUDA
  results/
    raw/
    charts/
  reports/
    saturation-point.md
    paged-kv-profile.md
    feature-study.md
```

README 只展示可复现命令、核心结果和导航，不再堆新的长篇定义。

## 11. 10 周建议节奏

| 周 | 目标 | 必须产物 |
|---|---|---|
| 1 | 基线 + KV/token | environment、KV calculator |
| 2～3 | 请求链路 + Scheduler | 单/双请求 trace、流程图 |
| 4 | Scheduler simulator | Go 代码与测试 |
| 5～6 | benchmark/排障 | 原始数据、P99/吞吐曲线 |
| 7～8 | CUDA paged KV | kernel、correctness、profile |
| 9 | 一个高级能力 | 单变量 feature report |
| 10 | 整理与讲解 | README、复现脚本、10 分钟讲稿 |

每天 1.5～2 小时可以完成；GPU 资源不连续时，先做 source trace、Go simulator、容量计算和离线数据分析。

## 12. 简历和面试应该展示什么？

比“熟悉 vLLM、PagedAttention、CUDA”更有证据的表达：

```text
基于 vLLM v0.25.0 构建可复现推理性能实验，追踪 OpenAI API 到
Scheduler/KVCacheManager/Model Runner 的请求生命周期；设计五类长度与
到达率 workload，定位吞吐饱和点和 TTFT/TPOT 瓶颈；实现 Go 流式计时
客户端及 CUDA paged-KV scatter/gather microbenchmark，并用 Nsight 验证
CPU/GPU overlap 与显存访问瓶颈。
```

所有数字必须来自保存的原始结果，不能先写简历结论再补实验。

## 13. 转岗验收标准

达到以下能力，才算从“会部署”进入推理工程：

1. 不看文档讲清一个 token 从 HTTP 到 SSE 的路径；
2. 根据模型结构估算权重和 KV 容量；
3. 读懂 Scheduler、KVCacheManager 和一个 Model Runner 的核心状态；
4. 用 TTFT/TPOT/queue/KV/GPU trace 定位瓶颈层；
5. 修改一个配置或小段代码，并设计回归验证；
6. 写并 profile 一个和推理路径有关的 CUDA kernel；
7. 用 workload 和 SLO 比较方案，而不是背框架宣传点。

## 14. v0.25.0 源码入口

- `vllm/entrypoints/openai/`：API 路由和 serving。
- `vllm/v1/engine/`：AsyncLLM、Engine Core 和输出。
- `vllm/v1/core/sched/`：Scheduler。
- `vllm/v1/core/kv_cache_manager.py`：KV 管理入口。
- `vllm/v1/core/block_pool.py`：block pool。
- `vllm/v1/worker/`：Worker 与 Model Runner。
- `vllm/v1/attention/`：backend 和 ops。
- `vllm/benchmarks/serve.py`：在线压测。

## 参考资料

- [vLLM v0.25.0 文档](https://docs.vllm.ai/en/v0.25.0/)
- [vLLM v0.25.0 Architecture Overview](https://docs.vllm.ai/en/v0.25.0/design/arch_overview/)
- [vLLM v0.25.0 Benchmark CLI](https://docs.vllm.ai/en/v0.25.0/benchmarking/cli/)
- [vLLM v0.25.0 Profiling](https://docs.vllm.ai/en/v0.25.0/contributing/profiling/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
