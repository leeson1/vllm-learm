# 13｜如何定位 vLLM v0.25.0 的 GPU、CPU、KV 与网络瓶颈

> 版本基线：vLLM v0.25.0，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。排障结论必须绑定模型、硬件、workload、Model Runner、async scheduling、attention backend 和 optimization level。

## 1. 第一原则：先定位时间花在哪一层

```text
客户端/压测器
  -> 网络/反向代理
  -> API、chat template、tokenizer
  -> Engine waiting queue
  -> Scheduler / KV allocation
  -> Worker input preparation
  -> GPU model/attention/collective
  -> sampler/detokenizer/SSE
```

“GPU utilization 低”只是现象，不是根因。可能是客户端发不满、排队策略、CPU 同步、batch 太小、数据准备、通信或日志开销。

可以先用这棵树缩小范围：

```text
请求慢/吞吐低
│
├─ 客户端实际 request rate 达标吗？
│   ├─ 否 → 客户端 CPU、连接、SSE 消费、网络/代理
│   └─ 是
│
├─ waiting / queue time 持续增长吗？
│   ├─ 是 → 服务已饱和；再看 GPU、KV 或通信限制
│   └─ 否
│
├─ 主要是 TTFT 高吗？
│   ├─ 是 → API/tokenizer → 排队 → prefill/APC
│   └─ 否
│
├─ 主要是 TPOT/ITL 高吗？
│   ├─ 是 → decode batch → KV 读取 → backend/launch
│   └─ 否
│
├─ KV usage / preemption 高吗？
│   ├─ 是 → 长度、max_num_seqs、KV dtype、容量/准入
│   └─ 否
│
└─ 上 Nsight Systems
    ├─ GPU 有空洞 → CPU preparation / sync / client
    ├─ kernel 连续 → compute 或 memory bound
    └─ NCCL 占主导 → TP/PP/互联与通信
```

## 2. 先固定可复现基线

保存：

- `vllm.__version__` 和 commit；
- 完整 server 命令及启动日志；
- GPU/驱动/CUDA/PyTorch；
- 模型 revision、dtype、quantization；
- 输入/输出长度与 arrival process；
- V1/V2 Model Runner；
- sync/async scheduling；
- attention backend；
- optimization/CUDA Graph 模式。

v0.25.0 的 V2、async 和 O2 可能自动选择或回退。如果不记录最终解析结果，同一条表面命令也可能落到不同路径。

## 3. 按指标症状分类

### TTFT 高

优先拆成：

```text
客户端等待
+ API/tokenize
+ request_queue_time
+ prefill_time
+ 首次输出交付
```

常见原因：arrival rate 超载、长 prompt、chunked prefill/token budget 不合适、CPU tokenizer、APC 未命中、GPU 已饱和。

### TPOT/ITL 高

优先看：

- decode batch 形状和上下文长度；
- KV Cache 读取与 attention backend；
- running 请求是否被长 prefill 干扰；
- CPU/GPU 是否有 launch/sync 空洞；
- CUDA Graph 是否实际 replay；
- streaming 客户端是否及时读取。

### Throughput 低

区分：

```text
GPU 没被喂满
GPU/通信已经饱和
KV 容量限制有效 batch
客户端实际到达率低于目标
```

### OOM 或频繁 preemption

OOM 阶段必须先分类：

```text
权重加载 OOM
compile/CUDA Graph/workspace OOM
KV Cache 建池 OOM
运行时临时 buffer/功能组合 OOM
```

降低 `gpu_memory_utilization` 只会减少 KV 预算，不能让放不下的权重自动变小。

## 4. 第一步：证明客户端不是瓶颈

检查：

- benchmark 实际 request rate 是否达到目标；
- 客户端 CPU、连接数和 event loop；
- server 与 client 时钟/时间口径；
- SSE 是否持续消费；
- 反向代理是否缓冲 stream；
- 错误/超时是否被统计为成功；
- 同机压测是否与 server 争 CPU/网络。

低成本验证：使用另一台压测机或第二个客户端实现，比较服务端 `prompt_tokens`/`generation_tokens` 增量与客户端统计。

## 5. 第二步：看 queue，而不是先看 kernel

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:num_requests_waiting_by_reason
vllm:request_queue_time_seconds
```

典型判断：

| 现象 | 更可能的方向 |
|---|---|
| waiting 持续增长，吞吐平台 | arrival rate 超过容量 |
| waiting 低但 TTFT 高 | prefill/API/首次输出本身慢 |
| running 接近上限、KV 高 | seq/KV 容量约束 |
| queue 呈 burst 后回落 | 突发流量或客户端节奏 |

如果已经超载，调 kernel 可能只改善少量峰值；先建立 backpressure、SLO goodput 和容量余量。

## 6. 第三步：看 KV Cache 和 preemption

```text
vllm:kv_cache_usage_perc
vllm:num_preemptions
vllm:prompt_tokens_cached
vllm:prefix_cache_queries
vllm:prefix_cache_hits
```

KV 高且 preemption 增长时：

- 限制输入和最大输出长度；
- 降低 `max_num_seqs`；
- 调整外层 arrival rate/并发；
- 增加安全的 KV 显存预算；
- 评估权重量化、KV FP8 或并行策略；
- 检查是否有客户端取消但 Core 未清理的异常。

不要把“prefix cache 未释放”简单当作泄漏：`ref_cnt=0` 的 cached blocks 本就在 free queue 中，可被 LRU 驱逐和重新分配。真正要看的是可用 blocks、ref counts、请求 ownership 和是否持续无法回收。

## 7. 第四步：区分 prefill 与 decode

做两个控制 workload：

```text
长输入短输出：4096 / 32
短输入长输出：128 / 512
```

如果前者恶化明显：

- 看 input tokens/s、prefill time；
- token budget 和 chunked prefill；
- APC 正/负对照；
- tokenizer/输入处理；
- prefill attention/GEMM kernel。

如果后者恶化明显：

- 看 TPOT/ITL 随上下文增长的曲线；
- decode batch size；
- attention backend/KV dtype；
- CUDA Graph replay 与 launch gaps；
- speculative decoding 是否适合该 QPS。

## 8. 第五步：看 CPU/GPU 时间线

`nvidia-smi` 只能给粗粒度 utilization。更可靠的是 Nsight Systems：

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end repeat \
  vllm serve <model> --profiler-config.profiler cuda
```

客户端用少量请求触发：

```bash
vllm bench serve ... --profile --num-prompts 2
```

重点看：

- GPU 是否有规则性空洞；
- CPU scheduling/input preparation 是否与 GPU 重叠；
- 是否存在频繁 device synchronization；
- NCCL/all-reduce 是否占主导；
- graph capture/replay 与 eager launch 的差异；
- 哪类 kernel 占 GPU 时间。

只有确认热点 kernel 后，才使用 Nsight Compute 查看 memory throughput、occupancy、tensor core、shared memory 和 warp stalls。

## 9. 用开关做二分诊断

### 去掉 CUDA Graph/compile 干扰

```bash
vllm serve <model> --enforce-eager
```

用于判断异常是否与 graph capture、shape、compile 或 replay 有关。它通常不是性能最优配置。

### 去掉异步时间线

```bash
vllm serve <model> --no-async-scheduling
```

用于让 schedule → execute → update 更易观察。如果同步后 bug 消失，继续检查 in-flight state/buffer 生命周期，不能直接把同步配置当最终修复。

### 固定 V1 Model Runner

```bash
VLLM_USE_V2_MODEL_RUNNER=0 vllm serve <model>
```

用于区分 V1/V2 实现差异。先确认模型/功能支持，不要在生产中随意切换后只比较一条请求。

### 显式 backend

```bash
vllm serve <model> --attention-backend <supported-backend>
```

只在已核对 v0.25.0 feature table 和启动日志后使用。backend 切换常同时改变 CUDA Graph/融合能力，要把这些联动记录下来。

## 10. 四种典型模式

### CPU 喂不饱 GPU

```text
GPU 有空洞
CPU 单核或少数线程很忙
小 batch、短请求更明显
关闭重日志后改善
```

检查 tokenizer、detokenizer、Python callbacks、metrics/logging、async overlap、NUMA 和压测客户端。

### GPU compute 饱和

```text
GPU 时间线连续
GEMM/attention kernel 占主导
吞吐随并发进入平台
queue 随 arrival rate 增长
```

检查模型/dtype/quantization、TP/PP/DP、batch shape；此时继续加并发只会增加尾延迟。

### Decode memory/KV-bound

```text
长输出或长上下文 TPOT 上升
attention/KV kernels 比例增加
算力指标未满但显存带宽压力大
```

检查 GQA/MLA 模型结构、KV dtype、backend、上下文长度和 speculative decoding。

### 通信瓶颈

```text
TP 增加后 latency 恶化
NCCL/all-reduce 占比高
跨卡互联利用率高
单卡能跑时反而更快
```

检查拓扑、NVLink/PCIe、TP 粒度、PP/DP 替代方案和 batch 大小。

## 11. 排障记录模板

```text
症状：
复现命令与 workload：
预期/实际：
客户端证据：
queue/KV 指标：
CPU/GPU 时间线：
当前 runner/scheduler/backend/graph：
单变量实验：
被证伪假设：
当前最小根因：
修复或容量建议：
回归验证：
```

## 12. 源码核对入口

- `vllm/v1/metrics/loggers.py`：queue、KV、latency、token 指标。
- `vllm/v1/core/sched/scheduler.py`：waiting、running、preemption。
- `vllm/v1/core/kv_cache_manager.py`：slots 和可用 block 判断。
- `vllm/v1/engine/core.py`：同步/异步 batch queue。
- `vllm/v1/worker/gpu_worker.py`：Worker 执行与 profiling。
- `vllm/v1/worker/gpu_model_runner.py`：V1 Runner。
- `vllm/v1/worker/gpu/model_runner.py`：V2 Runner。
- `vllm/v1/cudagraph_dispatcher.py`：CUDA Graph dispatch。
- `vllm/benchmarks/serve.py`：客户端计时与 arrival process。

## 13. 验收标准

给定一组 TTFT/TPOT/queue/KV/GPU 曲线，应能：

1. 提出最多三个有层次的假设；
2. 用最低成本的单变量实验逐个证伪；
3. 在需要时采一段最小 profiler trace；
4. 把根因定位到客户端、排队、KV、CPU preparation、GPU kernel 或通信层；
5. 用原始数据验证修复没有牺牲另一个关键 SLO。

## 参考资料

- [vLLM v0.25.0 Production Metrics](https://docs.vllm.ai/en/v0.25.0/usage/metrics/)
- [vLLM v0.25.0 Optimization and Tuning](https://docs.vllm.ai/en/v0.25.0/configuration/optimization/)
- [vLLM v0.25.0 Profiling](https://docs.vllm.ai/en/v0.25.0/contributing/profiling/)
- [vLLM v0.25.0 CUDA Graphs](https://docs.vllm.ai/en/v0.25.0/design/cuda_graphs/)
