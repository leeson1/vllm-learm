# 10｜v0.25.0 的 Prefix Caching、Speculative Decoding 与 Quantization

> 版本基线：vLLM v0.25.0，源码 commit `702f4814fe54fabff350d43cb753ae3e47c0c276`。三类功能插入的层次、适用 workload 和验收指标不同；本文提供版本准确的入口，不把它们合并成一条“开启即加速”的建议。

## 1. 先把三者放回主链路

```text
Automatic Prefix Caching
  -> KVCacheManager / BlockPool
  -> 跳过重复前缀的 prefill

Speculative Decoding
  -> Scheduler / proposer / target verification / sampler
  -> 一次 target pass 尝试接受多个 token

Quantization
  -> 权重加载、linear/MoE/attention kernels、KV Cache dtype
  -> 减少容量或带宽成本，可能改变精度和 kernel 路径
```

因此它们分别优先影响：

| 功能 | 最可能改善 | 首先验证 |
|---|---|---|
| Prefix caching | 重复长前缀的 TTFT、input throughput | cached tokens、命中率、TTFT |
| Spec decode | 中低 QPS、memory-bound decode 的 ITL | acceptance、TPOT、额外显存 |
| 权重量化 | 权重容量、部分 kernel 带宽 | 能否放下、吞吐、质量 |
| KV 量化 | KV 容量、长上下文 decode 带宽 | 可容纳 tokens、TPOT、质量 |

它们插入主链路的位置不同：

```text
HTTP/API
   │
   ▼
Scheduler ───────── speculative token 调度/验证 ─────────┐
   │                                                     │
   ▼                                                     │
KVCacheManager <──── APC hash / prefix hit               │
   │                                                     │
   ▼                                                     │
Model Runner ── quantized weights / proposer ────────────┤
   │                                                     │
   ▼                                                     │
Attention Backend ── quantized KV read/write             │
   │                                                     │
   ▼                                                     │
Sampler <──────────── accept / reject speculative tokens ┘
   │
   ▼
Output
```

## 2. Automatic Prefix Caching 的 v0.25.0 行为

Engine 参数未显式指定时，v0.25.0 根据模型是否支持 prefix caching 解析默认值；常见受支持的生成模型会默认开启。最终配置应以启动日志为准。

显式控制：

```bash
vllm serve <model> --enable-prefix-caching
vllm serve <model> --no-enable-prefix-caching
```

APC 使用 full-block 链式 hash：

```text
parent hash + block token IDs + extra keys
```

v0.25.0 默认 `--prefix-caching-hash-algo sha256`。还支持 CBOR/xxhash 变体；非密码学 hash 需要在性能与碰撞/隔离风险之间做明确选择，不能只因为“更快”就在多租户服务中替换。

只有 prefix 相同才命中：

```text
相同 system prompt + 相同长文档 + 不同尾部问题  -> 可复用
文本相同但出现在 prompt 中间                    -> 不可作为同一前缀命中
```

命中只能减少 prefill 计算，不会缩短长输出本身的 decode 循环。

## 3. APC 的容量和安全边界

请求结束后，完整 cached blocks 可以在 `ref_cnt=0` 时留在 free queue，直到复用或被 LRU 驱逐。因此 APC 并不是另开一块永不释放的显存；它复用同一个 KV block pool。

多租户环境可在请求中提供 `cache_salt`。salt 会进入首 block 的 hash，只有相同 salt 的请求才能共享该前缀，可用于划分信任域并降低基于命中时间的侧信道风险。

APC 实验至少要分两种 workload：

```text
shared-prefix：固定 4K/8K 前缀，只改末尾问题
unique-prefix：长度相同，但每个 prompt 内容不同
```

如果只用随机 prompt，就无法评价 APC。

## 4. Speculative Decoding 的真实目标

普通 decode：

```text
target forward -> 1 个新 token
```

投机解码：

```text
proposer 给出多个候选
  -> target 一次验证候选序列
  -> 接受连续正确部分
  -> 拒绝点后修正并继续
```

它的目标是减少昂贵 target model 的串行 decode 轮数。收益取决于：

- proposer 延迟；
- 接受长度/acceptance rate；
- target 验证多个 token 的成本；
- batch/QPS；
- 额外模型、hidden states 和 KV 的显存；
- 采样策略、模型族和 prompt 类型。

在高 QPS、target 已被大 batch 充分利用时，额外 proposer/verification 工作可能降低总吞吐。

## 5. v0.25.0 的 speculative 配置入口

统一使用 `--speculative-config` JSON：

```bash
vllm serve <target-model> \
  --speculative-config '{
    "method":"draft_model",
    "model":"<draft-model>",
    "num_speculative_tokens":5
  }'
```

v0.25.0 的 `SpeculativeConfig` 支持多种 proposer，包括 EAGLE/EAGLE3、MTP、draft model、MLP speculator、n-gram、suffix 等。不同方法有不同必需字段和兼容性，不要复用一份 JSON 猜配置。

还要检查 async scheduling：v0.25.0 对部分 speculative 方法支持异步调度，其他组合会自动关闭或在显式强制开启时失败。启动日志必须记录最终 `async_scheduling` 状态，不能把它作为未控制变量混进性能对比。

## 6. Spec decode 应怎样验收？

固定 target model、采样参数和 workload，对比：

```text
baseline
spec tokens = 1 / 3 / 5（在方法支持范围内）
```

至少记录：

- TTFT、TPOT/ITL、E2E；
- output tokens/s 与 requests/s；
- accepted tokens / proposed tokens；
- target verification batch 形状；
- 额外显存和启动时间；
- greedy 输出一致性或任务质量。

不要只看“每次接受几个 token”；系统目标是满足质量约束下的 latency/goodput。

## 7. 权重量化在 v0.25.0 中如何进入系统？

权重量化格式通常由 checkpoint 配置识别，也可通过 `--quantization` 指定：

```bash
vllm serve <quantized-model> --quantization <method>
```

v0.25.0 文档列出的实现包括 AWQ、GPTQModel、BitsAndBytes、LLM Compressor、ModelOpt、TorchAO 等；硬件支持矩阵不同。格式名称相同也不意味着在所有 GPU 上使用相同 kernel。

量化可能带来：

- 权重显存下降，模型更容易放入单卡；
- 为 KV Cache 留出更多显存；
- 某些硬件/形状上减少带宽并提高吞吐；
- 反量化、fallback kernel 或小 batch 下额外开销；
- 精度/质量变化。

所以“显存更小”不能直接推出“延迟更低”。

## 8. KV Cache 量化与权重量化不同

KV dtype 由 `--kv-cache-dtype` 控制：

```bash
vllm serve <model> --kv-cache-dtype fp8
```

标准 BF16/FP16 KV 变为 FP8 后，理论单 token KV 容量大致减半，因此同一 pool 可容纳更多 tokens。真实收益还受 backend、page padding、混合 attention、scale 和 kernel 支持影响。

量化 scale 应优先使用与 checkpoint/校准流程匹配的数据。默认 scale 或临时校准不应在未做质量评估时直接进入生产。

需要同时验证：

- 最大可容纳 token 数和 preemption；
- TTFT/TPOT/吞吐；
- 长上下文任务质量；
- 实际 attention backend 是否支持期望的 FP8 路径；
- 是否存在 dtype conversion/fallback。

## 9. 推荐学习顺序

三者不要同时开启。建议：

```text
1. APC：最容易构造确定性 shared-prefix workload
2. 权重或 KV 量化：先做容量与质量基线
3. Spec decode：最后分析 proposer、acceptance 与异步调度
```

每次只改变一个能力，否则无法判断指标变化来自：

```text
缓存命中
模型/KV dtype
backend/kernel
spec proposer
async scheduling
CUDA Graph
```

## 10. 三个最小实验

### 10.1 APC

- 4K 相同前缀、32 token 输出；
- APC on/off；
- 记录 cached tokens、TTFT、input throughput；
- 再用 unique-prefix 作为负对照。

### 10.2 KV FP8

- 相同权重 checkpoint；
- `kv_cache_dtype=auto` 对比 `fp8`；
- 固定 arrival rate 和长度分布；
- 比较容量、preemption、TPOT 与质量。

### 10.3 Spec decode

- 低 QPS 的长输出 workload；
- baseline 对比一种受支持 proposer；
- 记录 acceptance、TPOT、额外显存；
- 再提高 QPS，观察收益是否消失。

## 11. 源码核对入口

### Prefix caching

- `vllm/v1/core/kv_cache_utils.py`：block hash。
- `vllm/v1/core/block_pool.py`：hash mapping、free queue、eviction。
- `vllm/v1/core/kv_cache_manager.py`：命中查询与 slots 分配。
- `vllm/config/cache.py`：prefix hash 和 KV dtype 配置。

### Speculative decoding

- `vllm/config/speculative.py`：`SpeculativeConfig` 与方法兼容性。
- `vllm/v1/spec_decode/`：proposer、rejection sampling 和验证路径。
- `vllm/v1/core/sched/scheduler.py`：spec tokens 的调度与状态更新。
- `vllm/config/vllm.py`：spec decode 与 async scheduling 兼容性。

### Quantization

- `vllm/model_executor/layers/quantization/`：量化方法注册与实现。
- `vllm/config/model.py`：模型量化配置解析。
- `vllm/config/cache.py`：KV Cache dtype 与 scales。
- `vllm/v1/attention/backends/`：backend 对 KV dtype 的实际支持。

## 12. 自检题

1. APC 为什么主要降低 TTFT，而不是长输出 TPOT？
2. `ref_cnt=0` 的 cached block 为什么没有独占一份额外 cache pool？
3. speculative acceptance 高为什么仍不保证系统吞吐提高？
4. 权重量化与 KV 量化分别改变哪部分容量？
5. 为什么比较 spec decode 时必须记录 async scheduling 的最终状态？

## 参考资料

- [vLLM v0.25.0 Automatic Prefix Caching](https://docs.vllm.ai/en/v0.25.0/features/automatic_prefix_caching/)
- [vLLM v0.25.0 Prefix Caching Design](https://docs.vllm.ai/en/v0.25.0/design/prefix_caching/)
- [vLLM v0.25.0 Speculative Decoding](https://docs.vllm.ai/en/v0.25.0/features/speculative_decoding/)
- [vLLM v0.25.0 Quantization](https://docs.vllm.ai/en/v0.25.0/features/quantization/)
- [vLLM v0.25.0 Quantized KV Cache](https://docs.vllm.ai/en/v0.25.0/features/quantization/quantized_kvcache/)
