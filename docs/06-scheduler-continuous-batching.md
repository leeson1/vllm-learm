# 06｜Scheduler：continuous batching、prefill/decode 混排、chunked prefill

这篇开始进入第二阶段：深入 vLLM 的关键模块。

如果只用一句话概括 Scheduler：

```text
Scheduler 决定下一轮 GPU forward 跑哪些请求、每个请求跑多少 token、需要多少 KV block，以及是否要等待、抢占或释放资源。
```

对后端开发来说，它不是一个简单队列，而是一个“带资源约束的实时调度器”。

## 1. 为什么 LLM Serving 必须有 Scheduler？

普通 HTTP 服务里，请求通常是：

```text
request -> handler -> response
```

但是 LLM 生成不是一次算完：

```text
request
  -> prefill prompt
  -> decode token 1
  -> decode token 2
  -> decode token 3
  -> ...
```

一次请求会跨很多轮 GPU forward。

如果没有调度器，最直接的写法是：

```text
来一个请求 -> 单独跑完整个生成 -> 返回
```

这样 GPU 利用率会很差。因为 decode 阶段每次只生成一个 token，单请求 batch 太小，GPU 很容易吃不满。

所以 vLLM 需要把很多请求合并起来，让每一轮 GPU forward 都尽量有足够工作量。

## 2. 传统 batching 的问题

传统静态 batch 可以理解为：

```text
收集一批请求
  -> 一起 prefill
  -> 一起 decode
  -> 等全部结束
  -> 再收下一批
```

问题是 LLM 请求长度差异非常大：

```text
A：prompt 100 token，输出 50 token
B：prompt 3000 token，输出 100 token
C：prompt 200 token，输出 2000 token
```

静态 batch 会出现几个浪费：

1. 短请求早就结束了，但 batch slot 被长请求拖住。
2. 新请求来了，只能等当前 batch 完成。
3. 长 prompt 请求会阻塞 decode 请求，导致正在流式输出的用户卡顿。
4. batch 内部 token 数差异大，GPU 计算形态不稳定。

这就是为什么 vLLM 使用 continuous batching。

## 3. continuous batching 是什么？

continuous batching 可以理解为“动态 batch”：

```text
每一轮调度：
  1. 把已经完成的请求移出 batch
  2. 把新来的请求加入 batch
  3. 给每个请求分配本轮要处理的 token
  4. 组成一次 GPU forward
```

它不是“攒够 N 个请求再跑”，而是持续维护一个正在执行的请求集合。

简化图：

```text
step 1: A B C      -> GPU forward
step 2: A B C D    -> D 加入
step 3: B C D      -> A 完成
step 4: B C D E F  -> E/F 加入
```

后端类比：

```text
游戏服务器 tick：
  每一帧从各种队列里拿任务，根据预算执行一部分。

vLLM Scheduler：
  每一轮从 waiting/running 队列里拿请求，根据 token budget 和 KV block 预算执行一部分。
```

## 4. Scheduler 管什么资源？

Scheduler 至少要管三类资源。

### 4.1 请求状态

一个请求通常会经历：

```text
waiting -> running -> finished
           |
           v
        preempted / waiting for resource
```

不同阶段需要的资源不同：

- prefill：处理 prompt，通常一次处理很多输入 token。
- decode：每轮生成少量 token，通常每个序列一个 token。
- finished：释放 KV block、返回最终结果。

### 4.2 token budget

一次 GPU forward 不能无限大。

`max_num_batched_tokens` 可以粗略理解为：

```text
本轮最多处理多少 token
```

如果 budget 很大：

- prefill-heavy 场景吞吐可能更好。
- 单轮 GPU forward 变重。
- decode 请求可能等更久。

如果 budget 很小：

- 单轮耗时更短。
- 流式输出可能更平滑。
- 整体吞吐可能下降。

### 4.3 KV Cache block

每个请求处理 token 时，都需要 KV Cache。

prefill 会批量写入很多 KV；decode 每生成一个 token，也会追加新的 KV。

Scheduler 在决定是否调度某个请求前，需要判断：

```text
这轮要处理的 token 是否有足够 KV block 可用？
```

如果没有，就要等待、抢占，或者让其它请求先跑。

## 5. prefill 和 decode 的差异

vLLM Scheduler 的难点之一是：prefill 和 decode 的计算形态不同。

### 5.1 prefill

prefill 处理 prompt：

```text
输入：一大段 prompt token
输出：第一阶段 KV Cache + 可能的第一个 token
```

特点：

- token 数可能很多。
- 更像大矩阵计算。
- 主要影响 TTFT，也就是首 token 延迟。

### 5.2 decode

decode 逐 token 生成：

```text
输入：上一个 token + 历史 KV Cache
输出：下一个 token + 追加 KV Cache
```

特点：

- 每个请求每轮通常只处理少量 token。
- 会反复读取历史 KV Cache。
- 主要影响 TPOT，也就是每个输出 token 的间隔。

### 5.3 混排的矛盾

如果只顾 prefill：

```text
新请求 TTFT 可能好，但老请求流式输出会卡。
```

如果只顾 decode：

```text
老请求输出很平滑，但新请求一直排队，TTFT 变差。
```

所以 Scheduler 要在两者之间做权衡。

## 6. chunked prefill 解决什么问题？

假设来了一个超长 prompt：

```text
prompt = 12000 tokens
```

如果一次性 prefill，它可能占满本轮 token budget，导致其它 decode 请求无法及时执行。

chunked prefill 的思路是：

```text
不要一次性处理完整 prompt，而是切成多个 chunk。

12000 tokens
  -> chunk 1: 2048
  -> chunk 2: 2048
  -> chunk 3: 2048
  -> ...
```

这样 Scheduler 可以把长 prefill 拆开，插入到多个调度轮次中。

简化示意：

```text
round 1: decode A/B/C + prefill D chunk 1
round 2: decode A/B/C + prefill D chunk 2
round 3: decode A/B/C + prefill D chunk 3
```

它的核心价值：

1. 避免长 prompt 独占 GPU。
2. 降低 decode 请求的排队抖动。
3. 让 TTFT 和 TPOT 之间更容易折中。

## 7. 从队列视角理解 Scheduler

可以把 Scheduler 简化成几个队列：

```text
waiting_queue：新请求，尚未开始 prefill
running_queue：已经有 KV Cache，正在 decode 或继续 prefill
finished_queue：已经结束，等待回收资源和返回
```

每一轮调度大致做这些事：

```text
while token_budget 还有剩余:
    优先选择一部分 running 请求做 decode
    再选择 waiting 请求做 prefill
    如果 prompt 太长，就只处理一个 chunk
    检查 KV block 是否足够
    生成本轮要执行的 schedule
```

注意：真实 vLLM 的实现比这个复杂得多，比如多模态、LoRA、spec decode、prefix cache、分布式部署都会影响调度。但先用这个模型理解是足够的。

## 8. 和游戏服务器 tick 的类比

你可以把一次 GPU forward 理解成一帧 tick。

```text
游戏服务器 tick：
  固定时间预算内处理玩家输入、AI、定时器、网络包。

vLLM scheduler step：
  固定 token / seq / KV block 预算内处理 prefill、decode、释放和抢占。
```

两者都要面对：

- 队列积压。
- 长任务阻塞短任务。
- 资源预算不足。
- 吞吐和延迟的权衡。
- 公平性和优先级。

所以你以前做游戏后端的经验不是没用，而是可以迁移到 LLM Serving 的调度问题上。

## 9. 常见调优参数怎么影响 Scheduler？

### 9.1 `--max-num-batched-tokens`

控制单轮 token budget。

```bash
vllm serve <model> --max-num-batched-tokens 8192
```

直觉：

```text
更大 -> 单轮能塞更多 token，吞吐可能更高
更小 -> 单轮更轻，延迟可能更稳定
```

### 9.2 `--max-num-seqs`

控制单轮最多并发序列数。

```bash
vllm serve <model> --max-num-seqs 128
```

直觉：

```text
更大 -> 并发潜力更高，但 KV Cache 压力更大
更小 -> 更保守，尾延迟可能更可控
```

### 9.3 `--enable-chunked-prefill`

用于控制是否启用 chunked prefill。

直觉：

```text
长 prompt 场景下，它可以减少长 prefill 对 decode 的阻塞。
```

具体默认行为和参数细节要以当前 vLLM 版本文档为准，因为 vLLM V1 仍在持续演进。

## 10. 读源码时看什么？

读 Scheduler 不要一上来陷进所有分支。

建议先抓住 5 个问题：

1. 请求如何从 waiting 进入 running？
2. 每轮 token budget 如何扣减？
3. prefill 和 decode 如何混排？
4. KV block 分配失败时怎么办？
5. 请求完成后资源在哪里释放？

源码阅读路径可以先围绕这些关键词：

```text
scheduler
schedule
waiting
running
num_batched_tokens
num_seqs
block_manager / kv_cache_manager
```

不要把 Scheduler 当成一个孤立模块。它一定会和 KV Cache Manager、Model Runner、Output Processor 联动。

## 11. 本文小结

Scheduler 是 vLLM 的“资源调度中枢”。

它解决的不是“怎么调用模型”这么简单，而是：

```text
在有限 GPU 显存、有限 token budget、动态请求长度下，如何让 GPU 尽量忙，同时让用户延迟可接受。
```

你需要重点记住：

1. continuous batching 让请求可以动态进出 batch。
2. prefill 主要影响 TTFT，decode 主要影响 TPOT。
3. chunked prefill 把长 prompt 拆开，避免阻塞 decode。
4. Scheduler 的每个决策都受 KV Cache block 约束。
5. 学 vLLM 调度，本质是在学高性能服务端资源调度。

## 参考资料

- vLLM 官方文档：https://docs.vllm.ai/
- vLLM V1 文档：https://docs.vllm.ai/en/latest/usage/v1_guide.html
- vLLM Engine Arguments：https://docs.vllm.ai/en/latest/configuration/engine_args.html
- PagedAttention 论文：https://arxiv.org/abs/2309.06180
