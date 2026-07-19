# 02｜先跑起来：离线推理与 OpenAI-Compatible Server

> 版本基线：vLLM v0.25.0。本文的命令固定版本，避免将未来版本的参数默认值带进当前学习笔记。

这一篇的验收目标是跑通两条路径：

```text
离线：Python -> LLM.generate -> RequestOutput
在线：HTTP/OpenAI SDK -> vllm serve -> JSON 或 SSE
```

## 1. 先确认运行环境

v0.25.0 Quickstart 给出的通用前提是 Linux、Python 3.10–3.13。NVIDIA CUDA wheel 还要求 GPU compute capability 7.5 或更高；AMD、Intel、TPU 和 Apple Silicon 有各自安装路径，不能直接套用 CUDA 命令。

本文以 Linux + NVIDIA GPU 为例：

```bash
nvidia-smi
python --version
uv --version
```

同时确认：

- 模型权重能放入单卡，或已经规划 TP/PP。
- NVIDIA Driver 与要安装的 PyTorch backend 兼容。
- 下载模型所需的 Hugging Face 权限已经配置；受限模型要先接受许可并登录。
- 机器有足够的磁盘、主存和共享内存。

Apple Silicon 上官方文档指向独立的 vLLM-Metal 项目；它使用 MLX 和对应模型，不等价于本文的 CUDA 环境。

## 2. 安装固定版本

官方 v0.25.0 Quickstart 推荐用 `uv` 创建环境，并让它根据驱动选择 PyTorch backend。为了让本仓库示例可复现，这里显式固定 vLLM 版本：

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install "vllm==0.25.0" --torch-backend=auto
```

检查安装结果：

```bash
python -c "import vllm; print(vllm.__version__)"
python -m vllm.entrypoints.cli.main --help
```

第一条应该打印 `0.25.0`。如果拿到的不是这个版本，先修正环境，不要继续对照本仓库的默认值排错。

## 3. 离线批量推理

官方 Quickstart 用 `facebook/opt-125m` 演示基础 completion。它体积小，适合先验证 `LLM.generate`：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
]

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
)

llm = LLM(
    model="facebook/opt-125m",
    generation_config="vllm",
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("request_id:", output.request_id)
    print("prompt:", output.prompt)
    print("text:", output.outputs[0].text)
    print("finish_reason:", output.outputs[0].finish_reason)
```

这里有四个需要分清的对象：

- `LLM`：离线入口，初始化引擎、加载模型并驱动生成。
- `SamplingParams`：控制采样和停止条件，不控制模型加载。
- `LLM.generate`：接收一个或多个 prompt，内部加入引擎等待队列并执行。
- `RequestOutput`：每个请求的结果，包含 prompt、候选输出、token IDs 和结束原因等。

示例显式设置 `generation_config="vllm"`，是为了避免模型仓库的 `generation_config.json` 覆盖部分采样默认值。若不设置，vLLM 默认会应用模型作者提供的 generation config；这不是错误，但做对照实验时必须记录。

### 3.1 Chat 模型不要直接把 messages 传给 `generate`

`LLM.generate` 不会自动把 OpenAI `messages` 应用为 chat template。对 instruct/chat 模型应使用 `LLM.chat`，或先用 tokenizer 渲染模板：

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")
conversations = [
    [{"role": "user", "content": "用一句话解释 KV Cache"}],
    [{"role": "user", "content": "用一句话解释 continuous batching"}],
]

outputs = llm.chat(
    conversations,
    SamplingParams(temperature=0.0, max_tokens=64),
)

for output in outputs:
    print(output.outputs[0].text)
```

## 4. 启动在线服务

在线 Chat 示例使用 v0.25.0 Quickstart 同款 `Qwen/Qwen2.5-1.5B-Instruct`。它带 chat template，适合调用 `/v1/chat/completions`：

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --api-key token-abc123 \
  --generation-config vllm
```

先确认模型列表：

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer token-abc123"
```

再调用 Chat Completions：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [
      {"role": "user", "content": "用一句话解释 vLLM 是什么"}
    ],
    "temperature": 0,
    "max_tokens": 64
  }'
```

`--generation-config vllm` 让服务使用 vLLM 的采样默认值。如果希望沿用模型作者配置，可以删掉它，但压测和线上排障时应把选择记录下来。

### 4.1 用 OpenAI Python SDK 调用

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[
        {"role": "user", "content": "用一句话解释 PagedAttention"},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

### 4.2 验证流式输出

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "列出 vLLM 的三个核心组件"}],
    temperature=0,
    max_tokens=96,
    stream=True,
)

for chunk in stream:
    text = chunk.choices[0].delta.content
    if text:
        print(text, end="", flush=True)
print()
```

流式响应改变的是输出交付方式，不意味着服务端一次 forward 只服务这个连接。Engine Core 仍会把多个请求持续调度到执行批次中。

## 5. 离线与在线入口的边界

| 维度 | 离线 `LLM` | `vllm serve` |
|---|---|---|
| 调用方式 | 同一 Python 程序 | HTTP / OpenAI SDK |
| 典型用途 | 批处理、评测、实验 | 业务接入、多用户并发 |
| 输入处理 | 调用方直接传 prompt，chat 用 `LLM.chat` | API 层处理协议、chat template、tokenization |
| 输出 | `RequestOutput` 列表 | JSON 或 SSE stream |
| 额外关注 | 批量大小、生成参数 | 连接、鉴权、队列、指标和限流 |

二者共用 V1 引擎的核心调度与模型执行能力，但前端和进程拓扑不同。

## 6. 第一阶段必须理解的启动参数

### 6.1 `--dtype`

模型权重和计算使用的数据类型。`auto` 会根据模型配置选择。它与 `--kv-cache-dtype` 是不同配置：权重量化或权重 dtype 改变，不代表 KV Cache 自动使用同样的量化格式。

### 6.2 `--tensor-parallel-size`

把单个模型副本的层内参数切到多张 GPU：

```bash
vllm serve <model> --tensor-parallel-size 2
```

优先用于模型单卡放不下，或希望分摊每卡权重压力。它会增加卡间同步；如果模型本来能在单卡高效运行，不能假设 TP 一定提高总吞吐。

### 6.3 `--gpu-memory-utilization`

v0.25.0 默认值是 `0.92`。它表示当前 vLLM 实例为 model executor 使用的 GPU 显存比例，KV Cache 大小通常由该预算在启动 profiling 后推导。

```bash
vllm serve <model> --gpu-memory-utilization 0.9
```

这是“每个实例自己的上限”，不会感知同卡其他实例的预算。多个实例各设 `0.9` 并不会自动协调成安全总量。

### 6.4 `--kv-cache-memory-bytes`

如果显式设置，它直接指定每张 GPU 的 KV Cache 字节数，并覆盖 `gpu_memory_utilization` 对 KV Cache 大小的推导。它更精确，但也把容量规划责任交给使用者。

### 6.5 `--max-model-len`

限制单个序列的最大总长度，即 prompt token 与生成 token 的总和：

```bash
vllm serve <model> --max-model-len 8192
```

它是请求准入上限，不等于“为每个请求预分配 8192 token KV”。Paged KV 仍按 block 动态分配。降低它可以拒绝不需要的超长请求，并改变日志中的最大并发估算，但不会让模型权重变小。

### 6.6 `--max-num-batched-tokens` 与 `--max-num-seqs`

前者限制一次调度迭代处理的 token 数，后者限制一次迭代处理的序列数。它们是调度上限，不是对外部请求总并发或 HTTP 连接数的完整限流器。

## 7. 常见失败怎么定位？

### 7.1 启动阶段 OOM

先判断发生在加载权重、启动 profiling，还是建立 KV Cache：

- 权重放不下：换小模型/量化 checkpoint，或使用 TP/PP。
- 与其他进程争显存：清理占用，或给各实例规划互不冲突的预算。
- KV Cache 预算过大：适度降低 `gpu_memory_utilization`，但容量也会下降。
- 最大序列连一次都无法容纳：降低 `max_model_len` 或增加可用 KV 容量。

“降低 `gpu_memory_utilization`”不是所有 OOM 的通用答案；它不会压缩权重。

### 7.2 Chat API 报 chat template 错误

`/v1/chat/completions` 只适用于有 chat template 的生成模型。换用 instruct/chat 模型，或用 `--chat-template` 提供与模型匹配的模板。不要给基础 OPT 模型随意编一个模板并假设输出质量正确。

### 7.3 请求中的 model 不匹配

默认使用启动时的模型名。若配置了 `--served-model-name`，客户端应发送服务名。先查询 `/v1/models`，不要靠猜。

### 7.4 输出结果和预期默认采样不同

检查模型仓库的 `generation_config.json` 是否被应用，以及请求是否显式传了 `temperature`、`top_p`、`max_tokens` 等参数。做性能对比时固定这些值。

### 7.5 下载或启动很慢

区分模型下载、权重加载、torch.compile/CUDA Graph 初始化和 KV Cache profiling。首次启动可能包含多项一次性开销，不能直接当作稳态请求延迟。

## 8. 跑通后的观察清单

```bash
watch -n 1 nvidia-smi
curl http://localhost:8000/metrics | rg \
  'num_requests_(running|waiting)|kv_cache_usage_perc|time_to_first_token'
```

至少完成四组实验：

1. 单请求与 8 个并发请求。
2. 短 prompt 与长 prompt。
3. `max_tokens=16` 与 `max_tokens=256`。
4. 非流式与流式请求。

记录输入/输出 token 数、TTFT、E2E latency 和峰值显存。只看生成文本，无法建立 serving 性能直觉。

## 9. 本篇验收标准

- `import vllm` 显示版本 `0.25.0`。
- 离线脚本返回两个 `RequestOutput`。
- `/v1/models` 能看到服务模型。
- Chat Completions 的非流式和流式调用都成功。
- 能解释 chat template 与 `generation_config.json` 的作用。
- 能说清 `max-model-len` 为什么不是每请求 KV 预分配大小。

## 10. 源码核对入口

- `vllm/entrypoints/llm.py`：`LLM`、`generate`、`chat`。
- `vllm/entrypoints/cli/serve.py`：`vllm serve` 子命令。
- `vllm/entrypoints/openai/api_server.py`：在线 server 启动与前端。
- `vllm/v1/engine/async_llm.py`：在线异步引擎客户端。
- `vllm/engine/arg_utils.py`：Engine 参数定义与默认值解析。
- `vllm/config/cache.py`：`CacheConfig`，含 `gpu_memory_utilization` 和 KV Cache 配置。
- `vllm/config/model.py`：`max_model_len` 解析。

## 参考资料

- [vLLM v0.25.0 Quickstart](https://docs.vllm.ai/en/v0.25.0/getting_started/quickstart/)
- [vLLM v0.25.0 GPU Installation](https://docs.vllm.ai/en/v0.25.0/getting_started/installation/gpu/)
- [vLLM v0.25.0 OpenAI-Compatible Server](https://docs.vllm.ai/en/v0.25.0/serving/online_serving/openai_compatible_server/)
- [vLLM v0.25.0 Engine Arguments](https://docs.vllm.ai/en/v0.25.0/configuration/engine_args/)
