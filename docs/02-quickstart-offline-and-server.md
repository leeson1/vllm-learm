# 02｜先跑起来：离线推理与 OpenAI-Compatible Server

这一篇不深入源码，目标只有一个：先把 vLLM 跑起来，知道它最基本的两种使用方式。

```text
离线推理：Python 代码里直接 import vllm
在线服务：启动 OpenAI-compatible HTTP server
```

## 1. 环境前提

vLLM 主要面向 GPU 推理。实际安装前需要确认：

```bash
nvidia-smi
python --version
pip --version
```

通常你至少需要关注：

- GPU 型号和显存大小。
- NVIDIA Driver 版本。
- CUDA / PyTorch / vLLM wheel 的兼容性。
- 模型大小是否能放进显存。

如果只是学习，建议先用小模型，不要一开始就上 70B。

## 2. 安装 vLLM

最简单的方式是使用 pip：

```bash
pip install vllm
```

实际工程里更建议使用虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install vllm
```

安装后可以检查：

```bash
python -c "import vllm; print(vllm.__version__)"
```

如果这里失败，优先排查：

- Python 版本是否支持。
- PyTorch/CUDA 版本是否匹配。
- 是否装到了错误的虚拟环境。
- 机器是否真的有可用 GPU。

## 3. 第一种方式：离线推理

离线推理适合学习 API、验证模型、写脚本批处理。

最小例子：

```python
from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The capital of France is",
]

sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=64,
)

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print("prompt:", output.prompt)
    print("output:", output.outputs[0].text)
```

这里要理解几个对象：

### 3.1 LLM

`LLM` 是离线推理入口。它会负责加载模型、初始化引擎、申请显存、执行推理。

你可以先把它理解成：

```text
LLM = 本地模型推理客户端 + 推理引擎封装
```

### 3.2 SamplingParams

`SamplingParams` 描述生成策略，例如：

- `temperature`：随机性。
- `top_p`： nucleus sampling。
- `max_tokens`：最多生成多少 token。
- `stop`：遇到哪些字符串停止。

这些参数影响的是“怎么生成”，不是“模型怎么加载”。

### 3.3 generate

`generate` 接收一批 prompts。注意这里的“批”很重要，因为 vLLM 的性能优势来自 batch 和调度。

即使你先从单请求开始，也要尽快尝试多 prompt：

```python
prompts = [f"请解释概念 {i}" for i in range(32)]
```

观察吞吐和显存变化，会更容易理解 vLLM 的价值。

## 4. 第二种方式：在线服务

在线服务适合接业务系统。

启动服务：

```bash
vllm serve facebook/opt-125m \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --api-key token-abc123
```

然后用 OpenAI SDK 调用：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

resp = client.chat.completions.create(
    model="facebook/opt-125m",
    messages=[
        {"role": "user", "content": "用一句话解释 vLLM 是什么"},
    ],
    max_tokens=128,
)

print(resp.choices[0].message.content)
```

也可以直接 curl：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "facebook/opt-125m",
    "messages": [
      {"role": "user", "content": "hello"}
    ],
    "max_tokens": 64
  }'
```

## 5. 离线推理和在线服务的区别

| 维度 | 离线推理 | 在线服务 |
|---|---|---|
| 入口 | Python `LLM` 类 | `vllm serve` HTTP server |
| 适合场景 | 学习、批处理、评测 | 业务接入、服务化、流式输出 |
| 请求来源 | 本地代码 | 网络请求 |
| 重点 | API 和生成参数 | 并发、鉴权、监控、部署 |
| 对后端意义 | 理解引擎行为 | 理解生产部署 |

学习阶段建议两个都跑。

先用离线推理理解基本对象，再用在线服务理解服务化链路。

## 6. 常见启动参数

### 6.1 `--model`

模型名或本地路径：

```bash
vllm serve /data/models/Qwen2.5-7B-Instruct
```

### 6.2 `--dtype`

控制权重和计算精度。常见值：

```bash
--dtype auto
--dtype float16
--dtype bfloat16
```

如果不确定，先用 `auto`。

### 6.3 `--tensor-parallel-size`

多 GPU 张量并行：

```bash
vllm serve <model> --tensor-parallel-size 2
```

它表示把一个模型切到多张 GPU 上执行。适合单张卡放不下或希望提高吞吐的情况。

### 6.4 `--gpu-memory-utilization`

控制 vLLM 可以使用多少比例 GPU 显存：

```bash
--gpu-memory-utilization 0.9
```

这个参数很重要。它会影响 KV Cache 可用空间，也会影响最大并发和上下文长度。

### 6.5 `--max-model-len`

限制最大上下文长度：

```bash
--max-model-len 8192
```

上下文越长，KV Cache 可能越大。显存紧张时，不要盲目开很大。

## 7. 第一次跑不起来时怎么排查？

### 7.1 显存不够

现象可能是 OOM。

解决方向：

- 换小模型。
- 降低 `--max-model-len`。
- 降低 `--gpu-memory-utilization` 或释放其他进程。
- 使用量化模型。
- 多卡 tensor parallel。

### 7.2 模型下载慢或失败

可以提前下载模型到本地，然后用本地路径启动。

### 7.3 OpenAI SDK 调不通

检查：

- `base_url` 是否带 `/v1`。
- `api_key` 是否和 `--api-key` 一致。
- `model` 字段是否和服务端模型名匹配。
- 端口是否被占用。

### 7.4 启动慢

大模型加载权重本身就慢。第一次启动还可能涉及缓存、编译、初始化等工作。

## 8. 学习时建议观察什么？

跑起来以后，不要只看输出文本。建议同时观察：

```bash
watch -n 1 nvidia-smi
```

重点看：

- 显存占用。
- GPU utilization。
- 多请求并发时显存是否增长。
- 长 prompt 和短 prompt 的响应差异。
- `max_tokens` 增大后 decode 时间如何变化。

这会帮助你建立 LLM serving 的直觉。

## 9. 本文小结

vLLM 有两个最重要的入口：

```text
Python LLM 类：适合离线推理、脚本、学习
vllm serve：适合在线服务、业务接入、OpenAI-compatible API
```

下一篇开始进入核心原理：为什么 KV Cache 是 LLM 推理的关键资源，以及 PagedAttention 如何用“分页思想”管理它。

## 参考资料

- vLLM Quickstart：https://docs.vllm.ai/en/latest/getting_started/quickstart/
- OpenAI-Compatible Server：https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/
- vLLM Architecture Overview：https://docs.vllm.ai/en/latest/design/arch_overview/
