# 第四章：Token、Embedding、Logits、Sampling

本章从推理链路中最“入口”和最“出口”的部分讲起：

```text
文本 / 多模态输入
→ token ids / multimodal embeddings
→ embedding
→ Qwen3.6 language model
→ hidden states
→ lm_head
→ logits
→ logits processor
→ sampling
→ next token
→ detokenize
```

对传统 LLM 来说，这一章主要讲 token、embedding、logits、softmax、采样。对 **Qwen/Qwen3.6-35B-A3B** 来说，还必须额外注意两点：第一，它是 **Causal Language Model with Vision Encoder**，因此输入不只有纯文本 token；第二，它的 token embedding 和 LM output 都是 **248,320 padded**，hidden size 是 **2048**，所以 logits 维度非常大，sampling 本身也会成为真实 serving 中不可忽略的计算环节。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

可以把 LLM 看成一个“只认识编号、不认识文字”的工厂。

用户输入：

```text
帮我解释 KV Cache
```

模型内部不会直接处理这些汉字，而是先把它们变成 token id：

```text
[1532, 37046, 912, 128xxx, ...]
```

每个 token id 再去查一张“向量字典表”：

```text
1532   → 一个 2048 维向量
37046  → 一个 2048 维向量
912    → 一个 2048 维向量
```

这张字典表就是 **embedding table**。

模型经过 40 层 Qwen3.6 hybrid block 后，最后会输出一个 hidden state。这个 hidden state 再通过 **LM head** 映射成对整个词表的打分：

```text
“的”     8.7
“是”     7.9
“缓存”   7.4
“。”     6.1
...
```

这些分数叫 **logits**。logits 不是概率，softmax 后才是概率。sampling 再根据 temperature、top-k、top-p 等规则，从概率分布中选出下一个 token。

---

## 二、数学直觉

### 1. Token id 是离散符号

token id 本身没有连续数学意义。比如：

```text
token id = 100
token id = 101
```

不代表 101 比 100 “语义更大”。它只是查表索引。

### 2. Embedding 是查表

设 vocabulary size 是 $V$，hidden size 是 $H$，embedding table 是：

$$
E \in \mathbb{R}^{V \times H}
$$

某个 token id 为 $i$，它的 embedding 是：

$$
x_i = E[i]
$$

对 Qwen3.6-35B-A3B，模型卡给出 hidden dimension 为 2048，token embedding 为 248,320 padded，因此可以把 embedding table 理解成接近：

$$
E \in \mathbb{R}^{248320 \times 2048}
$$

注意这里的 248,320 是 padded vocabulary size，不一定等于 tokenizer 原始词表语义上的“自然词数”。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. LM head 把 hidden state 投回词表空间

最后一层 hidden state：

$$
h_t \in \mathbb{R}^{H}
$$

LM head 权重：

$$
W_{\text{lm}} \in \mathbb{R}^{V \times H}
$$

logits：

$$
z_t = W_{\text{lm}} h_t
$$

所以：

$$
z_t \in \mathbb{R}^{V}
$$

对 Qwen3.6，单个 token 的 logits 可以理解成：

$$
z_t \in \mathbb{R}^{248320}
$$

vLLM v0.20.1 的 `Qwen3_5ForCausalLMBase` 在最后一个 pipeline rank 上，如果没有 tie word embedding，会创建 `ParallelLMHead(config.vocab_size, config.hidden_size)`；随后 `compute_logits` 调用 `self.logits_processor(self.lm_head, hidden_states)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 4. Softmax 把 logits 变概率

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}
$$

如果加入 temperature：

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{V}\exp(z_j / \tau)}
$$

其中：

| temperature | 效果 |
|---:|---|
| 接近 0 | 更接近 greedy，选最大 logit |
| 1.0 | 原始分布 |
| 大于 1 | 分布更平，随机性更强 |

vLLM v0.20.1 的 `SamplingParams` 文档说明：temperature 控制随机性，较低更确定，较高更随机，0 表示 greedy sampling；top-p 控制考虑的累计概率，top-k 控制考虑的最高分 token 数量。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

---

## 三、张量形状

### 1. 传统教学形状

普通教学里常写：

```text
input_ids:      [B, T]
embeddings:     [B, T, H]
hidden_states:  [B, T, H]
logits:         [B, T, V]
next_logits:    [B, V]
```

其中：

| 符号 | 含义 |
|---|---|
| $B$ | batch size |
| $T$ | sequence length |
| $H$ | hidden size |
| $V$ | vocab size |

### 2. vLLM serving 里的 flattened 形状

vLLM 推理时更常见的是 flatten 后的 token 视角：

```text
input_ids:      [num_tokens]
positions:      [num_tokens]
hidden_states:  [num_tokens, hidden_size]
logits:         [num_tokens, vocab_size]
```

`qwen3_5.py` 中 forward docstring 明确说 `input_ids` 是 flattened / concatenated input ids，`positions` 也是 flattened / concatenated position ids；如果启用 mrope，positions 形状可能是 `(3, seq_len)`，否则是 `(seq_len,)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 3. 固定案例 Qwen3.6 的核心形状

```text
hidden_size = 2048
padded vocab / embedding size = 248320
LM output = 248320 padded
num_layers = 40
```

所以单步 decode 时，可以粗略理解成：

```text
hidden_states: [N, 2048]
logits:        [N, 248320]
sampled_ids:   [N]
```

这里 $N$ 是当前 batch 内要采样的 token 数，通常接近当前正在 decode 的请求数；如果是 prefill logprobs 或 prompt scoring，$N$ 也可能包含 prompt token。Qwen3.6 的这些模型尺寸来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 4. 多模态输入时的形状

Qwen3.6 是带 Vision Encoder 的 causal LM。vLLM v0.20.1 的 `Qwen3_5MoeForConditionalGeneration` 会构造 `Qwen3_VisionTransformer`，文本 embedding 先由 `self.language_model.embed_input_ids` 得到；如果存在 `multimodal_embeddings`，则调用 `_merge_multimodal_embeddings` 合并到输入 embedding 中。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

因此多模态时可以理解成：

```text
text input_ids
→ text embeddings: [N_text, 2048]

image / video
→ vision encoder
→ multimodal embeddings: [N_mm, 2048]

merge
→ input embeddings: [N_text + N_mm, 2048]
```

这里的核心是：**视觉内容最终也必须变成 language model 可消费的 hidden-size 向量**。但具体视觉 token 数量取决于图像/视频处理配置，本章不强行展开；原因是视觉 tokenization、grid、mrope 会在多模态部署章节中更自然。

---

## 四、计算流程

### 1. 输入到 embedding

```text
用户文本
→ tokenizer
→ input_ids
→ embedding table lookup
→ inputs_embeds
```

源码映射：

```text
Qwen3_5MoeForConditionalGeneration.embed_input_ids
→ _embed_text_input_ids
→ language_model.embed_input_ids
→ possible _merge_multimodal_embeddings
```

vLLM v0.20.1 源码确认：`embed_input_ids` 先调用 `_embed_text_input_ids`，使用 `self.language_model.embed_input_ids` 生成文本 embedding；若没有 multimodal embeddings，直接返回文本 embedding；否则再调用 `_merge_multimodal_embeddings`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 2. embedding 到 hidden states

```text
inputs_embeds
→ Qwen3.6 40 层语言主干
→ hidden_states
```

这里的 40 层不是普通 Transformer，而是：

```text
10 × (
  3 × (Gated DeltaNet → MoE)
  1 × (Gated Attention → MoE)
)
```

这一点来自 Qwen3.6 模型卡，后续第五章会进入 attention / gated attention / RoPE，第九章会进入 MoE。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. hidden states 到 logits

```text
hidden_states
→ lm_head
→ logits
→ logits processor
```

vLLM v0.20.1 源码中，`Qwen3_5ForCausalLMBase.forward` 调用 `self.model(...)` 得到 hidden states，`compute_logits` 再调用 `self.logits_processor(self.lm_head, hidden_states)`。`LogitsProcessor` 的职责说明包括：从 hidden states 收集 logits、必要时 scale logits、应用 logits processors。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

`LogitsProcessor._get_logits` 会调用 `lm_head.quant_method.apply(...)` 得到 logits，然后对 tensor parallel 的 logits 做 gather / all-gather，并去掉 vocab padding。也就是说，Qwen3.6 的 padded LM output 虽然是模型结构事实，但 vLLM 的 logits processor 有“remove paddings in vocab”的路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py))

### 4. logits 到 next token

vLLM v1 sampler 的 docstring 把采样流程写得很清楚：先可选计算 logprobs，把 logits 转成 float32，应用 allowed token ids、bad words、logit processors、penalties，然后采样；采样内部会处理 greedy、temperature、argmax-invariant processors、top-k/top-p，最后返回 `SamplerOutput`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

简化流程：

```text
logits
→ allowed token mask / bad words mask
→ repetition / frequency / presence penalties
→ temperature
→ top-k / top-p
→ softmax
→ sample token id
```

---

## 五、显存布局

### 1. Embedding table

Qwen3.6 的 token embedding 是 248,320 padded，hidden size 是 2048。按 BF16 粗略估算：

$$
248320 \times 2048 \times 2 \approx 1.02 \text{ GB}
$$

这只是 token embedding 一项的粗估，不包括 LM head 是否 tied、MoE expert 权重、attention/GDN 参数、vision encoder、KV cache、runtime workspace 等。Qwen3.6 的 embedding 和 LM output 维度来自模型卡；是否 tie word embeddings 则由 vLLM 源码中的 `config.tie_word_embeddings` 分支决定。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. Logits tensor

logits 的显存通常是临时显存，但 vocabulary 很大时，它仍然昂贵。

例如：

```text
N = 128 个当前要采样的 token
V = 248320
dtype = FP32 for sampling
```

那么 logits 大小粗略为：

$$
128 \times 248320 \times 4 \approx 127 \text{ MB}
$$

这解释了为什么 sampler 会关注 dtype、in-place 修改、top-k/top-p kernel、logprobs 是否返回等细节。vLLM v1 sampler 源码中明确会将 logits 转成 float32，并且 sampling / logprobs 过程会根据请求需求保留或处理 logits。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

### 3. Logprobs 会显著增加输出和计算负担

当用户请求 `logprobs` 或 `prompt_logprobs` 时，vLLM 需要保留更多概率信息。`SamplingParams` 文档说明：`logprobs` 会返回每个输出 token 的若干最可能 token 及所选 token 的 log probability；设置为 `-1` 时返回全部 vocab log probabilities。`prompt_logprobs` 对 prompt token 也有类似含义。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

工程结论：
**不要在高吞吐线上默认打开全量 logprobs。** 对 Qwen3.6 这种 248,320 padded output 维度的模型，全量 logprobs 的内存、带宽和序列化成本都很高。

### 4. 多模态 embedding 的显存

多模态时，vision encoder 产生的 embedding 会合并进语言模型输入序列。vLLM 源码确认了 `_merge_multimodal_embeddings` 路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

工程结论：
**图像/视频不是“免费 prompt 文本”**。它们会增加输入 token/embedding 数量，影响 prefill、position、KV/state cache，以及 batch 调度。

---

## 六、vLLM 源码映射

本章源码地图如下：

| 概念 | vLLM v0.20.1 路径 | 类/函数 | 作用 |
|---|---|---|---|
| 文本 embedding | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.embed_input_ids` | 调用底层 model 的 embedding |
| 多模态 embedding 合并 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration.embed_input_ids` | 文本 embedding 与 multimodal embeddings 合并 |
| Vision Encoder | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer` | 图像/视频 encoder |
| LM head | `vllm/model_executor/models/qwen3_5.py` | `ParallelLMHead` | hidden state 到 vocab logits |
| logits processor | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor` | gather logits、scale、去 padding |
| sampling params | `vllm/sampling_params.py` | `SamplingParams` | temperature、top-p、top-k、penalty、stop 等 |
| sampler | `vllm/v1/sample/sampler.py` | `Sampler` | logits 到 sampled token |
| top-k/top-p | `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` | top-k/top-p 过滤与随机采样 |

尤其要注意两个源码细节。

第一，`LogitsProcessor._get_logits` 会先通过 LM head 的 quant method 得到 logits，再做 tensor parallel gather / all-gather，最后去掉 vocab padding。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py))

第二，vLLM v1 sampler 的流程不是“直接 softmax 然后采样”这么简单，而是包含 allowed token mask、bad words、logits processors、penalties、temperature、top-k/top-p、logprobs gathering 等步骤。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

---

## 七、NVIDIA GPU 执行视角

### 1. Embedding lookup

embedding lookup 本质是按 token id 从 embedding table 读向量：

```text
input_ids: [N]
embedding table: [V, H]
output: [N, H]
```

它不是 Tensor Core 友好的大 GEMM，而更像不连续内存读取。batch 越乱、token id 越分散，越依赖 HBM 读取效率。

### 2. LM head GEMM

hidden states 到 logits 是一个大矩阵乘：

```text
hidden_states: [N, 2048]
lm_head:       [248320, 2048]
logits:        [N, 248320]
```

这一步非常适合 Tensor Core，但输出矩阵很大。对 Qwen3.6 而言，vocab 输出维度 248,320 padded，意味着 LM head 和 logits 都不是小开销。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Tensor Parallel 下的 logits gather

当使用 tensor parallel 时，LM head 通常按 vocabulary 或 hidden dimension 分片；vLLM 的 `LogitsProcessor` 源码中存在 `_gather_logits`，会根据平台选择 `tensor_model_parallel_all_gather` 或 `tensor_model_parallel_gather`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py))

工程含义：

```text
多卡并行不仅切模型计算
还要在 logits / sampling 前后处理跨卡结果
```

如果返回 logprobs 或 full logits，这部分通信和内存压力会更明显。

### 4. Sampling kernel

sampling 看似只是“选一个 token”，但真实步骤包括：

```text
logits float32
penalty
mask
temperature
top-k / top-p
softmax
random sample
```

vLLM v0.20.1 的 `TopKTopPSampler` 源码说明该模块执行可选 top-k/top-p filtering，然后 weighted random sampling；在 CUDA 平台上，FlashInfer sampling 是可选优化，需要设置 `VLLM_USE_FLASHINFER_SAMPLER=1` 才启用，否则走 native 路径。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/ops/topk_topp_sampler.py))

---

## 八、优化前 vs 优化后

### 优化前：朴素思路

```text
每步 decode:
  hidden → logits
  logits → softmax
  softmax → sample
```

问题：

| 问题 | 后果 |
|---|---|
| 每步都生成完整 logits | vocab 大时显存和带宽开销高 |
| 默认返回 logprobs | 输出序列化和 GPU/CPU 传输压力大 |
| top-p 需要排序 | 大 vocab 下排序成本高 |
| temperature 很低但仍走随机路径 | 浪费 sampling 计算 |
| 多卡 logits gather 不受控 | TP 下通信变重 |

### 优化后：工程化采样

```text
只在需要的位置计算 logits
尽量避免 full logprobs
temperature=0 时走 greedy
top-k/top-p 走专用路径
必要时使用 optimized sampler
```

vLLM v1 sampler 源码中，当所有请求都是 greedy 时，会直接返回 greedy sampled tokens；随机采样路径才应用 temperature、argmax-invariant processors、top-k/top-p 等步骤。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

### 对 Qwen3.6 的特殊提醒

Qwen3.6 输出维度大，logits 和 sampling 的成本更值得关注。它不是只在 attention/MoE 上复杂，**输出侧同样很重**：

```text
hidden_size = 2048
vocab padded = 248320
logits per token = 248320 scores
```

这些尺寸来自 Qwen3.6 模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 九、工程落地

### 1. 常用 API 参数理解

OpenAI-compatible 请求里常见参数：

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "logprobs": 5,
  "stop": ["</answer>"]
}
```

在 vLLM v0.20.1 中，这些最终会映射到 `SamplingParams` 一类参数。`SamplingParams` 定义了 `temperature`、`top_p`、`top_k`、`min_p`、`seed`、`stop`、`stop_token_ids`、`ignore_eos`、`max_tokens`、`logprobs`、`prompt_logprobs`、`detokenize`、`skip_special_tokens`、`allowed_token_ids`、`bad_words` 等字段。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

### 2. 推荐初始采样配置

对工程压测，建议先用稳定配置：

```json
{
  "temperature": 0,
  "max_tokens": 512
}
```

这样可以先排除随机采样带来的波动，观察：

```text
TTFT
TPOT
decode tokens/s
GPU utilization
KV/state cache capacity
```

再逐步加入：

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20
}
```

### 3. 什么时候不要打开 logprobs

以下场景不建议默认打开 logprobs：

```text
高并发 chat serving
长输出生成
多卡 TP 部署
Qwen3.6 这种大 vocab 输出模型
流式响应且用户不需要概率
```

因为 logprobs 会让系统保留和传输更多 vocab 维度的信息；`SamplingParams` 明确支持 `logprobs=-1` 返回全部 vocab log probabilities，但这在大 vocab 模型上代价很高。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

### 4. 多模态输入要先区分 text-only 与 image/video

Qwen3.6 是带 Vision Encoder 的 causal LM；vLLM 源码中也确实有 `Qwen3_VisionTransformer` 和 multimodal embedding merge 路径。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

工程上建议压测分三组：

```text
纯文本短 prompt
纯文本长 prompt
图像/视频 + 文本 prompt
```

不要把三者混在一起看平均值，否则你会分不清瓶颈来自 tokenizer、vision encoder、prefill、KV/state cache，还是 sampling。

---

## 十、本章总结

这一章建立了 LLM 推理的“入口/出口”闭环：

```text
text / multimodal input
→ token ids / multimodal embeddings
→ embeddings
→ hidden states
→ logits
→ sampling
→ next token
```

对传统 LLM，这就是最基础的 token-to-token 流程。对 Qwen3.6，要记住四个特殊点：

```text
1. 它是 Causal Language Model with Vision Encoder
2. hidden size 是 2048
3. token embedding 和 LM output 是 248320 padded
4. logits / sampling 在大 vocab 下不是免费操作
```

这些结构事实来自 Qwen3.6 模型卡；vLLM v0.20.1 源码则确认了 `embed_input_ids`、`ParallelLMHead`、`LogitsProcessor`、`Sampler`、`TopKTopPSampler` 等入口。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 小白总结

模型不直接看文字，而是看 token id。

```text
文字 → token id → embedding 向量 → 模型计算 → logits → 采样 → 新 token
```

logits 是“每个候选 token 的分数”，不是概率。softmax 后才是概率。temperature、top-p、top-k 会改变模型从这些概率里选下一个 token 的方式。

对 Qwen3.6 来说，词表输出很大：每个位置要对 248,320 个 padded vocab 位置打分，所以输出侧也很重。

---

## 工程师总结

工程视角下，本章最重要的链路是：

```text
Qwen3_5MoeForConditionalGeneration.embed_input_ids
→ Qwen3_5ForCausalLMBase.forward
→ Qwen3_5ForCausalLMBase.compute_logits
→ LogitsProcessor
→ Sampler
→ TopKTopPSampler
```

关键性能点：

```text
embedding lookup: HBM read
lm_head: large GEMM
logits gather: TP communication
sampling: FP32 logits + penalty + top-k/top-p + softmax/random
logprobs: potentially huge memory/output cost
```

对 Qwen3.6-35B-A3B，attention、Gated DeltaNet、MoE 固然重要，但输出侧的 `248320` padded vocab 也会直接影响 logits、sampling、logprobs 和 TP gather 成本。

---

## 关键公式

**1. Embedding lookup**

$$
x_i = E[i]
$$

**2. Hidden state 到 logits**

$$
z_t = W_{\text{lm}} h_t
$$

**3. Softmax**

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

**4. Temperature softmax**

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**5. Greedy decoding**

$$
x_{t+1} = \arg\max_i z_i
$$

**6. Top-k**

$$
\mathcal{S}_k = \mathrm{TopK}(z, k)
$$

只在 $\mathcal{S}_k$ 内采样。

**7. Top-p / nucleus sampling**

$$
\mathcal{S}_p =
\min \left\{
S:
\sum_{i \in S} p_i \ge p
\right\}
$$

只在累计概率达到阈值 $p$ 的候选集合中采样。

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 模型入口 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` |
| 文本 embedding | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.embed_input_ids` |
| 多模态 embedding 合并 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration.embed_input_ids` |
| LM head | `vllm/model_executor/models/qwen3_5.py` | `ParallelLMHead` |
| logits 计算 | `vllm/model_executor/models/qwen3_5.py` | `compute_logits` |
| logits processor | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor.forward`, `_get_logits`, `_gather_logits` |
| sampling 参数 | `vllm/sampling_params.py` | `SamplingParams` |
| sampler | `vllm/v1/sample/sampler.py` | `Sampler.forward`, `Sampler.sample` |
| top-k/top-p sampler | `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6-35B-A3B 的 hidden size 是 2048，token embedding 和 LM output 是 248320 padded | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6-35B-A3B 是 Causal Language Model with Vision Encoder | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM v0.20.1 中 Qwen3.6 兼容入口使用 `Qwen3_5MoeForConditionalGeneration` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` | 源码直接确认 | 是 |
| Qwen3.6 多模态路径会构造 `Qwen3_VisionTransformer` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer` | 源码直接确认 | 是 |
| 文本 embedding 由 language model 的 `embed_input_ids` 产生 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration.embed_input_ids` | 源码直接确认 | 是 |
| 存在 multimodal embeddings 时会调用 `_merge_multimodal_embeddings` 合并 | `vllm/model_executor/models/qwen3_5.py` | `_merge_multimodal_embeddings` | 源码直接确认 | 是 |
| vLLM v0.20.1 在最后 PP rank 上创建 `ParallelLMHead(config.vocab_size, config.hidden_size)`，或在 tie embeddings 时复用 embedding | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.__init__` | 源码直接确认 | 是 |
| Qwen3.6 logits 计算调用 `self.logits_processor(self.lm_head, hidden_states)` | `vllm/model_executor/models/qwen3_5.py` | `compute_logits` | 源码直接确认 | 是 |
| `LogitsProcessor` 会从 hidden states 获取 logits、必要时 scale，并应用 logits processors | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor` | 源码直接确认 | 是 |
| `LogitsProcessor._get_logits` 会调用 LM head quant method、gather TP logits、去掉 vocab padding | `vllm/model_executor/layers/logits_processor.py` | `_get_logits`, `_gather_logits` | 源码直接确认 | 是 |
| `SamplingParams` 包含 temperature、top-p、top-k、min-p、stop、logprobs 等采样参数 | `vllm/sampling_params.py` | `SamplingParams` | 源码直接确认 | 是 |
| vLLM v1 `Sampler` 的流程包括 logits float32、mask、logits processors、penalties、greedy/random、temperature、top-k/top-p、logprobs gathering | `vllm/v1/sample/sampler.py` | `Sampler.forward`, `Sampler.sample` | 源码直接确认 | 是 |
| `TopKTopPSampler` 执行 top-k/top-p filtering 和 weighted random sampling，并可选 FlashInfer CUDA 采样路径 | `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` | 源码直接确认 | 是 |
| tokenizer 的具体 Qwen3.6 special tokens、chat template、image/video placeholder token 细节 | HF tokenizer files / processor config | 待具体文件核验 | 待源码确认 | 否 |

---

## 源码阅读作业

按这个顺序读：

**1. 读 embedding 与 multimodal merge**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5MoeForConditionalGeneration.embed_input_ids
_embed_text_input_ids
_merge_multimodal_embeddings
```

目标：理解 text embedding 和 multimodal embedding 如何合并。

**2. 读 LM head 与 logits**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
ParallelLMHead
LogitsProcessor(config.vocab_size)
compute_logits
```

目标：理解 hidden states 到 logits 的路径。

**3. 读 logits processor**

```text
vllm/model_executor/layers/logits_processor.py
```

重点找：

```text
LogitsProcessor.forward
_get_logits
_gather_logits
```

目标：理解 TP gather 和 vocab padding removal。

**4. 读 sampler**

```text
vllm/v1/sample/sampler.py
vllm/v1/sample/ops/topk_topp_sampler.py
vllm/sampling_params.py
```

重点找：

```text
SamplingParams
Sampler.forward
Sampler.sample
TopKTopPSampler
```

目标：把 temperature、top-p、top-k、penalty 和源码步骤对应起来。

---

## 实验验证作业

### 实验 1：temperature=0 vs temperature=0.7

同一个 prompt，分别请求：

```json
{
  "temperature": 0,
  "max_tokens": 128
}
```

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 128
}
```

观察：

```text
输出是否稳定
TPOT 是否变化
tokens/s 是否变化
```

### 实验 2：logprobs 开关对性能的影响

分别请求：

```json
{
  "temperature": 0,
  "max_tokens": 128
}
```

```json
{
  "temperature": 0,
  "max_tokens": 128,
  "logprobs": 5
}
```

如果系统支持，再谨慎测试：

```json
{
  "temperature": 0,
  "max_tokens": 16,
  "logprobs": -1
}
```

记录：

```text
延迟
输出 payload 大小
GPU memory
CPU serialization 时间
```

### 实验 3：多模态 vs text-only

对 Qwen3.6 分别压测：

```text
纯文本 prompt
图像 + 文本 prompt
视频 + 文本 prompt
```

记录：

```text
TTFT
prefill 时间
GPU memory
vision encoder 是否成为瓶颈
```

---

## 3 个自测题

**1. logits 和 probability 有什么区别？**
logits 是未归一化分数；probability 是 softmax 后的概率。

**2. 为什么 Qwen3.6 的 logits / sampling 成本不能忽略？**
因为它的 LM output 是 248,320 padded，每个待采样 token 都要面对非常大的 vocab 维度。

**3. temperature=0 时为什么通常可以走 greedy？**
因为 temperature 接近 0 时，分布趋向于最大 logit token；vLLM `SamplingParams` 也把 0 temperature 视为 greedy sampling。

---

## 下一章预告

第五章进入：

```text
Attention / Gated Attention / RoPE 数学
```

我们会从传统 attention 的：

```text
Q, K, V
softmax(QK^T / sqrt(d))V
causal mask
KV cache
```

讲到 Qwen3.6 的：

```text
Gated Attention
Q heads = 16
KV heads = 2
head dim = 256
RoPE dim = 64
```

并特别说明：为什么 Qwen3.6 的 full attention 层只是 40 层中的一部分，不能把所有层都当普通 attention。

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py)
