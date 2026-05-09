# 第三章：Qwen3.6 架构总览
## Gated DeltaNet、Gated Attention、MoE、MTP

本章目标：把第二章的“传统 LLM 推理骨架”替换成 **Qwen/Qwen3.6-35B-A3B 的真实结构地图**。

先给一句总括：

> Qwen3.6-35B-A3B 不是“40 层普通 self-attention + MLP”的 Transformer Decoder，而是一个 **带 Vision Encoder 的 causal LM**，语言主干是 **Gated DeltaNet / Gated Attention / MoE / MTP** 组合结构。

模型卡确认：它是 **Causal Language Model with Vision Encoder**，总参数 35B、每 token 激活 3B，hidden size 2048，40 层，padded token embedding 和 LM output 都是 248,320；hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`；MoE 是 256 experts，每 token 激活 8 routed experts + 1 shared expert；MTP trained with multi-steps；原生上下文 262,144 tokens，可扩展到 1,010,000 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

传统 Transformer Decoder 像一个固定流程的工厂：

```text
每一层：
  全局读一遍历史上下文
  然后过一个固定的 MLP 加工站
```

Qwen3.6 更像一个混合型专家工厂：

```text
输入：
  文本 token
  或图像 / 视频经过 Vision Encoder 后形成的视觉 embedding

语言主干：
  快速记忆工位：Gated DeltaNet
  专家加工站：MoE
  快速记忆工位：Gated DeltaNet
  专家加工站：MoE
  快速记忆工位：Gated DeltaNet
  专家加工站：MoE
  全局注意力工位：Gated Attention
  专家加工站：MoE

上述结构重复 10 次
```

所以它不是“每层都完整翻阅历史记录”。更准确的直觉是：

| 模块 | 类比 | 作用 |
|---|---|---|
| Vision Encoder | 图像/视频前台接待员 | 把视觉输入转成语言模型可消费的 embedding |
| Gated DeltaNet | 快速滚动笔记 | 用状态化/线性注意力风格处理长序列信息 |
| Gated Attention | 全局复盘会议 | 用 full attention 看历史上下文 |
| MoE | 256 人专家团 | 每个 token 只找 8 个 routed experts + 1 个 shared expert |
| MTP | 草稿预测员 | 辅助提前预测多个未来 token，用于 speculative decoding |

vLLM recipe 也把 Qwen3.5/Qwen3.6 描述为 multimodal MoE models，并指出它们具有 gated delta networks 架构；同一 recipe 给出的 Qwen3.6 serving 命令使用 `--tensor-parallel-size 8`、`--max-model-len 262144` 和 `--reasoning-parser qwen3`，还给出 MTP speculative decoding 配置。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 二、数学直觉

### 1. 传统 Decoder 层的抽象

传统 Transformer Decoder 可以粗略写成：

$$
h_{\ell+1}
=
h_{\ell}
+
\text{MLP}(
  h_{\ell}
  +
  \text{Attention}(h_{\ell})
)
$$

更常见的 pre-norm 写法是：

$$
u_{\ell}=h_{\ell}+\text{Attention}(\text{Norm}(h_{\ell}))
$$

$$
h_{\ell+1}=u_{\ell}+\text{MLP}(\text{Norm}(u_{\ell}))
$$

这只是传统模型骨架，不是 Qwen3.6 的完整结构。

### 2. Qwen3.6 的层级抽象

Qwen3.6 的一个 4 层小周期可以抽象为：

```text
Layer A: Gated DeltaNet  → MoE
Layer B: Gated DeltaNet  → MoE
Layer C: Gated DeltaNet  → MoE
Layer D: Gated Attention → MoE
```

这个小周期重复 10 次，得到 40 层。模型卡明确给出这个 hidden layout。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Gated DeltaNet 的直觉

传统 full attention 的核心是：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

它要显式地把当前 query 和历史所有 key 做交互。

Gated DeltaNet 可以先理解成“状态化的线性注意力风格模块”：它不应该被简化成普通 attention。源码中 `GatedDeltaNetAttention` 属于 `PluggableLayer`，并继承 `MambaBase`；它有 `mamba_type = "gdn_attention"`，状态 dtype/shape 通过 `MambaStateDtypeCalculator.gated_delta_net_state_dtype` 和 `MambaStateShapeCalculator.gated_delta_net_state_shape` 计算。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

本章只做架构总览，不强行展开 Gated DeltaNet 的完整递推公式；原因是它的具体 kernel、state layout、chunked computation 和 speculative decoding 交互需要到第六、十一、十二章结合源码和显存布局一起讲。

### 4. Gated Attention 的直觉

Gated Attention 可以理解为 full attention 加了输出门控。vLLM v0.20.1 的 `Qwen3NextAttention` 源码中会读取 `config.num_attention_heads`、`config.num_key_value_heads`、`config.head_dim`，并有 `attn_output_gate = getattr(config, "attn_output_gate", True)`；这说明它不是一个只含普通 QKV 投影的最简 attention 模块。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

抽象成公式可以写成：

$$
a_t = \text{Attention}(q_t, K_{\le t}, V_{\le t})
$$

$$
y_t = g_t \odot a_t
$$

其中 $g_t$ 是门控信号，$\odot$ 是逐元素乘法。这个公式是教学抽象，不代表 vLLM 源码中的逐行实现。

### 5. MoE 的直觉

传统 MLP 是：

$$
y = \text{MLP}(h)
$$

MoE 是：

$$
y =
\sum_{e \in \text{TopK}(r(h))}
w_e \cdot E_e(h)
+
E_{\text{shared}}(h)
$$

其中 $r(h)$ 是 router，$E_e$ 是第 $e$ 个 expert，$w_e$ 是路由权重。Qwen3.6 的模型卡确认有 256 experts，每 token 激活 8 routed experts + 1 shared expert，expert intermediate dimension 为 512。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

vLLM v0.20.1 中，MoE block 对应 `Qwen3NextSparseMoeBlock`；源码中它读取 `config.num_experts` 作为 routed experts 数，并使用 expert parallel group 信息，如 `get_ep_group()`、`ep_rank`、`ep_size`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 6. MTP 的直觉

传统 autoregressive decoding 是：

```text
主模型每次预测 1 个 token
接受后再预测下一个
```

MTP / speculative decoding 的直觉是：

```text
辅助头或辅助路径先草拟多个未来 token
主模型验证其中可以接受的 token
一次推进多个位置
```

Qwen3.6 模型卡写明 MTP trained with multi-steps；vLLM recipe 给出 Qwen3.6 的 MTP speculative decoding 命令，使用 `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 三、张量形状

为了统一后续章节，我们用 vLLM serving 中常见的 flattened token 视角：

| 符号 | 含义 |
|---|---|
| $N$ | 当前 batch 内被合并后的 token 总数 |
| $B$ | 请求数 / sequence 数 |
| $T$ | 单个请求的序列长度 |
| $H$ | hidden size |
| $V$ | vocab size |
| $E$ | expert 数 |
| $K$ | 每 token routed experts 数 |

对 Qwen3.6-35B-A3B：

```text
H = 2048
V = 248320 padded
layers = 40
experts = 256
routed experts per token = 8
shared experts per token = 1
native context = 262144
```

这些值来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 1. 文本输入

```text
input_ids:       [N]
hidden_states:   [N, 2048]
logits:          [N, 248320]
```

vLLM 中很多模型 forward 会把 batch 内 token flatten 后处理，所以工程上经常看到 `[num_tokens, hidden_size]`，而不是教学里常见的 `[B, T, H]`。

### 2. Vision Encoder 输入

多模态输入时：

```text
image / video
→ Vision Encoder
→ multimodal_embeddings
→ merge into text embeddings
→ language model
```

vLLM v0.20.1 的 `qwen3_5.py` 中，Qwen3.5/3.6 多模态类会构造 `Qwen3_VisionTransformer`，并构造 `Qwen3_5MoeForCausalLM` 作为 language model；`embed_input_ids` 会先得到文本 embedding，再在存在 multimodal embeddings 时调用 `_merge_multimodal_embeddings` 合并。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 3. Gated DeltaNet 形状

模型卡给出 Gated DeltaNet 的头部配置：

```text
V-side linear attention heads: 32
QK-side linear attention heads: 16
head dim: 128
```

这说明它的 QK 和 V 侧 head 组织并不等同于普通 MHA 的 “Q/K/V heads 完全对齐” 形式。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

vLLM 源码中，`GatedDeltaNetAttention` 读取 `linear_num_value_heads`、`linear_num_key_heads`、`linear_key_head_dim`、`linear_value_head_dim`，并计算 `key_dim` 等内部尺寸。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

### 4. Gated Attention 形状

模型卡给出 Gated Attention 的配置：

```text
Q heads: 16
KV heads: 2
head dim: 256
RoPE dim: 64
```

这更接近 GQA：query heads 多，KV heads 少。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

传统 attention 的直觉形状是：

```text
Q: [N, num_q_heads, head_dim]
K: [N, num_kv_heads, head_dim]
V: [N, num_kv_heads, head_dim]
```

对 Qwen3.6 的 Gated Attention：

```text
Q: [N, 16, 256]   before tensor parallel partition
K: [N, 2, 256]    before tensor parallel partition
V: [N, 2, 256]    before tensor parallel partition
```

源码中 `Qwen3NextAttention` 会根据 tensor parallel size 切分 `total_num_heads` 和 `total_num_kv_heads`，如果 KV heads 少于 TP size，则会复制 KV heads；如果 KV heads 大于等于 TP size，则跨 TP 分区。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 5. MoE 形状

MoE 的 router logits 可以理解为：

```text
hidden_states:   [N, 2048]
router_logits:   [N, 256]
topk_indices:    [N, 8]
expert_outputs:  [N, 8, 2048]
shared_output:   [N, 2048]
final_output:    [N, 2048]
```

vLLM `Qwen3NextSparseMoeBlock` 的 forward 片段显示：router logits 形状注释为 `(num_tokens, n_experts)`，随后调用 `self.experts(hidden_states=hidden_states, router_logits=router_logits)`；在 sequence parallel 场景下还会做 `tensor_model_parallel_all_gather`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 四、计算流程

Qwen3.6 推理流程可以分成两条路径：多模态前处理路径和语言模型主干路径。

### 1. 多模态路径

```text
image / video
→ Qwen3_VisionTransformer
→ multimodal embeddings
→ merge with text embeddings
→ language model
```

vLLM v0.20.1 中，`Qwen3_5MoeForConditionalGeneration` 注册了 `Qwen3VLMultiModalProcessor`，构造 `Qwen3_VisionTransformer`，再构造 `Qwen3_5MoeForCausalLM`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 2. 语言主干路径

```text
input embeddings
→ 40 layers
→ final norm
→ lm_head
→ logits
→ sampling
```

但 40 层不是同构的普通 Transformer。Qwen3.6 的真实周期是：

```text
repeat 10 times:
  Gated DeltaNet  → MoE
  Gated DeltaNet  → MoE
  Gated DeltaNet  → MoE
  Gated Attention → MoE
```

模型卡确认了这个 hidden layout。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. vLLM 层选择路径

vLLM v0.20.1 中，`Qwen3_5DecoderLayer` 根据 `layer_type` 做分支：

```text
if layer_type == "linear_attention":
    use GatedDeltaNetAttention
elif layer_type == "full_attention":
    use Qwen3NextAttention
else:
    raise ValueError
```

同一个 decoder layer 之后再根据 config model type 选择 MoE 或 dense MLP；对于 `qwen3_5_moe_text`，使用 `Qwen3NextSparseMoeBlock`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

这就是本章最重要的源码映射：**Gated DeltaNet / Gated Attention 是 attention 路径分支，MoE 是每层后面的 MLP 替代物。**

---

## 五、显存布局

Qwen3.6 的显存不能只按传统 Transformer 看。

### 1. 权重显存

35B 总参数意味着 BF16 权重粗略是 70GB 量级；FP8 checkpoint 会显著降低权重占用。Qwen3.6 官方 FP8 模型卡说明：该仓库包含 FP8-quantized weights/config，采用 fine-grained FP8 quantization，block size 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

### 2. KV Cache

Gated Attention 层仍然需要 full attention 风格的 KV cache。因为它有 Q heads、KV heads、head dim 和 RoPE dim，decode 时需要读取历史 K/V。

### 3. Gated DeltaNet / Mamba-style state

Gated DeltaNet 不应简单归入普通 KV cache。vLLM 源码把它作为 `MambaBase` 风格的模块处理，并通过 mamba state dtype/shape calculator 计算状态。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

这意味着后续讲第六章“KV Cache 与长上下文显存问题”时，我们必须区分：

```text
full attention KV cache
Gated DeltaNet / Mamba-like state cache
MoE expert weights
MTP speculative token 额外占用
```

### 4. Vision Encoder 显存

多模态服务还要加载 vision encoder。vLLM recipe 对 Qwen3.5 的 text-only throughput 配置说明：`--language-model-only` 可以跳过 vision encoder，把更多显存留给 KV cache；对于 Qwen3.6，这个思路同样是部署规划中必须考虑的方向，但具体命令和可用性要以 v0.20.1 实测为准。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 六、vLLM 源码映射

本章的核心源码地图如下。

| 架构概念 | vLLM v0.20.1 源码入口 | 说明 |
|---|---|---|
| Qwen3.6 multimodal MoE 模型 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` |
| Vision Encoder | `vllm/model_executor/models/qwen3_5.py` | 构造 `Qwen3_VisionTransformer` |
| Language Model | `vllm/model_executor/models/qwen3_5.py` | 构造 `Qwen3_5MoeForCausalLM` |
| Hybrid decoder layer | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Full / Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Sparse MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| MTP | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP`, `Qwen3NextMultiTokenPredictor` |
| GDN attention backend metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder` 等后续再展开 |

特别注意：虽然文件名里是 `qwen3_5.py`，但 vLLM v0.20.1 的该文件包含 Qwen3.5/Qwen3.6 兼容路径。源码中 `Qwen3_5MoeForConditionalGeneration` 构造 `Qwen3_VisionTransformer` 和 `Qwen3_5MoeForCausalLM`，并通过 `set_moe_parameters()` 收集 MoE 层信息。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 七、NVIDIA GPU 执行视角

从 GPU 看，Qwen3.6 不是一条简单的 dense GEMM 流水线，而是四类计算混在一起：

| 模块 | GPU 计算特征 | 主要瓶颈 |
|---|---|---|
| Gated DeltaNet | 状态化/线性注意力风格计算，包含投影、卷积/状态更新、门控 | kernel 选择、状态读写、chunk 组织 |
| Gated Attention | QKV projection + full attention + output gate | KV cache 读带宽、attention backend |
| MoE | router + token dispatch + expert GEMM + combine | expert dispatch、GEMM 分组、通信 |
| MTP | 额外 draft token 路径 + 主模型验证 | 接受率、额外 KV/state 占用、吞吐/延迟权衡 |

MTP 并不是免费午餐。vLLM recipe 的配置说明指出，MTP-1 可降低 per-token latency，但在高并发下会降低 text throughput，因为 speculative tokens 会消耗 KV cache capacity，降低有效 batch size；`num_speculative_tokens` 可调，但接受率和吞吐存在权衡。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

对 NVIDIA GPU 优化来说，Qwen3.6 的难点是：

```text
不是只优化 attention
也不是只优化 GEMM
而是要同时优化：
  long context memory
  MoE expert dispatch
  tensor/expert parallel communication
  GDN state cache
  full attention KV cache
  MTP speculative path
```

---

## 八、优化前 vs 优化后

### 优化前：把它当普通 Transformer 跑

错误心智模型：

```text
40 layers × (Attention + MLP)
```

会导致几个误判：

| 误判 | 后果 |
|---|---|
| 以为每层都有完整 KV cache | 高估或误判 Gated DeltaNet 层缓存结构 |
| 以为 MLP 是 dense | 忽略 router、expert dispatch、shared expert |
| 以为 FP8 只影响权重 | 忽略 KV/state cache 和 MoE dispatch 的瓶颈 |
| 以为 MTP 一定提升吞吐 | 忽略 speculative tokens 消耗缓存容量 |
| 以为多模态只是输入格式变化 | 忽略 vision encoder 显存和吞吐影响 |

### 优化后：按真实模块分治

正确心智模型：

```text
Vision Encoder
+
10 × (
  3 × (Gated DeltaNet → Sparse MoE)
  1 × (Gated Attention → Sparse MoE)
)
+
MTP optional path
```

对应优化策略：

| 模块 | 优化方向 |
|---|---|
| Vision Encoder | text-only 时考虑跳过；多模态时考虑 encoder data parallel |
| Gated DeltaNet | 关注 mamba/GDN state、chunk、backend |
| Gated Attention | 关注 KV cache、attention backend、RoPE |
| MoE | 关注 expert parallel、router top-k、dispatch、expert GEMM |
| MTP | 关注接受率、TPOT、KV/state 额外占用 |
| 262K context | 关注 max model len、KV capacity、prefix caching、chunked prefill |

---

## 九、工程落地

如果你要在 NVIDIA GPU 上部署 Qwen3.6-35B-A3B，第三章阶段先建立以下落地判断。

### 1. 最小正确启动命令

vLLM recipe 给出的 Qwen3.6 基础命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

带 MTP speculative decoding 的命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

这两个命令均来自 vLLM Qwen3.5/Qwen3.6 recipe。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 2. FP8 单卡不是“完整等价配置”

Qwen3.6 有官方 FP8 checkpoint，权重量化方式是 fine-grained FP8、block size 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

但工程上要区分：

```text
能加载权重
≠ 能跑 262K 上下文
≠ 能高并发服务
≠ 能多模态 + 长上下文 + MTP 同时打开
```

FP8 主要降低权重和部分 GEMM 压力；长上下文下，full attention KV cache、GDN state、MTP speculative tokens 和 workspace 仍然可能成为显存瓶颈。

### 3. text-only 与 multimodal 要分开规划

Qwen3.6 是带 Vision Encoder 的 causal LM。多模态模式下，要考虑 vision encoder 显存、预处理缓存和视觉 token 数；text-only 模式下，应尽量把显存留给 language model 和 KV/state cache。vLLM recipe 对 Qwen3.5 text-only 说明了 `--language-model-only` 可跳过 vision encoder 并释放 KV cache 显存，这给 Qwen3.6 的部署规划提供了明确方向。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 4. MTP 要按 workload 开关

低并发、追求 TPOT 时，可以试 MTP；高并发、追求总吞吐时，MTP 可能因为额外 speculative tokens 占用 cache 而降低有效 batch size。这个权衡在 vLLM recipe 的配置提示中已有明确说明。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 十、本章总结

Qwen3.6-35B-A3B 的核心结构可以压缩成一句话：

```text
多模态输入通过 Vision Encoder 进入语言模型；
语言模型不是普通 Transformer，而是 10 个周期的
3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)；
推理时还可以使用 MTP speculative decoding。
```

传统 Transformer 是本课程的“入门参照物”，但从第三章开始，我们要把真实服务中的问题拆成四条主线：

```text
1. Gated DeltaNet：长上下文状态化计算
2. Gated Attention：full attention + KV cache + RoPE
3. MoE：router + experts + dispatch + parallelism
4. MTP：draft tokens + verification + latency/throughput trade-off
```

---

## 小白总结

Qwen3.6 不是一个普通“读上下文然后回答”的模型。

它更像一个混合系统：

```text
有看图/视频的前端
有快速记忆模块 Gated DeltaNet
有全局注意力模块 Gated Attention
有 256 个专家组成的 MoE
还有提前打草稿的 MTP
```

所以后面讲显存、速度、部署、源码时，不能只用“普通 Transformer”解释。

---

## 工程师总结

工程上，Qwen3.6 的复杂性来自四个方向叠加：

```text
Hybrid attention:
  Gated DeltaNet + Gated Attention

Sparse computation:
  256 experts, top-8 routed + shared expert

Long context:
  native 262K context

Speculative decoding:
  MTP trained with multi-steps
```

在 vLLM v0.20.1 中，关键源码入口已经确认：

```text
qwen3_5.py:
  Qwen3_5DecoderLayer
  Qwen3_5MoeForConditionalGeneration
  Qwen3_5MoeForCausalLM

qwen3_next.py:
  Qwen3NextAttention
  Qwen3NextSparseMoeBlock

gdn_linear_attn.py:
  GatedDeltaNetAttention

qwen3_next_mtp.py:
  Qwen3NextMTP
```

---

## 关键公式

**1. 传统 attention**

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

**2. Gated Attention 教学抽象**

$$
a_t = \text{Attention}(q_t, K_{\le t}, V_{\le t})
$$

$$
y_t = g_t \odot a_t
$$

**3. MoE 路由抽象**

$$
r = \text{Router}(h)
$$

$$
\mathcal{E}_{topk} = \text{TopK}(r, k=8)
$$

$$
y =
\sum_{e \in \mathcal{E}_{topk}}
w_e E_e(h)
+
E_{\text{shared}}(h)
$$

**4. Qwen3.6 层级结构**

$$
10 \times
\left[
3 \times
(\text{GatedDeltaNet} \rightarrow \text{MoE})
+
1 \times
(\text{GatedAttention} \rightarrow \text{MoE})
\right]
$$

**5. MTP 推理直觉**

$$
\text{draft tokens}
=
(\hat{x}_{t+1}, \hat{x}_{t+2}, \ldots, \hat{x}_{t+k})
$$

$$
\text{accepted prefix}
\subseteq
\text{draft tokens}
$$

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 多模态 MoE 总入口 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` |
| Vision Encoder | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer` |
| Language Model | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForCausalLM` |
| Hybrid decoder layer | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Sparse MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| MTP | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP` |
| GDN backend metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder` 等，后续确认展开 |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 是带 Vision Encoder 的 causal LM，35B total / 3B activated，40 层，hidden size 2048 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MoE 是 256 experts，8 routed + 1 shared | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MTP trained with multi-steps | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM recipe 给出 Qwen3.6 TP=8、262144 max model len、qwen3 reasoning parser 命令 | vLLM Qwen3.5/Qwen3.6 recipe | `vllm serve Qwen/Qwen3.6-35B-A3B ...` | recipe 确认 | 是 |
| vLLM recipe 给出 Qwen3.6 MTP speculative decoding 配置 | vLLM Qwen3.5/Qwen3.6 recipe | `--speculative-config '{"method": "mtp", ...}'` | recipe 确认 | 是 |
| vLLM v0.20.1 中 Qwen3.6 MoE multimodal 总入口继承 multimodal conditional generation 与 MoE protocol | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 multimodal 路径构造 `Qwen3_VisionTransformer` | `vllm/model_executor/models/qwen3_5.py` | `self.visual = Qwen3_VisionTransformer(...)` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 language model 路径构造 `Qwen3_5MoeForCausalLM` | `vllm/model_executor/models/qwen3_5.py` | `self.language_model = Qwen3_5MoeForCausalLM(...)` | 源码直接确认 | 是 |
| `Qwen3_5DecoderLayer` 根据 `layer_type` 选择 Gated DeltaNet 或 Qwen3NextAttention | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| linear attention 路径使用 `GatedDeltaNetAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.linear_attn = GatedDeltaNetAttention(...)` | 源码直接确认 | 是 |
| full attention 路径使用 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.self_attn = Qwen3NextAttention(...)` | 源码直接确认 | 是 |
| MoE text config 下使用 `Qwen3NextSparseMoeBlock` | `vllm/model_executor/models/qwen3_5.py` | `self.mlp = Qwen3NextSparseMoeBlock(...)` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 读取 `config.num_experts` 并使用 expert parallel group | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 读取 Q heads、KV heads、head dim，并有 `attn_output_gate` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 注册为 pluggable layer，并返回 `mamba_type = "gdn_attention"` | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| vLLM v0.20.1 中存在 Qwen3Next MTP 模型入口 | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP` | 源码直接确认 | 是 |
| `Qwen3NextMTP` 当前不支持 `mamba_cache_mode="all"`，提示使用 `--mamba-cache-mode=align` | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP.__init__` | 源码直接确认 | 是 |
| GDN backend metadata 具体如何构造 block table、spec decode metadata | `vllm/v1/attention/backends/gdn_attn.py` | `build(...)` | 待源码确认 | 否 |
| MoE kernel 最终选择 CUTLASS / Triton / fused_moe 的条件 | `vllm/model_executor/layers/fused_moe/` | 待定 | 待源码确认 | 否 |

---

## 源码阅读作业

按下面顺序读，不要求一次读懂 kernel。

**第一步：读模型总入口**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5MoeForConditionalGeneration
Qwen3_5MoeForCausalLM
Qwen3_5DecoderLayer
embed_input_ids
set_moe_parameters
```

目标：确认 Vision Encoder、language model、MoE 参数收集是如何串起来的。

**第二步：读 hybrid layer 分支**

```text
Qwen3_5DecoderLayer.__init__
```

目标：确认：

```text
linear_attention → GatedDeltaNetAttention
full_attention   → Qwen3NextAttention
qwen3_5_moe_text → Qwen3NextSparseMoeBlock
```

**第三步：读 MoE block**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextSparseMoeBlock
self.gate
self.experts
router_logits
```

**第四步：读 MTP 入口**

```text
vllm/model_executor/models/qwen3_next_mtp.py
```

重点找：

```text
Qwen3NextMTP
Qwen3NextMultiTokenPredictor
mamba_cache_mode
compute_logits
```

---

## 实验验证作业

实验重点不是追求性能，而是验证“结构差异”。

### 实验 1：查看 config

下载或查看 HF config，确认：

```text
hidden_size
num_hidden_layers
num_experts
num_experts_per_tok
linear_num_key_heads
linear_num_value_heads
num_attention_heads
num_key_value_heads
head_dim
```

目标：把模型卡上的结构参数和 config 字段对应起来。

### 实验 2：启动基础 serving

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

记录：

```text
启动日志中的 model config
GPU memory usage
KV cache capacity
是否加载 vision encoder
是否出现 GDN / Mamba cache 相关日志
```

### 实验 3：启动 MTP 对比

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

对比：

```text
TTFT
TPOT
tokens/s
显存占用
可承载并发数
```

结论不要只看单请求延迟；还要看高并发下吞吐是否下降。

---

## 3 个自测题

**1. 为什么 Qwen3.6-35B-A3B 不能被称为普通 Transformer Decoder？**
答案要包含：Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP。

**2. Qwen3.6 的 40 层 hidden layout 如何组织？**
答案应能写出：

```text
10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))
```

**3. 为什么 MTP 可能降低高并发吞吐？**
提示：speculative tokens 会消耗 KV/state cache capacity，降低有效 batch size；接受率也会影响收益。

---

## 下一章预告

第四章进入：

```text
Token、Embedding、Logits、Sampling
```

我们会从最基础的 token id 开始，讲清楚：

```text
文本如何变成 input_ids
input_ids 如何变成 hidden_states
hidden_states 如何变成 logits
logits 如何经过 temperature / top-k / top-p 变成 next token
```

并且会特别注意 Qwen3.6 的两个差异：

```text
1. 多模态 embedding 如何合并进语言序列
2. vocab / lm_head 的 248320 padded 输出维度如何影响 logits 与 sampling
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vllm/vllm/model_executor/layers/mamba/gdn_linear_attn.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py)
