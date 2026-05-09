# 第五章：Attention / Gated Attention / RoPE 数学

本章只把 **Qwen3.6-35B-A3B 中的 Gated Attention 层** 放到 Attention 数学框架里讲。
请先记住边界：Qwen3.6 的 40 层不是 40 层普通 attention。模型卡给出的 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，因此可推算为 **30 个 Gated DeltaNet 层 + 10 个 Gated Attention 层**；Gated DeltaNet 不应被强行套成普通 softmax attention。Qwen3.6 的 Gated Attention 配置是 Q heads = 16、KV heads = 2、head dim = 256、RoPE dim = 64。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

传统 Attention 像是在开卷考试：

```text
当前 token = 正在答题的人
历史 tokens = 之前所有笔记
Q = 我现在想找什么信息
K = 每条笔记的标签
V = 每条笔记的内容
```

当前 token 会拿自己的 query 去和历史所有 key 做匹配，匹配分数越高，就越多读取那个 token 的 value。

例如句子：

```text
小明把书放进书包，因为它很重。
```

当模型生成“它”后面的内容时，需要判断“它”指的是“书”还是“书包”。Attention 就是在做这种“当前词应该重点看历史上哪些词”的加权读取。

Qwen3.6 的 Gated Attention 还多了一道“阀门”。普通 Attention 是：

```text
读完历史信息 → 直接输出
```

Gated Attention 更像：

```text
读完历史信息 → 经过 gate 判断哪些维度该放大/抑制 → 再输出
```

vLLM v0.20.1 的 `Qwen3NextAttention` 源码里确实存在 `attn_output_gate` 分支：QKV projection 会额外产生 gate，forward 中对 gate 做 `torch.sigmoid(gate)`，然后执行 `attn_output = attn_output * gate`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 二、数学直觉

### 1. 传统 Scaled Dot-Product Attention

核心公式是：

\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}+M\right)V

其中：

| 符号 | 含义 |
|---|---|
| $Q$ | query，当前 token 想找什么 |
| $K$ | key，历史 token 的索引标签 |
| $V$ | value，历史 token 的内容 |
| $d$ | head dimension |
| $M$ | causal mask，禁止看未来 token |

拆成单个 token 的形式：

$$
s_{t,j} = \frac{q_t \cdot k_j}{\sqrt{d}}
$$

$$
\alpha_{t,j} = \frac{\exp(s_{t,j})}{\sum_{i \le t}\exp(s_{t,i})}
$$

$$
o_t = \sum_{j \le t}\alpha_{t,j}v_j
$$

这里 $j \le t$ 表示 causal attention 只能看当前和历史 token，不能看未来 token。

### 2. 为什么要除以 $\sqrt{d}$

如果 $q$ 和 $k$ 的每个维度方差近似为 1，那么点积 $q \cdot k$ 的方差大约随 $d$ 增长。head dim 越大，logits 越容易变得极大，softmax 越容易饱和。除以 $\sqrt{d}$ 是为了让 attention score 的尺度更稳定。

Qwen3.6 的 Gated Attention head dim 是 256，所以 full attention 层中的缩放因子是：

$$
\frac{1}{\sqrt{256}} = \frac{1}{16}
$$

vLLM v0.20.1 的 `Qwen3NextAttention` 源码也直接设置 `self.scaling = self.head_dim**-0.5`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. Gated Attention 的教学抽象

普通 Attention 输出：

$$
a_t = \operatorname{Attention}(q_t, K_{\le t}, V_{\le t})
$$

Gated Attention 可以抽象为：

$$
y_t = a_t \odot \sigma(g_t)
$$

其中：

| 符号 | 含义 |
|---|---|
| $a_t$ | attention 输出 |
| $g_t$ | gate 分支输出 |
| $\sigma$ | sigmoid |
| $\odot$ | 逐元素乘法 |

这个公式是教学抽象，但和 vLLM v0.20.1 的 forward 逻辑一致：源码中先从 `q_gate` 中 split 出 `q` 和 `gate`，然后 attention 输出乘以 sigmoid 后的 gate。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 4. RoPE 的数学直觉

Attention 本身只比较 $q$ 和 $k$ 的内容相似度。RoPE 的作用是把位置信息编码进 $q$ 和 $k$。

对二维向量 $(x_1, x_2)$，旋转角度 $\theta$ 后：

$$
\begin{bmatrix}
x_1' \\
x_2'
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

RoPE 就是把 head 里的部分维度两两成对旋转；位置越靠后，旋转角越不同。这样 $q_m^\top k_n$ 里会自然包含相对位置 $m-n$ 的信息。

Qwen3.6 的 Gated Attention head dim 是 256，但 RoPE dim 是 64。这意味着不是整个 256 维 head 都参与 RoPE 旋转，而是其中的 64 维参与旋转，其余维度透传。RoPE dim = 64 来自 Qwen3.6 模型卡；vLLM 的 rotary embedding 基类也确实把 query/key 切成 `query_rot = query[..., :rotary_dim]` 与 `query_pass = query[..., rotary_dim:]`，只对 `rotary_dim` 部分做旋转，再拼回去。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 三、张量形状

### 1. 传统 Multi-Head Attention

教学形状：

```text
hidden_states: [B, T, H]

Q: [B, T, num_q_heads, head_dim]
K: [B, T, num_kv_heads, head_dim]
V: [B, T, num_kv_heads, head_dim]
```

如果是标准 MHA：

```text
num_q_heads = num_kv_heads
```

如果是 GQA：

```text
num_q_heads > num_kv_heads
```

Qwen3.6 的 Gated Attention 是 GQA 风格：Q heads = 16，KV heads = 2，所以每 8 个 query heads 共享一组 KV heads。模型卡确认 Qwen3.6 的 Gated Attention 为 16 个 Q heads、2 个 KV heads、head dim 256、RoPE dim 64。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. Qwen3.6 Gated Attention 的逻辑形状

不考虑 tensor parallel 切分时：

```text
hidden_states: [N, 2048]

Q:    [N, 16, 256]
K:    [N,  2, 256]
V:    [N,  2, 256]
Gate: [N, 16, 256]
```

这里 $N$ 是当前 batch 中 flatten 后的 token 数。注意一个有趣的地方：Q heads × head dim = $16 \times 256 = 4096$，大于 hidden size 2048。这并不矛盾，因为 QKV projection 可以把 2048 维 hidden state 投影到更高维的 Q/K/V 空间，再由 output projection 投回 hidden size。

vLLM v0.20.1 的 `Qwen3NextAttention` 中，`qkv_proj` 使用 `QKVParallelLinear(config.hidden_size, self.head_dim, self.total_num_heads * (1 + self.attn_output_gate), self.total_num_kv_heads, ...)`。当 `attn_output_gate=True` 时，Q 侧会多出 gate 对应的维度；forward 中也确实把 `q_gate` split 成 `q` 与 `gate`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. Tensor Parallel 下的形状变化

在 tensor parallel 下，每张 GPU 只持有部分 heads。vLLM 源码中：

```text
self.total_num_heads = config.num_attention_heads
self.num_heads = self.total_num_heads // tp_size
self.total_num_kv_heads = config.num_key_value_heads
```

如果 KV heads 数量大于等于 TP size，就把 KV heads 跨 GPU 分区；如果 KV heads 少于 TP size，就复制 KV heads。Qwen3.6 的 KV heads 只有 2，因此当 TP size = 8 时，KV heads 会被复制，而不是每卡只分到 1/4 个 KV head。这个行为由 `Qwen3NextAttention` 源码中的判断逻辑确认。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 四、计算流程

Qwen3.6 的 full-attention 层，也就是 Gated Attention 层，在 vLLM v0.20.1 中大致是：

```text
hidden_states
→ qkv_proj
→ split q / gate / k / v
→ q_norm, k_norm
→ RoPE(q, k)
→ Attention(q, k, v)
→ sigmoid(gate)
→ attn_output *= gate
→ o_proj
```

源码对应：

1. `Qwen3_5DecoderLayer` 遇到 `layer_type == "full_attention"` 时创建 `Qwen3NextAttention`；遇到 `linear_attention` 时创建 `GatedDeltaNetAttention`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))
2. `Qwen3NextAttention.__init__` 读取 attention heads、KV heads、head dim，创建 `qkv_proj`、`o_proj`、`rotary_emb` 和 `Attention`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))
3. `Qwen3NextAttention.forward` 中先做 `qkv_proj`，再 split 出 `q`、`gate`、`k`、`v`，对 Q/K 做 RMSNorm 和 RoPE，然后调用 `self.attn(q, k, v)`，最后用 sigmoid gate 乘 attention 输出并通过 `o_proj`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 关键边界：Gated DeltaNet 不走这个流程

Qwen3.6 的 `linear_attention` 层走 `GatedDeltaNetAttention`，不是 `Qwen3NextAttention`。因此下面这个标准 attention 流程：

```text
QK^T
→ softmax
→ read V
→ KV cache
```

只适合解释 Qwen3.6 的 **Gated Attention 层**，不适合解释所有 40 层。vLLM v0.20.1 的 `Qwen3_5DecoderLayer` 对 `linear_attention` 和 `full_attention` 做了明确分支。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 五、显存布局

### 1. Attention 的中间张量

Prefill 阶段，传统 attention 的逻辑 attention score 是：

```text
scores: [B, num_heads, T, T]
```

如果真的显式 materialize 这个矩阵，长上下文会爆炸。所以现代 attention kernel 通常不会把完整 $T \times T$ scores 存下来，而是分块计算 softmax 和 value 聚合。

### 2. Decode 阶段的 KV Cache

decode 阶段每次只来一个新 token，但它要看历史所有 K/V：

```text
new q:     [B, num_q_heads, head_dim]
K cache:   [B, T_past, num_kv_heads, head_dim]
V cache:   [B, T_past, num_kv_heads, head_dim]
```

Qwen3.6 的 full attention 层 KV heads = 2，head dim = 256，因此每个 token、每个 Gated Attention 层的 KV 元素量为：

$$
2 \times \text{num\_kv\_heads} \times \text{head\_dim}
=
2 \times 2 \times 256
=
1024
$$

这里第一个 2 是 K 和 V 两份缓存。若使用 BF16/FP16，则每 token、每 Gated Attention 层约为：

$$
1024 \times 2\ \text{bytes} = 2048\ \text{bytes}
$$

Qwen3.6 按 hidden layout 推算有 10 个 Gated Attention 层，所以仅 full-attention KV cache 的粗略量级为：

$$
2048\ \text{bytes}
\times 10
\times \text{tokens}
$$

这个估算只覆盖 Gated Attention 的 K/V，不包括 Gated DeltaNet state、MoE 权重、vision encoder、runtime workspace、padding、block metadata 或量化差异。Qwen3.6 的 head 配置、layout 和 262K 原生上下文来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. RoPE 的缓存

RoPE 本身不保存每个 token 的 K/V，而是保存或计算位置相关的 cos/sin。vLLM v0.20.1 的 `RotaryEmbeddingBase` 会计算 `inv_freq`，再用位置 $t$ 与 `inv_freq` 生成 `freqs`，并缓存 cos/sin；forward 时通过 `positions` 从 `cos_sin_cache` 中 select 对应位置。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/rotary_embedding/base.py))

### 4. Gated DeltaNet state 另算

本节不强行展开 Gated DeltaNet 的显存布局，原因是它不是普通 KV cache，而是 vLLM 中 Mamba/GDN 风格的 state cache 路径。后续第六章讲 KV Cache 与长上下文时，会专门区分：

```text
Gated Attention KV cache
Gated DeltaNet / Mamba-style state
PagedAttention block metadata
MTP speculative token 额外 cache
```

---

## 六、vLLM 源码映射

本章最重要的源码映射是：

| 概念 | vLLM v0.20.1 源码入口 | 说明 |
|---|---|---|
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` 按 `layer_type` 选择 `GatedDeltaNetAttention` 或 `Qwen3NextAttention` |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| QKV projection | `vllm/model_executor/models/qwen3_next.py` | `QKVParallelLinear` |
| Q/K RMSNorm | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextRMSNorm` |
| RoPE 构造 | `vllm/model_executor/layers/rotary_embedding/__init__.py` | `get_rope` |
| RoPE 基类 | `vllm/model_executor/layers/rotary_embedding/base.py` | `RotaryEmbeddingBase`, `RotaryEmbedding` |
| Attention operator | `vllm/attention` 体系与 `vllm/v1/attention/backends/` | 具体 backend 第十一、十二章展开 |
| Attention backends | `vllm/v1/attention/backends/` | 包含 `flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`gdn_attn.py` 等 |

`Qwen3NextAttention` 的源码证据非常关键：它设置 Q heads / KV heads / head dim / scaling，创建 `rotary_emb = get_rope(...)`，再创建 `self.attn = Attention(...)`；forward 中调用 `self.rotary_emb(positions, q, k)`，然后调用 `self.attn(q, k, v)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 七、NVIDIA GPU 执行视角

从 GPU 看，Attention 不是一个单一数学公式，而是一组 kernel / fused kernel 的组合。

### 1. QKV projection

```text
hidden_states [N, 2048]
→ QKV projection
→ q, gate, k, v
```

这是 GEMM，适合 Tensor Core。Qwen3.6 的 Q 侧有 16 heads × 256 dim，同时 gate 也在 Q 侧一起投影，因此 Q/Gate 输出维度很大。vLLM 的 `qkv_proj` 参数中 `self.total_num_heads * (1 + self.attn_output_gate)` 正是为了在开启 gate 时多生成 gate 维度。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 2. RoPE kernel

RoPE 对 Q/K 的前 `rotary_dim` 维做 cos/sin 旋转，本质是 elementwise 操作，主要吃显存带宽和 kernel launch / fusion 效率。vLLM 的基类实现中会把 query/key 切成旋转部分和透传部分，只对旋转部分应用 rotary embedding。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/rotary_embedding/base.py))

### 3. Attention kernel

Attention kernel 的核心工作是：

```text
读 Q
读 K cache
读 V cache
计算 QK
做 causal softmax
聚合 V
写 output
```

在 prefill 中，计算量更大；在 decode 中，读取历史 KV cache 的带宽压力更明显。Qwen3.6 原生上下文是 262,144 tokens，这让 full-attention 层的 KV cache 读取和管理成为长上下文推理的关键问题之一。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 4. Gate 和 output projection

```text
attn_output
→ sigmoid(gate)
→ elementwise multiply
→ o_proj GEMM
```

gate 的乘法本身不如 GEMM 重，但会增加一次额外的 elementwise 处理和数据流。vLLM 源码中 `gate = torch.sigmoid(gate)` 后执行 `attn_output = attn_output * gate`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 八、优化前 vs 优化后

### 优化前：朴素 Attention

```text
scores = Q @ K.T
scores += causal_mask
probs = softmax(scores)
out = probs @ V
```

问题：

| 问题 | 后果 |
|---|---|
| 显式构造 $T \times T$ scores | 长 prompt 下显存爆炸 |
| 每步 decode 重新计算历史 K/V | 完全不可接受 |
| KV cache 连续分配 | 易碎片化，难以高并发 |
| 小 batch decode | GPU 利用率低 |
| 不区分 Q heads / KV heads | 忽略 GQA 对 KV cache 的降低作用 |

### 优化后：工程化 Attention

```text
prefill:
  分块 attention kernel
  避免 materialize 全量 scores

decode:
  复用 KV cache
  每步只算新 token 的 Q/K/V
  从 paged KV cache 读取历史 K/V

serving:
  continuous batching
  paged KV cache
  prefix caching
  chunked prefill
```

对 Qwen3.6 还有一个额外优化点：因为 Gated Attention 只有 10 层，而另外 30 层是 Gated DeltaNet，所以不能按“40 层 full attention KV cache”估算显存。正确做法是分开估算：

```text
10 层 Gated Attention KV cache
+
30 层 Gated DeltaNet state/cache
+
MoE expert weights / dispatch buffer
+
MTP speculative cache
```

这个“10 full-attention 层”的判断来自模型卡给出的 40 层 hidden layout。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 九、工程落地

### 1. 先确认 max model len

Qwen3.6 原生上下文是 262,144 tokens，模型卡还说明可扩展到 1,010,000 tokens。vLLM recipe 中 Qwen3.6 的 serving 命令使用 `--max-model-len 262144`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

### 2. 不要用普通 Transformer 的 KV 估算套 Qwen3.6

错误估算：

```text
40 layers × full-attention KV cache
```

更合理的估算：

```text
10 Gated Attention layers × KV cache
+
30 Gated DeltaNet layers × state/cache
```

Gated DeltaNet 的 state 不是本章要展开的 KV cache；后续第六章会把这两类 cache 分开讲。

### 3. 关注 TP 下 KV heads 复制

Qwen3.6 的 KV heads 只有 2。vLLM 源码显示，如果 KV heads 少于 tensor parallel size，就复制 KV heads。对于 `--tensor-parallel-size 8`，这意味着 KV heads 复制逻辑会生效。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

工程含义：

```text
TP 不只是把 heads 平均切开；
当 KV heads 很少时，KV heads 可能在多个 GPU 上复制。
```

这会影响 KV cache 显存估算、attention 通信模式和 backend 行为。

### 4. 关注 RoPE dim

Qwen3.6 的 head dim 是 256，但 RoPE dim 是 64。工程上不要默认把整个 head 都旋转。vLLM 的 `get_rope` 会从 `rope_parameters` 中取 `rope_dim`，如果没有则根据 `partial_rotary_factor` 计算；`RotaryEmbedding` forward 也只旋转 `rotary_dim` 前缀维度。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/rotary_embedding/__init__.py))

### 5. 注意 backend 不在本章下结论

本章不强行判断 Qwen3.6 在你机器上最终选择 FlashAttention、FlashInfer、Triton 还是其他 backend，原因是 backend 选择依赖 vLLM 配置、GPU、dtype、cache layout、环境变量和运行时条件。第十一章和第十二章会结合 `vllm/v1/attention/backends/` 以及运行日志确认。

---

## 十、本章总结

传统 Attention 的主线是：

```text
Q, K, V
→ QK^T / sqrt(d)
→ causal softmax
→ weighted sum of V
```

Qwen3.6 的 Gated Attention 在这条主线上增加了：

```text
Q/K RMSNorm
RoPE on Q/K
GQA: 16 Q heads, 2 KV heads
Output gate: sigmoid(gate) * attention_output
```

但 Qwen3.6 不是 40 层 full attention。它的真实 layout 是：

```text
10 × (
  3 × (Gated DeltaNet → MoE)
  1 × (Gated Attention → MoE)
)
```

所以 Attention 数学只解释其中的 Gated Attention 层；Gated DeltaNet 要用另一套状态化/线性注意力视角来讲。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 小白总结

Attention 就是“当前词去历史里找重点”。

```text
Q = 我想找什么
K = 历史信息的标签
V = 历史信息的内容
```

RoPE 是把“位置信息”旋进 Q 和 K 里，让模型知道谁在前、谁在后。
Qwen3.6 的 Gated Attention 还多了一个 gate，像一个阀门，决定 attention 输出的哪些维度应该通过、哪些维度应该被压低。

---

## 工程师总结

对 Qwen3.6-35B-A3B，Attention 工程判断必须分层：

```text
Gated Attention 层:
  Q heads = 16
  KV heads = 2
  head dim = 256
  RoPE dim = 64
  output gate = sigmoid(gate) * attn_output

Gated DeltaNet 层:
  不按普通 softmax attention 建模
  不直接套普通 KV cache 估算

整体:
  40 层 = 30 Gated DeltaNet + 10 Gated Attention
```

vLLM v0.20.1 中，`Qwen3_5DecoderLayer` 明确区分 `linear_attention` 与 `full_attention`；full-attention 路径走 `Qwen3NextAttention`，其中包含 `qkv_proj`、`get_rope`、`Attention`、gate 和 `o_proj`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 关键公式

**1. Scaled Dot-Product Attention**

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + M
\right)V
$$

**2. 单 token attention score**

$$
s_{t,j} = \frac{q_t \cdot k_j}{\sqrt{d}}
$$

**3. attention 权重**

$$
\alpha_{t,j}
=
\frac{\exp(s_{t,j})}
{\sum_{i \le t}\exp(s_{t,i})}
$$

**4. value 聚合**

$$
o_t = \sum_{j \le t}\alpha_{t,j}v_j
$$

**5. Gated Attention 教学抽象**

$$
y_t = o_t \odot \sigma(g_t)
$$

**6. RoPE 二维旋转**

$$
\begin{bmatrix}
x_1' \\
x_2'
\end{bmatrix}
=
\begin{bmatrix}
\cos \theta & -\sin \theta \\
\sin \theta & \cos \theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
$$

**7. Qwen3.6 Gated Attention KV cache 粗估**

$$
\text{KV elements per token per Gated Attention layer}
=
2 \times 2 \times 256
=
1024
$$

**8. Qwen3.6 full-attention 层数推算**

$$
10 \times 1 = 10\ \text{Gated Attention layers}
$$

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| QKV projection | `vllm/model_executor/models/qwen3_next.py` | `QKVParallelLinear` |
| Q/K norm | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextRMSNorm` |
| RoPE 构造 | `vllm/model_executor/layers/rotary_embedding/__init__.py` | `get_rope` |
| RoPE 基类 | `vllm/model_executor/layers/rotary_embedding/base.py` | `RotaryEmbeddingBase`, `RotaryEmbedding` |
| Attention operator | `vllm/model_executor/models/qwen3_next.py` 引用的 attention 层 | `Attention` |
| Attention backends | `vllm/v1/attention/backends/` | `flash_attn.py`, `flashinfer.py`, `triton_attn.py`, `gdn_attn.py` 等 |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 Gated Attention 配置为 16 Q heads、2 KV heads、head dim 256、RoPE dim 64 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| `Qwen3_5DecoderLayer` 按 `layer_type` 区分 `linear_attention` 和 `full_attention` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| `linear_attention` 路径使用 `GatedDeltaNetAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.linear_attn = GatedDeltaNetAttention(...)` | 源码直接确认 | 是 |
| `full_attention` 路径使用 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.self_attn = Qwen3NextAttention(...)` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 读取 total Q heads、total KV heads，并按 TP size 切分或复制 KV heads | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 设置 `self.scaling = self.head_dim**-0.5` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 使用 `QKVParallelLinear`，并在 gate 开启时增加 Q 侧输出 | `vllm/model_executor/models/qwen3_next.py` | `self.qkv_proj = QKVParallelLinear(...)` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 构造 `rotary_emb = get_rope(...)` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention.forward` 调用 `self.rotary_emb(positions, q, k)` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.forward` | 源码直接确认 | 是 |
| `Qwen3NextAttention.forward` 调用 `self.attn(q, k, v)` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.forward` | 源码直接确认 | 是 |
| `Qwen3NextAttention.forward` 中 gate 经过 sigmoid 后乘到 attention output 上 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.forward` | 源码直接确认 | 是 |
| vLLM `get_rope` 会从 `rope_parameters` 读取 `rope_dim`，否则用 `partial_rotary_factor` 推出 `rotary_dim` | `vllm/model_executor/layers/rotary_embedding/__init__.py` | `get_rope` | 源码直接确认 | 是 |
| vLLM `RotaryEmbedding` 只对 query/key 的前 `rotary_dim` 做旋转，其余维度透传 | `vllm/model_executor/layers/rotary_embedding/base.py` | `RotaryEmbedding.forward_static` | 源码直接确认 | 是 |
| Qwen3.6 在当前机器上最终使用的具体 attention backend | `vllm/v1/attention/backends/` | `flash_attn.py` / `flashinfer.py` / `triton_attn.py` 等 | 待源码确认 | 否 |
| Gated DeltaNet 的完整数学递推和状态缓存布局 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/v1/attention/backends/gdn_attn.py` | 待展开 | 待源码确认 | 否 |

---

## 源码阅读作业

按下面顺序读：

**1. 先读 layer 分流**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5DecoderLayer
if self.layer_type == "linear_attention"
elif self.layer_type == "full_attention"
```

目标：确认 Qwen3.6 不是所有层都走 `Qwen3NextAttention`。

**2. 再读 Gated Attention**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextAttention.__init__
self.total_num_heads
self.total_num_kv_heads
self.head_dim
self.scaling
self.attn_output_gate
self.qkv_proj
self.rotary_emb
self.attn
```

**3. 读 forward**

```text
Qwen3NextAttention.forward
```

重点找：

```text
qkv, _ = self.qkv_proj(hidden_states)
q_gate, k, v = qkv.split(...)
q, gate = torch.chunk(...)
q, k = self.rotary_emb(positions, q, k)
attn_output = self.attn(q, k, v)
gate = torch.sigmoid(gate)
attn_output = attn_output * gate
```

**4. 读 RoPE**

```text
vllm/model_executor/layers/rotary_embedding/__init__.py
vllm/model_executor/layers/rotary_embedding/base.py
```

重点找：

```text
get_rope
rope_dim
partial_rotary_factor
_compute_inv_freq
_compute_cos_sin_cache
forward_static
query_rot / query_pass
key_rot / key_pass
```

---

## 实验验证作业

### 实验 1：确认 attention 层数量

查看 Qwen3.6 config 中的：

```text
layer_types
num_hidden_layers
```

目标：确认 40 层里哪些是 `linear_attention`，哪些是 `full_attention`。

期望结构应与模型卡一致：

```text
3 个 linear_attention
1 个 full_attention
重复 10 次
```

### 实验 2：确认 Q/K/V/Gate 形状

在本地源码中给 `Qwen3NextAttention.forward` 临时加 debug log，打印：

```text
hidden_states.shape
q.shape
k.shape
v.shape
gate.shape
```

目标：验证：

```text
Q heads = 16
KV heads = 2
head dim = 256
gate 与 Q 侧维度匹配
```

### 实验 3：比较 max_model_len 对 KV capacity 的影响

分别启动：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3
```

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

记录：

```text
GPU memory usage
KV/state cache capacity
maximum concurrency
TTFT
TPOT
```

实验目的：看到长上下文不是“改一个参数就免费获得”。

---

## 3 个自测题

**1. 为什么 attention 要除以 $\sqrt{d}$？**
因为 $q \cdot k$ 的尺度会随 head dim 增大而增大，除以 $\sqrt{d}$ 可以稳定 softmax 输入尺度。

**2. Qwen3.6 的 Gated Attention 和普通 Attention 至少有哪三个差异？**
答案应包含：GQA，RoPE 只作用于部分维度，attention output 经过 sigmoid gate 逐元素调制。

**3. 为什么不能按 40 层 full attention 来估算 Qwen3.6 的 KV cache？**
因为 Qwen3.6 的 40 层中按模型卡布局只有 10 个 Gated Attention 层，另外 30 层是 Gated DeltaNet，后者不是普通 full-attention KV cache 路径。

---

## 下一章预告

第六章进入：

```text
KV Cache 与长上下文显存问题
```

下一章会回答四个工程问题：

```text
1. KV cache 到底缓存了什么？
2. 为什么 decode 一定要复用 KV cache？
3. 262K context 为什么会迅速吃掉显存？
4. Qwen3.6 中 Gated Attention KV cache 和 Gated DeltaNet state 应该如何分开估算？
```

第六章会开始把 Attention 数学落到真正的显存账本上。

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
