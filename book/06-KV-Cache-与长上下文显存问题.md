# 第六章：KV Cache 与长上下文显存问题

本章开始进入真正的推理显存账本。
对传统 Transformer，KV Cache 是 decode 阶段加速的核心；对 **Qwen/Qwen3.6-35B-A3B**，必须更谨慎：它的 40 层不是 40 层 full attention，而是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，因此普通 KV Cache 公式主要适用于其中的 **10 个 Gated Attention 层**；另外 30 个 Gated DeltaNet 层要按 Mamba/GDN-style state 单独理解。Qwen3.6 的 Gated Attention 是 16 Q heads、2 KV heads、head dim 256、RoPE dim 64，原生上下文长度 262,144 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

把 decode 想象成你在写一篇长文章。每写一个新字，你都要回顾前文。

**不使用 KV Cache** 时，每生成一个新 token，都像重新把整篇文章从头读一遍，重新整理所有历史信息：

```text
第 1 步：读 prompt，生成 token 1
第 2 步：重新读 prompt + token 1，生成 token 2
第 3 步：重新读 prompt + token 1 + token 2，生成 token 3
...
```

这会极其浪费。

**使用 KV Cache** 时，模型在第一次读到每个历史 token 时，就把它的 key/value 记到“索引卡片”里。后面生成新 token 时，不需要重新为所有历史 token 计算 K/V，只需要：

```text
1. 为当前新 token 算 Q/K/V
2. 把新 K/V 追加进 cache
3. 当前 Q 去读历史 K/V cache
4. 得到 attention 输出
```

这就是 KV Cache 的核心价值：**用显存换计算**。

但是长上下文下，问题反过来了：
你确实少算了很多重复计算，但你要把越来越多历史 token 的 K/V 长期放在 GPU 显存里。对 Qwen3.6 这种 262K 原生上下文模型，KV/state cache 不是小配角，而是服务容量的核心约束。Qwen 模型卡也明确提示：默认上下文是 262,144 tokens，遇到 OOM 可以降低 context window，但建议至少保持 128K 以保留复杂任务能力。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 二、数学直觉

### 1. 没有 KV Cache 的重复计算

传统 attention 中，每层都会从 hidden state 投影出：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

如果当前序列长度是 $T$，每次 decode 都重新处理全部 $T$ 个 token，那么历史 token 的 K/V 会被反复计算。

第 $t$ 步 decode 时，理论上你只新增了一个 token，但如果不用 cache，就会重新算：

$$
K_{1:t}, V_{1:t}
$$

这会造成大量重复计算。

### 2. 有 KV Cache 的 decode

使用 KV Cache 后，历史部分已经保存：

$$
K_{\le t-1}, V_{\le t-1}
$$

当前步只需要算新 token：

$$
q_t, k_t, v_t
$$

然后追加：

$$
K_{\le t} = [K_{\le t-1}; k_t]
$$

$$
V_{\le t} = [V_{\le t-1}; v_t]
$$

attention 输出：

$$
o_t =
\operatorname{softmax}
\left(
\frac{q_t K_{\le t}^{\top}}{\sqrt{d}}
\right)
V_{\le t}
$$

所以 decode 阶段的计算从“反复重算历史 K/V”变成“读取历史 K/V”。

**瓶颈从 compute 转向 memory bandwidth。**

### 3. KV Cache 显存公式

对一个 full-attention 层，KV cache 粗略显存是：

$$
\text{KV bytes}
=
T
\times
2
\times
H_{kv}
\times
D
\times
B_{\text{dtype}}
$$

其中：

| 符号 | 含义 |
|---|---|
| $T$ | token 数 |
| $2$ | K 和 V 两份 cache |
| $H_{kv}$ | KV heads 数 |
| $D$ | head dim |
| $B_{\text{dtype}}$ | 每个元素字节数，例如 BF16/FP16 为 2 bytes |

对多层模型：

$$
\text{Total KV bytes}
=
T
\times
L_{\text{attn}}
\times
2
\times
H_{kv}
\times
D
\times
B_{\text{dtype}}
$$

对 Qwen3.6，**只按 Gated Attention 层算**：

```text
L_attn = 10
H_kv = 2
D = 256
dtype = BF16/FP16 ≈ 2 bytes
```

所以每 token 的 Gated Attention KV cache 粗略为：

$$
10
\times
2
\times
2
\times
256
\times
2
=
20480\ \text{bytes}
$$

也就是每 token 约 **20 KB**，仅包含 10 个 Gated Attention 层的 K/V cache。

如果上下文是 262,144 tokens：

$$
262144 \times 20480
\approx
5.37\ \text{GB}
$$

这是 **单条序列、仅 Gated Attention KV、BF16/FP16、未计 block metadata / padding / fragmentation / GDN state / MoE / workspace / 多模态 / TP 复制细节** 的粗略账本。Qwen3.6 的层布局、KV heads、head dim 和上下文长度来自模型卡；vLLM 源码确认 Qwen3.6 兼容路径会把 `linear_attention` 与 `full_attention` 分流，full attention 才走 `Qwen3NextAttention`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 4. 为什么不能按 40 层 full attention 算

如果你错误地按 40 层 full attention 估算：

$$
40
\times
2
\times
2
\times
256
\times
2
=
81920\ \text{bytes/token}
$$

262K tokens 就会粗略变成：

$$
262144 \times 81920
\approx
21.47\ \text{GB}
$$

这个数字对 Qwen3.6 的 **普通 full-attention KV cache** 是错误心智模型，因为它把 30 个 Gated DeltaNet 层也当成了 full attention KV cache。正确做法是：

```text
10 个 Gated Attention 层：按 KV cache 估算
30 个 Gated DeltaNet 层：按 GDN/Mamba-style state 另算
```

---

## 三、张量形状

### 1. 普通 full attention 的 KV Cache

教学形状：

```text
K cache: [num_layers, batch, seq_len, num_kv_heads, head_dim]
V cache: [num_layers, batch, seq_len, num_kv_heads, head_dim]
```

vLLM 的实际布局不会这么简单，它会为了 PagedAttention、block 管理和 backend 访问效率重新组织 memory layout。vLLM Paged Attention 文档给出的 kernel 形状示例是：

```text
q:       [num_seqs, num_heads, head_size]
k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
v_cache: [num_blocks, num_kv_heads, head_size, block_size]
```

该文档同时带有 warning：它是历史设计文档，不再完整描述当前 vLLM 代码；所以本课程把它作为 paged KV cache 概念说明，具体实现以 vLLM v0.20.1 源码为准。([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html))

### 2. Qwen3.6 Gated Attention 的逻辑形状

不考虑 tensor parallel 时，Qwen3.6 的 Gated Attention 可以理解为：

```text
Q: [N, 16, 256]
K: [N,  2, 256]
V: [N,  2, 256]
```

decode 时，每个新 token 追加：

```text
new K: [num_new_tokens, 2, 256]
new V: [num_new_tokens, 2, 256]
```

历史 cache 逻辑上是：

```text
K cache: [seq_len, 2, 256]
V cache: [seq_len, 2, 256]
```

每个 Gated Attention 层都需要自己的 K/V cache。Qwen3.6 的 Q heads、KV heads、head dim、RoPE dim 来自模型卡；vLLM `Qwen3NextAttention` 源码确认它读取 `config.num_attention_heads`、`config.num_key_value_heads`、`config.head_dim`，并创建 `Attention(..., num_kv_heads=self.num_kv_heads, cache_config=cache_config, ...)`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Tensor Parallel 下的 KV heads

vLLM v0.20.1 的 `Qwen3NextAttention` 逻辑是：

```text
如果 total_num_kv_heads >= tp_size:
    KV heads 跨 TP ranks 切分
否则:
    KV heads 在 TP ranks 之间复制
```

源码注释明确写道：当 KV heads 数量小于 TP size 时，会 replicate KV heads across tensor parallel GPUs。Qwen3.6 的 total KV heads 是 2；如果用 recipe 中常见的 `--tensor-parallel-size 8`，就会进入“KV heads 少于 TP size”的复制逻辑。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

这对工程估算很重要：
**TP 并不总是把 KV cache 按 GPU 数量线性切薄。** 当 KV heads 很少时，某些 KV 数据可能复制，显存和通信行为要以源码和实际日志为准。

### 4. Gated DeltaNet 的 state 形状

本节不强行给出 Gated DeltaNet 的完整 state 公式，原因是它不是普通 K/V cache。vLLM v0.20.1 源码中，`GatedDeltaNetAttention` 被注册为 `@PluggableLayer.register("gated_delta_net_attention")`，继承 `MambaBase`，`mamba_type` 返回 `"gdn_attention"`；它的 state dtype 和 state shape 分别由 `MambaStateDtypeCalculator.gated_delta_net_state_dtype` 与 `MambaStateShapeCalculator.gated_delta_net_state_shape` 计算。源码还显示其计算路径包含 `q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, chunk_indices, chunk_offsets` 等参数，并调用 `fla_chunk_gated_delta_rule`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

所以后续显存账本要分成：

```text
Full-attention KV cache
Gated DeltaNet state cache
MoE expert weights / dispatch buffers
MTP speculative cache
Vision encoder / multimodal embeddings
Runtime workspace
```

---

## 四、计算流程

### 1. Prefill 阶段

Prefill 输入完整 prompt：

```text
input_ids: [prompt_len]
```

每个 full-attention 层会为所有 prompt token 生成 K/V，并写入 cache：

```text
for each Gated Attention layer:
    K_prompt, V_prompt = project(hidden_states)
    write K_prompt/V_prompt into KV cache
```

同时，prefill 还要算 prompt 内部 token 之间的 causal attention。长 prompt 时，prefill 往往是 compute-heavy + memory-heavy。

对 Qwen3.6，还要同时注意 Gated DeltaNet 层的 state 更新，而不是只看 Gated Attention 的 K/V。vLLM 源码确认 `Qwen3_5DecoderLayer` 遇到 `linear_attention` 使用 `GatedDeltaNetAttention`，遇到 `full_attention` 使用 `Qwen3NextAttention`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 2. Decode 阶段

Decode 每步新增 token 数通常远小于 prompt_len：

```text
new token
→ embedding
→ 40 层 forward
```

在 Gated Attention 层：

```text
1. 为新 token 计算 q, k, v
2. 把新 k/v 写入 KV cache
3. 当前 q 读取历史 K/V
4. 得到 attention output
```

在 Gated DeltaNet 层：

```text
1. 读取前一 state
2. 用当前 token 更新 state
3. 输出当前 token 的 hidden state
4. 保存新的 state
```

这里第二段是概念解释；具体 GDN state layout 和 kernel 行为以 `gdn_linear_attn.py`、`gdn_attn.py` 后续源码导读为准。vLLM 当前源码显示 GDN 走 MambaBase / state shape calculator 路径，而不是普通 KV cache 路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

### 3. vLLM 中的 KV slot 分配

vLLM v0.20.1 的 `KVCacheManager` 是核心入口之一。源码中 `KVCacheManager.__init__` 接收 `kv_cache_config`、`max_model_len`、`hash_block_size`、`max_num_batched_tokens`、`enable_caching` 等参数；`allocate_slots(...)` 用于给请求追加新 token 分配 cache slots。其参数里包括 `num_new_tokens`、`new_computed_blocks`、`num_lookahead_tokens`、`num_encoder_tokens` 等，源码注释说明 `num_lookahead_tokens` 用于带 KV-cache 的 spec decode proposer。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py))

这说明在真实 serving 中，KV cache 不是“一个 Python list 追加 K/V”这么简单，而是由 scheduler、block pool、prefix caching、spec decode、encoder tokens 等共同参与的资源分配问题。

---

## 五、显存布局

### 1. 显存账本总览

Qwen3.6 推理显存至少包括：

| 类别 | 内容 | 是否随上下文增长 |
|---|---|---|
| 权重 | embedding、Gated DeltaNet、Gated Attention、MoE、LM head、Vision Encoder | 否 |
| Gated Attention KV cache | 10 个 full-attention 层的 K/V | 是 |
| Gated DeltaNet state | 30 个 GDN 层的 state/cache | 是，但不是普通 KV 公式 |
| MoE runtime buffer | router logits、expert dispatch、expert GEMM workspace | 随 batch/token 动态变化 |
| MTP speculative cache | speculative tokens / lookahead 相关 cache | 随配置变化 |
| 多模态 embedding / encoder buffer | image/video encoder 输出与临时激活 | 随多模态输入变化 |
| CUDA workspace / graph memory pool | kernel 临时区、graph capture 等 | 动态变化 |

### 2. Qwen3.6 Gated Attention KV 粗算

仅算 10 个 Gated Attention 层，BF16/FP16：

```text
layers = 10
K/V = 2
kv_heads = 2
head_dim = 256
bytes = 2
```

每 token：

```text
10 × 2 × 2 × 256 × 2 = 20480 bytes ≈ 20 KB
```

不同上下文长度下，单序列仅 Gated Attention KV 粗略为：

| Context tokens | 仅 Gated Attention KV 粗估 |
|---:|---:|
| 32K | 约 0.67 GB |
| 64K | 约 1.34 GB |
| 128K | 约 2.68 GB |
| 262K | 约 5.37 GB |
| 1M | 约 20.48 GB |

这些是十进制 GB 粗估；实际部署会受 block size、padding、并发请求、TP 复制、KV dtype、backend、MTP、多模态和 workspace 影响。

### 3. 并发会线性放大 cache 压力

如果每条请求平均上下文 $T$，并发请求数是 $B$，那么 full-attention KV cache 粗略变成：

$$
B \times T \times L_{\text{attn}} \times 2 \times H_{kv} \times D \times B_{\text{dtype}}
$$

这就是为什么长上下文和高并发天然冲突。

一个 262K 的单请求可能还能勉强装下；但多个 262K 请求同时 decode，cache 会很快吃掉显存。

### 4. FP8 权重量化不能解决全部 cache 问题

FP8 checkpoint 可以降低权重显存压力，但 KV cache / GDN state / runtime buffers 仍要单独看。Qwen3.6 模型卡给出的 vLLM text-only 命令说明 `--language-model-only` 可以跳过 vision encoder 和 multimodal profiling，从而为额外 KV cache 释放内存；这也间接说明长上下文 serving 的显存瓶颈不只在权重。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 六、vLLM 源码映射

本章源码映射如下：

| 概念 | 文件路径 | 类/函数 | 说明 |
|---|---|---|---|
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` | 区分 `linear_attention` 与 `full_attention` |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | 创建 `Attention(..., cache_config=...)` |
| Gated DeltaNet state | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | MambaBase / state dtype / state shape |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` | 管理 request 到 cache blocks / slots 的分配 |
| slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 为请求追加 token 分配 cache slots |
| PagedAttention 概念文档 | vLLM docs | Paged Attention | 解释 paged KV cache 的 block 概念，但文档标注为历史文档 |

特别注意：vLLM v0.20.1 的 `Qwen3NextAttention` 在初始化时会把 `cache_config` 传给 `Attention`，这说明 full-attention 层接入 vLLM attention/cache 体系；而 `GatedDeltaNetAttention` 则通过 MambaBase 和 state shape calculator 管理状态，不能简单归入同一个 KV cache 公式。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 七、NVIDIA GPU 执行视角

### 1. Decode 变成 HBM bandwidth 问题

decode 阶段每个新 token 都要访问历史 K/V：

```text
q_t: 当前 token，小
K/V cache: 历史 token，大
```

当上下文从 32K 增长到 262K，单步 decode 要读取的历史 cache 也随之增长。即使每步只生成一个 token，GPU 也要不断从 HBM 读越来越长的 K/V。

所以 decode 常见瓶颈是：

```text
HBM bandwidth
L2 cache 命中
KV cache memory layout
attention backend 读取效率
batch size 是否足够大
```

### 2. Prefill 更像大吞吐计算

prefill 处理长 prompt 时，输入 token 多，矩阵乘和 attention 工作量大，更容易吃满 Tensor Core。但长 prompt 也会生成大量 K/V 或 GDN state，需要写入 cache。

### 3. Paged KV Cache 的 GPU 访问

Paged KV Cache 把连续长序列拆成 blocks。attention kernel 需要根据 block table 找到物理 block，再读取 K/V。vLLM Paged Attention 文档说明，paged KV cache 中 key 和 value cache 存在 separate blocks，并且 kernel 依赖特殊 memory layout 和访问方式；不过文档也明确标注为 historical document，不再完整描述当前代码，所以本章只把它作为概念基础，第七章再讲 PagedAttention。([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html))

### 4. GDN state 的 GPU 视角

Gated DeltaNet 层不是每步读取全部历史 K/V，而是读取和更新状态。这是它适合长上下文的关键直觉之一：它把历史压缩进状态，而不是像 full attention 那样每步显式扫全部历史 K/V。

但本节不强行展开 GDN kernel 细节，原因是 vLLM v0.20.1 的 GDN 路径涉及 `fla_chunk_gated_delta_rule`、chunk indices、state shape calculator、Mamba cache dtype 等，需要到第十一、十二章结合源码和 kernel 后端讲。当前只确认：它不是普通 KV cache 路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

---

## 八、优化前 vs 优化后

### 优化前：朴素 KV Cache

```text
每条请求分配一大段连续 KV memory
请求结束后释放
不同长度请求混合运行
```

问题：

| 问题 | 后果 |
|---|---|
| 长短请求混合 | 显存碎片严重 |
| 预留 max_model_len | 大量未使用空间浪费 |
| prompt 重复 | 相同 prefix 重复占用 cache |
| speculative decoding | lookahead tokens 额外占 cache |
| 多模态输入 | encoder tokens / multimodal embeddings 增加预算复杂度 |

### 优化后：vLLM 风格管理

```text
KV cache 分 block
请求按需分配 slots
prefix cache 复用已计算 blocks
scheduler 根据可用 blocks 决定是否接纳更多 token
```

vLLM v0.20.1 的 `KVCacheManager.allocate_slots` 就体现了这个方向：它不是简单按请求分配一整块连续内存，而是根据 request、new tokens、computed blocks、lookahead tokens、encoder tokens 等决定本次追加 token 的 cache 分配。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py))

### 对 Qwen3.6 的优化后心智

错误：

```text
Qwen3.6 = 40 层 attention，所以 KV cache 巨大
```

正确：

```text
Qwen3.6 =
  10 层 Gated Attention KV cache
  + 30 层 Gated DeltaNet state
  + 40 层 MoE expert computation
  + optional MTP speculative cache
  + optional Vision Encoder memory
```

这不是说 Qwen3.6 的 cache 压力小，而是说 **压力来源更复杂**，不能只用普通 dense Transformer 的 KV cache 公式一把尺量到底。

---

## 九、工程落地

### 1. 启动时先控制 max_model_len

vLLM / Qwen 模型卡给出的标准命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

模型卡同时提示，如果 OOM，可以降低 context window，但建议至少 128K 以保留复杂任务能力。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

工程建议：第一次部署不要直接追求 262K + 高并发 + MTP + 多模态全开。建议分阶段：

```text
阶段 1：32K，纯文本，MTP off
阶段 2：128K，纯文本，MTP off
阶段 3：262K，纯文本，MTP off
阶段 4：262K，纯文本，MTP on
阶段 5：262K，多模态
```

### 2. text-only 时考虑释放 vision encoder 显存

Qwen3.6 是带 Vision Encoder 的 causal LM。模型卡给出的 vLLM Text-Only 命令使用：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

模型卡说明该模式会跳过 vision encoder 和 multimodal profiling，以释放更多内存给 KV cache。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. MTP 会增加 cache 预算复杂度

Qwen3.6 模型卡给出的 vLLM MTP 命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

从 cache 管理角度，speculative tokens 不是免费的。vLLM `allocate_slots` 的参数中存在 `num_lookahead_tokens`，源码注释说明它用于带 KV-cache 的 speculative decode proposers；虽然这里不能直接把该注释等同于 Qwen3 MTP 的完整实现细节，但它说明 spec decode 会进入 cache slot 预算。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 4. 排障优先看这些日志/指标

部署 Qwen3.6 时，建议优先观察：

```text
max_model_len
GPU memory utilization
KV cache capacity
每张 GPU 可用 blocks
prefix cache hit rate
num running requests
num waiting requests
TTFT
TPOT
decode tokens/s
OOM 发生在启动、prefill 还是 decode
```

如果 OOM：

| OOM 位置 | 常见原因 | 优先动作 |
|---|---|---|
| 启动时 | 权重 + vision encoder + profiling + cache 预留太大 | 使用 FP8 / text-only / 降低 max_model_len |
| 长 prompt prefill | prompt 太长，prefill 激活和 cache 写入压力大 | 降低 max_model_len / chunked prefill / 限制请求长度 |
| decode 中 | 并发 + 已生成 token 让 cache 持续增长 | 降低并发 / max_tokens / max_model_len |
| MTP 开启后 | speculative tokens 额外占 cache | 降低 `num_speculative_tokens` 或关闭 MTP |
| 多模态请求 | vision encoder 和视觉 token 增加预算 | 分离 text-only 与 multimodal 服务池 |

---

## 十、本章总结

KV Cache 的本质是：

```text
用显存保存历史 K/V
换取 decode 阶段不重复计算历史 token 的 K/V
```

传统 Transformer 可以用统一公式估算所有层的 KV cache，但 Qwen3.6 不行。它的真实结构要求我们拆账：

```text
10 个 Gated Attention 层：
  用 KV cache 公式估算

30 个 Gated DeltaNet 层：
  用 GDN/Mamba-style state 估算，不能强行套普通 KV cache

MoE：
  主要是 expert weights、router、dispatch buffer、expert GEMM workspace

MTP：
  speculative tokens 会影响 cache slot 预算

Vision Encoder：
  text-only 可跳过以释放更多 cache 空间
```

这就是 Qwen3.6 长上下文部署的核心：**不是问“模型权重能不能放下”，而是问“在目标上下文长度和并发下，cache/state/工作区还能不能放下”。**

---

## 小白总结

KV Cache 就像把读过的书签和笔记存下来。
没有它，每写一个新字都要重新读完整篇文章；有了它，只要查之前保存的笔记。

但文章越长，笔记越多。Qwen3.6 支持 262K 原生上下文，所以这些“笔记”会非常大。更复杂的是，Qwen3.6 只有一部分层是普通 K/V 笔记，另一部分 Gated DeltaNet 层用的是另一种状态笔记，不能混在一起算。

---

## 工程师总结

工程上，Qwen3.6 cache 账本应拆成：

```text
Gated Attention KV:
  10 layers × K/V × 2 KV heads × 256 dim × dtype bytes × tokens

Gated DeltaNet state:
  30 layers × GDN state shape × dtype

MTP:
  lookahead/speculative tokens 额外 cache budget

Multimodal:
  vision encoder + multimodal embeddings + encoder tokens

Runtime:
  block metadata + workspace + graph memory + dispatch buffers
```

vLLM v0.20.1 中，`Qwen3_5DecoderLayer` 明确分流 `linear_attention` 和 `full_attention`；`Qwen3NextAttention` 接入 `Attention(..., cache_config=...)`；`GatedDeltaNetAttention` 则是 MambaBase / state shape calculator 路径；`KVCacheManager.allocate_slots` 负责为请求追加 tokens 分配 cache slots。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 关键公式

**1. 单层 KV cache**

$$
\text{KV bytes}
=
T
\times
2
\times
H_{kv}
\times
D
\times
B_{\text{dtype}}
$$

**2. 多层 KV cache**

$$
\text{Total KV bytes}
=
T
\times
L_{\text{attn}}
\times
2
\times
H_{kv}
\times
D
\times
B_{\text{dtype}}
$$

**3. Qwen3.6 Gated Attention KV 每 token 粗估**

$$
10
\times
2
\times
2
\times
256
\times
2
=
20480\ \text{bytes/token}
$$

**4. Qwen3.6 262K Gated Attention KV 粗估**

$$
262144
\times
20480
\approx
5.37\ \text{GB}
$$

**5. 并发下的 KV cache**

$$
\text{Total KV bytes}
=
B
\times
T
\times
L_{\text{attn}}
\times
2
\times
H_{kv}
\times
D
\times
B_{\text{dtype}}
$$

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Full attention / KV cache 接入 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention`, `Attention` |
| Gated DeltaNet state | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| cache slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| Paged KV cache 概念 | vLLM docs | Paged Attention design doc |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 hidden layout 为 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 Gated Attention 为 16 Q heads、2 KV heads、head dim 256、RoPE dim 64 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 原生上下文为 262,144 tokens，可扩展到 1,010,000 tokens | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 vLLM 标准部署命令使用 `--tensor-parallel-size 8 --max-model-len 262144` | Hugging Face model card | vLLM serving command | 官方文档确认 | 是 |
| Qwen3.6 text-only 命令可用 `--language-model-only` 跳过 vision encoder，为 KV cache 释放内存 | Hugging Face model card | vLLM Text-Only command | 官方文档确认 | 是 |
| `Qwen3_5DecoderLayer` 按 `layer_type` 区分 `linear_attention` 与 `full_attention` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| `linear_attention` 使用 `GatedDeltaNetAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.linear_attn = GatedDeltaNetAttention(...)` | 源码直接确认 | 是 |
| `full_attention` 使用 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.self_attn = Qwen3NextAttention(...)` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 读取 Q heads、KV heads，并在 KV heads 少于 TP size 时复制 KV heads | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 创建 `Attention(..., cache_config=...)` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 继承 `MambaBase`，`mamba_type` 为 `gdn_attention` | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 的 state dtype / shape 由 `MambaStateDtypeCalculator` 和 `MambaStateShapeCalculator` 计算 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `get_state_dtype`, `get_state_shape` | 源码直接确认 | 是 |
| `KVCacheManager` 管理 KV cache 配置、max model len、cache blocks 等 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.__init__` | 源码直接确认 | 是 |
| `KVCacheManager.allocate_slots` 为请求追加 token 分配 cache slots，并包含 `num_lookahead_tokens`、`num_encoder_tokens` 等参数 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 源码直接确认 | 是 |
| Paged KV cache 中 key/value cache 以 blocks 管理的概念 | vLLM Paged Attention docs | Paged Attention design doc | 官方文档确认 | 是，但该文档标注为历史文档 |
| Qwen3.6 Gated DeltaNet state 的精确逐字段显存公式 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py`, `vllm/v1/attention/backends/gdn_attn.py` | 待展开 | 待源码确认 | 否 |
| Qwen3.6 MTP 对 `allocate_slots` / lookahead tokens 的完整调用链 | speculative decoding 相关源码 | 待展开 | 待源码确认 | 否 |

---

## 源码阅读作业

按这个顺序读：

**1. 先读 Qwen3.6 层分流**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5DecoderLayer
self.layer_type == "linear_attention"
self.layer_type == "full_attention"
GatedDeltaNetAttention
Qwen3NextAttention
```

目标：确认普通 KV cache 只适合解释 full-attention 分支。

**2. 读 Gated Attention cache 接入**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextAttention.__init__
self.total_num_heads
self.total_num_kv_heads
self.num_kv_heads
Attention(... cache_config=cache_config ...)
```

目标：理解 KV heads、TP size、cache_config 如何进入 attention 层。

**3. 读 GDN state 入口**

```text
vllm/model_executor/layers/mamba/gdn_linear_attn.py
```

重点找：

```text
GatedDeltaNetAttention
mamba_type
get_state_dtype
get_state_shape
fla_chunk_gated_delta_rule
```

目标：确认 GDN 是 state/cache 路径，不是普通 K/V cache。

**4. 读 KV cache manager**

```text
vllm/v1/core/kv_cache_manager.py
```

重点找：

```text
KVCacheManager
allocate_slots
num_new_tokens
new_computed_blocks
num_lookahead_tokens
num_encoder_tokens
```

目标：理解 vLLM 不是按请求分配一整段连续 KV，而是按 blocks/slots 管理。

---

## 实验验证作业

### 实验 1：max_model_len 对显存的影响

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
  --max-model-len 131072 \
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
启动显存
KV/cache capacity
可接纳并发
TTFT
TPOT
```

### 实验 2：text-only 对 cache capacity 的影响

对比：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

和：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

观察：

```text
GPU memory available for KV/cache
启动耗时
多模态能力是否被关闭
最大并发是否变化
```

### 实验 3：MTP 对 cache 和吞吐的影响

对比：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

和：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

记录：

```text
TPOT
tokens/s
可承载并发
cache capacity
OOM 是否更早出现
```

---

## 3 个自测题

**1. KV Cache 为什么能加速 decode？**
因为历史 token 的 K/V 已经保存，decode 每步只需要计算新 token 的 Q/K/V，并读取历史 K/V，而不是反复重算所有历史 token 的 K/V。

**2. 为什么不能把 Qwen3.6 按 40 层 full attention 来估算 KV cache？**
因为 Qwen3.6 的 40 层中，按模型卡 layout 只有 10 个 Gated Attention 层；另外 30 个是 Gated DeltaNet，要按 GDN/Mamba-style state 单独估算。

**3. 为什么 FP8 权重不能彻底解决长上下文显存问题？**
因为 FP8 主要降低权重显存，长上下文的 KV cache、GDN state、MTP speculative cache、多模态 buffer 和 runtime workspace 仍会随上下文和并发增长。

---

## 下一章预告

第七章进入：

```text
PagedAttention
```

下一章会把本章的“KV cache 很大、不能连续粗暴分配”进一步工程化，回答：

```text
1. 为什么 KV cache 要分页？
2. block table 是什么？
3. vLLM 如何像操作系统管理虚拟内存一样管理 KV cache？
4. PagedAttention 如何降低碎片并支持 continuous batching？
5. Qwen3.6 的 Gated Attention KV cache 与 Gated DeltaNet state 在分页管理中有什么边界？
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
- [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
