# 第二章：传统 LLM 推理总览

> 本章先建立“传统 decoder-only LLM 推理”的骨架：**token → embedding → 多层 decoder → logits → sampling → 下一个 token → 循环**。
> 但要始终记住：这只是骨架。固定案例 **Qwen/Qwen3.6-35B-A3B** 不是普通 Transformer Decoder，它是带 Vision Encoder 的 causal LM，语言主干包含 Gated DeltaNet、Gated Attention、MoE、MTP，并且是 35B 总参数、每 token 激活 3B、原生 262K 上下文的 hybrid 架构。模型卡明确给出 hidden size 2048、40 层、padded vocab 248,320，以及 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` 的 hidden layout。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

把 LLM 推理想成一个“逐字接龙机器”。

用户给一句话：

> “北京的冬天通常”

模型不是一次性写完整答案，而是先预测下一个 token：

> “很”

然后把它接回上下文：

> “北京的冬天通常很”

再预测下一个 token：

> “冷”

再接回去：

> “北京的冬天通常很冷”

这就是 **autoregressive decoding**：每一步只预测下一个 token，预测完再把这个 token 作为未来上下文的一部分。

传统 LLM 推理可以粗略分成两个阶段：

| 阶段 | 类比 | 做什么 |
|---|---|---|
| Prefill | 读完整题目 | 把用户 prompt 一次性送进模型，算出所有 prompt token 的隐藏状态和 KV cache |
| Decode | 一个字一个字写答案 | 每次只输入最新 token，复用历史 KV cache，预测下一个 token |

在真实服务中，vLLM 这类 serving engine 的价值不只是“跑模型”，而是让很多用户请求可以一起高效跑。vLLM 官方文档把它的核心能力列为 PagedAttention 管理 KV memory、continuous batching、chunked prefill、prefix caching、CUDA/HIP graph、优化 attention kernel、优化 GEMM/MoE kernel 等。([vLLM](https://docs.vllm.ai/))

---

## 二、数学直觉

传统 causal LM 的目标是建模条件概率：

$$
P(x_{1:T})=\prod_{t=1}^{T}P(x_t \mid x_{<t})
$$

生成时，我们已经有上下文 $x_{1:t}$，模型输出下一个 token 的分布：

$$
P(x_{t+1} \mid x_{1:t})
$$

模型并不直接“理解句子”，而是做这个流程：

$$
\text{tokens}
\rightarrow
\text{embeddings}
\rightarrow
\text{decoder hidden states}
\rightarrow
\text{logits}
\rightarrow
\text{probabilities}
\rightarrow
\text{sample next token}
$$

最后一层 hidden state $h_t$ 会被投影到 vocabulary 维度：

$$
\text{logits}_t = h_t W_{\text{lm\_head}}^\top
$$

然后通常经过 temperature、top-k、top-p、presence penalty、repetition penalty 等采样策略，选出下一个 token。

对传统 dense Transformer 来说，每层大致是：

$$
h \leftarrow h + \text{SelfAttention}(\text{Norm}(h))
$$

$$
h \leftarrow h + \text{MLP}(\text{Norm}(h))
$$

但对 Qwen3.6-35B-A3B，这个“SelfAttention + MLP”只能当作入门抽象。实际语言主干是 Gated DeltaNet / Gated Attention 与 MoE 的混合层，且每 token 只激活部分 expert。模型卡确认其 MoE 为 256 experts，激活 8 routed experts + 1 shared expert。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 三、张量形状

先用传统 decoder-only LLM 的形状建立直觉：

| 符号 | 含义 | 典型形状 |
|---|---|---|
| $B$ | batch size / 当前并发请求数 | 标量 |
| $T$ | 当前序列长度 | 标量 |
| $H$ | hidden size | 标量 |
| $V$ | vocab size | 标量 |
| `input_ids` | token id | `[B, T]` |
| `hidden_states` | 每个 token 的隐藏向量 | `[B, T, H]` |
| `logits` | 每个 token 对词表的打分 | `[B, T, V]` |
| `next_token_logits` | 最后一个位置的输出分布 | `[B, V]` |

固定案例 Qwen3.6-35B-A3B 中，模型卡给出的关键形状参数是：hidden dimension = 2048，token embedding = 248,320 padded，LM output = 248,320 padded，number of layers = 40。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

所以对纯文本推理可以记成：

```text
input_ids:      [B, T]
embeddings:     [B, T, 2048]
hidden_states:  [B, T, 2048]
logits:         [B, T, 248320]
next_logits:    [B, 248320]
```

但注意两点。

第一，Qwen3.6 是 **Causal Language Model with Vision Encoder**，多模态输入时会先经过 vision encoder，再把视觉 embedding 合并进语言模型输入序列。vLLM v0.20.1 tag 源码中，`qwen3_5.py` 的 multimodal 类会构造 `Qwen3_VisionTransformer`，然后构造 `Qwen3_5MoeForCausalLM` 作为 language model。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py))

第二，Qwen3.6 的语言层并不是普通 dense MLP，而是 MoE。vLLM v0.20.1 源码中 `Qwen3_5DecoderLayer` 根据 `layer_type` 创建 `GatedDeltaNetAttention` 或 `Qwen3NextAttention`，并在 MoE text config 下创建 `Qwen3NextSparseMoeBlock`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 四、计算流程

传统 LLM 推理主流程如下：

```text
1. tokenizer:
   text -> input_ids

2. embedding:
   input_ids -> hidden_states

3. prefill forward:
   对 prompt 中所有 token 并行跑完整模型
   生成最后位置 logits
   同时写入每层 KV cache

4. sampling:
   logits -> next_token

5. decode loop:
   while not stop:
       输入上一步生成的 token
       读取历史 KV cache
       只计算当前 token 的新 hidden state
       写入新的 K/V
       生成 logits
       sampling 得到下一个 token

6. detokenizer:
   token ids -> text
```

Prefill 和 decode 的性能瓶颈不同：

| 阶段 | 主要输入长度 | 主要计算特征 | 常见瓶颈 |
|---|---:|---|---|
| Prefill | prompt 长度 $T$ | 大矩阵乘、attention 建 KV | compute + HBM bandwidth |
| Decode | 每步 1 个新 token | 读历史 KV、算当前 token attention | KV cache 读带宽、batching 效率 |
| Long context decode | 历史上下文很长 | 每步都要看更长历史 | KV cache 显存和读带宽 |

传统 Transformer 的 attention 在 prefill 中会看到 $T \times T$ 的因果关系；decode 时每个新 token 只新算一个 query，但要和历史所有 key/value 交互。也就是说，decode 的单步计算看似小，但会反复读取越来越长的 KV cache。

Qwen3.6 的差异在于：不是所有层都走 full attention。模型卡显示它的 40 层由 Gated DeltaNet → MoE 与 Gated Attention → MoE 的模式组成，因此后续讲 attention 时必须区分 full attention 层、linear attention/Gated DeltaNet 层和 MoE 层，不能把它当成“40 层普通 attention + MLP”。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 五、显存布局

传统 LLM 推理显存主要分四类：

| 显存类别 | 内容 | 生命周期 |
|---|---|---|
| Weights | 模型参数 | 常驻 |
| Activations | 当前 forward 临时中间结果 | 每步释放或复用 |
| KV Cache | 每层历史 token 的 key/value | 随上下文增长 |
| Runtime workspace | CUDA kernel 临时 workspace、通信 buffer、graph memory pool 等 | 运行时动态变化 |

一个常见误区是：只看模型权重大小，不看 KV cache。

例如 dense BF16 权重大约是：

$$
\text{weight memory} \approx \text{num parameters} \times 2 \text{ bytes}
$$

35B 参数的 BF16 权重粗略是 70GB 级别；FP8 checkpoint 则显著降低权重显存压力。Qwen3.6 有官方 FP8 版本，vLLM recipe 也把 Qwen3.6-35B-A3B / FP8 列为 Hugging Face 入口。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

但长上下文下，KV cache 会随：

$$
\text{layers} \times \text{tokens} \times \text{kv heads} \times \text{head dim}
$$

增长。Qwen3.6 原生上下文是 262,144 tokens，模型卡还说可扩展到 1,010,000 tokens；这意味着即使权重量化，长上下文 KV cache 仍可能成为主要显存压力。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

vLLM 的 PagedAttention 就是为了解决 KV cache 的管理问题。vLLM Paged Attention 文档说明，vLLM 的 paged KV cache 会把 key/value cache 存在分离的 block 中；其设计文档还明确给出 kernel 输入形状，例如 `q: [num_seqs, num_heads, head_size]`，`k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]`，`v_cache: [num_blocks, num_kv_heads, head_size, block_size]`。([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/))

---

## 六、vLLM 源码映射

本章只做总览，不展开 Scheduler / Block Manager / Attention Backend 的内部实现；这些留到第十二章。这里先建立“传统推理流程”与 vLLM v0.20.1 源码入口的对应关系。

| 推理阶段 | 概念 | vLLM v0.20.1 对 Qwen3.6 的源码入口 |
|---|---|---|
| Embedding | token id → hidden state | `vllm/model_executor/models/qwen3_5.py` 中 `Qwen3_5ForCausalLMBase.embed_input_ids` |
| Decoder forward | hidden state 逐层变换 | `Qwen3_5Model` / `Qwen3_5DecoderLayer` |
| Hybrid layer selection | linear attention / full attention | `Qwen3_5DecoderLayer.__init__` 中根据 `layer_type` 选择 `GatedDeltaNetAttention` 或 `Qwen3NextAttention` |
| MoE block | routed experts / shared expert | `Qwen3NextSparseMoeBlock` |
| LM head | hidden state → logits | `Qwen3_5ForCausalLMBase.compute_logits` |
| Vision encoder | image/video → multimodal embeddings | `Qwen3_VisionTransformer` |
| MTP | 多 token 预测辅助路径 | `qwen3_next_mtp.py` 中 `Qwen3NextMTP`，后续第 3、13 章展开 |

vLLM v0.20.1 的 `qwen3_5.py` 确认：`Qwen3_5ForCausalLMBase` 创建 `ParallelLMHead` 或复用 embedding 作为 lm head，并使用 `LogitsProcessor` 计算 logits；`compute_logits` 调用 `self.logits_processor(self.lm_head, hidden_states)`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py))

源码还确认：Qwen3.6 这种 Qwen3.5/3.6 兼容模型在 vLLM 中是 hybrid：`Qwen3_5DecoderLayer` 会在 `linear_attention` 时创建 `GatedDeltaNetAttention`，在 `full_attention` 时创建 `Qwen3NextAttention`，MoE text config 下创建 `Qwen3NextSparseMoeBlock`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 七、NVIDIA GPU 执行视角

从 GPU 视角看，传统 LLM 推理不是“一个大函数”，而是一串 kernel：

```text
embedding lookup
RMSNorm / LayerNorm
QKV projection GEMM
attention kernel
output projection GEMM
MLP GEMM
activation kernel
lm_head GEMM
sampling kernel
```

在 NVIDIA GPU 上，性能主要受三类资源约束：

| 资源 | 直觉 | LLM 推理中的表现 |
|---|---|---|
| HBM bandwidth | 显存读写速度 | decode 读 KV cache、读权重 |
| Tensor Core | 矩阵乘吞吐 | QKV projection、MLP/MoE GEMM、LM head |
| SM occupancy / scheduling | 并行度是否吃满 | 小 batch decode 时容易吃不满 GPU |

Prefill 阶段通常更像“大矩阵乘 + attention”的高吞吐任务；decode 阶段每个请求每步只生成 1 个 token，如果 batch 很小，GPU 会“吃不饱”。所以 serving engine 要做 continuous batching：不断把不同请求的 decode step 拼在一起，让 GPU 始终有活干。vLLM 官方首页明确把 continuous batching、chunked prefill、prefix caching、PagedAttention、优化 attention kernels、优化 GEMM/MoE kernels 列为核心能力。([vLLM](https://docs.vllm.ai/))

对 Qwen3.6，还要额外考虑 MoE dispatch：router 先决定 token 去哪些 experts，然后专家计算可能被分散到不同 GPU；后续第九、十、十一章会专门讲 router、expert dispatch、expert parallel、NCCL all-to-all / all-gather 等。

---

## 八、优化前 vs 优化后

传统、朴素的 LLM serving 方式大致是：

```text
一个请求进来
单独 prefill
单独 decode
生成完再处理下一个请求
```

问题很明显：

| 问题 | 后果 |
|---|---|
| 请求长短不同 | GPU 等待短请求/长请求，batch 利用率差 |
| KV cache 连续分配 | 显存碎片严重 |
| decode 每步 token 少 | GPU Tensor Core/SM 利用率低 |
| prompt 重复 | 相同前缀重复计算 |
| 长上下文 | KV cache 占用巨大 |

优化后的 serving engine 会做：

| 优化 | 解决什么 |
|---|---|
| Continuous batching | 动态合批，让 decode 阶段持续喂满 GPU |
| PagedAttention | 把 KV cache 分块管理，降低碎片与搬迁压力 |
| Chunked prefill | 把长 prompt 分块，避免 prefill 抢占 decode 太久 |
| Prefix caching | 复用相同 prefix 的 KV cache |
| Tensor Parallel | 把大矩阵切到多 GPU |
| Expert Parallel | MoE experts 分布到多 GPU |
| Speculative decoding / MTP | 用辅助预测减少主模型逐 token 等待成本 |

vLLM 的 Qwen3.6 recipe 给出的基础 serving 命令就是多 GPU tensor parallel：`vllm serve Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3`；同一 recipe 也给出 MTP speculative decoding 配置 `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 九、工程落地

如果你今天要在 NVIDIA GPU 上服务 Qwen3.6-35B-A3B，第二章阶段只需要建立这几个工程判断：

第一，**不要先纠结 kernel**。先把请求生命周期看懂：

```text
HTTP request
-> tokenizer / processor
-> prefill
-> KV cache allocation
-> decode loop
-> sampling
-> streaming response
```

第二，**不要把 max_model_len 当成普通参数**。Qwen3.6 原生 262K context，vLLM recipe 的 Qwen3.6 示例也直接使用 `--max-model-len 262144`；这会显著影响 KV cache 预算、并发上限、chunked prefill 策略和是否需要降级上下文长度。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

第三，**不要把 FP8 只理解成“模型变小”**。FP8 checkpoint 主要降低权重和 GEMM 相关压力，但长上下文下 KV cache 仍然会增长；所以 FP8 单卡可作为降级/验证配置，但 262K 满上下文、高并发和多模态 workload 往往需要更系统的并行与内存规划。

第四，**不要忽略多模态开销**。Qwen3.6 是带 Vision Encoder 的 causal LM；当只做文本服务时，官方 recipe 对 Qwen3.5 的吞吐配置提示 `--language-model-only` 可跳过 vision encoder、释放更多 KV cache 显存，这个思想对理解 Qwen3.6 的 text-only 部署也很关键，但具体配置仍需以 Qwen3.6 的 recipe 与 vLLM v0.20.1 实测为准。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

第五，**先建立观测指标**。最基本的压测指标应该包括：

| 指标 | 说明 |
|---|---|
| TTFT | time to first token，主要受 prefill / queue 影响 |
| TPOT | time per output token，主要受 decode / KV / batching 影响 |
| throughput | tokens/s |
| GPU memory used | 权重 + KV cache + workspace |
| KV cache utilization | block 使用率、可接纳 token 数 |
| request queue time | scheduler 排队时间 |
| prefill/decode ratio | prefill 是否拖慢 decode |

---

## 十、本章总结

传统 LLM 推理的核心是：

```text
prompt tokens
-> prefill
-> KV cache
-> decode one token
-> sampling
-> append token
-> repeat
```

但在 Qwen3.6-35B-A3B 上，这个流程只是“外壳”。真正的内部层结构不是普通 Transformer，而是：

```text
Vision Encoder, when multimodal
+
10 × (
    3 × (Gated DeltaNet → MoE)
    →
    1 × (Gated Attention → MoE)
)
+
MTP capability
```

因此后续课程会用“传统 LLM 推理流程”作为主线，但每遇到一个传统概念，都要问：

1. Qwen3.6 这里是不是被 Gated DeltaNet 替代了？
2. 这里是不是走 Gated Attention 而不是普通 attention？
3. MLP 是不是已经变成 MoE？
4. token 是否经过 8 routed experts + 1 shared expert？
5. MTP 是否改变 decode 策略？
6. 262K context 是否让 KV cache 成为主瓶颈？

---

## 小白总结

LLM 推理就是“读题 + 接龙”。

读题阶段叫 **prefill**，模型把 prompt 全部看完，并存下以后要用的 KV cache。接龙阶段叫 **decode**，模型每次生成一个 token，再把它接回上下文继续生成。

传统 Transformer 可以帮助我们理解基本流程，但 Qwen3.6-35B-A3B 更复杂：它不是普通文本 decoder，而是带 Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP 的 hybrid 模型。

---

## 工程师总结

从工程角度，传统 LLM serving 的核心瓶颈不是单纯“模型能不能跑”，而是：

```text
权重显存
+ KV cache 显存
+ prefill/decode 调度
+ batch 利用率
+ kernel 吞吐
+ 多 GPU 通信
```

vLLM 的价值在于把这些问题系统化：PagedAttention 管 KV cache，continuous batching 提高 decode 利用率，chunked prefill 降低长 prompt 对 decode 的阻塞，prefix caching 复用公共前缀，parallelism 和 optimized kernels 提升多 GPU/单 GPU 执行效率。vLLM 官方文档也把这些能力列为其核心特性。([vLLM](https://docs.vllm.ai/))

---

## 关键公式

**1. 自回归分解**

$$
P(x_{1:T})=\prod_{t=1}^{T}P(x_t \mid x_{<t})
$$

**2. 下一个 token 分布**

$$
P(x_{t+1}\mid x_{\le t})=\text{softmax}(\text{logits}_t)
$$

**3. LM head**

$$
\text{logits}_t = h_t W_{\text{lm\_head}}^\top
$$

**4. Temperature**

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**5. KV cache 粗略增长**

$$
\text{KV memory} \propto
\text{num layers}
\times
\text{sequence length}
\times
\text{num kv heads}
\times
\text{head dim}
\times
2
\times
\text{bytes per element}
$$

这里的 $2$ 来自 K 和 V 两份缓存。

---

## 关键源码入口

| 模块 | vLLM v0.20.1 路径 | 作用 |
|---|---|---|
| Qwen3.5/3.6 兼容模型入口 | `vllm/model_executor/models/qwen3_5.py` | Qwen3.5 / Qwen3.6 推理模型定义 |
| Causal LM base | `vllm/model_executor/models/qwen3_5.py` | embedding、forward、logits |
| Hybrid decoder layer | `vllm/model_executor/models/qwen3_5.py` | 根据 layer type 选择 Gated DeltaNet 或 full attention |
| Qwen3Next attention / MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention`、`Qwen3NextSparseMoeBlock` |
| Gated DeltaNet layer | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | Gated DeltaNet attention 实现入口 |
| Logits processor | `vllm/model_executor/layers/logits_processor.py` | hidden state 到 logits 的处理 |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | KV cache 分配与管理，后续第六、七、十二章展开 |
| Scheduler | `vllm/v1/core/sched/scheduler.py` | 请求调度，后续第八、十二章展开 |
| Attention backends | `vllm/v1/attention/backends/` | attention backend，后续第十一、十二章展开 |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 在 vLLM v0.20.1 中通过 Qwen3.5/3.6 兼容模型入口承载 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration`, `Qwen3_5ForConditionalGeneration` | 源码直接确认 | 是 |
| Qwen3.6 的 multimodal 路径包含 vision tower 与 language model | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer`, `Qwen3_5MoeForCausalLM` | 源码直接确认 | 是 |
| Qwen3.6 的 decoder layer 会按 layer type 选择 Gated DeltaNet 或 full attention | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| linear attention 路径使用 `GatedDeltaNetAttention` | `vllm/model_executor/models/qwen3_5.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| full attention 路径使用 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | 源码直接确认 | 是 |
| MoE text config 下使用 sparse MoE block | `vllm/model_executor/models/qwen3_5.py`, `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | 源码直接确认 | 是 |
| logits 计算由 LM head + LogitsProcessor 完成 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.compute_logits` | 源码直接确认 | 是 |
| vLLM 的 PagedAttention 使用 paged KV cache block 布局 | vLLM docs design page | Paged Attention design document | 官方文档确认 | 是 |
| Qwen3.6 recipe 使用 TP=8、262144 max model len、qwen3 reasoning parser | vLLM Qwen3.5 & Qwen3.6 recipe | `vllm serve Qwen/Qwen3.6-35B-A3B ...` | recipe 确认 | 是 |
| Qwen3.6 的 MTP speculative decoding 可通过 recipe 的 speculative config 启用 | vLLM Qwen3.5 & Qwen3.6 recipe | `--speculative-config '{"method": "mtp", ...}'` | recipe 确认 | 是 |

---

## 源码阅读作业

读 vLLM v0.20.1 tag 中这几个入口，不要求一次读懂，只要求建立“流程地图”：

1. `vllm/model_executor/models/qwen3_5.py`
   重点找：
   - `Qwen3_5ForCausalLMBase`
   - `embed_input_ids`
   - `forward`
   - `compute_logits`
   - `Qwen3_5DecoderLayer`

2. `vllm/model_executor/models/qwen3_next.py`
   重点找：
   - `Qwen3NextAttention`
   - `Qwen3NextSparseMoeBlock`

3. `vllm/v1/core/kv_cache_manager.py`
   只看类名和函数名，先不用深入。

4. `vllm/v1/core/sched/scheduler.py`
   只看 scheduler 的输入输出概念，先不用深入。

---

## 实验验证作业

本章不要求你先跑满 Qwen3.6-35B-A3B。建议先用较小模型验证推理生命周期，再迁移到 Qwen3.6。

实验 1：观察 prefill 和 decode 的差异。

```bash
vllm serve Qwen/Qwen3-8B \
  --reasoning-parser qwen3
```

然后分别请求：

```text
短 prompt + 生成 256 tokens
长 prompt + 生成 256 tokens
短 prompt + 生成 2048 tokens
```

记录：

```text
TTFT
tokens/s
GPU memory
GPU utilization
```

实验 2：理解 batch 对 decode 的影响。

同时发 1、4、16、64 个请求，观察：

```text
单请求延迟
整体吞吐 tokens/s
GPU utilization
```

实验 3：为 Qwen3.6 做预案。

按 recipe 的 Qwen3.6 命令作为目标配置：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

先不要盲目跑满。先估算：

```text
GPU 数量
单卡显存
权重 dtype
KV cache dtype
max_model_len
是否 text-only
是否启用 prefix caching
是否启用 MTP
```

---

## 3 个自测题

**1. 为什么 LLM 推理要分 prefill 和 decode？**
提示：prompt 阶段可以并行处理多个历史 token；decode 阶段每一步依赖上一步生成的 token。

**2. 为什么长上下文场景下，权重不是唯一显存瓶颈？**
提示：KV cache 会随 sequence length 增长，262K context 会显著放大这个问题。

**3. 为什么不能把 Qwen3.6-35B-A3B 当成普通 Transformer Decoder？**
提示：它有 Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP，并且 MoE 是 256 experts、8 routed + 1 shared expert。

---

## 下一章预告

第三章正式进入 **Qwen3.6 架构总览**。

我们会把传统的：

```text
Self-Attention → MLP
```

替换成 Qwen3.6 的真实结构：

```text
Gated DeltaNet → MoE
Gated Attention → MoE
MTP
Vision Encoder
```

重点回答四个问题：

1. Gated DeltaNet 为什么不是普通 attention？
2. Gated Attention 和传统 attention 有什么关系？
3. MoE 的 router、routed experts、shared expert 在推理中怎么工作？
4. MTP 为什么可能改变 decode 的吞吐与延迟结构？

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vLLM](https://docs.vllm.ai/)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py)
