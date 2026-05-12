# 第九章：MoE：Router、Expert、Shared Expert、Top-k、Expert Dispatch

本章把传统 Transformer 里的 **MLP / FFN** 替换成 Qwen3.6 的真实 **MoE** 路径。

对固定案例 **Qwen/Qwen3.6-35B-A3B**，模型卡确认：它是 35B total / 3B activated 的模型，hidden size 2048，40 层，hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`；MoE 部分有 **256 experts**，每 token 激活 **8 routed experts + 1 shared expert**，expert intermediate dimension 是 **512**。这意味着 Qwen3.6 的每一层都不是传统 dense MLP，而是 attention/GDN 后接一个 sparse MoE block。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

传统 MLP 像一个固定加工车间：

```text
每个 token 都送进同一个 MLP
同一套权重处理所有 token
```

MoE 像一个大型专家医院：

```text
每个 token 先挂号
router 判断它该看哪些专家
从 256 个专家里选出 8 个 routed experts
再额外经过 1 个 shared expert
最后把专家意见加权合并
```

比如一个 token 和代码相关，router 可能更偏向“代码专家”；一个 token 和数学推理相关，router 可能更偏向“数学专家”；一个 token 是通用语法连接词，也会经过 shared expert 保留通用能力。这里的“专家”不是人类意义上的可解释专家，而是不同参数子网络；它们是否真的对应某种语义专长，需要实证分析，不能望文生义。

这就是 MoE 的核心：**总参数很多，但每个 token 只激活一部分参数**。Qwen3.6 的总参数是 35B，但每 token 激活约 3B；MoE 的 256 experts 里，每个 token 只走 8 个 routed experts 加 1 个 shared expert。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 二、数学直觉

### 1. Dense MLP

传统 dense Transformer 的 FFN / MLP 可以写成：

$$
y = W_2 \, \phi(W_1 h)
$$

每个 token 都经过同一组 $W_1, W_2$。

### 2. MoE Router

MoE 先用 router 给每个 token 算专家分数：

$$
r = h W_{\text{router}}
$$

其中：

$$
h \in \mathbb{R}^{2048}
$$

$$
r \in \mathbb{R}^{256}
$$

对 Qwen3.6，router logits 的维度是 256，因为模型卡确认有 256 experts；vLLM v0.20.1 的 `Qwen3NextSparseMoeBlock` 中 `self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, ...)`，forward 中注释也写明 `router_logits: (num_tokens, n_experts)`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. Top-k 选择专家

Router 不会让每个 token 跑 256 个专家，而是选 top-k：

$$
\mathcal{E}_{topk}(h)=\mathrm{TopK}(r, k=8)
$$

然后通常会对 top-k 分数做 softmax 或 renormalize，得到专家权重：

$$
w_e =
\frac{\exp(r_e)}
{\sum_{j \in \mathcal{E}_{topk}}\exp(r_j)}
$$

Qwen3.6 的 $k=8$ 来自模型卡；vLLM v0.20.1 中 `Qwen3NextSparseMoeBlock` 构造 `FusedMoE` 时传入 `top_k=config.num_experts_per_tok`，而 `FusedMoE` 的文档说明 `top_k` 是每个 token 选择的 expert 数。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 4. Routed Experts

每个 routed expert 可以理解成一个小 MLP：

$$
E_e(h)=W_{2,e}\,\phi(W_{1,e}h)
$$

MoE 输出是 top-k experts 的加权和：

$$
y_{\text{routed}}
=
\sum_{e \in \mathcal{E}_{topk}}
w_e E_e(h)
$$

Qwen3.6 的 expert intermediate dimension 是 512，所以每个 expert 的 MLP 中间维度远小于传统 dense MLP 常见的大 intermediate size；但因为有 256 个 experts，总参数量仍然很大。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 5. Shared Expert

Qwen3.6 每个 token 还会经过 1 个 shared expert。教学上可以写成：

$$
y =
y_{\text{routed}}
+
y_{\text{shared}}
$$

shared expert 的直觉是：所有 token 都能走一条通用专家路径，避免完全依赖 router 的稀疏选择。vLLM v0.20.1 的 `Qwen3NextSparseMoeBlock` 中存在 `shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, ...)`，当 `config.shared_expert_intermediate_size > 0` 时会创建 `self.shared_expert = Qwen3NextMLP(...)`，并把 `shared_experts=self.shared_expert` 传入 `FusedMoE`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 6. 最终 MoE 抽象

合起来：

$$
r = \mathrm{Router}(h)
$$

$$
\mathcal{E}_{top8} = \mathrm{TopK}(r, 8)
$$

$$
y =
\sum_{e \in \mathcal{E}_{top8}}
w_e E_e(h)
+
E_{\text{shared}}(h)
$$

这就是本章要掌握的核心数学直觉。

---

## 三、张量形状

设当前 vLLM step 中 flatten 后的 token 数是 $N$。

### 1. 输入 hidden states

```text
hidden_states: [N, 2048]
```

Qwen3.6 hidden size 是 2048。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. Router logits

```text
router_logits: [N, 256]
```

vLLM 源码中 `Qwen3NextSparseMoeBlock.forward` 注释明确写了 `router_logits: (num_tokens, n_experts)`，并由 `self.gate(hidden_states)` 产生；Qwen3.6 的 `n_experts = 256` 来自模型卡。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Top-k indices / weights

```text
topk_indices: [N, 8]
topk_weights: [N, 8]
```

这里的 8 是 Qwen3.6 的 routed experts per token。vLLM 中 top-k 由 `config.num_experts_per_tok` 传入 `FusedMoE`，`FusedMoE` 文档说明 `top_k` 表示每个 token 选择的专家数量。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 4. Expert 输入重排后的形状

MoE dispatch 后，可以把 token 按专家分组理解成：

```text
expert_0_tokens: [N_0, 2048]
expert_1_tokens: [N_1, 2048]
...
expert_255_tokens: [N_255, 2048]
```

并且：

$$
\sum_{e=0}^{255} N_e = N \times 8
$$

因为每个 token 会被复制或路由到 8 个 routed experts。注意这只是 routed expert 部分，不包括 shared expert。

### 5. Expert 输出合并

每个 expert 输出仍回到 hidden size：

```text
expert_outputs: [N, 8, 2048]
weighted_sum:   [N, 2048]
shared_output:  [N, 2048]
final_output:   [N, 2048]
```

`Qwen3NextSparseMoeBlock.forward` 最后返回 `final_hidden_states.view(orig_shape)`，也就是说 MoE block 的输入输出形状保持一致。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 四、计算流程

Qwen3.6 每层的总体结构是：

```text
Gated DeltaNet 或 Gated Attention
→ MoE
```

在 vLLM v0.20.1 中，`Qwen3_5DecoderLayer` 会根据 `layer_type` 选择 `GatedDeltaNetAttention` 或 `Qwen3NextAttention`，然后根据 `config.model_type == "qwen3_5_moe_text"` 创建 `Qwen3NextSparseMoeBlock` 作为 `self.mlp`。这证明 Qwen3.6 的 “MLP” 位置在 vLLM 中被 sparse MoE block 替代，而不是 dense MLP。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 1. Router 计算

```text
hidden_states: [N, 2048]
→ gate / router linear
→ router_logits: [N, 256]
```

源码路径：

```text
Qwen3NextSparseMoeBlock.forward
→ router_logits, _ = self.gate(hidden_states)
```

vLLM 源码同时处理一种 `self.experts.is_internal_router` 的情况；如果不是 internal router，就在 `Qwen3NextSparseMoeBlock` 中显式调用 `self.gate(hidden_states)`，再把 `router_logits` 传给 `self.experts(...)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 2. Top-8 routed experts

```text
router_logits
→ top-k selection
→ topk_indices / topk_weights
```

这一步在 `FusedMoE` / router runner 内部完成。`FusedMoE` 初始化中会创建 fused MoE router：`create_fused_moe_router(top_k=top_k, global_num_experts=..., renormalize=..., ...)`，并构造 `FusedMoEConfig`，其中 `experts_per_token=top_k`、`num_experts=self.global_num_experts`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 3. Expert dispatch

```text
token hidden states
→ 按 topk_indices 分发到 experts
→ 每个 expert 处理自己的 tokens
```

这一步是 MoE 的工程难点。因为每个 expert 收到的 token 数 $N_e$ 由 router 动态决定，可能非常不均匀。某些 expert 可能很热，某些 expert 可能几乎没有 token。

### 4. Expert GEMM

每个 expert 本质上执行 MLP GEMM：

```text
[N_e, 2048]
→ up/gate projection
→ activation
→ down projection
→ [N_e, 2048]
```

vLLM 的 `FusedMoE` 文档说明该层包含 `MergedColumnParallel` 权重，即 gate/up projection，也包含 `RowParallelLinear` 权重，即 down projection；这对应常见 MoE expert 的 w1/w3 与 w2 权重组织。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 5. Combine

```text
expert_outputs
→ 按 topk_weights 加权
→ scatter 回原 token 顺序
→ 加上 shared expert 输出
→ final_hidden_states
```

vLLM 的 `Qwen3NextSparseMoeBlock.forward` 把 `hidden_states` 和 `router_logits` 传给 `self.experts(...)` 后得到 `final_hidden_states`，如果启用了 sequence parallel，还会做 `tensor_model_parallel_all_gather`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 五、显存布局

MoE 的显存和 dense MLP 不一样。

### 1. Dense MLP 显存

Dense MLP 只有一套权重：

```text
W_gate / W_up / W_down
```

每个 token 都访问同一套权重。

### 2. MoE 权重显存

MoE 有很多套 expert 权重：

```text
expert_0:   W_gate_0, W_up_0, W_down_0
expert_1:   W_gate_1, W_up_1, W_down_1
...
expert_255: W_gate_255, W_up_255, W_down_255
```

Qwen3.6 有 256 routed experts，每个 expert intermediate dimension 是 512。总权重显存会因为 experts 数量变大而变大，但每 token 只激活 8 routed experts + 1 shared expert，所以计算量不是 256 个专家全跑。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Router / Dispatch 临时显存

每一层 MoE 至少会有这些临时数据：

```text
router_logits: [N, 256]
topk_indices:  [N, 8]
topk_weights:  [N, 8]
expert token permutation / routing metadata
expert intermediate buffers
combined output: [N, 2048]
```

这些 buffer 会随 `N = max_num_batched_tokens`、top-k、专家数量和并行策略增长。vLLM `FusedMoEConfig` 中记录了 `max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens`，说明 MoE backend 的最大 token 规模和 scheduler batch token budget 是耦合的。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 4. Shared Expert 显存

shared expert 是额外的一条通用 MLP 路径。vLLM 中 `Qwen3NextSparseMoeBlock` 创建 `shared_expert_gate`，并在 `shared_expert_intermediate_size > 0` 时创建 `Qwen3NextMLP` 作为 shared expert，然后传入 `FusedMoE`。这意味着 shared expert 不是“router 选出来的第 9 个 routed expert”，而是结构上单独存在的共享路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 5. Expert Parallel 下的权重分布

在 expert parallel 下，expert 权重可以分布到不同 GPU 上。vLLM v0.20.1 的 `Qwen3NextSparseMoeBlock` 读取 `get_ep_group()`，设置 `ep_rank`、`ep_size`，并根据 `n_physical_experts // ep_size` 计算每个 EP rank 的 local physical experts 范围；`FusedMoE` 中也有 `ep_size`、`ep_rank`、`local_num_experts`、`expert_map` 等机制。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

---

## 六、vLLM 源码映射

本章核心源码路径如下。

| 概念 | 文件路径 | 类/函数 | 说明 |
|---|---|---|---|
| Qwen3.6 decoder layer | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` | 每层 attention/GDN 后接 MLP/MoE |
| MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | Qwen3.6 的 sparse MoE block |
| Router | `vllm/model_executor/models/qwen3_next.py` | `self.gate = ReplicatedLinear(...)` | hidden → num_experts router logits |
| Shared expert gate | `vllm/model_executor/models/qwen3_next.py` | `self.shared_expert_gate` | hidden → 1 gate |
| Shared expert | `vllm/model_executor/models/qwen3_next.py` | `self.shared_expert = Qwen3NextMLP(...)` | shared expert MLP |
| Fused experts | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | routed experts 的 fused MoE 层 |
| Fused MoE router | `vllm/model_executor/layers/fused_moe/layer.py` | `create_fused_moe_router(...)` | top-k routing / renormalize 等 |
| Expert parallel metadata | `vllm/model_executor/layers/fused_moe/layer.py` | `ep_size`, `ep_rank`, `expert_map` | EP 下 expert 放置与映射 |
| MoE model protocol | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5_MoeMixtureOfExperts` | 收集 MoE layers、expert metadata |

源码中的关键事实是：`Qwen3_5DecoderLayer` 对 `qwen3_5_moe_text` 使用 `Qwen3NextSparseMoeBlock`，而 `Qwen3NextSparseMoeBlock` 内部创建 router gate、shared expert gate、shared expert、`FusedMoE`，并在 forward 中把 `router_logits` 传给 `self.experts(...)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 七、NVIDIA GPU 执行视角

MoE 在 GPU 上的难点不是单个 GEMM，而是 **动态稀疏 + 分发 + 合并**。

### 1. Router 是小 GEMM

```text
hidden_states: [N, 2048]
router weight: [2048, 256]
router_logits: [N, 256]
```

这一步相对规则，适合 GEMM。

### 2. Top-k 是选择与排序问题

```text
router_logits: [N, 256]
→ top8 experts
```

这个步骤包含按行 top-k、权重归一化、生成 expert indices。它不是普通大 GEMM，更像并行 selection / reduction。

### 3. Dispatch 是内存重排问题

每个 token 走 8 个 routed experts，所以需要把 token hidden state 分发到多个 expert 的输入队列：

```text
[N, 2048]
→ dispatch
→ [sum_e N_e, 2048]
```

这里会发生 token copy、indexing、scatter/gather、可能的跨 GPU 通信。MoE 的性能常常卡在这里，而不是卡在“专家 MLP 的数学公式”。

### 4. Expert GEMM 是不规则 batched GEMM

每个 expert 收到的 $N_e$ 不同：

```text
expert 7:   2048 tokens
expert 19:   12 tokens
expert 88:  400 tokens
expert 121:   0 tokens
```

这会造成负载不均衡。vLLM 的 `FusedMoE` 支持 expert parallel load balancer 相关配置，`Qwen3NextSparseMoeBlock` 读取 `parallel_config.enable_eplb`，并设置 logical/physical/redundant experts；`FusedMoE` 文档参数里也包含 `enable_eplb`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 5. Combine 是加权 scatter

专家输出要按原 token 顺序合并：

```text
expert outputs
→ multiply topk_weights
→ sum 8 routed outputs
→ add shared output
→ [N, 2048]
```

这个阶段主要是内存带宽、scatter/gather 和融合效率问题。

### 6. 多卡 EP 会引入 NCCL / All-to-all

如果 expert 分布在不同 GPU 上，某个 token 的 top-8 experts 可能不都在本卡。此时需要把 token hidden state 发送给对应 expert 所在 GPU，专家算完后再把结果送回来。第十章会系统讲 Tensor Parallel、Expert Parallel 和 NCCL 通信；本章只建立直觉：**MoE 的多卡通信不是可选细节，而是高性能 MoE 推理的核心问题。**

---

## 八、优化前 vs 优化后

### 优化前：朴素 MoE

```text
for token in tokens:
    router_logits = router(token)
    top8 = topk(router_logits)
    for expert in top8:
        output += expert(token)
```

问题：

| 问题 | 后果 |
|---|---|
| token-by-token 调用 expert | kernel launch 极多 |
| 每个 expert 单独小 GEMM | Tensor Core 利用率低 |
| 动态路由导致 token 分布不均 | 热专家拖慢整体 |
| scatter/gather 未融合 | HBM 带宽浪费 |
| 多卡 experts 分布不当 | All-to-all 通信重 |

### 优化后：FusedMoE

vLLM 的 `FusedMoE` 层把 gate/up projection 和 down projection 的 expert 权重组织在一个专门的 fused MoE 层里，并创建 fused MoE router 与 MoE runner；配置里包含 experts 数、experts per token、hidden dim、intermediate size、local experts、backend、router logits dtype、max tokens 等。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

优化后的心智是：

```text
1. 批量计算 router
2. 批量 top-k
3. 对 token 按 expert 重排
4. 用 fused / grouped MoE GEMM 处理 experts
5. 合并 routed outputs
6. 融合 shared expert 或单独执行 shared expert
```

### 对 Qwen3.6 的特殊优化点

Qwen3.6 的 MoE 特点是：

```text
256 experts
top-8 routed experts
1 shared expert
expert intermediate dim 512
40 层都接 MoE
```

这意味着 MoE 不是偶尔出现的模块，而是每层主干计算的一部分。优化 Qwen3.6 推理时，不能只盯着 attention backend；MoE dispatch、FusedMoE backend、expert parallel placement 和 batch token budget 同样关键。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 九、工程落地

### 1. 先观察 MoE 是否成为瓶颈

Qwen3.6 的每层都有 MoE，因此压测时要把瓶颈拆开看：

```text
prefill 阶段：
  router + expert GEMM 大量并行
  MoE GEMM 可能成为吞吐核心

decode 阶段：
  每步 token 少
  expert dispatch 小批量、不均匀更明显
```

如果 decode batch 很小，MoE expert GEMM 可能变成很多小而碎的任务，GPU 利用率下降。continuous batching 可以缓解这个问题。

### 2. `max_num_batched_tokens` 会影响 MoE

`FusedMoEConfig` 里记录 `max_num_tokens=vllm_config.scheduler_config.max_num_batched_tokens`。这意味着 scheduler 的 batch token budget 不只是影响 attention 和 prefill，也会影响 MoE runner / backend 对最大 token 数的规划。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

工程上：

```text
小 max_num_batched_tokens:
  decode 延迟可能更稳
  MoE GEMM 可能更碎

大 max_num_batched_tokens:
  MoE GEMM 更容易做大
  prefill 吞吐更好
  但 TTFT / ITL 可能变化
```

### 3. Expert Parallel 不等于自动更快

Expert parallel 能把 experts 分布到多 GPU，降低单卡权重压力，增加可服务的专家容量。但如果 token 路由跨 GPU 很多，all-to-all 通信可能变成瓶颈。vLLM `Qwen3NextSparseMoeBlock` 和 `FusedMoE` 都有 EP rank、EP size、local experts、expert map 等元数据，说明 vLLM v0.20.1 的 MoE 路径已经显式支持 expert placement / EP 管理。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 4. Shared Expert 的部署含义

Shared expert 是每个 token 都会走的通用路径。它带来更稳定的通用表达能力，但也意味着即使 routed experts 很稀疏，shared expert 仍然是 dense-like 的额外计算。vLLM 源码中 shared expert 被显式创建并传给 `FusedMoE`，不是推理时可以随便忽略的分支。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 5. 不要把 3B activated 简化成“只跑 3B 参数”

“3B activated”是模型结构层面的概括，工程上实际延迟还受到：

```text
router top-k
expert dispatch
expert load balance
shared expert
kernel backend
FP8/BF16 dtype
TP/EP 通信
batch token 分布
```

这些因素共同影响。尤其是 decode 小 batch 时，激活参数少不一定代表 GPU 利用率高。

---

## 十、本章总结

Qwen3.6 的 MoE 路径可以压缩成：

```text
hidden_states
→ router logits [N, 256]
→ top-8 routed experts
→ expert dispatch
→ expert GEMM
→ weighted combine
→ + shared expert
→ output [N, 2048]
```

核心区别是：

```text
Dense MLP:
  每个 token 走同一套 MLP

MoE:
  每个 token 从 256 个 experts 里选 8 个 routed experts
  再走 1 个 shared expert
```

vLLM v0.20.1 源码中，Qwen3.6 的 MoE block 对应 `Qwen3NextSparseMoeBlock`，router 是 `ReplicatedLinear(hidden_size, num_experts)`，routed experts 由 `FusedMoE` 承载，shared expert 由 `shared_expert_gate` 和 `Qwen3NextMLP` 路径承载。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 小白总结

MoE 就是“不是所有 token 都找同一个老师”。

Qwen3.6 有 256 个专家。每个 token 会先让 router 判断该找哪些专家，然后只找 8 个 routed experts，再额外找 1 个 shared expert。这样模型可以有很多总参数，但每个 token 只激活一部分参数。

---

## 工程师总结

Qwen3.6 的 MoE 工程关键点是：

```text
router:
  [N, 2048] → [N, 256]

top-k:
  每 token 选 8 routed experts

dispatch:
  token 按 expert 重排，形成 expert-local batches

expert GEMM:
  每个 expert 做 gate/up/down MLP

shared expert:
  每 token 都走的额外共享路径

combine:
  routed outputs 按 top-k weights 加权合并，再与 shared expert 输出融合
```

vLLM v0.20.1 中，`Qwen3_5DecoderLayer` 在 `qwen3_5_moe_text` 下创建 `Qwen3NextSparseMoeBlock`；该 block 创建 `self.gate`、`self.shared_expert_gate`、`self.shared_expert` 和 `self.experts = FusedMoE(...)`；forward 时生成 `router_logits` 并调用 `self.experts(hidden_states=hidden_states, router_logits=router_logits)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)) ([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 关键公式

**1. Router logits**

$$
r = h W_{\text{router}}
$$

$$
r \in \mathbb{R}^{256}
$$

**2. Top-k expert selection**

$$
\mathcal{E}_{top8}
=
\mathrm{TopK}(r, 8)
$$

**3. Top-k 权重归一化**

$$
w_e =
\frac{\exp(r_e)}
{\sum_{j \in \mathcal{E}_{top8}}\exp(r_j)}
$$

**4. Routed experts 输出**

$$
y_{\text{routed}}
=
\sum_{e \in \mathcal{E}_{top8}}
w_e E_e(h)
$$

**5. Shared expert 合并**

$$
y =
y_{\text{routed}}
+
E_{\text{shared}}(h)
$$

**6. Expert token 数关系**

$$
\sum_{e=0}^{255}N_e = N \times 8
$$

这里 $N_e$ 是 routed expert $e$ 收到的 token 数，$N$ 是本轮 MoE 输入 token 数。

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 decoder layer | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Qwen3.6 sparse MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| Router gate | `vllm/model_executor/models/qwen3_next.py` | `self.gate = ReplicatedLinear(...)` |
| Shared expert gate | `vllm/model_executor/models/qwen3_next.py` | `self.shared_expert_gate` |
| Shared expert MLP | `vllm/model_executor/models/qwen3_next.py` | `self.shared_expert = Qwen3NextMLP(...)` |
| Fused routed experts | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` |
| Fused MoE router | `vllm/model_executor/layers/fused_moe/layer.py` | `create_fused_moe_router(...)` |
| MoE config | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoEConfig` |
| Expert map / EP metadata | `vllm/model_executor/layers/fused_moe/layer.py` | `ensure_round_robin_expert_routing_tables`, `update_expert_map` |
| Qwen MoE metadata collection | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5_MoeMixtureOfExperts.set_moe_parameters` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 MoE 有 256 experts，每 token 激活 8 routed + 1 shared expert，expert intermediate dim 为 512 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是每层 attention/GDN 后接 MoE | Hugging Face model card | Hidden Layout | 官方文档确认 | 是 |
| `Qwen3_5DecoderLayer` 在 `qwen3_5_moe_text` 下使用 `Qwen3NextSparseMoeBlock` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 读取 `config.num_experts` 作为 routed experts 数 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.__init__` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 使用 `get_ep_group()` 设置 EP rank、EP size | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.__init__` | 源码直接确认 | 是 |
| Router 是 `ReplicatedLinear(config.hidden_size, config.num_experts)` | `vllm/model_executor/models/qwen3_next.py` | `self.gate` | 源码直接确认 | 是 |
| Shared expert gate 是 `ReplicatedLinear(config.hidden_size, 1)` | `vllm/model_executor/models/qwen3_next.py` | `self.shared_expert_gate` | 源码直接确认 | 是 |
| 当 `shared_expert_intermediate_size > 0` 时创建 `Qwen3NextMLP` 作为 shared expert | `vllm/model_executor/models/qwen3_next.py` | `self.shared_expert` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 构造 `FusedMoE` 并传入 `num_experts`、`top_k`、`hidden_size`、`intermediate_size`、`renormalize` 等 | `vllm/model_executor/models/qwen3_next.py` | `self.experts = FusedMoE(...)` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock.forward` 中 router logits 形状为 `(num_tokens, n_experts)` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.forward` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock.forward` 调用 `self.experts(hidden_states=hidden_states, router_logits=router_logits)` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.forward` | 源码直接确认 | 是 |
| `FusedMoE` 文档说明它包含 gate/up projection 和 down projection 权重 | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | 源码直接确认 | 是 |
| `FusedMoE` 的 `top_k` 表示每 token 选择的 expert 数 | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE.__init__` docstring | 源码直接确认 | 是 |
| `FusedMoE` 创建 fused MoE router，并把 `experts_per_token=top_k` 写入 `FusedMoEConfig` | `vllm/model_executor/layers/fused_moe/layer.py` | `create_fused_moe_router`, `FusedMoEConfig` | 源码直接确认 | 是 |
| `FusedMoE` 支持 EP metadata，例如 `ep_size`、`ep_rank`、local experts、expert map | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE`, expert routing table helpers | 源码直接确认 | 是 |
| Qwen3.6 在特定 NVIDIA GPU 上最终选择哪一种 MoE backend / kernel | `vllm/model_executor/layers/fused_moe/` | backend dispatch 相关 | 待源码确认 | 否 |
| Qwen3.6 shared expert 与 routed expert 是否在当前配置下完全融合到同一个 kernel | `vllm/model_executor/layers/fused_moe/` | quant method / runner | 待源码确认 | 否 |
| Expert parallel 下具体使用 all-to-all、DeepEP、NCCL 还是其他通信 backend | EP / distributed / fused_moe runner 源码 | 待展开 | 待源码确认 | 否 |

---

## 源码阅读作业

按这个顺序读。

**1. 先读 Qwen3.6 layer 如何接 MoE**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5DecoderLayer
config.model_type == "qwen3_5_moe_text"
self.mlp = Qwen3NextSparseMoeBlock(...)
```

目标：确认 Qwen3.6 的 MLP 位置不是 dense MLP，而是 sparse MoE block。

**2. 读 Qwen3NextSparseMoeBlock**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextSparseMoeBlock
self.n_routed_experts
self.gate
self.shared_expert_gate
self.shared_expert
self.experts = FusedMoE(...)
forward
router_logits
```

目标：理解 router、shared expert、FusedMoE 的连接方式。

**3. 读 FusedMoE**

```text
vllm/model_executor/layers/fused_moe/layer.py
```

重点找：

```text
class FusedMoE
top_k
renormalize
create_fused_moe_router
FusedMoEConfig
local_num_experts
expert_map
```

目标：理解 vLLM 把 MoE 作为一个专用 fused layer，而不是 Python for-loop。

**4. 读 expert parallel 相关**

```text
vllm/model_executor/layers/fused_moe/layer.py
```

重点找：

```text
ep_size
ep_rank
expert_placement_strategy
ensure_round_robin_expert_routing_tables
update_expert_map
```

目标：为第十章 Tensor Parallel / Expert Parallel / NCCL 通信做准备。

---

## 实验验证作业

### 实验 1：打印 router logits 和 top-k 分布

在 `Qwen3NextSparseMoeBlock.forward` 附近临时加 debug，记录：

```text
router_logits.shape
topk_indices.shape
topk_weights.shape
每个 expert 收到的 token 数
```

目标：验证：

```text
router_logits: [N, 256]
topk_indices:  [N, 8]
expert token counts 分布不均
```

### 实验 2：观察 batch size 对 MoE GEMM 的影响

固定模型，分别压测：

```text
并发 1
并发 8
并发 32
并发 128
```

记录：

```text
tokens/s
TPOT
GPU utilization
MoE kernel time
expert token imbalance
```

预期：并发太小时，expert GEMM 可能很碎；并发增大后，专家 batch 更大，Tensor Core 利用率更好，但通信和调度也更重。

### 实验 3：对比 MTP off / on 时 MoE 压力

分别启动：

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
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

记录：

```text
tokens/s
TPOT
accepted speculative tokens
MoE kernel time
KV/state cache usage
```

目标：观察 speculative tokens 是否让每步 MoE token batch 增大，以及吞吐是否真的改善。

### 实验 4：Expert 热点分析

统计一段真实 workload 中：

```text
每层 expert hit count
top-10 hottest experts
bottom-10 coldest experts
每个 step expert token count variance
```

目标：判断是否存在严重 expert imbalance，为第十章 Expert Parallel 和 EPLB 做准备。

---

## 3 个自测题

**1. Qwen3.6 的 MoE 中，router 输出的形状是什么？**
如果当前 batch 有 $N$ 个 token，router logits 是 `[N, 256]`，因为 Qwen3.6 有 256 个 routed experts。

**2. 为什么 MoE 不等于“每个 token 跑 256 个 MLP”？**
因为每个 token 只选择 top-8 routed experts，再额外经过 1 个 shared expert；没有把 256 个 routed experts 全部计算一遍。

**3. 为什么 MoE 的 GPU 优化难点不只是 GEMM？**
因为还要做 router top-k、token dispatch、expert 分组、负载均衡、expert output combine，以及多卡 expert parallel 下的跨 GPU 通信。

---

## 下一章预告

第十章进入：

```text
Tensor Parallel、Expert Parallel、NCCL 通信
```

下一章会把本章的 MoE 从单卡视角扩展到多卡：

```text
1. Tensor Parallel 如何切 QKV、LM head、expert weights？
2. Expert Parallel 如何把 256 experts 分布到多 GPU？
3. 为什么 top-k routing 会引入 all-to-all 通信？
4. TP 和 EP 同时开时，通信顺序和瓶颈在哪里？
5. Qwen3.6 在 vLLM v0.20.1 中如何通过 EP metadata、expert map、FusedMoEConfig 进入多卡执行路径？
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
