# 第十章：Tensor Parallel、Expert Parallel、NCCL 通信

本章从“单 GPU 上怎么跑一层”扩展到“多 GPU 上怎么把一层拆开跑”。

在 Qwen/Qwen3.6-35B-A3B 上，多卡并行不能只理解成“把模型平均切到 8 张卡”。它同时涉及：

```text
Tensor Parallel:
  把同一层里的矩阵乘权重切到多张 GPU 上。

Expert Parallel:
  把 MoE 的 experts 分布到多张 GPU 上。

NCCL / collective communication:
  在 GPU 之间做 all-reduce、all-gather、reduce-scatter、all-to-all 等通信。

Qwen3.6 特殊点:
  Gated Attention + Gated DeltaNet + MoE + MTP + 262K context
```

Qwen3.6-35B-A3B 的模型卡确认它是 35B total / 3B activated，hidden size 2048，40 层，MoE 为 256 experts，每 token 激活 8 routed experts + 1 shared expert；vLLM recipe 给出的 Qwen3.6 基础 serving 命令使用 `--tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

Tensor Parallel 像把一道大菜切给多个厨师一起做：

```text
同一个矩阵乘太大
→ 把权重矩阵切成几块
→ 每张 GPU 负责其中一块
→ 最后把结果拼起来或加起来
```

Expert Parallel 像把 256 位专家分布到不同办公室：

```text
GPU 0: experts 0-31
GPU 1: experts 32-63
...
GPU 7: experts 224-255
```

某个 token 被 router 选中 experts：

```text
token A → experts [3, 19, 88, 102, 140, 155, 200, 241]
```

那么这个 token 的 hidden state 可能要被送到多个 GPU，因为这些 experts 不一定在本卡。算完后，expert 输出还要被送回来合并。

所以多卡 MoE 推理最核心的区别是：

```text
Tensor Parallel:
  同一个 token 的同一层计算被切开，需要同步 partial results。

Expert Parallel:
  不同 experts 放在不同 GPU，tokens 要被路由到专家所在 GPU。
```

vLLM 的 Expert Parallel 部署文档也把 EP 描述为 MoE 模型的专用并行方式：expert layers 会被 sharded across EP ranks；attention layers 则根据 TP size 决定复制或 TP 切分。文档还提醒 EP 是实验性功能，参数名和默认值未来可能变化。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

---

## 二、数学直觉

### 1. Tensor Parallel：列切和行切

一个线性层：

```math
Y = XW
```

假设：

```math
X \in \mathbb{R}^{N \times H}
```

```math
W \in \mathbb{R}^{H \times O}
```

#### Column Parallel

把 $W$ 的输出维度切开：

```math
W = [W_1, W_2, \ldots, W_p]
```

每张 GPU 算：

```math
Y_i = XW_i
```

最后：

```math
Y = [Y_1, Y_2, \ldots, Y_p]
```

这通常需要 **all-gather**，除非后续计算可以直接消费 shard 后的输出。vLLM v0.20.1 的 `ColumnParallelLinear` 源码注释明确写道，矩阵 $A$ 沿第二维并行化为 `[A_1, ..., A_p]`；如果 `gather_output=True`，forward 会对 partition 输出做 `tensor_model_parallel_all_gather`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/linear.py))

#### Row Parallel

把 $W$ 的输入维度切开，同时输入 $X$ 也按 hidden 维度切开：

```math
X = [X_1, X_2, \ldots, X_p]
```

```math
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_p
\end{bmatrix}
```

每张 GPU 算 partial output：

```math
Y_i = X_iW_i
```

最终：

```math
Y = \sum_{i=1}^{p}Y_i
```

这通常需要 **all-reduce**。vLLM v0.20.1 的 `RowParallelLinear` 源码注释写明 $A$ 沿第一维并行化、$X$ 沿第二维切分；如果 `reduce_results=True` 且 TP size 大于 1，forward 会调用 `tensor_model_parallel_all_reduce`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/linear.py))

### 2. Qwen3.6 Gated Attention 中的 TP

对 Qwen3.6 的 Gated Attention，模型卡给出：

```text
Q heads  = 16
KV heads = 2
head dim = 256
```

vLLM v0.20.1 的 `Qwen3NextAttention` 会读取 tensor parallel world size，并计算：

```text
num_heads = total_num_heads // tp_size
num_kv_heads = max(1, total_num_kv_heads // tp_size)
```

源码还明确处理了一个关键情况：如果 KV heads 数量小于 TP size，则 KV heads 会在多个 TP GPU 上复制，而不是继续切分。Qwen3.6 的 KV heads 只有 2；当 recipe 使用 TP=8 时，就会进入“KV heads 少于 TP size”的复制逻辑。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. Expert Parallel：按 expert 切，而不是按矩阵切

MoE 的输出可以抽象为：

```math
y =
\sum_{e \in \mathrm{TopK}(r(h), 8)}
w_eE_e(h)
+
E_{\text{shared}}(h)
```

Tensor Parallel 会把同一个 expert 的矩阵切到多 GPU；Expert Parallel 则会把不同 experts 放到不同 GPU。

如果有 256 experts，EP size = 8，最简单的线性放置是：

```math
\text{experts per rank} = 256 / 8 = 32
```

vLLM v0.20.1 的 `determine_expert_map` 源码说明，它会根据 `ep_size`、`ep_rank`、`global_num_experts` 计算本 rank 的 local experts，并生成从 global expert id 到 local expert index 的 `expert_map`；当 `ep_size=1` 时返回所有 experts 本地，否则按 linear 或 round-robin 策略分配。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

### 4. 通信数学直觉

| 通信 | 数学含义 | 常见位置 |
|---|---|---|
| All-reduce | 每张卡有 partial result，最后求和并让每张卡拿到完整结果 | RowParallelLinear output |
| All-gather | 每张卡有一片结果，最后拼成完整结果 | ColumnParallelLinear output / logits gather |
| Reduce-scatter | 先求和，再把结果切分给不同 ranks | 某些优化 TP / sequence parallel 路径 |
| All-to-all | 每张卡把不同 token 发给不同目标卡 | Expert Parallel token dispatch |

vLLM v0.20.1 的 `GroupCoordinator` 封装了 `all_reduce`、`all_gather`、`reduce_scatter`、broadcast、send/recv 等通信操作；它内部持有 CPU group 和 device group，device communication backend 在 CUDA-like 平台上通常对应 GPU collective 通信。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/distributed/parallel_state.py))

---

## 三、张量形状

### 1. TP 下的 Attention 形状

Qwen3.6 Gated Attention 的全局逻辑形状：

```text
hidden_states: [N, 2048]

Q global: [N, 16, 256]
K global: [N,  2, 256]
V global: [N,  2, 256]
```

如果 TP=8：

```text
Q heads per rank = 16 / 8 = 2
KV heads global = 2 < TP size
```

所以每 rank 的 Q 是：

```text
Q local: [N, 2, 256]
```

KV heads 则会复制到多个 ranks，源码中明确写了“KV heads 少于 TP size 时复制 KV heads across tensor parallel GPUs”。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

这带来一个工程结论：

```text
Q heads 可以平均切；
KV heads 太少时不能继续均切，只能复制。
```

### 2. TP 下的 RowParallel output

以 output projection 为例，Qwen3NextAttention 中 `o_proj` 是 `RowParallelLinear(total_num_heads * head_dim, hidden_size)`。其输入来自各 TP rank 的 attention output shard，输出要回到 hidden size 2048。`RowParallelLinear` 的源码说明它把 input dimension 切分；如果 `reduce_results=True`，会通过 all-reduce 聚合 partial output。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

逻辑上：

```text
rank 0: [N, local_q_heads * head_dim] → partial [N, 2048]
rank 1: [N, local_q_heads * head_dim] → partial [N, 2048]
...
all-reduce sum
→ [N, 2048]
```

### 3. EP 下的 MoE 形状

Qwen3.6 MoE：

```text
hidden_states:  [N, 2048]
router_logits:  [N, 256]
topk_indices:   [N, 8]
topk_weights:   [N, 8]
```

如果 EP=8，简单均分时每 rank 大约 32 experts。每个 token 的 top-8 experts 可能分布在多个 ranks，所以 dispatch 后每个 rank 收到：

```text
rank r input:
  [tokens routed to local experts, 2048]
```

每个 local expert 再执行自己的 MLP，然后输出被 combine 回原 token 顺序。vLLM v0.20.1 的 `Qwen3NextSparseMoeBlock` 源码中会读取 `get_ep_group()`，记录 `ep_rank`、`ep_size`，计算 `n_local_physical_experts`、`physical_expert_start`、`physical_expert_end`，然后构造 `FusedMoE`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 4. TP + EP 同时存在时

如果 TP 和 EP 同时打开，不能简单说“一张卡负责 1/8 模型”。你要分别问：

```text
Attention / Gated Attention:
  QKV、O projection 怎么按 TP 切？

MoE experts:
  experts 是按 TP 切矩阵，还是按 EP 切专家？

Shared expert:
  是 replicated、TP shard，还是融合进 FusedMoE 路径？

Gated DeltaNet:
  state shape 是否随 TP 切分？

LM head:
  vocab 或 hidden 维度如何切分，logits 是否需要 gather？
```

vLLM 最新官方优化文档对并行策略做了一个高层总结：TP 会把模型参数跨 GPU shard；EP 是 MoE 模型的专用并行方式，启用 `enable_expert_parallel=True` 后会对 MoE layers 使用 expert parallelism 而不是 tensor parallelism，并使用与 tensor parallelism 相同的并行度；数据并行则复制完整模型处理不同请求 batch。这里是官方文档层面的部署说明，具体到 v0.20.1 的实现仍要以 tag 源码为准。([vLLM](https://docs.vllm.ai/en/latest/configuration/optimization/))

---

## 四、计算流程

### 1. Gated Attention 的 TP 流程

Qwen3.6 的 full-attention 层在 vLLM v0.20.1 中由 `Qwen3NextAttention` 承载。它创建：

```text
qkv_proj = QKVParallelLinear(...)
o_proj   = RowParallelLinear(...)
```

forward 中：

```text
hidden_states
→ qkv_proj
→ split q / gate / k / v
→ q_norm, k_norm
→ RoPE
→ Attention(q, k, v)
→ gate sigmoid
→ attn_output *= gate
→ o_proj
```

源码确认 `Qwen3NextAttention` 使用 `QKVParallelLinear` 生成 Q/K/V/Gate，使用 `RowParallelLinear` 做 output projection，并将 `cache_config` 传入 `Attention`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 2. MoE 的 EP 流程

MoE 的多卡流程可以抽象为：

```text
hidden_states
→ router logits
→ top-8 experts
→ token dispatch to expert ranks
→ local expert GEMM
→ expert output combine
→ return hidden_states
```

vLLM v0.20.1 的 `Qwen3NextSparseMoeBlock` 中，router 是 `ReplicatedLinear(hidden_size, num_experts)`；shared expert gate 是 `ReplicatedLinear(hidden_size, 1)`；routed experts 由 `FusedMoE` 承载，构造时传入 `num_experts`、`top_k`、`hidden_size`、`intermediate_size`、`enable_eplb`、`num_redundant_experts` 等参数。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. FusedMoE 的 expert placement

vLLM v0.20.1 的 `FusedMoE` 源码说明该层包含 gate/up projection 和 down projection 的专家权重；它的参数包括 `num_experts`、`top_k`、`hidden_size`、`intermediate_size`、`enable_eplb`、`router_logits_dtype`、`routed_scaling_factor` 等。`determine_expert_map` 负责根据 EP size 和 rank 计算 local experts 与 expert map。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

这说明 vLLM 的 MoE 不是 Python for-loop：

```text
for expert in experts:
  expert(tokens)
```

而是一个专用的 fused MoE 层，里面包含 router、expert placement、local expert map、MoE runner / backend 等机制。

### 4. 通信路径

多卡执行时，通信大致分四类：

```text
Attention TP:
  QKV projection 后可能保留 shard
  O projection 后通常 all-reduce

LM head / logits:
  vocab parallel 或 logits gather

MoE EP:
  token dispatch 到 expert rank
  expert output 回传
  可能使用 all-to-all / allgather-reducescatter / DeepEP 等 backend

Pipeline / DP:
  层间传 hidden states 或不同副本处理不同 requests
```

vLLM 的 EP 部署文档列出了 EP all-to-all backend：`allgather_reducescatter`、`deepep_high_throughput`、`deepep_low_latency`、`flashinfer_nvlink_one_sided`、`flashinfer_nvlink_two_sided`，并分别说明适合 prefill、多节点 decode、NVLink 系统等场景。该文档是部署层面的说明；对 v0.20.1 的具体 backend 选择仍需结合源码和运行日志确认。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

---

## 五、显存布局

### 1. TP 对权重显存的影响

Tensor Parallel 的主要收益之一是降低单卡权重显存：

```text
ColumnParallelLinear:
  每张卡保存部分 output columns

RowParallelLinear:
  每张卡保存部分 input rows
```

所以权重近似按 TP size 切薄。但注意，某些小模块、norm、router、部分 KV heads、shared structures 可能复制，不会严格按 TP size 等比例下降。vLLM 的 `ReplicatedLinear` 就是一个显式复制的 linear 类型；Qwen3NextSparseMoeBlock 的 router gate 和 shared expert gate 都是 `ReplicatedLinear`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/linear.py))

### 2. KV Cache 不一定按 TP size 等比例下降

对 Qwen3.6 的 Gated Attention：

```text
KV heads = 2
TP = 8
```

由于 KV heads 少于 TP size，vLLM 源码会复制 KV heads across tensor parallel GPUs。这意味着不能简单说：

```text
KV cache per GPU = total KV cache / 8
```

至少对 KV heads 很少的 GQA/MQA 模型，要看源码里的复制逻辑和实际 backend layout。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. EP 对 expert 权重显存的影响

Qwen3.6 有 256 routed experts。如果 EP=8 且线性均分，则每 rank 大约保存 32 个 experts 的权重。vLLM `determine_expert_map` 源码说明 experts 会尽可能均匀分布到 EP ranks；如果不能整除，剩余 experts 分配给靠前 ranks 或按策略处理。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

这降低了单卡 expert 权重压力，但会增加 token dispatch 通信压力。

### 4. EPLB 的显存代价

Expert Parallel Load Balancer 通过冗余 experts 让热门 experts 更容易本地命中或均衡负载，但冗余 experts 会占额外显存。vLLM EP 部署文档说明 EPLB 可通过 `--enable-eplb` 开启，并会收集 load statistics、周期性 rebalance expert distribution；文档还明确说冗余 experts 会增加 GPU memory footprint，在 KV cache 空间紧张时可能不适合。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

### 5. Qwen3.6 的显存账本

对 Qwen3.6，单卡显存应拆成：

```text
TP-sharded dense / attention / GDN weights
+
EP-sharded expert weights
+
replicated router / norm / small weights
+
Gated Attention KV cache
+
Gated DeltaNet state
+
MoE dispatch buffers
+
NCCL communication buffers
+
MTP lookahead cache
+
Vision Encoder / multimodal buffers
```

这就是为什么“35B FP8 能不能单卡放下”不是完整问题；你还要问目标上下文、并发、MTP、多模态和 EP/TP 通信如何配置。

---

## 六、vLLM 源码映射

| 概念 | 文件路径 | 类/函数 | 说明 |
|---|---|---|---|
| Tensor parallel group | `vllm/distributed/parallel_state.py` | `GroupCoordinator`, all-reduce/all-gather/reduce-scatter | TP/EP/PP 等 group 通信封装 |
| Column TP | `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear` | 权重按输出维切分，可选 all-gather |
| Row TP | `vllm/model_executor/layers/linear.py` | `RowParallelLinear` | 权重按输入维切分，可选 all-reduce |
| Qwen3.6 QKV TP | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention`, `QKVParallelLinear` | QKV/Gate projection |
| Qwen3.6 O projection TP | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention`, `RowParallelLinear` | attention output projection |
| Qwen3.6 MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | router、shared expert、FusedMoE |
| Expert group | `vllm/model_executor/models/qwen3_next.py` | `get_ep_group()` | 获取 EP group、rank、size |
| Fused MoE | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | routed expert fused layer |
| Expert map | `vllm/model_executor/layers/fused_moe/layer.py` | `determine_expert_map` | global expert → local expert 映射 |
| EPLB | `vllm/model_executor/layers/fused_moe/layer.py` | `enable_eplb`, redundant experts | expert load balancing 相关 |
| Qwen MoE metadata | `vllm/model_executor/models/qwen3_next.py` | `QwenNextMixtureOfExperts.set_moe_parameters` | 收集 MoE layer 和 expert metadata |

两个关键源码点：

第一，`Qwen3NextAttention` 明确把 Q heads 按 TP 切分，同时在 KV heads 少于 TP size 时复制 KV heads，这对 Qwen3.6 的 TP=8 非常关键。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

第二，`Qwen3NextSparseMoeBlock` 通过 `get_ep_group()` 获取 EP group，并把 routed experts 交给 `FusedMoE`；`FusedMoE` 通过 expert map 决定当前 rank 持有哪些 local experts。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 七、NVIDIA GPU 执行视角

### 1. TP 的 GPU 代价

Tensor Parallel 把大 GEMM 切小，但引入 collective：

```text
Column parallel:
  GEMM shard 快
  需要时 all-gather

Row parallel:
  每卡算 partial output
  需要 all-reduce 求和
```

在单机 NVLink / NVSwitch 上，这通常可接受；跨节点时，如果网络慢，TP 通信可能明显拖慢。vLLM 分布式 serving 文档也提示，要让 tensor parallel 高性能，需要高效节点间通信，例如 InfiniBand。([vLLM](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html?utm_source=chatgpt.com))

### 2. EP 的 GPU 代价

Expert Parallel 的计算集中在 local experts，但 token dispatch 需要跨 GPU：

```text
router top-k
→ 计算目标 expert rank
→ token hidden state 发给对应 rank
→ local expert GEMM
→ expert output 发回原 rank
→ weighted combine
```

如果 router 分布非常不均，某些 GPU 上的 experts 会成为热点。vLLM EP 文档说明 EPLB 会收集每次 forward 的 load statistics 并周期性 rebalance expert distribution。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

### 3. NCCL 的现实瓶颈

NCCL 通信不是“一个函数调用”这么简单，它受到：

```text
GPU 拓扑:
  PCIe / NVLink / NVSwitch / IB

消息大小:
  小消息 latency-bound
  大消息 bandwidth-bound

通信频率:
  每层一次、每 token 一次、每 MoE block 一次

overlap:
  通信是否能和计算重叠

collective 类型:
  all-reduce / all-gather / reduce-scatter / all-to-all
```

vLLM 的 `GroupCoordinator` 把 device communicator 封装在 group 中，并提供 all-reduce、all-gather、reduce-scatter、broadcast、send/recv 等接口；这些操作在 world size 为 1 时会绕过，在多 GPU 时走 device communicator。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/distributed/parallel_state.py))

### 4. 对 Qwen3.6 的 GPU 重点

Qwen3.6 多卡性能不是只看 attention：

```text
Gated Attention:
  TP Q heads + KV heads replication + KV cache read

Gated DeltaNet:
  TP 影响 state shape，Mamba/GDN state 还要看 mamba_cache_mode

MoE:
  256 experts + top-8 routing + shared expert + dispatch

MTP:
  speculative tokens 会放大每 step 的 token 数和 cache/state 预算

262K context:
  长上下文让 KV/state 和通信 buffer 更敏感
```

源码中 `Qwen3NextForCausalLM.get_mamba_state_shape_from_config` 会把 `parallel_config.tensor_parallel_size`、linear key/value heads、head dim、conv kernel dim、speculative token 数传入 Gated DeltaNet state shape calculator，说明 TP 和 speculative config 也会影响 GDN state shape。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 八、优化前 vs 优化后

### 优化前：只开 TP

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

这是 vLLM recipe 给出的 Qwen3.6 基础命令。它的好处是简单，所有层都按 TP 视角处理；但对 MoE 来说，只用 TP 不一定是最优，因为 experts 本身天然适合按 expert 维度分布。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 优化后：按模块选择并行

更完整的思路：

```text
Attention / Gated Attention:
  TP 切 Q/O projection，注意 KV heads 复制

MoE:
  如果启用 EP，把 experts 分布到 EP ranks

多副本吞吐:
  DP 复制 attention/dense 路径，处理不同请求 batch

跨节点:
  避免过重 TP 跨节点通信，必要时使用 DP/EP/PP 组合
```

vLLM 官方 optimization 文档建议：TP 适合模型太大无法单卡放下、需要降低单卡内存压力；EP 专用于 MoE 模型，启用后对 MoE layers 使用 expert parallelism；DP 则复制整个模型处理不同 request batches。([vLLM](https://docs.vllm.ai/en/latest/configuration/optimization/))

### 对 Qwen3.6 的优化心智

错误心智：

```text
8 张 GPU → 所有东西都除以 8 → 性能提升 8 倍
```

正确心智：

```text
权重:
  有些 TP 切，有些 EP 切，有些 replicated

KV cache:
  KV heads 少时可能复制

MoE:
  expert 权重可切，但 token dispatch 会通信

GDN state:
  随 TP 和 speculative tokens 改变 state shape

性能:
  受 GEMM、KV bandwidth、all-reduce、all-to-all、expert imbalance 共同决定
```

---

## 九、工程落地

### 1. 基础 TP 配置

按 vLLM Qwen3.6 recipe：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

这是最适合作为第一条“可验证路径”的命令。先不要同时加入 EP、MTP、多模态和极端 batch 参数。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 2. EP 配置要谨慎验证

vLLM EP 部署文档给出的通用方式是使用 `--enable-expert-parallel`，并说明 EP size 由 `TP_SIZE × DP_SIZE` 自动计算；MoE expert layers 会分布到 EP ranks，attention layers 则取决于 TP size。文档同时明确标注 EP 是实验性功能，所以在固定 vLLM v0.20.1 分析时，CLI、backend 和行为必须用实际版本日志与源码核对。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

对 Qwen3.6，你可以把 EP 验证分成两步：

```text
步骤 1:
  TP=8 基线，确认可启动、可服务、吞吐稳定。

步骤 2:
  开启 EP，观察 expert placement、all2all backend、tokens/s、TPOT、OOM、EPLB 日志。
```

### 3. 选择 all-to-all backend

vLLM EP 文档列出多个 all-to-all backend：默认 `allgather_reducescatter` 适合通用场景；`deepep_high_throughput` 偏 prefill-dominated workload；`deepep_low_latency` 偏 decode-dominated workload；FlashInfer NVLink backend 面向特定 NVLink 多节点系统。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

工程上不要只看单请求：

```text
prefill-heavy:
  长 prompt，短输出

decode-heavy:
  短 prompt，长输出

mixed:
  真实 chat/RAG/agent workload
```

不同 workload 的最佳 EP backend 可能不同。

### 4. 监控指标

建议同时看：

```text
GPU:
  SM utilization
  HBM bandwidth
  Tensor Core utilization
  NVLink / PCIe / IB bandwidth

vLLM:
  tokens/s
  TTFT
  TPOT / ITL
  kv_cache_usage_perc
  num_requests_running
  num_requests_waiting

MoE:
  expert hit count
  per-rank expert tokens
  expert imbalance
  all-to-all time
  EPLB rebalance events
```

### 5. 常见排障

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| TP=8 后性能不升反降 | all-reduce / all-gather 太重，batch 太小 | NCCL 拓扑、batch size、RowParallel all-reduce 时间 |
| KV cache 没按 8 倍下降 | Qwen3.6 KV heads=2，小于 TP=8，KV heads 复制 | `Qwen3NextAttention` KV head replication 逻辑 |
| EP 后吞吐下降 | all-to-all 通信大于 expert GEMM 收益 | all2all backend、expert token distribution |
| 某些 GPU 很忙 | expert 热点 / load imbalance | expert hit count、EPLB |
| MTP 开启后显存紧张 | speculative tokens 增加 KV/GDN state budget | `num_speculative_tokens`、cache usage |
| 多模态 OOM | Vision Encoder / multimodal buffers 额外占用 | text-only 对照、`--language-model-only` |

---

## 十、本章总结

本章核心是三句话：

```text
Tensor Parallel:
  把同一层的矩阵切到多 GPU，靠 all-gather / all-reduce 合并。

Expert Parallel:
  把 MoE experts 分到多 GPU，靠 token dispatch / all-to-all 把 token 送到专家所在 GPU。

Qwen3.6:
  Attention、GDN、MoE、MTP、KV/state cache 各自对并行方式有不同约束，不能用“8 卡就除以 8”一把尺估算。
```

vLLM v0.20.1 源码证据最关键的三个点是：

```text
1. ColumnParallelLinear / RowParallelLinear 明确实现 TP 权重切分与 collective。
2. Qwen3NextAttention 明确处理 TP head 切分与 KV head 复制。
3. Qwen3NextSparseMoeBlock + FusedMoE 明确接入 EP group、local experts、expert map。
```

---

## 小白总结

多 GPU 推理不是简单“多几张卡就快几倍”。

Tensor Parallel 是“大家一起做同一个大矩阵乘”；Expert Parallel 是“不同专家住在不同 GPU，token 要被送到对应专家那里”；NCCL 通信就是 GPU 之间传结果、拼结果、求和的高速物流系统。

Qwen3.6 有 256 个 experts，每个 token 只找 8 个 routed experts + 1 个 shared expert，所以 MoE 很适合 expert parallel，但也会带来 token dispatch 和 all-to-all 通信。

---

## 工程师总结

Qwen3.6 多卡推理的工程分解：

```text
Gated Attention:
  QKVParallelLinear
  RowParallelLinear
  KV heads=2 under TP=8 → KV heads replicated

MoE:
  router replicated linear
  FusedMoE
  EP group
  expert_map
  local_num_experts
  possible EPLB

Communication:
  all-gather for column-parallel outputs when needed
  all-reduce for row-parallel outputs
  all-to-all / allgather-reducescatter for EP token dispatch
  reduce-scatter in optimized TP/SP paths

GDN:
  state shape depends on TP size and speculative tokens
```

不要只调 `--tensor-parallel-size`。真正的性能来自：合适的 TP/EP/DP 组合、合理的 `max_num_batched_tokens`、长上下文 cache 预算、MoE expert balance，以及 GPU 拓扑和 NCCL backend。

---

## 关键公式

**1. Column Parallel**

```math
W = [W_1, W_2, \ldots, W_p]
```

```math
Y_i = XW_i
```

```math
Y = [Y_1, Y_2, \ldots, Y_p]
```

**2. Row Parallel**

```math
X = [X_1, X_2, \ldots, X_p]
```

```math
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_p
\end{bmatrix}
```

```math
Y = \sum_{i=1}^{p}X_iW_i
```

**3. Expert Parallel 均分**

```math
\text{experts per rank}
=
\frac{\text{num experts}}{\text{EP size}}
```

Qwen3.6 简单均分例子：

```math
256 / 8 = 32
```

**4. MoE routed output**

```math
y_{\text{routed}}
=
\sum_{e \in \mathrm{TopK}(r(h), 8)}
w_eE_e(h)
```

**5. Shared expert 合并**

```math
y =
y_{\text{routed}} + E_{\text{shared}}(h)
```

**6. 通信量直觉**

```math
\text{EP dispatch traffic}
\propto
N
\times
k
\times
H
\times
\text{bytes}
```

其中 $N$ 是 token 数，$k=8$，$H=2048$。

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| 分布式通信封装 | `vllm/distributed/parallel_state.py` | `GroupCoordinator` |
| All-reduce | `vllm/distributed/parallel_state.py` | `GroupCoordinator.all_reduce` |
| All-gather | `vllm/distributed/parallel_state.py` | `GroupCoordinator.all_gather` |
| Reduce-scatter | `vllm/distributed/parallel_state.py` | `GroupCoordinator.reduce_scatter` |
| Column TP | `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear` |
| Row TP | `vllm/model_executor/layers/linear.py` | `RowParallelLinear` |
| Qwen3.6 Attention TP | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| QKV projection | `vllm/model_executor/layers/linear.py` | `QKVParallelLinear` |
| Qwen3.6 MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| Fused MoE | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` |
| Expert placement | `vllm/model_executor/layers/fused_moe/layer.py` | `determine_expert_map` |
| MoE metadata | `vllm/model_executor/models/qwen3_next.py` | `QwenNextMixtureOfExperts.set_moe_parameters` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 recipe 使用 `--tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3` | vLLM Qwen3.5/Qwen3.6 recipe | serving command | recipe 确认 | 是 |
| Qwen3.6 有 256 experts，每 token 激活 8 routed + 1 shared expert | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM 的 distributed state 封装了 process group、device group 和 collective communication | `vllm/distributed/parallel_state.py` | `GroupCoordinator` | 源码直接确认 | 是 |
| `GroupCoordinator` 提供 `all_reduce`、`all_gather`、`reduce_scatter` 等操作 | `vllm/distributed/parallel_state.py` | `all_reduce`, `all_gather`, `reduce_scatter` | 源码直接确认 | 是 |
| `ColumnParallelLinear` 沿输出维切权重，必要时 all-gather 输出 | `vllm/model_executor/layers/linear.py` | `ColumnParallelLinear` | 源码直接确认 | 是 |
| `RowParallelLinear` 沿输入维切权重，必要时 all-reduce partial output | `vllm/model_executor/layers/linear.py` | `RowParallelLinear` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 根据 TP size 切分 Q heads | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 在 KV heads 少于 TP size 时复制 KV heads | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 使用 `QKVParallelLinear` 和 `RowParallelLinear` | `vllm/model_executor/models/qwen3_next.py` | `qkv_proj`, `o_proj` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 使用 `get_ep_group()` 获取 EP group / rank / size | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.__init__` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 的 router 是 `ReplicatedLinear(hidden_size, num_experts)` | `vllm/model_executor/models/qwen3_next.py` | `self.gate` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 构造 `FusedMoE` 并传入 `num_experts`、`top_k`、`hidden_size`、`intermediate_size`、`enable_eplb` 等 | `vllm/model_executor/models/qwen3_next.py` | `self.experts = FusedMoE(...)` | 源码直接确认 | 是 |
| `FusedMoE` 包含 gate/up projection 和 down projection 专家权重 | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` docstring | 源码直接确认 | 是 |
| `determine_expert_map` 根据 EP size / rank / global experts 生成 local experts 与 expert map | `vllm/model_executor/layers/fused_moe/layer.py` | `determine_expert_map` | 源码直接确认 | 是 |
| vLLM 官方 EP 部署文档说明 EP layers 会在 EP ranks 上分片，attention layers 根据 TP size 复制或切分 | vLLM Expert Parallel Deployment docs | Layer Behavior with EP Enabled | 官方文档确认 | 是 |
| vLLM 官方 EP 文档列出 all-to-all backend，包括 allgather/reducescatter、DeepEP、FlashInfer NVLink backend | vLLM Expert Parallel Deployment docs | Backend Selection Guide | 官方文档确认 | 是 |
| vLLM 官方 EP 文档明确 EP 是实验性功能，参数名和默认值可能变化 | vLLM Expert Parallel Deployment docs | Warning | 官方文档确认 | 是 |
| Qwen3.6 在 v0.20.1 下开启 EP 后最终选择的 all-to-all backend | `vllm/model_executor/layers/fused_moe/runner`, runtime config | 待展开 | 待源码确认 | 否 |
| Qwen3.6 shared expert 在 EP 下是否与 routed experts 完全融合到同一个 kernel | `fused_moe` backend / quant method | 待展开 | 待源码确认 | 否 |
| 具体 NVIDIA 拓扑下 NCCL all-reduce / all-to-all 是否成为瓶颈 | 运行日志 / profiler | 实验验证 | 否 |

---

## 源码阅读作业

按这个顺序读。

**1. 读 TP linear**

```text
vllm/model_executor/layers/linear.py
```

重点找：

```text
ColumnParallelLinear
RowParallelLinear
gather_output
reduce_results
tensor_model_parallel_all_gather
tensor_model_parallel_all_reduce
```

目标：确认列切、行切分别对应 all-gather 和 all-reduce。

**2. 读 Qwen3NextAttention**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextAttention
total_num_heads
total_num_kv_heads
tp_size
num_heads
num_kv_heads
QKVParallelLinear
RowParallelLinear
```

目标：理解 Qwen3.6 的 Q heads 怎么切，KV heads 为什么在 TP=8 下会复制。

**3. 读 EP group 和 MoE block**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextSparseMoeBlock
get_ep_group
ep_rank
ep_size
n_local_physical_experts
physical_expert_start
physical_expert_end
FusedMoE
```

目标：理解 Qwen3.6 MoE 怎么接入 Expert Parallel metadata。

**4. 读 FusedMoE expert map**

```text
vllm/model_executor/layers/fused_moe/layer.py
```

重点找：

```text
determine_expert_map
FusedMoE
local_num_experts
expert_map
enable_eplb
num_redundant_experts
```

目标：理解 global experts 如何映射到本 rank 的 local experts。

**5. 读 distributed group**

```text
vllm/distributed/parallel_state.py
```

重点找：

```text
GroupCoordinator
all_reduce
all_gather
reduce_scatter
device_group
device_communicator
```

目标：理解 vLLM 如何封装 NCCL / device collective。

---

## 实验验证作业

### 实验 1：TP=1 / 2 / 4 / 8 对比

在同一 workload 下测试：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only
```

再分别改成：

```text
--tensor-parallel-size 2
--tensor-parallel-size 4
--tensor-parallel-size 8
```

记录：

```text
启动显存
KV/cache capacity
TTFT
TPOT
tokens/s
NCCL all-reduce 时间
GPU utilization
```

目标：确认 TP 增大并不总是线性加速。

### 实验 2：验证 KV heads 复制影响

在 `Qwen3NextAttention.__init__` 打印：

```text
tp_size
total_num_heads
num_heads
total_num_kv_heads
num_kv_heads
```

在 TP=8 下确认：

```text
total_num_kv_heads = 2
num_kv_heads = 1
KV heads 被复制到多个 TP ranks
```

目标：不要把 KV cache 错误估算成严格除以 8。

### 实验 3：EP 开关对 MoE 性能影响

对比：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

和 EP 配置：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-expert-parallel
```

记录：

```text
tokens/s
TPOT
all-to-all 时间
expert token imbalance
per-rank GPU utilization
显存变化
```

目标：判断 EP 是否真的改善你当前硬件和 workload。

### 实验 4：EPLB 热点验证

如果启用 EPLB，记录：

```text
每层 expert hit count
每 rank expert tokens
rebalance events
redundant experts 数量
显存变化
```

目标：验证“专家热点”是否是瓶颈，以及冗余 experts 是否值得。

### 实验 5：NCCL 拓扑检查

运行：

```bash
nvidia-smi topo -m
```

并记录：

```text
GPU-GPU: NVLink / NVSwitch / PCIe
GPU-NIC: 是否靠近 InfiniBand HCA
跨 socket 情况
```

目标：解释为什么同样 TP/EP 配置在不同机器上性能差异很大。

---

## 3 个自测题

**1. ColumnParallelLinear 和 RowParallelLinear 的通信差异是什么？**
Column parallel 沿输出维切权重，必要时 all-gather 拼输出；Row parallel 沿输入维切权重，每张卡产生 partial output，必要时 all-reduce 求和。

**2. 为什么 Qwen3.6 在 TP=8 时不能简单认为 KV cache 除以 8？**
因为 Qwen3.6 Gated Attention 的 KV heads 只有 2；vLLM 源码在 KV heads 少于 TP size 时会复制 KV heads across tensor parallel GPUs。

**3. Expert Parallel 为什么会引入 all-to-all？**
因为 token 的 top-8 experts 可能分布在不同 GPU 上，hidden state 必须被 dispatch 到 expert 所在 rank，expert 输出也要回传合并。

---

## 下一章预告

第十一章进入：

```text
NVIDIA GPU Kernel：
HBM、SM、Tensor Core、CUDA Kernel、MoE GEMM、Attention Backend
```

下一章会从硬件执行角度回答：

```text
1. HBM、L2、SM、Tensor Core 分别在 LLM 推理中做什么？
2. 为什么 prefill 和 decode 的 GPU 瓶颈不同？
3. Attention backend 为什么决定长上下文性能？
4. MoE GEMM 为什么容易碎片化？
5. Qwen3.6 的 Gated Attention、Gated DeltaNet、MoE、MTP 分别会触发哪些 kernel 类型？
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Expert Parallel Deployment - vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/linear.py)
