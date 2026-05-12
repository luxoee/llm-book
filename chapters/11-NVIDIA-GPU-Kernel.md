# 第十一章：NVIDIA GPU Kernel

## 本章解决什么问题

- 把前面学过的 Attention、Gated DeltaNet、MoE、MTP、Scheduler 放到 NVIDIA GPU 执行视角下重新理解。
- 解释 HBM、SM、Tensor Core、CUDA kernel、MoE GEMM 和 attention backend 分别影响哪类瓶颈。
- 帮你判断 prefill / decode / MoE / GDN 慢时应该看 compute、memory bandwidth、occupancy 还是通信。
- 为第十二章按 vLLM 源码路径追踪 backend 和 scheduler 铺垫硬件语言。

先给结论：

```text
LLM 推理性能不是“模型参数量”单独决定的，
而是由 GPU 上每一步 kernel 如何读写 HBM、占用 SM、使用 Tensor Core、
以及通信/调度是否能隐藏延迟共同决定。
```

对固定案例 **Qwen/Qwen3.6-35B-A3B**，这一点尤其重要。它不是普通 dense Transformer，而是 35B total / 3B activated 的 multimodal MoE 模型，40 层结构为 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，包含 30 个 Gated DeltaNet 层、10 个 Gated Attention 层、每层 MoE、256 experts、8 routed experts + 1 shared expert、262K 原生上下文。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

把 GPU 想成一个超级工厂：

| GPU 部件 | 类比 | LLM 推理里的角色 |
|---|---|---|
| HBM | 巨大的仓库 | 存权重、KV cache、GDN state、临时 buffer |
| L2 Cache | 工厂内部中转站 | 缓存热点数据，减少反复去 HBM |
| SM | 生产车间 | 执行 CUDA blocks / warps |
| CUDA Core | 普通工人 | 做通用算术、elementwise、索引、控制 |
| Tensor Core | 专用矩阵乘机器 | 做 GEMM / matmul / MoE expert GEMM |
| CUDA Kernel | 一张生产工单 | 一次 GPU 上的并行任务 |
| NCCL | 跨车间物流 | 多 GPU 间 all-reduce / all-gather / all-to-all |

NVIDIA 官方 CUDA 编程资料说明，NVIDIA GPU 架构围绕一组可扩展的多线程 **SM** 构建；CUDA kernel grid 的线程块会被分发到可用 SM 上执行，一个 SM 可以并发执行多个线程块。资料还说明，SM 以 32 个线程组成的 **warp** 为单位创建、管理、调度和执行线程。([NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/))

在 LLM 推理里，“快不快”常常取决于这张工单到底是哪种类型：

```text
大 GEMM:
  很适合 Tensor Core，prefill 阶段多见

长上下文 attention decode:
  不一定算力瓶颈，常常是 HBM 读 KV cache 瓶颈

MoE dispatch:
  不只是 GEMM，还要 router、top-k、重排、跨卡发送

Sampling:
  不是 Tensor Core 主战场，而是 logits 处理、mask、top-k/top-p、softmax
```

---

## 二、数学直觉

### 1. Roofline 直觉：算力瓶颈 vs 带宽瓶颈

一个 kernel 的性能可以粗略看两个量：

```math
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}
{\text{Bytes moved from/to memory}}
```

如果一个 kernel 每读 1 byte 能做很多 FLOPs，它更可能是 **compute-bound**；如果读很多数据但每个数据只算很少，它更可能是 **memory-bound**。

### 2. Prefill 为什么更容易 compute-bound

Prefill 一次处理很多 prompt tokens。以线性层为例：

```math
Y = XW
```

其中：

```text
X: [N, H]
W: [H, O]
Y: [N, O]
```

当 $N$ 很大时，GEMM 很大，Tensor Core 更容易吃满。NVIDIA 性能文档说明，Tensor Cores 是为加速矩阵乘加操作而引入的硬件单元，用于机器学习和科学计算中的矩阵乘累加；NVIDIA 的矩阵乘性能文档也强调，Tensor Cores 用于最大化 tensor multiply / matrix multiply 的速度。([docs.nvidia.com](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)) ([docs.nvidia.com](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html))

### 3. Decode 为什么更容易 memory-bound

Decode 每步每个请求通常只新增 1 个 token。GEMM 的 $N$ 变小，但 attention 要读历史 KV cache：

```math
o_t =
\mathrm{softmax}
\left(
\frac{q_t K_{\le t}^{\top}}{\sqrt{d}}
\right)
V_{\le t}
```

当上下文是 262K，当前 query 很小，历史 K/V 很大。于是 decode 的瓶颈经常变成：

```text
从 HBM 读取历史 K/V
而不是 Tensor Core 算不够快
```

对 Qwen3.6，这个判断还要拆开：只有 10 个 Gated Attention 层走普通 K/V attention；另外 30 个 Gated DeltaNet 层走 GDN / Mamba-style state 路径。模型卡给出了 40 层 layout，vLLM v0.20.1 源码中 `Qwen3NextAttention` 负责 full-attention 层，`GatedDeltaNetAttention` 负责 GDN 路径。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 4. MoE 为什么不是普通 GEMM

MoE expert 的数学上像很多 MLP：

```math
E_e(h)=W_{2,e}\phi(W_{1,e}h)
```

但执行上不是一个大 dense GEMM，而是：

```math
\text{router}
\rightarrow
\text{top-k}
\rightarrow
\text{dispatch}
\rightarrow
\text{expert GEMM}
\rightarrow
\text{combine}
```

Qwen3.6 每 token 选 8 个 routed experts，再走 1 个 shared expert。也就是说，本轮如果有 $N$ 个 tokens，routed expert 的 token-expert 对数量是：

```math
N \times 8
```

这会产生动态重排和不规则 expert GEMM，而不仅仅是一个 `[N, H] @ [H, O]`。vLLM v0.20.1 的 `Qwen3NextSparseMoeBlock` 中 router 是 `ReplicatedLinear(hidden_size, num_experts)`，然后把 `router_logits` 交给 `FusedMoE`；`FusedMoE` 目录下包含 router、experts、runner、prepare/finalize、all2all utils、flashinfer/cutlass MoE 等子路径，说明 MoE 是专门的执行子系统。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/model_executor/layers/fused_moe))

---

## 三、张量形状

### 1. Qwen3.6 的主要 kernel 输入形状

固定案例的关键尺寸：

```text
hidden_size = 2048
layers = 40
layout = 30 Gated DeltaNet + 10 Gated Attention
Gated Attention: Q heads 16, KV heads 2, head dim 256
Gated DeltaNet: V heads 32, QK heads 16, head dim 128
MoE: 256 experts, top-8 routed + 1 shared, expert intermediate dim 512
vocab padded = 248320
```

这些结构参数来自 Qwen 模型卡。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. Dense / projection GEMM

常见投影：

```text
hidden_states: [N, 2048]
QKV projection: [2048, qkv_output_dim]
O projection:   [local_attn_dim, 2048]
LM head:        [2048, 248320]
```

其中 $N$ 是本轮被 scheduler 合并后的 token 数。prefill 时 $N$ 大，decode 时 $N$ 小。

### 3. Gated Attention kernel 形状

逻辑形状：

```text
Q: [N, 16, 256]
K: [N,  2, 256]
V: [N,  2, 256]
```

在 TP=8 时，Q heads 会被切分；vLLM 源码中 `Qwen3NextAttention` 根据 TP size 计算本 rank 的 `num_heads`，并在 KV heads 少于 TP size 时复制 KV heads。Qwen3.6 的 KV heads 是 2，因此 TP=8 时不能简单认为 KV cache 按 8 等分。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 4. Gated DeltaNet state / kernel 形状

Gated DeltaNet 不是普通 attention 的 QKᵀ-softmax-V 路径。vLLM v0.20.1 的 `GatedDeltaNetAttention` 继承 `MambaBase`，`mamba_type` 返回 `"gdn_attention"`；它的 state dtype/shape 由 `MambaStateDtypeCalculator.gated_delta_net_state_dtype` 和 `MambaStateShapeCalculator.gated_delta_net_state_shape` 计算；底层计算调用 `fla_chunk_gated_delta_rule`，输入包括 `q, k, v, g, beta, initial_state, output_final_state, cu_seqlens, chunk_indices, chunk_offsets` 等。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

### 5. MoE token-expert 形状

```text
hidden_states:  [N, 2048]
router_logits:  [N, 256]
topk_indices:   [N, 8]
topk_weights:   [N, 8]
expert inputs:  [sum_e N_e, 2048]
expert outputs: [sum_e N_e, 2048]
final output:   [N, 2048]
```

`Qwen3NextSparseMoeBlock.forward` 的注释明确 router logits 形状为 `(num_tokens, n_experts)`，并调用 `self.experts(hidden_states=hidden_states, router_logits=router_logits)`；这说明 top-k、dispatch、expert GEMM、combine 的细节被交给 `FusedMoE` 层处理。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 四、计算流程

### 1. 一个 decode step 的 kernel 级流程

以一个 token step 为例，Qwen3.6 大致经历：

```text
embedding lookup
→ 30 个 Gated DeltaNet 层中的 state update / GDN kernel
→ 10 个 Gated Attention 层中的 QKV projection / RoPE / attention backend / KV cache read-write
→ 40 个 MoE block 中的 router / top-k / expert dispatch / expert GEMM / combine
→ final norm
→ LM head GEMM
→ logits processor / sampler
```

这里的“30 + 10 + 40”来自模型结构：每 4 层里 3 个 GDN 层、1 个 Gated Attention 层，重复 10 次，每层后接 MoE。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. Gated Attention backend

vLLM v0.20.1 的 `vllm/v1/attention/backends/` 目录包含多个 backend 文件，如 `flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`gdn_attn.py`、`linear_attn.py`、`mamba*_attn.py`、`turboquant_attn.py` 等。这个目录结构说明 vLLM v0.20.1 的 attention 执行不是一个固定 kernel，而是按模型、平台、配置选择 backend。([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends))

FlashAttention backend 中的 `do_kv_cache_update` 会把 key/value cache 拆成 `key_cache, value_cache`，并使用 `slot_mapping` 进行 cache 写入；FlashInfer backend 声明支持 dtype 为 FP16/BF16，KV cache dtype 支持 auto、float16、bfloat16、fp8、fp8_e4m3、fp8_e5m2，并返回支持的 kernel block sizes `[16, 32, 64]`。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/attention/backends/flash_attn.py)) ([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/attention/backends/flashinfer.py))

### 3. Gated DeltaNet backend

GDN 路径不是 `softmax(QKᵀ)V`。`GatedDeltaNetAttention` 的源码显示其底层调用 `fla_chunk_gated_delta_rule`，并携带 `chunk_indices`、`chunk_offsets`、`initial_state` 和 `output_final_state` 等参数；这说明它更像 state/chunk 化 kernel 路径，而不是普通 paged KV attention kernel。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

### 4. MoE kernel 流程

MoE 的 kernel 流程不是单个 GEMM：

```text
router linear
→ top-k selection
→ prepare / dispatch
→ grouped / batched expert GEMM
→ finalize / combine
```

vLLM v0.20.1 的 `fused_moe` 目录包含 `router`、`experts`、`runner`、`prepare_finalize`、`all2all_utils.py`、`flashinfer_cutlass_moe.py`、`fused_batched_moe.py`、`fused_marlin_moe.py` 等文件/目录，说明 MoE 路径在 vLLM 中有专门的 routing、expert、runner 和通信/准备收尾模块。([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/model_executor/layers/fused_moe))

---

## 五、显存布局

### 1. GPU 显存层级

工程上最常见的层级可以这样记：

```text
HBM:
  最大、最慢；存权重、KV cache、GDN state、workspace

L2:
  片上缓存；缓存热点权重/激活/cache line

Shared Memory:
  block 内线程共享；常用于 tiled GEMM / attention tile

Registers:
  每个线程私有；最快但容量小

Tensor Core fragment / accumulator:
  矩阵乘微块计算
```

NVIDIA CUDA 资料说明，SM 会同时执行大量线程，使用 SIMT 架构；每个 SM 有寄存器和并行数据缓存/共享内存，kernel 的 occupancy 会受到寄存器和共享内存使用量影响。([NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/))

### 2. Qwen3.6 显存大头

Qwen3.6 推理显存至少包括：

```text
1. 权重:
   GDN / Gated Attention / MoE experts / LM head / Vision Encoder

2. Gated Attention KV cache:
   10 层 full-attention K/V

3. Gated DeltaNet state:
   30 层 GDN state / conv state / chunk state

4. MoE runtime:
   router logits、top-k、dispatch map、expert buffers

5. logits:
   [N, 248320]，尤其 logprobs 时昂贵

6. MTP:
   speculative tokens / lookahead cache

7. NCCL / backend workspace:
   all-reduce / all-to-all / FlashAttention / FlashInfer / MoE runner buffer
```

模型结构、vocab 和 MoE 尺寸来自 Qwen 模型卡；vLLM recipe 的 Qwen3.6 命令使用 TP=8 与 262144 max model len，MTP 示例使用 `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([docs.vllm.ai](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 3. Prefill 和 decode 的显存行为不同

Prefill：

```text
大 batch token
大 GEMM
大量写 KV/GDN state
临时 activation / workspace 峰值高
```

Decode：

```text
小 batch token
反复读历史 KV/GDN state
LM head 和 sampling 每步都执行
KV/state 随输出增长
```

### 4. MoE 显存的特殊性

MoE 不仅有 expert 权重，还有 routing metadata：

```text
router_logits: [N, 256]
topk_indices:  [N, 8]
topk_weights:  [N, 8]
expert permutation / inverse permutation
expert workspace
possibly all-to-all buffers
```

这解释了为什么 `max_num_batched_tokens` 会影响 MoE kernel 的显存和性能：$N$ 越大，router、dispatch、expert GEMM 的临时 buffer 越大，但 expert GEMM 也更容易吃满 Tensor Core。

---

## 六、vLLM 源码映射

| 概念 | 文件路径 | 类/函数 | 作用 |
|---|---|---|---|
| Qwen3.6 full attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | QKV/Gate projection、RoPE、Attention、O projection |
| Qwen3.6 GDN | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | GDN state、chunked delta rule |
| Qwen3.6 MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | Router、shared expert、FusedMoE |
| Attention backends | `vllm/v1/attention/backends/` | `flash_attn.py`, `flashinfer.py`, `triton_attn.py`, `gdn_attn.py` 等 | Attention/GDN 后端 |
| FlashAttention KV update | `vllm/v1/attention/backends/flash_attn.py` | `do_kv_cache_update` | 使用 `slot_mapping` 写 K/V cache |
| FlashInfer backend | `vllm/v1/attention/backends/flashinfer.py` | `FlashInferBackend` | 支持 FP16/BF16 与 FP8 KV cache dtype |
| Fused MoE 子系统 | `vllm/model_executor/layers/fused_moe/` | `router`, `experts`, `runner`, `prepare_finalize`, `all2all_utils.py` 等 | MoE routing / dispatch / expert runner |
| TP/EP 通信 | `vllm/distributed/parallel_state.py` | `GroupCoordinator` | all-reduce / all-gather / reduce-scatter 等 |

源码证据的主线是：`Qwen3NextAttention` 明确实现 full-attention 分支；`GatedDeltaNetAttention` 明确是 MambaBase / GDN state 路径；`Qwen3NextSparseMoeBlock` 明确创建 router、shared expert 和 `FusedMoE`；vLLM v1 attention backend 目录明确存在 FlashAttention、FlashInfer、Triton、GDN、linear/mamba 等多种 backend。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py)) ([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends))

---

## 七、NVIDIA GPU 执行视角

### 1. HBM 视角

HBM 负责喂数据。decode 长上下文时，很多 kernel 的核心问题不是“能不能算”，而是：

```text
能不能足够快地把历史 KV / GDN state / expert weights 喂给 SM
```

如果一个 kernel 每读很多 bytes 只做少量计算，它会被 HBM bandwidth 限制。

### 2. SM / Warp 视角

CUDA kernel 会被拆成 thread blocks，blocks 分发到 SM 上；SM 内部再以 warp 为单位调度执行。warp 是 32 个线程组成的执行单元；当 warp 内线程走不同分支时，会降低效率。([NVIDIA CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/))

这对 LLM kernel 的含义是：

```text
规则 GEMM:
  warp 行为规则，容易高效

top-k / dispatch / scatter:
  分支和不规则内存访问更多

attention:
  需要精心分块，避免显式 materialize T×T

sampling:
  mask / penalty / top-p 可能有较多不规则逻辑
```

### 3. Tensor Core 视角

Tensor Core 适合矩阵乘累加。NVIDIA 文档说明 Tensor Cores 用于加速 matrix multiply-and-accumulate 操作；对于 LLM，最能使用 Tensor Core 的通常是 QKV projection、O projection、MoE expert GEMM、LM head 等 GEMM。([docs.nvidia.com](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html))

但 Tensor Core 不是万能的：

```text
KV cache 读取:
  主要是 memory movement

top-k:
  selection / reduction

dispatch:
  gather / scatter / all-to-all

RoPE / RMSNorm / activation:
  elementwise 或 reduction

sampling:
  logits 后处理
```

### 4. Kernel launch / fusion 视角

很多小 kernel 会有 launch overhead 和中间 tensor HBM 往返。高性能 LLM kernel 常做融合：

```text
RMSNorm + QKV
QKV + RoPE + cache write
Attention softmax + value aggregation
MoE prepare + expert GEMM + finalize
Sampling mask + penalty + top-k/top-p
```

vLLM 的代码结构也体现了这种方向：attention backend、FlashAttention/FlashInfer、FusedMoE、fused_moe prepare/finalize、router/runner 子系统都是为了把多个逻辑步骤变成更高效的 backend 执行。([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/model_executor/layers/fused_moe)) ([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends))

---

## 八、优化前 vs 优化后

### 优化前：朴素 kernel 串

```text
RMSNorm kernel
Q projection GEMM
K projection GEMM
V projection GEMM
RoPE kernel
cache write kernel
QK matmul
softmax kernel
prob @ V matmul
gate sigmoid kernel
mul kernel
O projection GEMM
router GEMM
top-k kernel
dispatch kernel
expert_0 GEMM
expert_1 GEMM
...
combine kernel
```

问题：

| 问题 | 后果 |
|---|---|
| kernel 太碎 | launch overhead 高 |
| 中间 tensor 反复落 HBM | 带宽浪费 |
| attention 显式 scores | 长上下文显存爆炸 |
| expert GEMM 太小 | Tensor Core 利用率低 |
| dispatch 不规则 | HBM / L2 / all-to-all 成瓶颈 |
| decode batch 太小 | SM 占用低 |

### 优化后：backend 化 / fused 化

```text
QKVParallelLinear
→ backend RoPE / cache update
→ FlashAttention / FlashInfer / Triton backend
→ FusedMoE router / dispatch / expert runner
→ optimized sampler
```

具体到 vLLM v0.20.1：

```text
Attention:
  vllm/v1/attention/backends/

MoE:
  vllm/model_executor/layers/fused_moe/

GDN:
  gdn_linear_attn.py + gdn_attn.py

Scheduler:
  continuous batching + chunked prefill 让 N 更适合 GPU
```

### 对 Qwen3.6 的优化重点

错误优化思路：

```text
只优化 FlashAttention 就够了
```

正确思路：

```text
Gated Attention:
  关注 KV cache layout、attention backend、RoPE、decode bandwidth

Gated DeltaNet:
  关注 GDN state、chunked rule、mamba cache mode、state update kernel

MoE:
  关注 router top-k、dispatch、expert GEMM、EP all-to-all、load balance

LM head / sampling:
  关注 248320 vocab logits、logprobs、top-k/top-p

MTP:
  关注 speculative tokens 带来的额外 compute/cache 与接受率
```

---

## 九、工程落地

### 1. 基础部署路径

vLLM recipe 给出的 Qwen3.6 基础命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

MTP speculative decoding 命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

这两个命令来自 vLLM Qwen3.5/Qwen3.6 recipe。([docs.vllm.ai](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 2. 性能分析工具

建议从粗到细：

```text
nvidia-smi / DCGM:
  GPU utilization、HBM、power、NVLink

vLLM metrics:
  TTFT、TPOT、tokens/s、kv_cache_usage、running/waiting

Nsight Systems:
  kernel timeline、CPU/GPU overlap、NCCL overlap

Nsight Compute:
  单 kernel 的 SM occupancy、HBM throughput、Tensor Core utilization、warp stalls

PyTorch profiler:
  高层 op 和 kernel 对应关系
```

### 3. 判断瓶颈的经验表

| 现象 | 可能瓶颈 | 检查方向 |
|---|---|---|
| Prefill 慢，GPU compute 高 | GEMM / attention compute | Tensor Core utilization、GEMM shape |
| Decode 慢，HBM 高 | KV/GDN state 读取 | HBM bandwidth、KV cache dtype、context length |
| MoE kernel 时间高 | expert dispatch / grouped GEMM | expert token counts、MoE runner、all-to-all |
| GPU utilization 低 | batch 太小 / kernel 太碎 | continuous batching、max_num_batched_tokens |
| NCCL 时间高 | TP/EP 通信 | all-reduce / all-to-all timeline |
| sampling 时间高 | vocab 大 / logprobs | logprobs、top-k/top-p、vocab size |
| MTP 开后吞吐降 | speculative cache / 接受率 | accepted tokens、KV usage、batch size |

### 4. 调参顺序

建议顺序：

```text
1. 先固定 workload:
   prompt len、output len、concurrency、text-only/multimodal

2. 先关复杂项:
   MTP off、prefix cache off/on 分开测、多模态分开测

3. 先看宏观:
   tokens/s、TTFT、TPOT、GPU/HBM/NVLink

4. 再看 kernel:
   attention backend、MoE runner、LM head、sampling

5. 最后调并行:
   TP、EP、EPLB、all-to-all backend
```

### 5. 不要混淆“backend 存在”和“实际被选中”

`vllm/v1/attention/backends/` 里有 FlashAttention、FlashInfer、Triton、GDN 等 backend 文件，但某次运行到底选哪个 backend，要看当前 GPU、dtype、KV cache dtype、模型结构、配置、环境变量和 vLLM 运行日志。本章只确认源码入口存在，不在没有运行日志的情况下断言你机器上一定使用某个 backend。([github.com](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends))

---

## 十、本章总结

本章把 LLM 推理从“模型结构”落到了“GPU kernel”。

核心结论：

```text
Prefill:
  更像大 GEMM + attention compute，Tensor Core 更容易吃满。

Decode:
  更像反复读取 KV/GDN state，长上下文下 HBM bandwidth 关键。

Gated Attention:
  走 attention backend，关注 KV cache、slot_mapping、FlashAttention/FlashInfer/Triton。

Gated DeltaNet:
  走 GDN / Mamba-style state path，关注 chunk、initial_state、output_final_state。

MoE:
  不是普通 MLP，而是 router + top-k + dispatch + expert GEMM + combine。

MTP:
  可能降低 TPOT，但会增加 speculative token compute/cache，收益取决于接受率和并发。
```

对 Qwen3.6，性能优化不能只盯着一个模块。它是：

```text
30 层 Gated DeltaNet
+ 10 层 Gated Attention
+ 40 层 MoE
+ 262K context
+ 248320 padded vocab
+ optional Vision Encoder
+ optional MTP
```

---

## 小白总结

GPU 不是“一个很快的计算器”，而是很多小车间组成的工厂。

有些任务像大矩阵乘，Tensor Core 很擅长；有些任务像查很长的历史笔记，主要卡在从显存搬数据；MoE 则像把 token 分发给不同专家，难点在分发、合并和负载均衡。

Qwen3.6 的推理快不快，不只看模型参数量，也要看每一步 GPU kernel 是否吃满 Tensor Core、是否被 HBM 卡住、是否被 MoE dispatch 或多卡通信拖慢。

---

## 工程师总结

Qwen3.6 的 kernel 级瓶颈地图：

```text
Gated Attention:
  QKV GEMM → RoPE → KV cache update → attention backend → O GEMM
  重点: HBM, KV cache layout, backend selection

Gated DeltaNet:
  projection / conv / state update / fla_chunk_gated_delta_rule
  重点: state shape, chunk metadata, mamba cache mode

MoE:
  router GEMM → top-k → dispatch → expert GEMM → combine
  重点: grouped GEMM, expert imbalance, all-to-all

LM head:
  [N, 2048] × [2048, 248320]
  重点: vocab parallel, logits gather, logprobs

Sampling:
  logits mask / penalty / top-k/top-p / random
  重点: 大 vocab 下避免不必要 logprobs
```

vLLM v0.20.1 源码入口：`Qwen3NextAttention`、`GatedDeltaNetAttention`、`Qwen3NextSparseMoeBlock`、`vllm/v1/attention/backends/`、`vllm/model_executor/layers/fused_moe/`。这些入口分别对应 full attention、GDN state、MoE 和 backend/kernel 子系统。([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)) ([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

---

## 关键公式

**1. 算术强度**

```math
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}
{\text{Bytes moved}}
```

**2. GEMM FLOPs 粗估**

```math
\text{FLOPs}
\approx
2M N K
```

对：

```math
C_{M \times N} = A_{M \times K} B_{K \times N}
```

**3. Attention decode 读取量直觉**

```math
\text{KV read bytes}
\propto
T_{\text{context}}
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
```

**4. Qwen3.6 Gated Attention KV 每 token 粗估**

```math
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
```

**5. MoE routed token-expert 对数量**

```math
N_{\text{token-expert}}
=
N_{\text{tokens}}
\times
8
```

**6. MoE dispatch traffic 粗估**

```math
\text{traffic}
\propto
N
\times
k
\times
H
\times
B_{\text{dtype}}
```

其中 Qwen3.6 中：

```math
k=8,\quad H=2048
```

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 full attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| QKV projection | `vllm/model_executor/models/qwen3_next.py` / `linear.py` | `QKVParallelLinear` |
| O projection | `vllm/model_executor/models/qwen3_next.py` / `linear.py` | `RowParallelLinear` |
| Attention operator | `vllm/model_executor/models/qwen3_next.py` | `Attention` |
| Attention backend 目录 | `vllm/v1/attention/backends/` | `flash_attn.py`, `flashinfer.py`, `triton_attn.py`, `gdn_attn.py` 等 |
| FlashAttention KV update | `vllm/v1/attention/backends/flash_attn.py` | `do_kv_cache_update` |
| FlashInfer backend | `vllm/v1/attention/backends/flashinfer.py` | `FlashInferBackend` |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| GDN kernel call | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `fla_chunk_gated_delta_rule` |
| Qwen3.6 MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| Fused MoE | `vllm/model_executor/layers/fused_moe/` | `FusedMoE`, `router`, `experts`, `runner`, `prepare_finalize` |
| Distributed collectives | `vllm/distributed/parallel_state.py` | `GroupCoordinator` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 是 35B total / 3B activated，40 层，layout 为 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 Gated Attention 为 16 Q heads、2 KV heads、head dim 256、RoPE dim 64 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MoE 为 256 experts、8 routed + 1 shared、expert intermediate dim 512 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM recipe 给出 Qwen3.6 TP=8、262144 max model len、qwen3 reasoning parser 命令 | vLLM Qwen3.5/Qwen3.6 recipe | Running Qwen3.6 | recipe 确认 | 是 |
| vLLM recipe 给出 Qwen3.6 MTP speculative decoding 命令 | vLLM Qwen3.5/Qwen3.6 recipe | `--speculative-config` | recipe 确认 | 是 |
| NVIDIA GPU 架构围绕 SM 阵列构建，thread blocks 被分发到 SM 执行，SM 以 warp 为单位调度 | NVIDIA CUDA hardware blog | SM / SIMT sections | 官方文档确认 | 是 |
| Tensor Cores 用于加速矩阵乘累加操作 | NVIDIA GPU performance docs | Tensor Core background | 官方文档确认 | 是 |
| `Qwen3NextAttention` 按 TP size 切分 heads，并在 KV heads 少于 TP size 时复制 KV heads | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 创建 router、shared expert gate、shared expert，并构造 `FusedMoE` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock.forward` 中 router logits 形状为 `(num_tokens, n_experts)`，并调用 `self.experts(...)` | `vllm/model_executor/models/qwen3_next.py` | `forward` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 继承 `MambaBase`，`mamba_type` 为 `gdn_attention` | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` state dtype/shape 由 Mamba state calculator 计算 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `get_state_dtype`, `get_state_shape` | 源码直接确认 | 是 |
| GDN 底层调用 `fla_chunk_gated_delta_rule`，输入包含 q/k/v/g/beta/state/chunk metadata | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `fla_chunk_gated_delta_rule` call | 源码直接确认 | 是 |
| vLLM v0.20.1 attention backend 目录包含 FlashAttention、FlashInfer、Triton、GDN、linear/mamba 等 backend 文件 | `vllm/v1/attention/backends/` | backend files | 源码直接确认 | 是 |
| FlashAttention backend 的 KV cache update 使用 `slot_mapping` 写入 key/value cache | `vllm/v1/attention/backends/flash_attn.py` | `do_kv_cache_update` | 源码直接确认 | 是 |
| FlashInfer backend 支持 FP16/BF16，并支持 fp8 KV cache dtype；支持 kernel block sizes `[16, 32, 64]` | `vllm/v1/attention/backends/flashinfer.py` | `FlashInferBackend` | 源码直接确认 | 是 |
| vLLM v0.20.1 `fused_moe` 目录包含 router、experts、runner、prepare_finalize、all2all、flashinfer/cutlass MoE 等组件 | `vllm/model_executor/layers/fused_moe/` | directory structure | 源码直接确认 | 是 |
| Qwen3.6 在你的 NVIDIA GPU 上实际选择 FlashAttention、FlashInfer、Triton 或其他 attention backend | 运行日志 / backend registry | 待运行确认 | 待运行确认 | 否 |
| Qwen3.6 MoE 在你的 GPU 上实际选择的 runner/kernel/backend | `fused_moe/runner` 与运行日志 | 待运行确认 | 待运行确认 | 否 |
| GDN kernel 在特定硬件上的 bottleneck 是 HBM、SM occupancy 还是 state layout | Nsight / profiling | N/A | 需按环境验证 | 否 |

---

## 源码阅读作业

**1. 读 Attention full path**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextAttention
QKVParallelLinear
RowParallelLinear
self.attn = Attention(...)
self.rotary_emb(...)
```

目标：把 QKV、RoPE、attention backend、O projection 连起来。

**2. 读 Attention backend**

```text
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/flashinfer.py
vllm/v1/attention/backends/triton_attn.py
```

重点找：

```text
block_table
slot_mapping
do_kv_cache_update
supported_dtypes
supported_kv_cache_dtypes
get_supported_kernel_block_sizes
```

目标：理解 backend 如何处理 KV cache 和 kernel block size。

**3. 读 Gated DeltaNet**

```text
vllm/model_executor/layers/mamba/gdn_linear_attn.py
vllm/v1/attention/backends/gdn_attn.py
```

重点找：

```text
GatedDeltaNetAttention
get_state_dtype
get_state_shape
fla_chunk_gated_delta_rule
chunk_indices
chunk_offsets
initial_state
output_final_state
```

目标：确认 GDN 是 state/chunk 路径，不是普通 softmax attention。

**4. 读 MoE**

```text
vllm/model_executor/models/qwen3_next.py
vllm/model_executor/layers/fused_moe/
```

重点找：

```text
Qwen3NextSparseMoeBlock
FusedMoE
router
experts
runner
prepare_finalize
all2all_utils
flashinfer_cutlass_moe.py
fused_batched_moe.py
```

目标：把 MoE 从公式拆到 router、dispatch、expert GEMM、combine。

**5. 读 distributed / communication**

```text
vllm/distributed/parallel_state.py
```

重点找：

```text
GroupCoordinator
all_reduce
all_gather
reduce_scatter
device_communicator
```

目标：为第十二章 Scheduler / Backend 源码导读做准备。

---

## 实验验证作业

### 实验 1：Prefill vs Decode profiler

固定 Qwen3.6 text-only：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

分别压测：

```text
A: 32K prompt + 128 output
B: 1K prompt + 4096 output
```

记录：

```text
prefill kernel 时间
decode kernel 时间
HBM bandwidth
Tensor Core utilization
SM occupancy
attention backend kernel 名称
MoE kernel 名称
```

### 实验 2：MoE expert GEMM 观察

统计不同并发：

```text
concurrency = 1, 8, 32, 128
```

观察：

```text
expert token counts
MoE runner kernel 时间
dispatch / combine 时间
all-to-all 时间
Tensor Core utilization
```

目标：判断瓶颈是 expert GEMM 太碎，还是 dispatch / all-to-all 太重。

### 实验 3：Attention backend 观察

在启动日志和 profiler 中确认：

```text
使用哪个 backend:
  FlashAttention?
  FlashInfer?
  Triton?
  GDN backend?
```

记录：

```text
KV cache dtype
block size
slot_mapping 写 cache kernel
decode attention kernel 时间
```

目标：不要根据源码目录猜 backend，要以运行日志和 profiler 为准。

### 实验 4：MTP 对 kernel 批大小的影响

对比 MTP off/on：

```bash
# off
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only

# on
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

记录：

```text
TPOT
accepted speculative tokens
KV/GDN state usage
MoE kernel batch size
decode kernel 时间
tokens/s
```

### 实验 5：logprobs 对 logits/sampling 的影响

请求：

```json
{"temperature": 0, "max_tokens": 256}
```

对比：

```json
{"temperature": 0, "max_tokens": 256, "logprobs": 5}
```

再谨慎测试：

```json
{"temperature": 0, "max_tokens": 16, "logprobs": -1}
```

观察：

```text
LM head 时间
logits processor 时间
sampler 时间
GPU/CPU transfer
response serialization
```

目标：验证大 vocab 输出模型上 logprobs 的代价。

---

## 3 个自测题

**1. 为什么 prefill 通常比 decode 更容易吃满 Tensor Core？**
因为 prefill 一次处理多个 prompt token，GEMM 的 $N$ 更大，矩阵乘更规则；decode 每步新增 token 少，更多时间花在读取历史 KV/GDN state。

**2. 为什么 MoE kernel 优化不等于只优化 GEMM？**
因为 MoE 还包含 router、top-k、token dispatch、expert grouping、expert output combine，以及 EP 下的 all-to-all 通信。

**3. 为什么不能看到 `vllm/v1/attention/backends/flashinfer.py` 就断言一定用了 FlashInfer？**
因为 backend 选择依赖 GPU、dtype、KV cache dtype、模型结构、配置、环境变量和运行时条件；源码目录只能证明 backend 存在，实际选择要看运行日志和 profiler。

---

## 下一章预告

第十二章进入：

```text
vLLM Scheduler、Block Manager、Attention Backend 源码导读
```

下一章会把第八章和第十一章合起来，按源码路径读：

```text
Scheduler.schedule
→ KVCacheManager.allocate_slots
→ BlockPool
→ Attention metadata builder
→ Attention backend forward
→ KV cache update
→ GDN metadata / state
```

重点回答：

```text
1. Scheduler 一轮到底输出什么？
2. KV blocks 和 slot_mapping 如何进入 backend？
3. FlashAttention / FlashInfer backend 的输入是什么？
4. GDN attention backend 如何不同于普通 KV attention？
5. Qwen3.6 的 hybrid 架构在 vLLM v0.20.1 中如何被调度和执行？
```

---

**Sources:**

- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [NVIDIA Tensor Cores](https://www.nvidia.com/en-us/data-center/tensor-cores/)
- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vLLM v0.20.1 attention backends](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends)
- [vLLM v0.20.1 fused MoE layers](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/model_executor/layers/fused_moe)
