# 第十三章：工程部署：FP8 单卡、BF16 多卡、Expert Parallel、降级配置

本章把前 12 章的原理落成部署决策。核心目标不是“给一条万能命令”，而是建立一套判断框架：

```text
1. 先能启动
2. 再能稳定跑目标上下文
3. 再能承载目标并发
4. 最后再追求吞吐、延迟和成本最优
```

对 **Qwen/Qwen3.6-35B-A3B**，部署时必须始终记住：它是带 Vision Encoder 的 causal LM，35B total / 3B activated，hidden size 2048，40 层，结构是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，MoE 为 256 experts、8 routed + 1 shared expert，原生上下文 262,144 tokens，可扩展到 1,010,000 tokens。官方模型卡还明确提示：如果遇到 OOM，可以降低 context window，但建议至少保持 128K 以保留复杂任务能力。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

## 本章解决什么问题

- 把前 12 章的模型结构、显存、调度、MoE、多卡和 GPU 视角落成部署选择。
- 帮你区分单卡 FP8、BF16 TP=8、text-only、多模态、MTP、Expert Parallel 的适用边界。
- 给出上线前要检查的显存、上下文、并发、TTFT、TPOT、OOM 和回滚条件。
- 明确哪些结论来自官方命令，哪些必须在你的硬件和 workload 上验证。

---

## 一、生活类比

部署 Qwen3.6 像安排一支大型混合专家团队进办公室。

**BF16 多卡** 像给整支团队安排一栋大楼：

```text
8 张 GPU
→ 权重分摊
→ KV/GDN state 有更多空间
→ MoE experts 可以更合理分布
→ 适合 262K context、高并发、生产服务
```

**FP8 单卡** 像把文件压缩后塞进一个小办公室：

```text
权重显存压力下降
但会议室、白板、临时工位仍然有限
```

也就是说，FP8 能降低权重占用，但不能让 KV cache、Gated DeltaNet state、MoE dispatch buffer、MTP lookahead tokens、Vision Encoder buffer 自动消失。Qwen3.6 官方 FP8 checkpoint 是 fine-grained FP8 quantization，block size 为 128，模型卡称其性能指标接近原模型；但它仍然是同一个 35B / 3B activated / 262K context / Vision Encoder / MoE / MTP 架构。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

所以部署思路应该是：

```text
FP8：
  主要解决权重显存和部分 GEMM/带宽压力

max_model_len：
  主要决定 KV/GDN state 预算

language-model-only：
  主要释放 vision encoder / multimodal profiling 相关显存

MTP：
  可能降低 TPOT，但增加 speculative token cache/state 预算

Expert Parallel：
  主要解决 MoE experts 分布和专家计算/通信效率问题
```

### 部署决策树

| 目标场景 | 优先配置 | 主要收益 | 主要风险 | 验证重点 |
|---|---|---|---|---|
| 本地功能验证 / 小上下文服务 | FP8 单卡 | 降低权重显存门槛 | KV/GDN state、workspace、Vision Encoder 仍会占显存 | 是否能启动、短上下文 TTFT/TPOT、OOM 边界 |
| 纯文本生产服务 | BF16/FP8 + `--language-model-only` | 释放多模态相关显存，减少 profiling 压力 | 不能服务 image/video prompt | text-only workload 的 P50/P99 与 KV usage |
| 多模态服务 | 保留 Vision Encoder，单独分池 | 支持 image/video 输入 | encoder budget、processor cache、显存波动 | text-only 与 multimodal 分开压测 |
| 262K 长上下文主服务 | BF16 多卡 TP=8 起步 | 最接近官方 recipe 的长上下文路径 | KV/GDN state、NCCL、backend 选择 | 128K/262K prompt、并发阶梯、preemption |
| MoE 多卡优化 | TP + EP 候选 | expert 分布更灵活 | all-to-all、expert imbalance、实验性参数 | EP off/on 对照、per-rank util、all-to-all 时间 |
| 低延迟 decode 优化 | MTP off/on 对照 | 可能降低 TPOT | speculative tokens 占 cache/state，可能伤高并发吞吐 | accepted tokens、TPOT、KV usage、吞吐回归 |

这张表不是替代压测，而是帮助你决定先测哪条路径。

---

## 二、数学直觉

### 1. 权重显存

粗略地说：

```math
\text{weight memory}
\approx
\text{num parameters}
\times
\text{bytes per parameter}
```

对 35B 参数：

```text
BF16 / FP16:
  35B × 2 bytes ≈ 70 GB

FP8:
  35B × 1 byte ≈ 35 GB
  加上 scale、metadata、padding、runtime workspace 后会更高
```

这说明 FP8 对“能不能加载权重”非常有帮助。但这只是权重，不是完整推理显存。

### 2. Qwen3.6 的 Gated Attention KV cache 粗算

Qwen3.6 不是 40 层 full attention。按模型卡 layout，40 层中只有 10 个 Gated Attention 层；Gated Attention 的 KV heads = 2，head dim = 256。若 KV dtype 按 BF16/FP16 粗估，每 token 的 Gated Attention KV cache 是：

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

其中：

```text
10 = Gated Attention 层数
第一个 2 = K 和 V
第二个 2 = KV heads
256 = head dim
最后 2 = BF16/FP16 bytes
```

262,144 tokens 时，仅单序列 Gated Attention KV cache 粗略为：

```math
262144 \times 20480 \approx 5.37\ \text{GB}
```

这还没有算 30 个 Gated DeltaNet 层的 state、MTP lookahead、MoE runtime buffer、block metadata、多模态、workspace、TP 下 KV heads 复制等。Qwen3.6 的层布局、Gated Attention 参数和 262K context 均来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. 并发放大

如果平均每个请求占用 $T$ 个 active tokens，并发数是 $B$，则仅 Gated Attention KV cache 就近似按：

```math
B
\times
T
\times
20480
```

增长。

这就是为什么：

```text
单请求 262K 能跑
≠
多请求 262K 高并发能跑
```

### 4. MTP 的额外预算

MTP speculative decoding 会让 scheduler 预留 lookahead / speculative tokens。vLLM v0.20.1 的 Scheduler 源码中，如果存在 speculative config，会设置 `num_spec_tokens` 和 `num_lookahead_tokens`，并在 `allocate_slots()` 时把 `num_lookahead_tokens` 传入 KV cache manager。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py))

所以 MTP 的工程判断是：

```math
\text{收益}
=
\text{accepted speculative tokens 带来的 TPOT 降低}
-
\text{额外 compute/cache/state/调度成本}
```

---

## 三、张量形状

### 1. Qwen3.6 的部署关键形状

```text
hidden_size = 2048
layers = 40
Gated DeltaNet layers = 30
Gated Attention layers = 10
Gated Attention Q heads = 16
Gated Attention KV heads = 2
Gated Attention head dim = 256
MoE experts = 256
routed experts per token = 8
shared experts per token = 1
vocab / LM output = 248320 padded
context = 262144 native
```

这些尺寸直接决定权重切分、KV/state 显存、MoE dispatch、LM head logits、sampling 成本。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. BF16 多卡的核心张量

用官方 recipe 的 TP=8 命令时：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

vLLM recipe 明确给出这个 Qwen3.6 基础命令。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

在 TP=8 下，Gated Attention 的 Q heads 可以切分：

```text
global Q heads = 16
TP = 8
local Q heads ≈ 2
```

但 KV heads 只有 2。vLLM v0.20.1 的 `Qwen3NextAttention` 会根据 TP size 切分 attention heads，并在 KV heads 少于 TP size 时复制 KV heads；因此不能简单认为 KV cache 严格按 8 等分。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. Expert Parallel 的核心形状

MoE 输入：

```text
hidden_states: [N, 2048]
router_logits: [N, 256]
topk_indices:  [N, 8]
```

如果 EP size = 8，直觉上 256 experts 可以约分成每 rank 32 experts。但真实映射要看 vLLM 的 expert map。vLLM v0.20.1 的 `FusedMoE` 会根据 `ep_size`、`ep_rank`、`global_num_experts`、expert placement strategy 等生成 local expert map；源码中还包含 EPLB、redundant experts 和 expert placement 逻辑。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py))

---

## 四、计算流程

### 1. 基础 BF16 多卡流程

适合第一条生产基线：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

执行链路：

```text
加载 BF16/默认 dtype 权重
→ TP=8 切分 attention / dense / LM head 等张量
→ 建立 KV/GDN state cache
→ Scheduler 接收请求
→ prefill / decode / chunked prefill / prefix cache
→ Qwen3.6 hybrid layer:
    Gated DeltaNet 或 Gated Attention
    → MoE
→ logits / sampling
```

这是“最接近官方 262K 满上下文推荐路径”的部署起点，因为 vLLM recipe 明确给出 TP=8 + 262144 max model len。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 2. Text-only 流程

如果只服务文本，应该优先考虑：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

Qwen 模型卡说明 text-only 命令会跳过 vision encoder 和 multimodal profiling，从而释放更多内存给 KV cache。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

这不是“小优化”，而是部署形态选择：

```text
纯文本服务池：
  --language-model-only

多模态服务池：
  保留 Vision Encoder
  单独压测 image/video prompt
```

不要把 text-only 和 multimodal workload 混在同一个平均值里分析。

### 3. MTP 流程

Qwen 模型卡推荐的 vLLM MTP 命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --port 8000 \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

vLLM recipe 中也给出 MTP speculative decoding 配置，不过 recipe 使用的是 `{"method": "mtp", "num_speculative_tokens": 2}`。这两个写法在 v0.20.1 的完整解析路径需要继续读 speculative config 源码确认；部署时应以当前安装版本的日志和启动成功路径为准。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

MTP 流程：

```text
主模型当前 token
→ MTP proposer 草拟未来 token
→ scheduler 预留 lookahead slots
→ 主路径验证
→ 接受若干 draft tokens
→ 回退或继续
```

### 4. Expert Parallel 流程

启用 EP 后，MoE expert layers 会按 expert parallelism 分布到 EP ranks，而 attention layers 根据 TP size 复制或 tensor parallel 切分。vLLM EP 文档说明，`--enable-expert-parallel` 会启用 EP，EP size 自动计算为 `TP_SIZE × DP_SIZE`；文档也明确提醒 EP 是实验性功能，参数名和默认值可能变化。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

通用形态：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-expert-parallel
```

这条命令是基于官方 EP 参数机制和 Qwen3.6 recipe 组合出的工程候选配置；是否优于纯 TP，要看你的 GPU 拓扑、expert token 分布、all-to-all backend、并发和 workload。

---

## 五、显存布局

### 1. 单卡 FP8 显存账本

FP8 单卡常见误区是：

```text
只要 FP8 权重能放下，模型就能完整服务 262K。
```

更准确的账本是：

```text
FP8 weights
+ scale / metadata
+ Gated Attention KV cache
+ Gated DeltaNet state
+ MoE runtime buffer
+ LM head / logits / sampling workspace
+ scheduler / block metadata
+ CUDA graph / backend workspace
+ optional Vision Encoder
+ optional MTP lookahead cache
```

Qwen3.6 FP8 checkpoint 的模型卡说明它包含 FP8-quantized weights/config，采用 fine-grained FP8 quantization，block size 为 128；但模型仍然保留 262K context、Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP 等结构。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

因此单卡 FP8 更适合定位为：

```text
1. 本地功能验证
2. 小上下文在线服务
3. 低并发内部工具
4. 成本敏感的降级配置
5. 对 262K 不做满血承诺的服务池
```

不应默认承诺：

```text
单卡 FP8 + 262K + 多模态 + MTP + 高并发
```

### 2. BF16 多卡显存账本

BF16 多卡更适合作为 262K 主服务形态：

```text
权重：
  TP/EP 分摊

Gated Attention KV:
  随 context 和并发增长

Gated DeltaNet state:
  随 context、mamba_cache_mode、spec tokens 增长

MoE expert weights:
  TP 或 EP 分布

MoE runtime:
  router/top-k/dispatch/expert GEMM workspace

Vision Encoder:
  multimodal 服务时额外加载

MTP:
  num_speculative_tokens 增加 lookahead cache/state 预算
```

vLLM v0.20.1 的 Qwen3Next 源码中，GDN state dtype/shape 通过 Mamba state calculator 计算；`get_mamba_state_shape_from_config` 会读取 parallel config 和 HF text config。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 3. 降级时优先动哪些旋钮

显存不够时，优先按这个顺序降级：

```text
1. text-only:
   加 --language-model-only，跳过 vision encoder

2. max_model_len:
   262K → 128K → 64K → 32K

3. MTP:
   关闭 MTP 或降低 num_speculative_tokens

4. 并发:
   限制 max running requests / 外部网关限流

5. 输出长度:
   限制 max_tokens

6. logprobs:
   默认关闭，禁止 logprobs=-1

7. FP8:
   使用 Qwen3.6 FP8 checkpoint 降低权重显存

8. 多模态分池:
   text-only 和 vision-language 分开部署
```

Qwen 模型卡建议 OOM 时降低 context window，但也建议至少保持 128K；所以从 262K 降到 128K 是较自然的第一档，而不是直接降到 8K 或 16K。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

---

## 六、vLLM 源码映射

| 部署问题 | vLLM v0.20.1 源码入口 | 结论 |
|---|---|---|
| TP 下 attention heads 如何切 | `vllm/model_executor/models/qwen3_next.py` → `Qwen3NextAttention` | Q heads 按 TP 切，KV heads 少于 TP size 时复制 |
| GDN state 如何估算 | `qwen3_next.py` → `get_mamba_state_shape_from_config` | GDN state shape 与 parallel config / model config / cache config 相关 |
| MTP 如何影响 cache | `vllm/v1/core/sched/scheduler.py` | speculative config 设置 `num_spec_tokens` / `num_lookahead_tokens`，并传给 `allocate_slots` |
| MoE 如何接 EP | `qwen3_next.py` → `Qwen3NextSparseMoeBlock` | 读取 EP group、rank、size，构造 FusedMoE |
| FusedMoE 如何放置 experts | `fused_moe/layer.py` → `FusedMoE` | 根据 EP size/rank、placement strategy 生成 local experts / expert map |
| EPLB 是否存在 | `fused_moe/layer.py` | `enable_eplb`、expert placement、redundant experts 等逻辑存在 |
| 官方 EP 参数 | vLLM EP docs | `--enable-expert-parallel`，EP size = TP × DP，EP 仍为实验性功能 |

源码上最重要的事实是：Qwen3.6 的 full-attention、GDN、MoE、MTP 并不是一个统一“Transformer block”能解释完的路径。`Qwen3NextAttention`、GDN state calculator、`Qwen3NextSparseMoeBlock`、`FusedMoE`、Scheduler 的 lookahead token 机制分别对应部署中的 TP、GDN cache、EP/MoE 和 MTP 预算问题。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 七、NVIDIA GPU 执行视角

### 1. FP8 单卡

FP8 单卡的收益：

```text
权重更小
HBM 读权重压力下降
某些 GEMM 更快或更省带宽
启动门槛降低
```

FP8 单卡的风险：

```text
KV/GDN state 不一定随权重 FP8 等比例下降
长上下文仍然吃 cache
MoE dispatch buffer 仍然存在
LM head logits 仍然是 248320 padded vocab
多模态 vision encoder 仍然占显存
MTP 仍然增加 lookahead state
```

所以单卡 FP8 应该先测：

```text
max_model_len = 32768
max_model_len = 65536
max_model_len = 131072
```

确认稳定后，再尝试更长上下文。

### 2. BF16 多卡

BF16 多卡的优势：

```text
数值路径更稳健
权重分摊
KV/GDN state 空间更大
TP/EP 能利用多 GPU
适合生产 262K 服务
```

主要瓶颈：

```text
TP all-reduce / all-gather
EP all-to-all
KV heads 复制导致 KV cache 不严格除以 TP
长上下文 HBM 读
MoE expert imbalance
```

### 3. Expert Parallel

EP 的收益来自：

```text
专家权重分布到多 GPU
MoE local expert GEMM 更大
热门 workload 下可能提升 MoE 效率
```

EP 的代价来自：

```text
token dispatch
all-to-all / allgather-reducescatter
expert imbalance
EPLB 额外显存和权重迁移
```

vLLM EP 文档说明，启用 EP 后 expert layers 会 across EP ranks 分片，attention layers 则由 TP size 决定复制或 TP 切分；文档还说明 EPLB 会收集每次 forward 的 load statistics 并周期性 rebalance expert distribution。([vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/))

### 4. MTP

MTP 适合：

```text
低并发
追求低 TPOT
draft 接受率高
cache 预算充足
```

MTP 不一定适合：

```text
高并发
KV/GDN state 已接近上限
长上下文请求很多
draft 接受率低
```

---

## 八、优化前 vs 优化后

### 优化前：一次性开满

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
  --enable-expert-parallel \
  --enable-prefix-caching
```

这类“一次性全开”不适合作为第一轮部署。问题是：

```text
OOM 了不知道是权重、KV、GDN state、MTP 还是 Vision Encoder
吞吐低不知道是 attention、MoE、NCCL 还是 sampler
延迟高不知道是 prefill、decode、chunked prefill 还是 EP all-to-all
```

### 优化后：分阶段建基线

推荐四条基线：

**基线 A：BF16 多卡标准**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

**基线 B：BF16 多卡 text-only**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

**基线 C：FP8 降级**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only
```

这里故意先用 32K，而不是 262K。原因是 FP8 单卡首先验证“能稳定服务”，再逐步拉长上下文。FP8 模型卡展示了可以用 vLLM serve 启动 `Qwen/Qwen3.6-35B-A3B-FP8`，但没有承诺单卡可满血 262K 高并发。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

**基线 D：MTP 对照**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

MTP 必须和非 MTP 对照压测，不能只看单请求演示。

---

## 九、工程落地

### 1. 推荐部署矩阵

| 场景 | 推荐起点 | 说明 |
|---|---|---|
| 生产 262K 文本服务 | BF16/默认 dtype，TP=8，`--language-model-only` | 优先保证 cache 空间 |
| 生产多模态服务 | BF16/默认 dtype，TP=8，不加 `--language-model-only` | 单独压测 vision encoder |
| 成本敏感小上下文 | FP8 checkpoint，32K/64K 起步 | 不承诺满血 262K |
| 低并发低 TPOT | BF16/FP8 + MTP | 看接受率和 cache 余量 |
| 高并发吞吐 | text-only + prefix caching + 合理 batch token | MTP 不一定收益 |
| MoE 瓶颈明显 | 尝试 `--enable-expert-parallel` | 观察 all-to-all 和 expert imbalance |
| 显存紧张 | text-only → 降 max_model_len → 关 MTP → FP8 | 不要先乱调 backend |

### 2. 降级配置阶梯

建议准备四档 SLA：

```text
P0 满血:
  262K context
  BF16/默认 dtype
  TP=8
  text-only 或 multimodal 分池
  prefix caching
  MTP 按 workload 开

P1 稳定:
  128K context
  TP=8
  text-only
  MTP off
  prefix caching on

P2 成本:
  64K context
  FP8
  text-only
  低并发

P3 应急:
  32K context
  FP8
  max_tokens 限制
  logprobs off
  MTP off
```

### 3. 观测指标

上线前至少记录：

```text
启动阶段:
  weight memory
  KV/GDN state capacity
  num_gpu_blocks
  backend selection
  mamba_cache_mode

请求阶段:
  TTFT
  TPOT / ITL
  tokens/s
  request_prefill_time
  request_decode_time
  kv_cache_usage_perc
  num_requests_running
  num_requests_waiting

MoE:
  expert token distribution
  all-to-all time
  FusedMoE kernel time
  EPLB rebalance events

MTP:
  accepted speculative tokens
  draft acceptance rate
  lookahead cache usage

多模态:
  vision encoder time
  multimodal processor cache hit
  image/video token count
```

### 4. 必须禁止的默认项

生产默认不建议：

```text
logprobs = -1
无限 max_tokens
262K + 高并发 + MTP + 多模态全开
text-only 和 multimodal 混池
没有 prefix cache 观测就假设命中
没有 profiler 就判断 backend 瓶颈
```

### 5. 上线流程

推荐上线流程：

```text
第 1 步：
  跑 32K text-only，确认可启动和基本输出

第 2 步：
  跑 128K text-only，确认 cache 稳定

第 3 步：
  跑 262K text-only，确认单请求与低并发

第 4 步：
  加并发，观察 waiting/running/KV usage

第 5 步：
  加 prefix caching，观察 hit rate 和 TTFT

第 6 步：
  对照 MTP off/on，观察 TPOT、吞吐、cache

第 7 步：
  如果 MoE 瓶颈明显，再测试 EP/EPLB

第 8 步：
  多模态单独建服务池和压测
```

### 6. 发布前验收表

| 验收项 | 最低记录内容 | 通过标准 | 不通过时的动作 |
|---|---|---|---|
| 启动成功 | 完整启动命令、vLLM 版本、GPU 型号、dtype、TP/EP 配置 | 无 OOM，日志中 backend / block 数清晰 | 降 `max_model_len`、改 text-only、降低并发或换多卡 |
| 上下文边界 | 32K / 128K / 262K 单请求结果 | 目标上下文能稳定返回，TTFT 可接受 | 降上下文、启用 chunked prefill、检查 cache 预算 |
| 并发阶梯 | concurrency 1 / 8 / 32 / 目标值 | waiting 不持续堆积，KV usage 不长期贴近 1 | 降并发、增加 GPU、缩短 output、拆服务池 |
| TTFT / TPOT | p50 / p90 / p99 与 workload | 达到业务 SLA，且瓶颈能解释 | 分别看 queue、prefill、decode、backend、MoE、NCCL |
| OOM 边界 | 启动、prefill、decode、MTP、多模态各阶段 | OOM 模式已知且有降级策略 | 限 prompt/output、关 MTP、降 max len、切 text-only |
| MTP off/on | accepted tokens、TPOT、throughput、KV usage | 低延迟收益大于 cache 成本 | 高并发吞吐回退时关闭或降低 speculative tokens |
| EP off/on | per-rank util、all-to-all time、tokens/s | EP 在目标拓扑和 workload 下有净收益 | 回退纯 TP 或换 all-to-all backend |
| 回滚方案 | 可执行的保守启动命令 | 出问题时能快速恢复服务 | 保留 P1/P2/P3 降级配置 |

这张表应该随压测结果一起保存，作为 GitBook 之外的上线记录。

---

## 十、本章总结

本章最重要的部署判断是：

```text
FP8 解决权重压力，
不等于解决 262K 长上下文压力。

TP 解决大矩阵切分，
不等于 KV cache 严格除以 TP。

EP 解决 expert 权重/计算分布，
不等于 all-to-all 免费。

MTP 可能降低 TPOT，
不等于高并发吞吐一定提升。

text-only 能释放 vision encoder 相关显存，
但会关闭多模态能力。
```

对 Qwen3.6-35B-A3B，推荐部署顺序是：

```text
先 BF16/默认 dtype TP=8 建 262K 基线
再 text-only 释放 cache 空间
再根据 workload 加 prefix caching
再单独评估 MTP
再单独评估 EP/EPLB
最后才把 FP8 作为成本/降级/小上下文方案推广
```

---

## 小白总结

FP8 就像把模型文件压缩了，能省很多空间；但模型推理还需要草稿纸、历史笔记、临时工作台。Qwen3.6 的历史上下文很长，还有专家系统和视觉编码器，所以不能只看“模型文件能不能放进显卡”。

最稳妥的部署方法是：先多卡跑通 262K，再一步步打开 text-only、prefix cache、MTP、EP；单卡 FP8 更适合小上下文或降级服务，不要一开始就承诺满血 262K 高并发。

---

## 工程师总结

Qwen3.6 部署的核心账本：

```text
weights:
  BF16/FP16 or FP8

cache/state:
  10 层 Gated Attention KV
  30 层 Gated DeltaNet state

MoE:
  256 experts
  top-8 routed + 1 shared
  FusedMoE / EP / all-to-all

scheduler:
  max_model_len
  max_num_batched_tokens
  prefix caching
  chunked prefill
  num_lookahead_tokens

runtime:
  attention backend
  GDN backend
  NCCL
  sampling/logprobs
  vision encoder
```

vLLM v0.20.1 的源码证据支持这些边界：`Qwen3NextAttention` 处理 Gated Attention / TP head 分片，GDN state shape 由 Mamba state calculator 计算，Scheduler 会把 speculative lookahead tokens 传入 `allocate_slots`，MoE 由 `Qwen3NextSparseMoeBlock` + `FusedMoE` + expert map 承载。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 关键公式

**1. 权重显存粗估**

```math
\text{weight memory}
\approx
\text{num parameters}
\times
\text{bytes per parameter}
```

**2. Qwen3.6 Gated Attention KV 每 token 粗估**

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

**3. 单序列 262K Gated Attention KV 粗估**

```math
262144
\times
20480
\approx
5.37\ \text{GB}
```

**4. 并发 cache 粗估**

```math
\text{KV bytes}
\propto
B
\times
T
\times
20480
```

**5. MoE token-expert 对数量**

```math
N_{\text{token-expert}}
=
N_{\text{tokens}}
\times
8
```

**6. MTP 收益判断**

```math
\text{Net gain}
=
\text{accepted tokens gain}
-
\text{extra compute/cache/state cost}
```

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| TP head / KV head 逻辑 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` |
| GDN state shape | `vllm/model_executor/models/qwen3_next.py` | `get_mamba_state_shape_from_config` |
| GDN kernel/state | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Scheduler lookahead | `vllm/v1/core/sched/scheduler.py` | `num_spec_tokens`, `num_lookahead_tokens` |
| KV slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| Qwen3.6 MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| Fused experts | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` |
| EP expert map | `vllm/model_executor/layers/fused_moe/layer.py` | `determine_expert_map`, `_expert_map` |
| EPLB | `vllm/model_executor/layers/fused_moe/layer.py` | `enable_eplb`, `EplbLayerState` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 是 Causal Language Model with Vision Encoder，35B total / 3B activated，40 层，262K context | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MoE 是 256 experts，8 routed + 1 shared，expert intermediate dim 512 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 FP8 checkpoint 使用 fine-grained FP8 quantization，block size 128 | Hugging Face FP8 model card | FP8 model overview | 官方文档确认 | 是 |
| Qwen3.6 标准 vLLM 命令使用 TP=8、max model len 262144、reasoning parser qwen3 | Qwen HF model card / vLLM recipe | vLLM serving command | recipe 确认 | 是 |
| Qwen3.6 text-only 命令使用 `--language-model-only`，用于跳过 vision encoder 和 multimodal profiling，释放更多 KV cache 内存 | Qwen HF model card | Text-Only command | 官方文档确认 | 是 |
| Qwen3.6 MTP 命令可使用 `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'` | Qwen HF model card | MTP command | 官方文档确认 | 是 |
| vLLM recipe 也给出 Qwen3.6 MTP speculative decoding 配置 `{"method":"mtp","num_speculative_tokens":2}` | vLLM Qwen recipe | MTP command | recipe 确认 | 是 |
| `Qwen3NextAttention` 按 TP size 切分 heads，并处理 KV heads 少于 TP size 时的复制 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| GDN state dtype/shape 通过 Mamba state calculator 计算 | `vllm/model_executor/models/qwen3_next.py` | `get_mamba_state_dtype_from_config`, `get_mamba_state_shape_from_config` | 源码直接确认 | 是 |
| Scheduler 会根据 speculative config 设置 `num_spec_tokens` 和 `num_lookahead_tokens` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.__init__` | 源码直接确认 | 是 |
| Scheduler 会把 `num_lookahead_tokens` 传入 `kv_cache_manager.allocate_slots` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Qwen3.6 MoE block 对应 `Qwen3NextSparseMoeBlock` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | 源码直接确认 | 是 |
| `FusedMoE` 是 vLLM MoE fused layer，包含 gate/up 与 down expert weights | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | 源码直接确认 | 是 |
| `FusedMoE` 中存在 `enable_eplb`、expert placement strategy、expert map、本地 expert 数等逻辑 | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` | 源码直接确认 | 是 |
| vLLM EP 文档说明 `--enable-expert-parallel` 启用 EP，EP size = TP × DP，EP 为实验性功能 | vLLM Expert Parallel Deployment docs | EP Configuration | 官方文档确认 | 是 |
| vLLM EP 文档说明启用 EP 后 MoE expert layers across EP ranks 分片，attention layers 由 TP size 决定 | vLLM Expert Parallel Deployment docs | Layer Behavior with EP Enabled | 官方文档确认 | 是 |
| 单卡 FP8 是否能承载 262K、高并发、多模态和 MTP | 运行环境 / 显存 / profiler | N/A | 需按环境验证 | 否 |
| 当前机器上 MTP `method="mtp"` 与 `method="qwen3_next_mtp"` 的解析差异 | speculative config 源码 / 启动日志 | 待运行确认 | 待运行确认 | 否 |
| 当前机器上 EP all-to-all backend 是否优于纯 TP | runtime benchmark | N/A | 需按环境验证 | 否 |

---

## 源码阅读作业

**1. 部署命令与模型结构**

读：

```text
Qwen/Qwen3.6-35B-A3B model card
Qwen/Qwen3.6-35B-A3B-FP8 model card
vLLM Qwen3.5 & Qwen3.6 recipe
```

目标：确认标准命令、MTP 命令、text-only 命令、FP8 checkpoint 和 262K context。

**2. TP / Gated Attention**

读：

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextAttention
tp_size
total_num_heads
total_num_kv_heads
num_heads
num_kv_heads
QKVParallelLinear
RowParallelLinear
```

目标：理解 TP=8 下 Q heads 和 KV heads 的行为。

**3. GDN state**

读：

```text
vllm/model_executor/models/qwen3_next.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
```

重点找：

```text
get_mamba_state_shape_from_config
MambaStateShapeCalculator.gated_delta_net_state_shape
GatedDeltaNetAttention
```

目标：理解为什么 GDN state 要单独估算。

**4. MTP / lookahead**

读：

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/kv_cache_manager.py
```

重点找：

```text
num_spec_tokens
num_lookahead_tokens
allocate_slots
```

目标：理解 MTP 为什么会影响 cache slot 预算。

**5. EP / MoE**

读：

```text
vllm/model_executor/models/qwen3_next.py
vllm/model_executor/layers/fused_moe/layer.py
```

重点找：

```text
Qwen3NextSparseMoeBlock
get_ep_group
FusedMoE
enable_eplb
determine_expert_map
local_num_experts
```

目标：理解 Expert Parallel 和 expert placement 的源码证据。

---

## 实验验证作业

### 实验 1：BF16/默认 dtype 多卡基线

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

记录：

```text
启动是否成功
GPU memory used
KV/GDN state capacity
num_gpu_blocks
TTFT
TPOT
tokens/s
backend selection
```

### 实验 2：text-only 对照

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

对比：

```text
启动显存
可用 KV/GDN cache
TTFT
TPOT
max concurrency
多模态能力是否关闭
```

### 实验 3：FP8 小上下文起步

```bash
vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only
```

逐步测试：

```text
32768
65536
131072
262144
```

每一档都记录 OOM 位置：启动、prefill、decode、MTP、multimodal。

### 实验 4：MTP off/on 对照

off：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

on：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```

记录：

```text
accepted speculative tokens
TPOT
tokens/s
KV/GDN state usage
waiting/running requests
```

### 实验 5：EP / EPLB 对照

基础 EP：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-expert-parallel
```

观察：

```text
MoE kernel time
all-to-all time
expert token imbalance
per-rank GPU utilization
```

如果 expert imbalance 明显，再测试 EPLB：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --enable-expert-parallel \
  --enable-eplb
```

---

## 3 个自测题

**1. FP8 checkpoint 为什么不能保证单卡 262K 高并发？**
因为 FP8 主要降低权重显存；262K 下的 KV cache、Gated DeltaNet state、MTP lookahead、多模态 buffer、MoE dispatch buffer 和 runtime workspace 仍然会消耗大量显存。

**2. 为什么 text-only 部署能释放显存？**
因为 Qwen3.6 是带 Vision Encoder 的 causal LM；`--language-model-only` 会跳过 vision encoder 和 multimodal profiling，把更多内存留给 KV cache / state。

**3. Expert Parallel 为什么需要实测？**
因为 EP 会把 experts 分到多个 GPU，可能降低 expert 权重压力并改善局部性，但也引入 all-to-all、expert imbalance、EPLB 显存代价和通信瓶颈；是否更快取决于硬件拓扑和 workload。

---

## 下一章预告

第十四章进入：

```text
压测、监控、排障
```

下一章会把部署后的问题系统化：

```text
1. 如何设计 prompt length / output length / concurrency 压测矩阵？
2. 如何区分 TTFT、TPOT、throughput、GPU utilization 的含义？
3. OOM 发生在启动、prefill、decode、MTP、多模态时分别怎么排？
4. 如何观察 KV cache、GDN state、MoE expert imbalance？
5. 如何定位 attention backend、MoE kernel、NCCL、sampling/logprobs 的瓶颈？
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vllm/vllm/v1/core/sched/scheduler.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
