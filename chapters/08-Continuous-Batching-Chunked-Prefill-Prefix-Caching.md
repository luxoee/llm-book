# 第八章：Continuous Batching、Chunked Prefill、Prefix Caching

本章把第七章的 **PagedAttention / block 管理** 接到 vLLM 的 **Scheduler** 上。

前三章你已经知道：

```text
prefill 负责读 prompt
decode 负责一个 token 一个 token 生成
KV cache / GDN state 会随上下文增长
PagedAttention 用 blocks 管理 Gated Attention 的 K/V
```

第八章要回答：

```text
多个用户请求同时进来时，vLLM 到底怎么决定：
  这一步算谁？
  算多少 token？
  先算 decode 还是 prefill？
  长 prompt 会不会卡住短请求？
  相同 prefix 能不能复用？
```

对固定案例 **Qwen/Qwen3.6-35B-A3B**，调度问题更尖锐：它原生上下文 262,144 tokens，结构是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，包含 Gated DeltaNet、Gated Attention、MoE、MTP，并且是带 Vision Encoder 的 Causal LM；所以调度器不仅要面对普通 KV blocks，还要面对 hybrid / Mamba-like state、MoE dispatch、多模态 encoder budget 和 speculative tokens。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

把 vLLM 想成一个 GPU 餐厅。

每个用户请求是一桌客人：

```text
短 prompt + 长输出：像点菜快、慢慢吃
长 prompt + 短输出：像菜单读很久、吃得很快
长 prompt + 长输出：从点菜到吃饭都很久
```

朴素 serving 像这样：

```text
一桌客人从点菜到吃完，GPU 厨房只服务这一桌。
```

这会浪费 GPU。因为 decode 阶段每桌每次只生成一个 token，单桌很难吃满 GPU。

**Continuous batching** 像餐厅把不同桌的“下一道菜”凑成一批一起做：

```text
A 桌要生成 1 个 token
B 桌要生成 1 个 token
C 桌刚进来，要 prefill 512 个 token
D 桌要生成 1 个 token

GPU 本轮 batch：
  A decode 1 token
  B decode 1 token
  D decode 1 token
  C prefill 一小块
```

**Chunked prefill** 像把超长菜单拆成几页读，不让一桌 262K prompt 的用户霸占厨房：

```text
长 prompt 不一次性吃完整个 batch budget
而是切成 chunk，穿插 decode 请求一起跑
```

**Prefix caching** 像老客户点了相同套餐前半部分：

```text
系统 prompt / 工具说明 / 长文档前缀一样
上次已经算过前缀 KV blocks
这次直接复用，不重复 prefill
```

vLLM 官方优化文档说明，chunked prefill 会把大 prefill 切成小块，并和 decode 请求同批处理；V1 中该策略会优先调度 decode 请求，剩余 `max_num_batched_tokens` 预算再安排 prefill，放不下的 prefill 会被自动 chunk。([docs.vllm.ai](https://docs.vllm.ai/en/latest/configuration/optimization/))

---

## 二、数学直觉

### 1. Scheduler 的核心变量

vLLM v0.20.1 Scheduler 源码中的关键注释是：调度器里没有严格的“decode phase”或“prefill phase”，每个请求只有：

```text
num_computed_tokens
num_tokens_with_spec
```

调度器每一步尝试给请求分配 tokens，让 `num_computed_tokens` 追上 `num_tokens_with_spec`；这个抽象可以覆盖 chunked prefill、prefix caching、speculative decoding，以及未来的 jump decoding。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

可以把它写成：

```math
\Delta_i =
N^{\text{target}}_i - N^{\text{computed}}_i
```

其中：

| 符号 | 含义 |
|---|---|
| $N^{\text{computed}}_i$ | 请求 $i$ 已经算过的 token 数 |
| $N^{\text{target}}_i$ | 请求 $i$ 当前应该追上的 token 数，包含 prompt、output、spec tokens |
| $\Delta_i$ | 当前请求还需要调度的 token 数 |

每个 scheduler step 有一个 token budget：

```math
\sum_i \Delta_i^{\text{scheduled}} \leq \mathrm{maxNumScheduledTokens}
```

vLLM v0.20.1 源码中，`self.max_num_scheduled_tokens` 来自 `scheduler_config.max_num_scheduled_tokens`，如果未设置则使用 `scheduler_config.max_num_batched_tokens`；`schedule()` 中用 `token_budget = self.max_num_scheduled_tokens` 控制本轮能安排多少 token。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 2. Continuous batching 的直觉公式

传统静态 batch：

```math
\text{batch} = \{\text{同一时刻开始的一组请求}\}
```

Continuous batching：

```math
\text{batch}_t =
\{\text{本轮能推进的 running requests}\}
\cup
\{\text{本轮能接纳的 waiting requests}\}
```

也就是说，batch 不是固定的。每个 GPU step 都可以加入新请求、继续旧请求、暂停或抢占请求。

### 3. Chunked prefill 的直觉公式

如果一个 prompt 还剩 $P$ 个 token 没 prefill，而本轮剩余 token budget 是 $B$，则本轮只安排：

```math
\Delta_{\text{prefill}} = \min(P, B)
```

如果配置了 `long_prefill_token_threshold`，源码还会进一步限制单次调度的 token 数：

```math
\Delta_{\text{prefill}}
=
\min(P, B, \mathrm{longPrefillTokenThreshold})
```

vLLM v0.20.1 Scheduler 源码在 running 与 waiting 调度路径中都检查 `long_prefill_token_threshold`，当阈值大于 0 且小于 `num_new_tokens` 时，会把 `num_new_tokens` 截断到该阈值。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 4. Prefix caching 的直觉公式

如果新请求 prompt 是：

```text
prefix + suffix
```

且 prefix 对应的 KV blocks 已经在 cache 中，那么实际需要重新 prefill 的 token 数从：

```math
T_{\text{prompt}}
```

变成：

```math
T_{\text{prompt}} - T_{\text{cached}}
```

vLLM 官方 prefix caching 文档说明，vLLM 会缓存已处理请求的 KV-cache blocks，新请求有相同 prefix 时复用这些 blocks；vLLM V1 使用 hash-based 方法，block hash 由 parent hash、block tokens，以及 LoRA ID、多模态输入 hash、cache salt 等 extra hashes 组成。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching/))

---

## 三、张量形状

本章调度层面不主要关心单个 tensor 的 `[B, T, H]`，而关心 **tokens、blocks、requests**。

### 1. Request 维度

```text
waiting:  尚未进入运行队列的请求
running:  当前正在被持续调度的请求
finished: 已完成或被 abort 的请求
```

vLLM v0.20.1 Scheduler 初始化中创建 `self.waiting`、`self.skipped_waiting`、`self.running`，并维护 `self.finished_req_ids`；它还用 `max_num_running_reqs` 限制同时运行请求数。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 2. Token budget 维度

每一轮 scheduler 输出：

```text
num_scheduled_tokens: dict[request_id, int]
total_num_scheduled_tokens: int
```

源码中 `schedule()` 会累计 `num_scheduled_tokens`，并断言 `total_num_scheduled_tokens <= self.max_num_scheduled_tokens`。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 3. KV block 维度

第七章讲过，KV cache 不是连续大数组，而是 blocks。vLLM v0.20.1 的 `KVCacheBlocks` 是 `KVCacheManager` 的 allocation result，作为 Scheduler 与 KVCacheManager 之间的接口；`blocks[i][j]` 表示第 $i$ 个 KV cache group 的第 $j$ 个 token block。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

因此调度器的真实输出不只是：

```text
本轮算多少 token
```

还包括：

```text
本轮哪些请求拿到了哪些 KV blocks
哪些 prefix blocks 被复用
哪些 requests 被 preempt
哪些 encoder inputs 被调度
哪些 speculative tokens 被调度
```

### 4. Qwen3.6 的特殊形状边界

Qwen3.6 有 10 个 Gated Attention 层和 30 个 Gated DeltaNet 层。Gated Attention 的 K/V 可以进入普通 KV block 视角；Gated DeltaNet 在 vLLM 中是 Mamba/GDN state 路径，Scheduler 源码里也出现了 `has_mamba_layers`、`mamba_cache_mode == "align"` 和 `_mamba_block_aligned_split(...)` 这类 hybrid / mamba 对齐逻辑。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

这说明对 Qwen3.6，调度张量形状要同时考虑：

```text
Attention KV blocks
GDN / Mamba state blocks
MoE routed expert metadata
MTP speculative token slots
multimodal encoder input budget
```

---

## 四、计算流程

### 1. 一个 scheduler step 的高层流程

vLLM v0.20.1 的 `schedule()` 大致可以读成：

```text
1. 初始化 token_budget
2. 先调度 running requests
3. 为 running requests 分配新增 KV slots
4. 如果 KV blocks 不足，可能 preempt 低优先级请求
5. 再调度 waiting requests
6. 对新请求检查 prefix cache hit
7. 对新请求分配 KV slots
8. 构造 SchedulerOutput
9. 更新 scheduler 内部状态
```

源码中 `schedule()` 先遍历 `self.running`，再在没有 preemptions 且未 pause 时调度 `waiting` / `skipped_waiting`；running 路径和 waiting 路径都会调用 `kv_cache_manager.allocate_slots(...)`。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 2. Continuous batching：先推进 running

对已经在生成的请求，调度器优先让它们继续推进。这个策略降低 inter-token latency，因为老请求不用等长 prompt 先完整 prefill 完。

源码中 running 调度路径会计算：

```text
num_new_tokens =
  request.num_tokens_with_spec
  + request.num_output_placeholders
  - request.num_computed_tokens
```

然后受 `long_prefill_token_threshold`、`token_budget`、`max_model_len`、encoder budget、Mamba block alignment、KV block 可用性等限制。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 3. Chunked prefill：长 prompt 分块进入

对 waiting 请求，源码会先尝试拿到已经 cache 的 prefix blocks：

```text
new_computed_blocks, num_new_local_computed_tokens =
    kv_cache_manager.get_computed_blocks(request)
```

然后剩余要计算的 token 数是：

```text
num_new_tokens = request.num_tokens - num_computed_tokens
```

如果超过 `long_prefill_token_threshold`，就截断；如果 chunked prefill 未启用且 `num_new_tokens > token_budget`，源码会停止调度该 waiting 请求；否则 `num_new_tokens = min(num_new_tokens, token_budget)`。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

官方优化文档也给出同样的策略描述：V1 中 chunked prefill 尽可能默认启用，调度策略优先 decode，然后在 `max_num_batched_tokens` 预算有剩余时安排 prefill；若 pending prefill 放不进预算，会自动 chunk。([docs.vllm.ai](https://docs.vllm.ai/en/latest/configuration/optimization/))

### 4. Prefix caching：先查已计算 blocks

prefix caching 的运行路径可以概括成：

```text
新请求进入 waiting
→ get_computed_blocks(request)
→ 根据 prompt block hash 查找已缓存 blocks
→ 命中的 blocks 视为已计算
→ allocate_slots 只为剩余 token 分配新 blocks
```

vLLM 官方 prefix caching 文档说明：新请求调度时，scheduler 会调用 `kv_cache_manager.get_computed_blocks()` 获取已经计算过的 blocks，然后调用 `allocate_slots()`；`allocate_slots()` 会计算新需要的 blocks、touch computed blocks、从 free queue 分配新 blocks，并在 block full 时加入 cache。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching/))

---

## 五、显存布局

### 1. Continuous batching 的显存视角

Continuous batching 提高 GPU 利用率，但它会让多个请求同时占用 cache：

```text
request A: 已生成 2K tokens
request B: prompt 32K 正在 chunked prefill
request C: 已生成 512 tokens
request D: prompt 128K 等待进入
```

显存压力不是单个请求决定的，而是：

```math
\text{cache usage}
=
\sum_i
\text{active tokens}_i
\times
\text{cache bytes per token}
```

vLLM metrics 文档列出 `vllm:kv_cache_usage_perc` 作为已使用 KV cache blocks 的比例，并暴露 `vllm:num_requests_running`、`vllm:num_requests_waiting`、TTFT、inter-token latency、prefill time、decode time 等指标。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/metrics/))

### 2. Chunked prefill 的显存视角

chunked prefill 不会让总 prompt 变短，但它能避免一次性把大 prompt 的全部计算塞进一个 step。它的主要价值是：

```text
把长 prefill 对 decode 的阻塞切小
让 compute-bound prefill 和 memory-bound decode 混合进同一个 batch
```

官方优化文档明确说，chunked prefill 可以通过更好平衡 compute-bound prefill 与 memory-bound decode 来改善吞吐和延迟。([docs.vllm.ai](https://docs.vllm.ai/en/latest/configuration/optimization/))

### 3. Prefix caching 的显存视角

prefix caching 不是“免费减少所有显存”。它是把已计算 KV blocks 留在 block pool / cache map 中，等待未来请求复用。

好处：

```text
相同 prefix 不需要重复 prefill
TTFT 下降
GPU compute 减少
```

代价：

```text
cached blocks 也占用 GPU cache 资源
block eviction 策略变重要
多租户场景要考虑 cache isolation
```

官方 prefix caching 文档说明，vLLM 只缓存 full blocks，并支持通过 `cache_salt` 限制 cache 共享范围以隔离多租户环境中的 timing side-channel 风险。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching/))

### 4. Qwen3.6 的 cache/state 显存边界

对 Qwen3.6：

```text
Gated Attention:
  prefix caching 复用的是 KV-cache blocks

Gated DeltaNet:
  需要看 GDN / Mamba state cache 的实现边界

Multimodal:
  prefix hash 需要区分图像/视频内容，不只是 placeholder tokens

MTP:
  speculative tokens 会进入 lookahead slot 预算
```

prefix caching 文档特别说明，多模态输入中 placeholder tokens 会被 image embeddings 替换，因此 prefix caching 需要把 image hash 编入 extra hash，以区分不同图像。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching/))

---

## 六、vLLM 源码映射

本章源码入口如下。

| 概念 | 文件路径 | 类/函数 | 说明 |
|---|---|---|---|
| Scheduler 主入口 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 一个调度 step 的核心 |
| running / waiting 队列 | `vllm/v1/core/sched/scheduler.py` | `self.running`, `self.waiting`, `self.skipped_waiting` | continuous batching 的请求队列 |
| token budget | `vllm/v1/core/sched/scheduler.py` | `self.max_num_scheduled_tokens`, `token_budget` | 控制本轮最多调度 tokens |
| chunked prefill | `vllm/v1/core/sched/scheduler.py` | `long_prefill_token_threshold`, `enable_chunked_prefill` | 限制长 prefill 单轮 token 数 |
| prefix cache lookup | `vllm/v1/core/sched/scheduler.py` | `kv_cache_manager.get_computed_blocks` | 新请求先查已计算 prefix blocks |
| slots 分配 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots` | 为新增 token / lookahead token / encoder token 分配 slots |
| block pool | `vllm/v1/core/block_pool.py` | `BlockPool` | free blocks、cached blocks、eviction |
| Mamba/GDN 对齐 | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` | hybrid 模型中对 Mamba state cache 做 block-aligned split |
| prefix cache block hash | `vllm/v1/core/block_pool.py` | `BlockHashToBlockMap`, `cached_block_hash_to_block` | hash 到 block 的缓存映射 |
| 指标 | vLLM metrics docs | `/metrics`, `vllm:*` | 观测 running、waiting、KV cache usage、TTFT、ITL 等 |

尤其要注意 Scheduler 源码中的一句注释：vLLM v0.20.1 调度器不硬编码“prefill phase / decode phase”，而是用 `num_computed_tokens` 追赶 `num_tokens_with_spec` 的统一抽象。这是理解 continuous batching、chunked prefill、prefix caching、spec decoding 同时存在的关键。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

---

## 七、NVIDIA GPU 执行视角

### 1. Decode 是 memory-bound

decode 阶段每个请求每步通常只生成 1 个 token，但要读历史 KV cache。对单个请求来说，GPU 并行度可能不够，SM 和 Tensor Core 吃不满。

Continuous batching 的作用是把多个请求的 decode token 合在一起：

```text
request A decode 1 token
request B decode 1 token
request C decode 1 token
...
合并成一个更大的 GPU batch
```

这样可以提高 GPU occupancy。

### 2. Prefill 是 compute-bound

prefill 一次处理多个 prompt tokens，GEMM 和 attention 工作量大，更容易吃满 Tensor Core。但长 prompt prefill 会占用大量 compute 时间，导致正在 decode 的请求等待。

Chunked prefill 的目的就是把长 prompt 切小：

```text
不要让 262K prompt 一口气占满 GPU step
而是把它拆成多个 chunk
穿插 decode 执行
```

官方文档也明确说，chunked prefill 能把 compute-bound prefill 和 memory-bound decode 放到同一 batch 中，从而改善 GPU 利用率。([docs.vllm.ai](https://docs.vllm.ai/en/latest/configuration/optimization/))

### 3. Qwen3.6 的 GPU 特殊性

Qwen3.6 的一个 step 不是单纯 dense Transformer：

```text
Gated DeltaNet:
  state update / chunk metadata / Mamba-like state

Gated Attention:
  paged KV read / attention backend

MoE:
  router / top-k / expert dispatch / expert GEMM

MTP:
  speculative token verification

Vision Encoder:
  多模态输入时还有 encoder compute / cache
```

Scheduler 源码中已经显式处理 `has_mamba_layers`、`need_mamba_block_aligned_split`、encoder budget、speculative config、`num_lookahead_tokens` 等变量，这说明调度器要服务的不是单一 dense attention workload。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 4. GPU 上的 trade-off

| 策略 | 对 TTFT | 对 TPOT / ITL | 对吞吐 | 对显存 |
|---|---:|---:|---:|---:|
| 更大 `max_num_batched_tokens` | 通常更好 | 可能变差 | 可能更好 | batch 临时压力更大 |
| 更小 `max_num_batched_tokens` | 可能变差 | 通常更好 | 不一定 | 更保守 |
| chunked prefill | 长 prompt TTFT 可能分摊 | 改善 decode 抖动 | 通常更稳 | 总 cache 不变 |
| prefix caching | 命中时显著改善 | 间接改善 | 命中 workload 更好 | cached blocks 占用资源 |
| MTP | 低并发 TPOT 可能改善 | 取决于接受率 | 高并发可能变差 | lookahead tokens 额外占 slots |

vLLM recipe 对 Qwen 系列也提示：MTP-1 在低并发、延迟敏感场景下可降低 TPOT，但在高负载下会牺牲吞吐。([docs.vllm.ai](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 八、优化前 vs 优化后

### 优化前：静态 batch + 整段 prefill

```text
请求 A 进来 → 完整 prefill → decode 到结束
请求 B 等待
请求 C 等待
```

问题：

| 问题 | 后果 |
|---|---|
| decode 单请求太小 | GPU 利用率低 |
| 长 prompt prefill 阻塞短请求 | TTFT 尾延迟高 |
| 相同 prefix 重复计算 | 浪费 prefill compute |
| KV cache 连续/粗暴分配 | 并发容量差 |
| MTP / 多模态 / GDN state 混在一起 | 调度复杂但不可见 |

### 优化后：vLLM V1 风格调度

```text
每个 step:
  1. 先推进 running requests
  2. 剩余 token budget 给 waiting prefills
  3. 长 prefill 自动切 chunk
  4. 新请求先查 prefix cache
  5. KV blocks 不足时可能 preempt
  6. 输出 SchedulerOutput 给 model runner
```

源码证据包括：

```text
Scheduler.schedule
KVCacheManager.allocate_slots
KVCacheManager.get_computed_blocks
BlockPool
_mamba_block_aligned_split
```

vLLM v0.20.1 源码确认 `schedule()` 会先调度 running requests，再调度 waiting requests；waiting 请求会先通过 `get_computed_blocks()` 获取本地 prefix cache 命中，再通过 `allocate_slots()` 分配 slots。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 对 Qwen3.6 的优化心智

错误心智：

```text
Qwen3.6 支持 262K，所以每个请求直接完整 prefill 262K。
```

正确心智：

```text
262K 只是模型能力上限；
服务系统要通过 continuous batching、chunked prefill、prefix caching、
PagedAttention、Mamba/GDN state 对齐、MTP slot budget
来控制 TTFT、TPOT、吞吐和显存。
```

---

## 九、工程落地

### 1. 基础 Qwen3.6 serving 命令

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

### 2. 调参顺序

建议按这个顺序调：

```text
1. 先固定 max_model_len
2. 再观察 KV cache usage 和 waiting/running requests
3. 再调 max_num_batched_tokens
4. 再打开 prefix caching
5. 再测试 MTP
6. 最后混入多模态 workload
```

不要一开始同时打开：

```text
262K context
高并发
MTP
多模态
prefix caching
最大 batch token
```

否则 TTFT、TPOT、OOM、preemption、GDN state 对齐和 MoE dispatch 会混在一起，排障困难。

### 3. Prefix caching 适合哪些 workload

适合：

```text
长 system prompt
长工具说明
RAG 文档前缀相同
多轮 agent 保留历史上下文
批量请求共享模板
```

不适合或收益低：

```text
每个请求 prompt 完全不同
用户输入很短
prefix 不按 block 对齐
cached blocks 很快被 eviction
```

vLLM 官方 prefix caching 文档说明只缓存 full blocks，因此 prefix 命中效果和 block 边界有关。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/prefix_caching/))

### 4. 观察指标

生产观测至少看：

```text
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:prefix_cache_queries
vllm:prefix_cache_hits
vllm:time_to_first_token_seconds
vllm:inter_token_latency_seconds
vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds
```

vLLM metrics 文档列出了这些 V1 Prometheus 指标，并说明它们用于观测引擎和请求级性能。([docs.vllm.ai](https://docs.vllm.ai/en/latest/design/metrics/))

### 5. 常见问题定位

| 现象 | 可能原因 | 优先检查 |
|---|---|---|
| TTFT 很高 | 长 prompt prefill 堵塞、prefix cache 未命中 | `request_prefill_time_seconds`, prefix hit rate |
| TPOT 抖动 | decode 被大 prefill chunk 拖慢 | `max_num_batched_tokens`, chunked prefill |
| running 少、waiting 多 | KV blocks 不足或 token budget 太小 | `kv_cache_usage_perc`, `num_requests_waiting` |
| MTP 开后吞吐下降 | speculative tokens 占 slots，接受率不够 | `num_speculative_tokens`, TPOT/tokens/s |
| 多模态请求拖慢 | vision encoder / multimodal hash / encoder budget | multimodal batch 与 text-only 分池 |
| prefix cache 命中低 | prefix 不一致、未满 block、extra hash 不同 | block size、cache salt、image hash |

---

## 十、本章总结

Continuous batching、chunked prefill、prefix caching 是 vLLM serving 的三根调度支柱：

```text
Continuous batching:
  每个 step 动态混合多个请求，提高 decode GPU 利用率。

Chunked prefill:
  把长 prompt 切成小块，避免长 prefill 阻塞 decode。

Prefix caching:
  复用相同 prefix 的 KV blocks，减少重复 prefill。
```

在 vLLM v0.20.1 源码里，它们不是三个孤立功能，而是统一落在 Scheduler 的 token 追赶模型里：

```text
num_computed_tokens
追赶
num_tokens_with_spec
```

同时，`KVCacheManager.allocate_slots()`、`get_computed_blocks()`、`BlockPool`、`long_prefill_token_threshold`、`num_lookahead_tokens`、encoder budget 和 Mamba block alignment 共同决定每一步能算什么。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

对 Qwen3.6，真正的工程重点是：

```text
262K 长上下文
+ Gated Attention KV blocks
+ Gated DeltaNet state
+ MoE dispatch
+ MTP speculative tokens
+ Vision Encoder / multimodal hash
```

这些共同决定 TTFT、TPOT、吞吐、显存和稳定性。

---

## 小白总结

vLLM 不会傻傻地让一个请求独占 GPU。

它会像餐厅一样：

```text
老请求继续上菜
新请求有空就接
长菜单拆开读
相同套餐前半部分直接复用
```

Continuous batching 让很多请求一起跑；chunked prefill 防止长 prompt 卡住别人；prefix caching 让相同前缀不用重复算。

---

## 工程师总结

vLLM v0.20.1 的 Scheduler 核心不是传统的“prefill 队列 + decode 队列”二分法，而是统一 token 进度模型：

```text
request.num_computed_tokens
request.num_tokens_with_spec
token_budget
KV block availability
encoder budget
lookahead tokens
Mamba/GDN alignment
```

源码重点：

```text
Scheduler.schedule:
  先 running
  后 waiting
  token_budget 控制本 step
  long_prefill_token_threshold 控制长 prefill chunk
  get_computed_blocks 支持 prefix caching
  allocate_slots 绑定 KV blocks / lookahead / encoder tokens
```

Qwen3.6 上，调度器必须同时兼顾 Gated Attention KV、Gated DeltaNet state、MTP、MoE 和多模态输入。

---

## 关键公式

**1. 请求剩余计算量**

```math
\Delta_i =
N^{\text{target}}_i
-
N^{\text{computed}}_i
```

**2. 每轮 token budget**

```math
\sum_i \Delta_i^{\text{scheduled}}
\leq
\mathrm{maxNumScheduledTokens}
```

**3. Chunked prefill**

```math
\Delta_{\text{prefill}}
=
\min(
P_{\text{remaining}},
\mathrm{tokenBudget},
\mathrm{longPrefillTokenThreshold}
)
```

**4. Prefix caching 节省量**

```math
T_{\text{recompute}}
=
T_{\text{prompt}}
-
T_{\text{cached}}
```

**5. KV cache 活跃占用粗略关系**

```math
\text{cache usage}
\propto
\sum_i
\text{active tokens}_i
\times
\text{cache bytes per token}
```

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Scheduler 主入口 | `vllm/v1/core/sched/scheduler.py` | `Scheduler`, `schedule` |
| running / waiting 队列 | `vllm/v1/core/sched/scheduler.py` | `self.running`, `self.waiting`, `self.skipped_waiting` |
| token budget | `vllm/v1/core/sched/scheduler.py` | `self.max_num_scheduled_tokens`, `token_budget` |
| chunked prefill 截断 | `vllm/v1/core/sched/scheduler.py` | `long_prefill_token_threshold` |
| chunked prefill 开关判断 | `vllm/v1/core/sched/scheduler.py` | `enable_chunked_prefill` |
| prefix cache lookup | `vllm/v1/core/sched/scheduler.py` | `kv_cache_manager.get_computed_blocks` |
| KV slots 分配 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.allocate_slots` |
| KV blocks 结果 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` |
| block pool | `vllm/v1/core/block_pool.py` | `BlockPool` |
| prefix block cache | `vllm/v1/core/block_pool.py` | `BlockHashToBlockMap`, `cached_block_hash_to_block` |
| Mamba/GDN 对齐 | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` |
| 指标观测 | vLLM metrics docs | `/metrics`, `vllm:*` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| vLLM v0.20.1 Scheduler 核心入口是 `vllm/v1/core/sched/scheduler.py` | `vllm/v1/core/sched/scheduler.py` | `Scheduler`, `schedule` | 源码直接确认 | 是 |
| Scheduler 中没有硬编码 prefill/decode phase，而是用 `num_computed_tokens` 追赶 `num_tokens_with_spec` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` 注释 | 源码直接确认 | 是 |
| Scheduler 用 `max_num_scheduled_tokens` / `max_num_batched_tokens` 形成本轮 token budget | `vllm/v1/core/sched/scheduler.py` | `Scheduler.__init__`, `schedule` | 源码直接确认 | 是 |
| Scheduler 先调度 running requests，再调度 waiting requests | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| running 和 waiting 路径都会调用 `kv_cache_manager.allocate_slots` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| waiting 新请求会调用 `kv_cache_manager.get_computed_blocks` 以获取本地 prefix cache 命中 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| `long_prefill_token_threshold` 会限制单轮调度的长 prefill token 数 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| 当 chunked prefill 未启用且 prefill 放不进 token budget，waiting 调度会停止 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 为 speculative decoding 设置 `num_lookahead_tokens`，并传入 `allocate_slots` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.__init__`, `schedule` | 源码直接确认 | 是 |
| Scheduler 对 hybrid / Mamba 模型有 `_mamba_block_aligned_split`，用于 Mamba cache mode align 下的 block 对齐切分 | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` | 源码直接确认 | 是 |
| `KVCacheBlocks` 是 KVCacheManager 的 allocation result，是 Scheduler 和 KVCacheManager 的接口 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` | 源码直接确认 | 是 |
| `KVCacheManager` 通过 coordinator 接入 block pool，并暴露 usage | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.__init__`, `usage` | 源码直接确认 | 是 |
| `BlockPool` 管理 KVCacheBlocks，负责 allocate、free、cache KV cache blocks | `vllm/v1/core/block_pool.py` | `BlockPool` | 源码直接确认 | 是 |
| `BlockPool` 维护 free block queue 和 cached block hash map | `vllm/v1/core/block_pool.py` | `BlockPool.__init__` | 源码直接确认 | 是 |
| vLLM 官方文档说明 chunked prefill 会将大 prefill 切小并与 decode batch 混合，V1 尽可能默认启用 | vLLM Optimization docs | Chunked Prefill | 官方文档确认 | 是 |
| vLLM 官方 prefix caching 文档说明 prefix caching 复用已处理请求的 KV-cache blocks，并使用 hash-based approach | vLLM Prefix Caching docs | Automatic Prefix Caching | 官方文档确认 | 是 |
| prefix caching 只缓存 full blocks | vLLM Prefix Caching docs | Note 1 | 官方文档确认 | 是 |
| prefix caching 的 hash 可包含 LoRA IDs、多模态 input hashes、cache salts 等 extra hashes | vLLM Prefix Caching docs | Hash components | 官方文档确认 | 是 |
| Qwen3.6 是 multimodal MoE，具有 gated delta networks 架构；vLLM recipe 给出 TP=8、262144 max model len、qwen3 reasoning parser 命令 | vLLM Qwen3.5/Qwen3.6 recipe | Running Qwen3.6 | recipe 确认 | 是 |
| Qwen3.6 hidden layout、Gated Attention、MoE、MTP、262K context | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 下 prefix caching 对 GDN state 的精确复用边界 | `vllm/v1/core/kv_cache_coordinator*`, `gdn_attn.py` | 待展开 | 待源码确认 | 否 |
| MTP `method="mtp"` 与 `method="qwen3_next_mtp"` 在 v0.20.1 中的完整解析路径 | speculative decoding config 源码 | 待展开 | 待源码确认 | 否 |

---

## 源码阅读作业

按这个顺序读：

**1. Scheduler 总体流程**

```text
vllm/v1/core/sched/scheduler.py
```

重点找：

```text
class Scheduler
def schedule(self)
self.running
self.waiting
self.skipped_waiting
token_budget
```

目标：看懂一次 scheduler step 如何构造 `SchedulerOutput`。

**2. Continuous batching**

```text
Scheduler.schedule
```

重点找：

```text
First, schedule the RUNNING requests
Next, schedule the WAITING requests
num_scheduled_tokens
scheduled_running_reqs
scheduled_new_reqs
scheduled_resumed_reqs
```

目标：理解为什么 vLLM 先推进正在 decode / running 的请求。

**3. Chunked prefill**

```text
Scheduler.schedule
```

重点找：

```text
long_prefill_token_threshold
enable_chunked_prefill
num_new_tokens = min(num_new_tokens, token_budget)
```

目标：理解长 prompt 如何被切 chunk。

**4. Prefix caching**

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
```

重点找：

```text
get_computed_blocks
allocate_slots
KVCacheBlocks
BlockPool
BlockHashToBlockMap
cached_block_hash_to_block
```

目标：理解 prefix blocks 如何命中、touch、复用、evict。

**5. Qwen3.6 hybrid 相关**

```text
vllm/v1/core/sched/scheduler.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
vllm/v1/attention/backends/gdn_attn.py
```

重点找：

```text
has_mamba_layers
need_mamba_block_aligned_split
_mamba_block_aligned_split
mamba_cache_mode
```

目标：确认 Qwen3.6 这种 GDN hybrid 模型不是普通 dense attention 模型。

---

## 实验验证作业

### 实验 1：观察 chunked prefill 对 TTFT / TPOT 的影响

固定模型：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

构造三类请求：

```text
A: 1K prompt + 512 output
B: 32K prompt + 512 output
C: 128K prompt + 512 output
```

观察：

```text
TTFT
TPOT / ITL
request_prefill_time
request_decode_time
num_requests_waiting
kv_cache_usage_perc
```

### 实验 2：prefix caching 命中实验

构造：

```text
共享 prefix: 16K tokens 的 system prompt / 文档
不同 suffix: 100 tokens 的用户问题
```

连续发送多次请求，比较：

```text
第一次 TTFT
第二次 TTFT
prefix_cache_queries
prefix_cache_hits
prefill tokens
```

目标：确认相同前缀减少重复 prefill。

### 实验 3：`max_num_batched_tokens` 调参实验

分别测试较小和较大 token budget：

```text
max_num_batched_tokens = 2048
max_num_batched_tokens = 8192
max_num_batched_tokens = 16384
```

观察：

```text
TTFT
TPOT
throughput tokens/s
GPU utilization
waiting queue
```

预期：小 budget 往往更照顾 decode ITL，大 budget 往往更有利于 prefill / TTFT / 总吞吐，但具体取决于 workload 和 GPU。

### 实验 4：MTP 与高并发

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
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

分别在低并发和高并发下记录：

```text
TPOT
tokens/s
kv_cache_usage_perc
waiting requests
accepted speculative tokens
```

目标：验证 MTP 的 latency / throughput trade-off。

---

## 3 个自测题

**1. vLLM v0.20.1 Scheduler 为什么说没有严格的 prefill phase / decode phase？**
因为它用统一的 `num_computed_tokens` 追赶 `num_tokens_with_spec` 模型来调度请求；prefill、decode、prefix caching、spec decode 都可以表示成“还差多少 token 没算”。

**2. Chunked prefill 为什么能改善 decode 延迟？**
因为长 prompt 不再一次性占满整个 GPU step，而是被切成 chunk；decode 请求可以优先被调度，并和 prefill chunk 混合执行。

**3. Prefix caching 为什么只缓存 full blocks？**
因为 vLLM 的 prefix cache 以 KV cache block 为基本单位，full block 可以稳定 hash、复用和管理；不完整 block 还会继续追加 tokens，不适合作为稳定 cache key。

---

## 下一章预告

第九章进入：

```text
MoE：Router、Expert、Shared Expert、Top-k、Expert Dispatch
```

我们会把 Qwen3.6 的 MLP 替换成真实 MoE 路径：

```text
hidden_states
→ router logits
→ top-8 routed experts
→ shared expert
→ expert GEMM
→ combine
```

重点回答：

```text
1. Router 到底算什么？
2. 256 experts 为什么每 token 只激活 8 + 1？
3. shared expert 和 routed expert 有什么区别？
4. expert dispatch 为什么会成为 GPU / NCCL / kernel 优化难点？
5. vLLM v0.20.1 的 Qwen3NextSparseMoeBlock 与 FusedMoE 如何映射？
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM v0.20.1 Scheduler source](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py)
- [vLLM v0.20.1 KV cache manager source](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py)
