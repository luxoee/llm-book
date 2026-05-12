# 第十二章：vLLM Scheduler、Block Manager、Attention Backend 源码导读

本章把前面几章串成一条 **vLLM v0.20.1 推理执行链**：

```text
Request
→ Scheduler.schedule()
→ KVCacheManager.allocate_slots()
→ BlockPool / KVCacheBlocks
→ Attention metadata builder
→ Attention backend
→ KV cache update / GDN state update
→ ModelRunner 执行 Qwen3.6 layer
```

先澄清一个版本边界：在 vLLM v0.20.1 中，本课程不使用旧版记忆里的 `BlockSpaceManager` 心智，而是以 `vllm/v1/core/kv_cache_manager.py`、`vllm/v1/core/block_pool.py`、`vllm/v1/core/sched/scheduler.py` 为准。`KVCacheManager` 通过 coordinator 接入 `block_pool`，`KVCacheBlocks` 是 Scheduler 与 KVCacheManager 之间的 allocation result 接口；`BlockPool` 管理 KV cache blocks、free block queue 和 prefix cache block map。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

对固定案例 **Qwen/Qwen3.6-35B-A3B**，Qwen 模型卡确认它是带 Vision Encoder 的 causal LM，语言主干为 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，共 40 层，Gated Attention 是 16 Q heads / 2 KV heads / head dim 256 / RoPE dim 64，MoE 是 256 experts、8 routed + 1 shared expert，原生上下文 262,144 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

## 本章解决什么问题

- 把前面章节的概念落到 vLLM v0.20.1 的 scheduler、KV cache manager、block pool 和 attention backend 源码路径上。
- 帮你从日志或 profiler 结果反查应该读哪段源码，而不是在仓库里盲搜。
- 说明 vLLM V1 中 prefill/decode、prefix cache、lookahead slots、GDN state 之间的接口边界。
- 为第十三章把源码理解转成部署配置决策做准备。

---

## 一、生活类比

把 vLLM 想成一个大型机场调度系统。

用户请求像一架架飞机：

```text
请求 A：已经起飞，正在 decode
请求 B：刚到跑道，长 prompt 需要 prefill
请求 C：有相同 prefix，可以复用之前的跑道记录
请求 D：MTP speculative decoding，需要预留额外 lookahead slots
请求 E：多模态，需要 encoder budget
```

Scheduler 像塔台：

```text
本轮 GPU 能处理多少 token？
哪些 running requests 先推进？
哪些 waiting requests 可以进来？
哪些长 prompt 要切 chunk？
KV blocks 够不够？
不够时要不要 preempt 某个请求？
```

KVCacheManager 像停机位管理器：

```text
这个请求新增 token 需要几个 cache slots？
哪些 prefix blocks 已经算过？
哪些 blocks 可以复用？
哪些 blocks 要新分配？
```

BlockPool 像停机位池：

```text
哪些 block 空闲？
哪些 block 被 running request 引用？
哪些 block 是 cached prefix？
哪些 block 可以 eviction？
```

Attention Backend 像真正的跑道执行系统：

```text
拿到 block_table / slot_mapping
把新的 K/V 写入 cache
从 paged KV cache 读历史 K/V
执行 FlashAttention / FlashInfer / Triton attention
```

对 Qwen3.6，还多一条特殊跑道：Gated DeltaNet 不走普通 softmax attention KV cache，而是走 GDN/Mamba-style state 和 `GDNAttentionMetadata` 路径。vLLM v0.20.1 的 GDN backend 明确声明 `GDNAttentionBackend.get_name() = "GDN_ATTN"`，且 `is_ssm()` 返回 true。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/gdn_attn.py))

---

## 二、数学直觉

vLLM v0.20.1 Scheduler 的核心不是“先 prefill 队列，再 decode 队列”的硬切分。源码注释明确说：Scheduler 中没有严格的 decoding phase 或 prefill phase；每个 request 只有 `num_computed_tokens` 和 `num_tokens_with_spec`，每一步都让前者追赶后者，这个抽象覆盖 chunked prefill、prefix caching、speculative decoding 和未来的 jump decoding。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

可以写成：

```math
\Delta_i =
N^{\mathrm{withSpec}}_i
-
N^{\text{computed}}_i
```

其中：

```text
N_computed      = 已经真正算完的 token 数
N_with_spec     = prompt tokens + output tokens + speculative tokens
Delta           = 当前 request 还需要追赶的 token 数
```

每个 scheduler step 有 token budget：

```math
\sum_i \Delta^{\text{scheduled}}_i
\le
\mathrm{maxNumScheduledTokens}
```

源码中 `max_num_scheduled_tokens` 来自 scheduler config；若没有单独设置，则 fallback 到 `max_num_batched_tokens`。`schedule()` 每轮用 `token_budget = self.max_num_scheduled_tokens` 控制本轮最多安排多少 token。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

对 KV cache，核心映射是：

```math
\text{Request tokens}
\rightarrow
\text{logical blocks}
\rightarrow
\text{physical KVCacheBlock}
\rightarrow
\mathrm{blockTable}/\mathrm{slotMapping}
\rightarrow
\text{Attention backend}
```

`KVCacheBlocks` 的源码注释说明，它是 `KVCacheManager` 的 allocation result，是 Scheduler 与 KVCacheManager 的接口；`blocks[i][j]` 表示第 i 个 KV cache group 的第 j 个 token block。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

---

## 三、张量形状

### 1. Scheduler 层面不是 `[B, T, H]`

Scheduler 主要处理这些“调度张量”：

```text
num_scheduled_tokens: dict[request_id, int]
req_to_new_blocks:   dict[request_id, KVCacheBlocks]
scheduled_encoder_inputs: dict[request_id, list[int]]
scheduled_spec_decode_tokens: dict[request_id, list[int]]
```

源码中 `schedule()` 会维护 `scheduled_new_reqs`、`scheduled_resumed_reqs`、`scheduled_running_reqs`、`preempted_reqs`、`req_to_new_blocks`、`num_scheduled_tokens`、`scheduled_encoder_inputs` 和 `scheduled_spec_decode_tokens`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 2. KVCacheBlocks 形状

`KVCacheBlocks` 是：

```text
blocks: tuple[Sequence[KVCacheBlock], ...]
```

外层 tuple 对应 KV cache groups，内层 sequence 对应该 group 中的 token blocks。源码还提供 `get_block_ids()`，把 `KVCacheBlocks` 转成 block id 列表，返回结构是 `tuple[list[int], ...]`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

### 3. FlashAttention metadata 形状

FlashAttention backend 的 metadata 中包含：

```text
num_actual_tokens
max_query_len
query_start_loc
max_seq_len
seq_lens
block_table
slot_mapping
```

其中 `block_table` 用于从逻辑序列定位物理 KV blocks，`slot_mapping` 用于把本轮新产生的 K/V 写入正确 cache slot。源码中 `FlashAttentionMetadata` 明确包含 `block_table: torch.Tensor` 和 `slot_mapping: torch.Tensor`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

### 4. GDN metadata 形状

GDN backend 的 metadata 与普通 attention 不同。`GDNAttentionMetadata` 包含：

```text
num_prefills
num_prefill_tokens
num_decodes
num_decode_tokens
num_spec_decodes
num_spec_decode_tokens
has_initial_state
spec_state_indices_tensor
non_spec_state_indices_tensor
chunk_indices
chunk_offsets
nums_dict
batch_ptr
token_chunk_offset_ptr
```

这说明 GDN 路径关注 state indices、spec decode state、chunk metadata 和 causal conv metadata，而不是普通 attention 的 `K/V block_table + softmax attention` 一套抽象。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/gdn_attn.py))

---

## 四、计算流程

### 1. Scheduler 一轮的主流程

vLLM v0.20.1 `Scheduler.schedule()` 的核心流程是：

```text
1. 初始化 token_budget
2. kv_cache_manager.new_step_starts()
3. 先调度 running requests
4. 对 running requests 调用 allocate_slots()
5. KV blocks 不足时，按策略 preempt 某个 running request
6. 再调度 waiting / skipped_waiting requests
7. 对新请求先 get_computed_blocks()，获取 prefix cache hit
8. 再 allocate_slots()
9. 记录 scheduled_encoder_inputs / scheduled_spec_decode_tokens
10. 返回 SchedulerOutput
```

源码中明确有 “First, schedule the RUNNING requests.”，随后才是 “Next, schedule the WAITING requests.”；running 和 waiting 路径都会调用 `kv_cache_manager.allocate_slots(...)`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 2. Running request 调度

running 请求当前可能处于 decode，也可能处于 chunked prefill 的中间。源码用：

```text
num_new_tokens =
    request.num_tokens_with_spec
    + request.num_output_placeholders
    - request.num_computed_tokens
```

再经过：

```text
long_prefill_token_threshold
token_budget
max_model_len
encoder budget
mamba block aligned split
KV slot availability
```

这些限制后，才真正安排本轮 token。源码还显示，如果启用 Mamba/GDN align mode，会调用 `_mamba_block_aligned_split()`，原因是 Mamba state cache 在 prefill 阶段需要 block-aligned splitting。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 3. Waiting request 调度

waiting 请求如果 `num_computed_tokens == 0`，会先尝试 prefix cache：

```text
new_computed_blocks, num_new_local_computed_tokens =
    kv_cache_manager.get_computed_blocks(request)
```

`KVCacheManager.get_computed_blocks()` 的注释说明，它返回“computed blocks”和“computed token 数”；并且 computed blocks 必须是 full blocks。源码还明确说明，如果全部 token 都命中 cache，仍然必须重算最后一个 token 以获得 logits。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

### 4. KV slot 分配

`KVCacheManager.allocate_slots()` 的职责是给 request 追加新 tokens 分配 slots。它的参数包括：

```text
request
num_new_tokens
num_new_computed_tokens
new_computed_blocks
num_lookahead_tokens
num_external_computed_tokens
delay_cache_blocks
num_encoder_tokens
```

源码注释明确指出：`num_lookahead_tokens` 是为带 KV-cache 的 speculative decode proposer 分配 speculative tokens 使用。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

### 5. Attention backend 执行

以 FlashAttention 为例，metadata builder 从 common attention metadata 中取：

```text
num_reqs
num_actual_tokens
max_query_len
max_seq_len
query_start_loc
seq_lens
block_table_tensor
slot_mapping
```

然后构造 `FlashAttentionMetadata`。这一步把 Scheduler / KVCacheManager 的 block 结果转换成 backend 可执行的 attention metadata。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

---

## 五、显存布局

### 1. BlockPool 是显存 block 池

`BlockPool` 源码注释说明：它管理 `KVCacheBlocks`，提供 allocate、free、cache KV cache blocks 的方法；`free_block_queue` 按 eviction order 存 free blocks，`cached_block_hash_to_block` 用 block hash 找 cached blocks。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/block_pool.py))

这对应前面 PagedAttention 的显存心智：

```text
逻辑 token 连续
物理 KV block 分散
block_table 负责映射
BlockPool 负责分配、释放、缓存、驱逐
```

### 2. Prefix caching 的 full block 约束

`BlockPool.cache_full_blocks()` 的注释说明，它缓存的是 full blocks；函数会更新 block hash metadata，并把 full block 插入 `cached_block_hash_to_block`。源码还说明一些 sparse attention 或 Mamba align mode 下的 null blocks 会被跳过。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/block_pool.py))

### 3. FlashAttention KV cache layout

FlashAttention backend 的 `get_kv_cache_shape()` 返回：

```text
(2, num_blocks, block_size, num_kv_heads, head_size)
```

其中第一维 2 对应 K/V。源码还根据 cache layout 返回不同 stride order，并要求 block size 是 16 的倍数。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

### 4. FlashInfer KV cache layout

FlashInfer backend 的 `get_kv_cache_shape()` 返回：

```text
(num_blocks, 2, block_size, num_kv_heads, head_size)
```

如果是 nvfp4，还有 packed layout。FlashInfer backend 支持 `auto`、`float16`、`bfloat16`、`fp8`、`fp8_e4m3`、`fp8_e5m2` 这些 KV cache dtype，并列出支持的 kernel block sizes 为 `[16, 32, 64]`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/flashinfer.py))

### 5. Qwen3.6 的特殊显存边界

Qwen3.6 的普通 Gated Attention KV cache 只覆盖 10 个 full-attention 层；30 个 Gated DeltaNet 层走 GDN/Mamba state。`Qwen3NextDecoderLayer` 源码中，`linear_attention` 使用 `GatedDeltaNetAttention`，`full_attention` 使用 `Qwen3NextAttention`；`Qwen3NextForCausalLM` 还明确通过 `MambaStateShapeCalculator.gated_delta_net_state_shape(...)` 计算 GDN state shape。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 六、vLLM 源码映射

本章最重要的源码链路如下：

| 阶段 | 文件路径 | 类/函数 |
|---|---|---|
| 调度入口 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` |
| token budget | `vllm/v1/core/sched/scheduler.py` | `self.max_num_scheduled_tokens`, `token_budget` |
| running / waiting 队列 | `vllm/v1/core/sched/scheduler.py` | `self.running`, `self.waiting`, `self.skipped_waiting` |
| Mamba/GDN 对齐 | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` |
| prefix cache lookup | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` |
| KV slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| KV block 结果 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` |
| block pool | `vllm/v1/core/block_pool.py` | `BlockPool` |
| prefix block map | `vllm/v1/core/block_pool.py` | `BlockHashToBlockMap` |
| FlashAttention metadata | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata`, `FlashAttentionMetadataBuilder` |
| FlashInfer metadata | `vllm/v1/attention/backends/flashinfer.py` | `FlashInferMetadata` |
| GDN metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadata`, `GDNAttentionMetadataBuilder` |
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextDecoderLayer` |
| Qwen3.6 full attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Qwen3.6 GDN | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Qwen3.6 MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |

这里有一个关键版本事实：`Qwen3NextDecoderLayer` 源码中按 `layer_type` 明确分流，`linear_attention` 创建 `GatedDeltaNetAttention`，`full_attention` 创建 `Qwen3NextAttention`；同一层后面再根据 MoE 配置创建 `Qwen3NextSparseMoeBlock` 或 dense MLP。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 七、NVIDIA GPU 执行视角

### 1. Scheduler 不直接跑 kernel

Scheduler 的产物不是 CUDA kernel，而是“本轮该跑什么”的执行计划：

```text
哪些 request
每个 request 算多少 token
分配了哪些 KV blocks
哪些 prefix blocks 命中
哪些 encoder inputs 被安排
哪些 speculative tokens 被安排
```

这些信息进入 model runner 后，才进一步构造 attention metadata，最终触发 FlashAttention、FlashInfer、Triton、GDN、MoE kernels。

### 2. FlashAttention / FlashInfer 的核心差异

FlashAttention backend 的 KV cache shape 是 `(2, num_blocks, block_size, num_kv_heads, head_size)`；FlashInfer backend 的 KV cache shape 是 `(num_blocks, 2, block_size, num_kv_heads, head_size)`，并且 FlashInfer 支持更多 KV cache dtype，包括 FP8。两者都用 block / page 思想处理 paged KV cache，但具体 layout、metadata 和 kernel wrapper 不同。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

### 3. GDN backend 不是 KV attention backend

GDN backend 的 builder 会先通过 `mamba_get_block_table_tensor(...)` 处理 Mamba/GDN state 的 block table，再区分 spec decode、non-spec decode、prefill，并在 prefill 时准备 `chunk_indices` 和 `chunk_offsets`。源码注释明确说 prefill batches 使用 FLA chunk ops，并预先在 CPU 上计算再异步拷贝到 GPU，以避免 `prepare_chunk_indices` 中的 GPU→CPU sync。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/gdn_attn.py))

### 4. MoE kernel 在 Scheduler 后面爆发

Scheduler 的 `max_num_batched_tokens` 决定一轮里合并多少 token；这个 N 直接影响 MoE 的 router logits `[N, 256]`、top-k `[N, 8]`、expert dispatch 和 expert GEMM。`Qwen3NextSparseMoeBlock` 中 router 是 `ReplicatedLinear(hidden_size, num_experts)`，输出 `router_logits` 后调用 `self.experts(...)`，而 `self.experts` 是 `FusedMoE`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 5. 真正的 GPU 性能瓶颈要靠 profiler 确认

源码可以告诉我们“可能路径”，但不能替代运行日志和 profiler。比如 `flash_attn.py`、`flashinfer.py`、`triton_attn.py`、`gdn_attn.py` 都存在，不代表你的机器一定选择某一个 backend；backend 选择依赖 GPU capability、dtype、KV cache dtype、block size、模型结构、环境变量和 vLLM runtime config。源码只能确认候选路径和接口。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

---

## 八、优化前 vs 优化后

### 优化前：错误的旧版心智

```text
Request
→ prefill queue
→ decode queue
→ BlockSpaceManager
→ one attention kernel
```

这对 vLLM v0.20.1 + Qwen3.6 是不够准确的。

原因：

```text
1. Scheduler 不硬分 prefill/decode phase
2. KVCacheManager / BlockPool 才是 v1 cache 管理核心
3. Qwen3.6 同时有 Attention KV cache 和 GDN state
4. Attention backend 不止一种
5. MTP / encoder budget / prefix cache 都进入 schedule()
```

### 优化后：v0.20.1 正确链路

```text
Scheduler.schedule()
  → running requests first
  → waiting requests second
  → token_budget
  → long_prefill_token_threshold
  → get_computed_blocks()
  → allocate_slots()
  → possible preemption
  → encoder inputs
  → speculative tokens

KVCacheManager
  → KVCacheBlocks
  → coordinator
  → BlockPool

BlockPool
  → free_block_queue
  → cached_block_hash_to_block
  → full block caching
  → block eviction / reuse

Attention Backend
  → CommonAttentionMetadata
  → FlashAttentionMetadata / FlashInferMetadata / GDNAttentionMetadata
  → block_table / slot_mapping / state_indices / chunk_indices
```

### 对 Qwen3.6 的优化后心智

```text
Gated Attention:
  走 Qwen3NextAttention → Attention → paged KV attention backend

Gated DeltaNet:
  走 GatedDeltaNetAttention → GDNAttentionMetadata → GDN state/chunk backend

MoE:
  走 Qwen3NextSparseMoeBlock → FusedMoE

MTP:
  通过 num_spec_tokens / num_lookahead_tokens 进入 scheduler 和 cache budget

多模态:
  通过 encoder budget / encoder cache manager 进入 scheduler
```

---

## 九、工程落地

### 1. 阅读运行日志时先找这些关键词

```text
max_num_scheduled_tokens
max_num_batched_tokens
enable_chunked_prefill
long_prefill_token_threshold
enable_prefix_caching
num_gpu_blocks
kv_cache_usage
mamba_cache_mode
GDN_ATTN
FLASH_ATTN
FLASHINFER
slot_mapping
block_table
speculative_config
num_lookahead_tokens
```

### 2. Qwen3.6 推荐先按 recipe 建立基线

vLLM Qwen recipe 给出的 Qwen3.6 基础命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

MTP 示例命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

这些命令来自 vLLM Qwen3.5/Qwen3.6 recipe；正式压测时，建议先不开 MTP，先建立 TP=8、262K、text-only 或 multimodal 的基线。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 3. 排查 Scheduler 相关问题

| 现象 | 优先检查 |
|---|---|
| TTFT 高 | waiting 队列、prefix cache hit、chunked prefill |
| TPOT 抖动 | running requests 是否被长 prefill chunk 干扰 |
| waiting 很多 | token budget 太小、KV blocks 不足、encoder budget 耗尽 |
| preemption 多 | KV blocks 不足、max_model_len/并发过高 |
| MTP 后显存紧 | `num_lookahead_tokens` 增加 cache slot 预算 |
| GDN 相关 cache miss | `mamba_cache_mode`、block-aligned split |

源码中 `_mamba_block_aligned_split()` 明确说明：为了启用 Mamba state 的 block-aligned caching，prefill 阶段 `num_new_tokens` 必须是 `block_size` 的倍数；如果小于 block size，则 state 不缓存。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 4. 排查 Attention Backend 问题

| 现象 | 优先检查 |
|---|---|
| decode attention 慢 | backend 是否合适、KV cache dtype、block size、context length |
| backend 不符合预期 | 运行日志、GPU capability、dtype、env vars |
| KV 写入异常 | `slot_mapping`、block_table、cache layout |
| FP8 KV 问题 | backend 是否支持 FP8 KV cache |
| GDN 慢 | GDN prefill backend 是 FlashInfer 还是 Triton/FLA |

`GatedDeltaNetAttention` 中的 `ChunkGatedDeltaRule` 会根据 `additional_config["gdn_prefill_backend"]` 和平台 capability 选择 FlashInfer GDN prefill kernel 或 Triton/FLA；源码还写明如果选择 FlashInfer 但平台不支持，会 warning 并 fallback 到 Triton/FLA。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

### 5. 排查 MoE 问题

| 现象 | 优先检查 |
|---|---|
| MoE kernel 时间高 | expert token imbalance、batch token 数、FusedMoE backend |
| 某 GPU 热点 | EP expert placement、EPLB |
| 高并发吞吐不升 | all-to-all / dispatch / combine 时间 |
| batch 太小 | continuous batching 是否有效，`max_num_batched_tokens` 是否过小 |
| shared expert 开销明显 | shared expert 路径是否成为 dense-like 成本 |

---

## 十、本章总结

第十二章的核心链路是：

```text
Scheduler.schedule()
→ KVCacheManager.get_computed_blocks()
→ KVCacheManager.allocate_slots()
→ BlockPool
→ KVCacheBlocks
→ Attention metadata builder
→ FlashAttention / FlashInfer / GDN backend
→ Qwen3NextAttention / GatedDeltaNetAttention / FusedMoE
```

对 vLLM v0.20.1，不要用旧版 Block Manager 记忆直接解释。当前源码事实是：

```text
Scheduler:
  vllm/v1/core/sched/scheduler.py

KV cache manager:
  vllm/v1/core/kv_cache_manager.py

Block pool:
  vllm/v1/core/block_pool.py

Attention backends:
  vllm/v1/attention/backends/

Qwen3.6 model path:
  qwen3_next.py
  gdn_linear_attn.py
```

对 Qwen3.6，尤其要记住：

```text
Gated Attention:
  paged KV cache + attention backend

Gated DeltaNet:
  Mamba/GDN state + GDN metadata + chunk metadata

MoE:
  FusedMoE + router + expert dispatch

MTP:
  num_spec_tokens / num_lookahead_tokens 影响 scheduler 和 cache budget

多模态:
  encoder budget / encoder cache manager 进入 scheduler
```

---

## 小白总结

vLLM 像一个调度系统。

Scheduler 决定“这一轮谁上 GPU、算多少 token”；KVCacheManager 决定“这些 token 的历史笔记放到哪些 block”；BlockPool 管理这些 block 的借出、归还和复用；Attention Backend 拿到 block table 和 slot mapping 后，真正去读写 KV cache 并执行 attention。

Qwen3.6 更复杂，因为它不是普通 Transformer：一部分层走 Gated Attention 的 KV cache，一部分层走 Gated DeltaNet 的 state，所有层还接 MoE。

---

## 工程师总结

源码主线：

```text
Scheduler.schedule
  running first
  waiting second
  token_budget
  prefix cache lookup
  allocate_slots
  preemption
  encoder budget
  spec decode tokens
  mamba block alignment

KVCacheManager
  get_computed_blocks
  allocate_slots
  KVCacheBlocks
  block_pool

BlockPool
  FreeKVCacheBlockQueue
  BlockHashToBlockMap
  cache_full_blocks
  full block prefix caching

Attention Backend
  FlashAttentionMetadata: block_table + slot_mapping
  FlashInferMetadata: prefill/decode paged KV wrapper metadata
  GDNAttentionMetadata: state_indices + chunk_indices + spec metadata
```

对 Qwen3.6 的源码映射：

```text
full_attention:
  Qwen3NextAttention

linear_attention:
  GatedDeltaNetAttention

MoE:
  Qwen3NextSparseMoeBlock + FusedMoE

Hybrid layer:
  Qwen3NextDecoderLayer
```

---

## 关键公式

**1. Scheduler token gap**

```math
\Delta_i =
N^{\mathrm{withSpec}}_i
-
N^{\text{computed}}_i
```

**2. 本轮 token budget**

```math
\sum_i \Delta^{\text{scheduled}}_i
\le
\mathrm{maxNumScheduledTokens}
```

**3. Prefix cache recompute tokens**

```math
T_{\text{recompute}}
=
T_{\text{request}}
-
T_{\text{cached}}
```

**4. Block id 映射**

```math
\mathrm{physicalBlock}
=
\mathrm{blockTable}[\text{request}, \mathrm{logicalBlock}]
```

**5. Slot mapping**

```math
\mathrm{cacheSlot}
=
\mathrm{slotMapping}[\mathrm{newTokenIndex}]
```

**6. Qwen3.6 cache 拆账**

```math
\text{Cache/State}
=
\text{GatedAttention KV}
+
\text{GatedDeltaNet State}
+
\text{Speculative Lookahead}
+
\text{Encoder/Multimodal Cache}
```

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` |
| Running / waiting 队列 | `vllm/v1/core/sched/scheduler.py` | `self.running`, `self.waiting`, `self.skipped_waiting` |
| Mamba/GDN 对齐 | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| Prefix cache lookup | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` |
| Slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| KV block result | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` |
| Block pool | `vllm/v1/core/block_pool.py` | `BlockPool` |
| Prefix block hash map | `vllm/v1/core/block_pool.py` | `BlockHashToBlockMap` |
| FlashAttention backend | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionBackend`, `FlashAttentionMetadataBuilder` |
| FlashInfer backend | `vllm/v1/attention/backends/flashinfer.py` | `FlashInferBackend`, `FlashInferMetadata` |
| GDN backend | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionBackend`, `GDNAttentionMetadataBuilder` |
| Qwen3.6 hybrid layer | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextDecoderLayer` |
| Qwen3.6 full attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Qwen3.6 GDN | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Qwen3.6 MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| vLLM v0.20.1 Scheduler 主入口是 `vllm/v1/core/sched/scheduler.py` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 不硬分 prefill/decode phase，而是让 `num_computed_tokens` 追赶 `num_tokens_with_spec` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` 注释 | 源码直接确认 | 是 |
| Scheduler 先调度 running requests，再调度 waiting requests | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 通过 `token_budget = self.max_num_scheduled_tokens` 控制本轮 token 数 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 在 hybrid/Mamba align mode 下会调用 `_mamba_block_aligned_split` | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` | 源码直接确认 | 是 |
| `KVCacheBlocks` 是 KVCacheManager 的 allocation result，是 Scheduler 与 KVCacheManager 的接口 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` | 源码直接确认 | 是 |
| `KVCacheManager` 通过 coordinator 接入 `block_pool` | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.__init__` | 源码直接确认 | 是 |
| `get_computed_blocks` 返回 computed full blocks 和 computed token 数 | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` | 源码直接确认 | 是 |
| `allocate_slots` 为新增 token、speculative lookahead tokens、encoder tokens 等分配 slots | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 源码直接确认 | 是 |
| `BlockPool` 管理 KVCacheBlocks，提供 allocate、free、cache blocks | `vllm/v1/core/block_pool.py` | `BlockPool` | 源码直接确认 | 是 |
| `BlockPool` 维护 free block queue 和 cached block hash map | `vllm/v1/core/block_pool.py` | `BlockPool.__init__` | 源码直接确认 | 是 |
| `cache_full_blocks` 只缓存 full blocks，并更新 block hash metadata | `vllm/v1/core/block_pool.py` | `cache_full_blocks` | 源码直接确认 | 是 |
| FlashAttention metadata 包含 `block_table` 和 `slot_mapping` | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata` | 源码直接确认 | 是 |
| FlashAttention KV cache shape 是 `(2, num_blocks, block_size, num_kv_heads, head_size)` | `vllm/v1/attention/backends/flash_attn.py` | `get_kv_cache_shape` | 源码直接确认 | 是 |
| FlashInfer backend 支持 FP16/BF16 计算和 auto/FP16/BF16/FP8 KV cache dtype | `vllm/v1/attention/backends/flashinfer.py` | `FlashInferBackend` | 源码直接确认 | 是 |
| FlashInfer KV cache shape 是 `(num_blocks, 2, block_size, num_kv_heads, head_size)` | `vllm/v1/attention/backends/flashinfer.py` | `get_kv_cache_shape` | 源码直接确认 | 是 |
| GDN backend 名称为 `GDN_ATTN`，且 `is_ssm()` 返回 true | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionBackend` | 源码直接确认 | 是 |
| GDN metadata 包含 spec/non-spec state indices、chunk indices、chunk offsets、causal conv metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadata` | 源码直接确认 | 是 |
| GDN prefill 会准备 FLA chunk metadata，避免 GPU→CPU sync | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder.build` | 源码直接确认 | 是 |
| `Qwen3NextDecoderLayer` 按 `layer_type` 分流到 GatedDeltaNetAttention 或 Qwen3NextAttention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextDecoderLayer` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 使用 QKVParallelLinear、RowParallelLinear、RoPE 和 Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 是 PluggableLayer + MambaBase，state shape 由 MambaStateShapeCalculator 计算 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 创建 router、shared expert 和 FusedMoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | 源码直接确认 | 是 |
| Qwen3.6 结构为 Vision Encoder + Gated DeltaNet + Gated Attention + MoE + MTP，原生上下文 262K | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 vLLM recipe 提供 TP=8、262144 max model len 和 MTP speculative decoding 示例 | vLLM Qwen recipe | serving commands | recipe 确认 | 是 |
| 当前机器上实际选择 FlashAttention、FlashInfer、Triton 或其他 backend | 运行日志 / profiler | 待运行确认 | 待运行确认 | 否 |
| 当前机器上 GDN prefill backend 实际为 FlashInfer 还是 Triton/FLA | 运行日志 / `gdn_prefill_backend` | 待运行确认 | 待运行确认 | 否 |
| Qwen3.6 MTP 在 v0.20.1 中 `method="mtp"` 与 `method="qwen3_next_mtp"` 的完整解析差异 | speculative decoding config 源码 | 待展开 | 待源码确认 | 否 |

---

## 源码阅读作业

**1. Scheduler 主线**

读：

```text
vllm/v1/core/sched/scheduler.py
```

重点找：

```text
Scheduler.__init__
Scheduler.schedule
token_budget
self.running
self.waiting
long_prefill_token_threshold
num_lookahead_tokens
_mamba_block_aligned_split
```

目标：理解一轮 scheduler step 产生了什么。

**2. KV cache 管理**

读：

```text
vllm/v1/core/kv_cache_manager.py
```

重点找：

```text
KVCacheBlocks
get_computed_blocks
can_fit_full_sequence
allocate_slots
usage
```

目标：理解 prefix cache hit、slot allocation、lookahead tokens 如何进入 cache budget。

**3. BlockPool**

读：

```text
vllm/v1/core/block_pool.py
```

重点找：

```text
BlockPool
BlockHashToBlockMap
free_block_queue
cached_block_hash_to_block
cache_full_blocks
get_cached_block
```

目标：理解 full block prefix caching 和 block lifecycle。

**4. FlashAttention / FlashInfer**

读：

```text
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/flashinfer.py
```

重点找：

```text
FlashAttentionMetadata
FlashInferMetadata
block_table
slot_mapping
get_kv_cache_shape
get_supported_kernel_block_sizes
supported_kv_cache_dtypes
```

目标：理解普通 Gated Attention KV cache 如何进入 backend。

**5. GDN backend**

读：

```text
vllm/v1/attention/backends/gdn_attn.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
```

重点找：

```text
GDNAttentionBackend
GDNAttentionMetadata
mamba_get_block_table_tensor
chunk_indices
chunk_offsets
GatedDeltaNetAttention
ChunkGatedDeltaRule
```

目标：理解 GDN state path 与 ordinary attention path 的边界。

---

## 实验验证作业

### 实验 1：打印 SchedulerOutput

在 `Scheduler.schedule()` 返回前临时打印：

```text
num_scheduled_tokens
req_to_new_blocks
scheduled_encoder_inputs
scheduled_spec_decode_tokens
preempted_reqs
```

用三类请求压测：

```text
短 prompt + 长输出
长 prompt + 短输出
共享 prefix 的多请求
```

观察 Scheduler 如何混合 prefill、decode、prefix cache 和 blocks。

### 实验 2：观察 prefix cache hit

构造共享 16K tokens prefix 的请求，连续发送：

```text
第 1 次：prefix cache miss
第 2 次：prefix cache hit
```

记录：

```text
get_computed_blocks 返回的 computed tokens
KVCacheBlocks block ids
TTFT
prefill time
prefix cache hit rate
```

### 实验 3：观察 Mamba/GDN block aligned split

启动 Qwen3.6，设置 `mamba_cache_mode=align`，打印：

```text
num_new_tokens before _mamba_block_aligned_split
num_new_tokens after _mamba_block_aligned_split
block_size
request.num_computed_tokens
request.num_prompt_tokens
```

目标：验证 GDN state caching 为什么要求 block-aligned chunk。

### 实验 4：确认 backend metadata

在 FlashAttention / FlashInfer / GDN metadata builder 中打印：

```text
backend name
num_prefills
num_decodes
num_actual_tokens
block_table shape
slot_mapping shape
chunk_indices shape
spec_state_indices shape
```

目标：不要猜 backend，直接看实际 metadata。

### 实验 5：Profiler 对照

用 Nsight Systems 观察：

```text
Scheduler step
KV cache update
FlashAttention / FlashInfer kernels
GDN chunk kernels
FusedMoE kernels
NCCL collectives
sampler kernels
```

将 profiler timeline 与 `SchedulerOutput` 对齐，理解“调度计划”如何变成 GPU kernel。

---

## 3 个自测题

**1. vLLM v0.20.1 Scheduler 为什么不直接说 prefill phase / decode phase？**
因为源码用 `num_computed_tokens` 追赶 `num_tokens_with_spec` 的统一模型，prefill、decode、prefix caching、speculative decoding 都可以表达成“还差多少 token 没算”。

**2. `block_table` 和 `slot_mapping` 的区别是什么？**
`block_table` 用来从请求的逻辑 block 找到物理 KV block；`slot_mapping` 用来把本轮新 token 的 K/V 写到对应 cache slot。

**3. 为什么 Qwen3.6 的 GDN 层不能直接套 FlashAttention KV cache 心智？**
因为 GDN backend 是 SSM/Mamba-style path，metadata 包含 state indices、chunk indices、chunk offsets、initial state 等；它不是普通 `QKᵀ → softmax → V` 的 paged K/V attention。

---

## 下一章预告

第十三章进入：

```text
工程部署：FP8 单卡、BF16 多卡、Expert Parallel、降级配置
```

下一章会把前面所有原理落成部署决策：

```text
1. FP8 checkpoint 能解决什么，不能解决什么？
2. 单卡 FP8 适合作为哪些场景的降级方案？
3. BF16 多卡如何规划 TP / EP / max_model_len？
4. 262K context 下如何逐步降级？
5. text-only、MTP、prefix caching、chunked prefill 如何组合？
6. Qwen3.6 在 vLLM v0.20.1 中哪些配置必须实测确认？
```

---

**Sources:**

- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py)
- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
