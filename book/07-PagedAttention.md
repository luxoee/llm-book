# 第七章：PagedAttention

本章回答一个核心问题：

```text
KV Cache 很大，为什么不能像普通 tensor 一样连续分配？
```

vLLM 的答案是：**把 KV Cache 像操作系统虚拟内存一样分页管理**。也就是把长序列的 K/V 拆成固定大小的 blocks，用 block table 把“逻辑 token 位置”映射到“物理 cache block”。

但对 **Qwen/Qwen3.6-35B-A3B** 要先划边界：PagedAttention 主要解释 **Gated Attention 层的 KV cache**；Qwen3.6 的 40 层结构是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，所以不能把 30 个 Gated DeltaNet 层也强行套进普通 PagedAttention 的 K/V cache 模型里。Qwen3.6 的模型卡确认它有 40 层、10 个重复周期、Gated Attention 配置为 16 Q heads / 2 KV heads / head dim 256 / RoPE dim 64，原生上下文 262,144 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

想象你在图书馆里做超长论文。

**朴素 KV Cache** 像给每个读者分一整排连续书架：

```text
读者 A 需要 262K tokens 的空间
→ 先给他预留一整排连续书架

读者 B 只需要 2K tokens
→ 也要找一段连续书架

读者 C 中途结束
→ 中间空出一块碎片
```

问题是：书架很快变得碎片化。明明总空位还很多，但没有足够长的连续空间给下一个长请求。

**PagedAttention** 像把书架拆成固定大小的小格子：

```text
每个 block 能放 16 个 token 的 K/V
请求 A 用 blocks: [7, 8, 21, 35, ...]
请求 B 用 blocks: [3, 4]
请求 C 结束后释放 blocks: [10, 11, 12]
```

逻辑上，请求 A 的上下文仍然是连续的：

```text
token 0, token 1, token 2, ...
```

但物理显存里，它们可以分散在不同 blocks 中。真正跑 attention kernel 时，靠 **block table** 告诉 GPU：

```text
逻辑第 i 个 block → 物理第 j 个 block
```

这就是 PagedAttention 的核心：**逻辑连续，物理分页**。

---

## 二、数学直觉

传统 KV Cache 可以想成一大块连续数组：

```text
K_cache[layer][seq][kv_head][head_dim]
V_cache[layer][seq][kv_head][head_dim]
```

PagedAttention 则把 `seq` 维度拆成 blocks：

```text
logical_block_id = token_position // block_size
offset_in_block  = token_position %  block_size
physical_block_id = block_table[sequence_id][logical_block_id]
```

然后访问：

```text
K = K_cache[physical_block_id, kv_head, offset_in_block, :]
V = V_cache[physical_block_id, kv_head, offset_in_block, :]
```

更抽象地说：

$$
\text{physical\_block} =
\text{block\_table}[\text{seq}, \lfloor t / B \rfloor]
$$

$$
\text{offset} = t \bmod B
$$

其中 $B$ 是 block size。

这个思想和操作系统虚拟内存很像：

| 操作系统 | PagedAttention |
|---|---|
| 虚拟地址 | token 的逻辑位置 |
| 物理页 | GPU KV cache block |
| 页表 | block table |
| 页内偏移 | token 在 block 内的 offset |
| 内存碎片管理 | KV cache block pool |

vLLM 的 Paged Attention 文档说明，paged KV cache 中 key cache 和 value cache 存在分离 blocks 中，并且 block 是 vLLM 的 KV cache block，不是 CUDA thread block；该文档也明确警告它是基于 vLLM 原始论文的历史文档，不再完整描述当前代码，所以本章把它作为概念说明，源码结论以 vLLM v0.20.1 tag 为准。([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html))

---

## 三、张量形状

### 1. 文档中的 PagedAttention 形状

vLLM Paged Attention 文档给出的 kernel 形状示例包括：

```text
q:       [num_seqs, num_heads, head_size]
k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
v_cache: [num_blocks, num_kv_heads, head_size, block_size]
```

这里最重要的不是具体 layout，而是 **`num_blocks` 取代了连续的 `seq_len` 维度**。这说明 KV cache 的物理组织不是按每条请求一整段连续序列存储，而是按 blocks 存储。([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html))

### 2. Qwen3.6 Gated Attention 的逻辑 KV 形状

Qwen3.6 的 Gated Attention 是：

```text
Q heads  = 16
KV heads = 2
head dim = 256
RoPE dim = 64
```

逻辑上，每个 Gated Attention 层的 K/V cache 可以理解成：

```text
K cache: [seq_len, 2, 256]
V cache: [seq_len, 2, 256]
```

但在 vLLM 中，物理上会映射到 paged blocks。前面第六章已经算过，仅 Qwen3.6 的 10 个 Gated Attention 层，BF16/FP16 下每 token 的 K/V 粗略是：

```text
10 × 2 × 2 × 256 × 2 bytes = 20,480 bytes/token
```

所以 262,144 tokens 的单序列，仅 Gated Attention KV 就是约 5.37GB 量级。这只是普通 full-attention KV 的粗算，不包括 Gated DeltaNet state、MoE buffer、MTP、block metadata、workspace、TP 复制等。

### 3. vLLM v0.20.1 中的 KVCacheBlocks

vLLM v0.20.1 的 `KVCacheBlocks` 注释说明，它是 `KVCacheManager` 的 allocation result，用来作为 Scheduler 和 KVCacheManager 之间的接口，隐藏 KVCacheManager 的内部数据结构；其中 `blocks[i][j]` 表示第 i 个 KV cache group 的第 j 个 token block。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

这意味着 vLLM v0.20.1 已经不是简单返回“一段连续 tensor 地址”，而是返回 **按 KV cache group 组织的 block 集合**。

---

## 四、计算流程

### 1. Prefill 写入 blocks

prefill 阶段输入一段 prompt，例如 65,536 tokens。如果 block size 是 16，那么逻辑上需要：

```text
65536 / 16 = 4096 blocks
```

流程可以理解成：

```text
prompt tokens
→ 逐层计算 K/V
→ 按 token 位置写入 slot
→ slot 属于某个 physical block
→ 记录 logical block 到 physical block 的映射
```

vLLM v0.20.1 的 `KVCacheManager.allocate_slots` 注释明确说它用于“给 request 追加新 tokens 分配 slots”；它的参数包括 `num_new_tokens`、`num_new_computed_tokens`、`new_computed_blocks`、`num_lookahead_tokens`、`num_external_computed_tokens`、`num_encoder_tokens` 等。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py))

### 2. Decode 追加 blocks

decode 阶段每步新增 token。多数时候只是在最后一个 block 里继续填 slot：

```text
block 10:
  token 160
  token 161
  ...
  token 175
```

如果最后一个 block 填满，再从 block pool 申请一个新 block：

```text
old blocks: [7, 8, 21]
new block:  [35]
block table: [7, 8, 21, 35]
```

vLLM v0.20.1 的 `BlockPool.get_new_blocks` 会从 free block pool 中取 blocks；如果请求数量超过 free blocks，会抛出无法获得 free blocks 的错误。源码还显示，如果 prefix caching 开启，分配前会在必要时 eviction cached block，并更新 block ref count。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/block_pool.py))

### 3. Attention 读取 block table

attention kernel 不是按连续地址扫一整条 sequence，而是需要 block table 告诉它每个逻辑 block 对应哪个物理 block。

vLLM v0.20.1 的 FlashAttention backend metadata 中包含 `block_table=block_table_tensor` 和 `slot_mapping=slot_mapping`；后续 attention 调用会把 `block_table` 传给 attention kernel / wrapper。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

这就是 PagedAttention 的关键执行路径：

```text
Scheduler / KVCacheManager
→ 分配 KV blocks
→ 生成 block_table / slot_mapping
→ Attention backend 依据 block_table 读 K/V
```

### 4. 请求结束释放 blocks

请求完成后，KV blocks 不一定立刻物理清零；它们会回到 block pool，或者在 prefix caching 开启时保留为可复用 cached blocks。

vLLM v0.20.1 的 `BlockPool.free_blocks` 会降低 block 的 `ref_cnt`，并把 ref count 归零且非 null 的 block 加回 free block queue；`touch` 会在 prefix cache hit 时增加 ref count，并可能从 free queue 中移除该 block。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/block_pool.py))

---

## 五、显存布局

PagedAttention 解决的不是“KV cache 变小”，而是 **KV cache 更可管理**。

### 1. 不分页时的问题

假设有三个请求：

```text
A: 100K tokens
B: 3K tokens
C: 50K tokens
```

如果每个请求都要连续显存：

```text
[A A A A A A ...][B B][C C C C ...]
```

当 B 结束后：

```text
[A A A A A A ...][空][C C C C ...]
```

中间的小空洞很难给新的长请求使用。这就是碎片问题。

### 2. 分页后的布局

PagedAttention 的物理布局更像：

```text
block 0: A
block 1: C
block 2: free
block 3: A
block 4: B
block 5: free
block 6: A
block 7: C
...
```

每个请求的逻辑序列由 block table 串起来：

```text
A block table: [0, 3, 6, ...]
B block table: [4]
C block table: [1, 7, ...]
```

这让 vLLM 可以把分散的 free blocks 重新分配给其他请求，降低连续大块分配带来的碎片。

### 3. BlockPool 的角色

vLLM v0.20.1 的 `BlockPool` 注释写明：它管理 KVCacheBlocks，提供 allocate、free、cache KV cache blocks 的方法；`free_block_queue` 以 eviction order 存储 free blocks，`cached_block_hash_to_block` 负责通过 block hash 查找 cached blocks，以支持 prefix caching。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/block_pool.py))

所以 BlockPool 同时承担两件事：

```text
1. 显存 block 分配/释放
2. prefix cache block 的查找/复用/驱逐
```

### 4. Qwen3.6 的特殊边界

对 Qwen3.6，PagedAttention 主要覆盖：

```text
10 个 Gated Attention 层的 K/V cache
```

但不应该强行覆盖：

```text
30 个 Gated DeltaNet 层的状态缓存
```

vLLM v0.20.1 源码确认，`Qwen3_5DecoderLayer` 遇到 `linear_attention` 会创建 `GatedDeltaNetAttention`，遇到 `full_attention` 才创建 `Qwen3NextAttention` 并传入 `cache_config`；这说明 Qwen3.6 在源码层面确实区分了两类 attention/state 路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 六、vLLM 源码映射

第七章的源码地图如下：

| 概念 | vLLM v0.20.1 文件路径 | 类/函数 | 作用 |
|---|---|---|---|
| KV cache 分配管理 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` | Scheduler 和 KV cache block 管理之间的核心接口 |
| KV block allocation result | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` | 封装每个 KV cache group 的 block 列表 |
| 分配 slots | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 给 request 的新增 tokens / speculative tokens / encoder tokens 分配 slots |
| block pool | `vllm/v1/core/block_pool.py` | `BlockPool` | 管理 free blocks、cached blocks、eviction |
| 申请 blocks | `vllm/v1/core/block_pool.py` | `get_new_blocks` | 从 free block pool 取新 blocks |
| 释放 blocks | `vllm/v1/core/block_pool.py` | `free_blocks` | 递减 ref count，把可释放 block 放回 free queue |
| prefix cache 触碰 | `vllm/v1/core/block_pool.py` | `touch` | prefix hit 时增加 ref count |
| attention metadata | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata` | 包含 `block_table`、`slot_mapping` 等 |
| KV cache 写入 | `vllm/v1/attention/backends/flash_attn.py` | `do_kv_cache_update` | 用 `reshape_and_cache_flash` 写入 key/value cache |
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` | 区分 `linear_attention` 与 `full_attention` |
| GDN state 路径 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | Mamba/GDN state，不是普通 K/V cache |

`KVCacheManager.__init__` 接收 `kv_cache_config`、`max_model_len`、`hash_block_size`、`max_num_batched_tokens`、`enable_caching` 等参数，并通过 `get_kv_cache_coordinator(...)` 构造 coordinator；它还通过 `self.block_pool = self.coordinator.block_pool` 接入 block pool。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py))

`allocate_slots` 的注释非常关键：它不是只处理普通 decode tokens，还处理 prefix cache hit、external computed tokens、lookahead speculative tokens、encoder tokens，以及 sliding window 下可移除 blocks；这解释了为什么真实 KV cache 管理必须比“append K/V tensor”复杂得多。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py))

---

## 七、NVIDIA GPU 执行视角

从 GPU 视角，PagedAttention 的好处和代价同时存在。

### 好处：减少显存碎片

PagedAttention 让请求不需要占用连续 KV cache 空间。只要有足够数量的 free blocks，请求就可以继续生成。这提高了显存利用率，也让 continuous batching 更容易运行。

### 代价：多一次 indirection

GPU attention kernel 读取 K/V 时，不再是：

```text
base_ptr + token_id * stride
```

而是类似：

```text
logical_block = token_id // block_size
offset        = token_id % block_size
physical_block = block_table[seq_id][logical_block]
ptr = cache_base + physical_block * block_stride + offset * token_stride
```

这意味着 kernel 要处理 block table、slot mapping、可能的非连续读取。vLLM v0.20.1 的 FlashAttention metadata 明确包含 `block_table` 和 `slot_mapping`，KV cache update 也使用 `slot_mapping` 把 key/value reshape and cache 到对应位置。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/attention/backends/flash_attn.py))

### 对 long context decode 的意义

Qwen3.6 原生 262K context，单个长请求需要大量 cache blocks。PagedAttention 不能让 attention 读取的历史变短，但它可以让这些历史 K/V 在显存中以 blocks 组织，便于多请求混合、释放、复用和调度。

换句话说：

```text
PagedAttention 不改变数学 attention
PagedAttention 改变 KV cache 的内存管理和访问方式
```

### 对 Gated DeltaNet 的边界

Gated DeltaNet 在 vLLM v0.20.1 中是 `GatedDeltaNetAttention(PluggableLayer, MambaBase)`，`mamba_type` 返回 `"gdn_attention"`，state dtype/shape 由 `MambaStateDtypeCalculator` 和 `MambaStateShapeCalculator` 计算；其底层路径调用 `fla_chunk_gated_delta_rule`，输入包括 `initial_state`、`output_final_state`、`chunk_indices`、`chunk_offsets` 等。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

所以在 GPU 视角上，Qwen3.6 的缓存/状态分成两类：

```text
Gated Attention:
  block table + paged K/V cache + attention backend

Gated DeltaNet:
  Mamba/GDN state + chunk metadata + GDN backend
```

---

## 八、优化前 vs 优化后

### 优化前：连续 KV Cache

```text
request A → 连续 100K token KV
request B → 连续 4K token KV
request C → 连续 60K token KV
```

问题：

| 问题 | 后果 |
|---|---|
| 请求长度差异大 | 显存碎片 |
| 请求结束时间不同 | 中间空洞难复用 |
| 高并发 decode | 很难动态插入新请求 |
| prefix 相同 | 难以共享已有 cache |
| speculative tokens | 额外预留更复杂 |

### 优化后：PagedAttention + BlockPool

```text
request A → blocks [0, 3, 7, 11, ...]
request B → blocks [1]
request C → blocks [2, 4, 8, ...]
```

优化点：

| 优化点 | 作用 |
|---|---|
| 固定大小 blocks | 降低外部碎片 |
| block table | 逻辑连续、物理分散 |
| free block queue | 快速回收可用 blocks |
| ref count | 支持共享和安全释放 |
| prefix block hash | 支持 prefix caching |
| scheduler + allocate_slots | 按本 step 的 token 需求分配 cache |

vLLM v0.20.1 的 `BlockPool` 源码说明，它维护所有 KV cache blocks、free block queue 和 cached block hash map；`touch` 增加 ref count，`free_blocks` 降低 ref count 并把可释放 blocks 放回 free queue。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/block_pool.py))

### 对 Qwen3.6 的优化后心智

错误心智：

```text
Qwen3.6 支持 262K，所以给每个请求预留 262K 连续 KV cache
```

正确心智：

```text
根据请求实际 token 数，按 blocks 分配
根据 prefix hit 复用 blocks
根据 decode step 动态追加 slots
根据 free blocks 决定是否调度更多 tokens
```

这就是第八章 continuous batching 和 chunked prefill 的基础。

---

## 九、工程落地

### 1. 看懂日志里的 cache capacity

部署 Qwen3.6 时，启动日志中关于 KV/cache capacity、GPU blocks、max concurrency 的信息非常关键。不要只看模型是否成功加载，要看：

```text
可用 KV blocks 数量
max_model_len
gpu_memory_utilization
prefix caching 是否开启
max_num_batched_tokens
是否启用 speculative decoding / MTP
```

### 2. max_model_len 不是免费参数

Qwen3.6 原生 262,144 context；如果 `--max-model-len 262144`，vLLM 需要为长上下文场景规划 cache 空间。模型卡给出 vLLM 标准 serving 命令时也使用了 `--max-model-len 262144`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

### 3. text-only 可以释放更多 cache 空间

Qwen3.6 是带 Vision Encoder 的 Causal LM；模型卡给出的 text-only vLLM 命令使用 `--language-model-only`，其说明是跳过 vision encoder 和 multimodal profiling，以释放更多内存给 KV cache。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

### 4. MTP 会改变 slot 预算

MTP / speculative decoding 会引入 lookahead tokens。`KVCacheManager.allocate_slots` 的参数里有 `num_lookahead_tokens`，源码注释说明它是为带 KV-cache 的 spec decode proposer 分配 speculative tokens 使用。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py))

工程判断：

```text
MTP 可能降低 TPOT
但会增加 cache slot 预算复杂度
高并发时不一定提高总吞吐
```

### 5. 排障顺序

如果 Qwen3.6 长上下文服务 OOM 或吞吐差，建议按这个顺序排查：

```text
1. max_model_len 是否过大
2. 是否同时开启多模态 + 262K + MTP
3. prefix caching 是否命中
4. 是否有大量长 prompt 占住 blocks
5. free blocks 是否不足导致 scheduler 无法接纳新 token
6. decode 是否被长上下文 KV 读取拖慢
7. GDN state 与 full-attention KV 是否分别估算
```

---

## 十、本章总结

PagedAttention 的本质是：

```text
把 KV cache 从“每条请求一大段连续显存”
变成“按固定大小 blocks 分页管理”
```

它解决的是：

```text
显存碎片
动态追加 token
请求结束释放
prefix cache 复用
continuous batching 下的 cache 调度
```

但它不改变：

```text
Attention 数学本身
长上下文需要读取大量历史 K/V 的事实
Qwen3.6 中 Gated DeltaNet 不是普通 KV cache 的事实
```

对 Qwen3.6，最重要的结论是：

```text
PagedAttention 主要管理 10 个 Gated Attention 层的 K/V cache；
30 个 Gated DeltaNet 层走 Mamba/GDN-style state 路径；
MoE、MTP、多模态和 runtime workspace 还要单独入账。
```

---

## 小白总结

KV Cache 像模型的“历史笔记”。
PagedAttention 就是把这本笔记拆成一页一页的小纸片，而不是放在一整本连续的大本子里。

这样做的好处是：
有人用完几页，就能把这几页还回去；新请求可以拿这些空页继续写，不需要找一整段连续空间。

---

## 工程师总结

vLLM v0.20.1 中，PagedAttention 相关的工程链路可以概括为：

```text
KVCacheManager
→ KVCacheBlocks
→ BlockPool
→ block_table / slot_mapping
→ attention backend
→ reshape_and_cache_flash / paged K/V read
```

对 Qwen3.6，源码边界是：

```text
full_attention:
  Qwen3NextAttention
  Attention(..., cache_config=...)
  paged K/V cache

linear_attention:
  GatedDeltaNetAttention
  MambaBase
  GDN state shape / dtype
```

这就是为什么后续讲 scheduler、chunked prefill、prefix caching、MTP 时，必须同时跟踪 **KV blocks** 和 **GDN state**，不能只看一种 cache。

---

## 关键公式

**1. token 到逻辑 block**

$$
\text{logical\_block} = \left\lfloor \frac{t}{B} \right\rfloor
$$

**2. token 在 block 内 offset**

$$
\text{offset} = t \bmod B
$$

**3. block table 映射**

$$
\text{physical\_block}
=
\text{block\_table}[\text{seq}, \text{logical\_block}]
$$

**4. Qwen3.6 Gated Attention KV 粗估**

$$
\text{KV bytes/token}
=
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
20480
$$

**5. blocks 数量粗估**

$$
\text{num\_blocks}
=
\left\lceil
\frac{\text{num\_tokens}}{\text{block\_size}}
\right\rceil
$$

---

## 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| KV cache 管理器 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| KV block 封装 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` |
| slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| block pool | `vllm/v1/core/block_pool.py` | `BlockPool` |
| block 申请 | `vllm/v1/core/block_pool.py` | `get_new_blocks` |
| block 释放 | `vllm/v1/core/block_pool.py` | `free_blocks` |
| prefix cache 引用 | `vllm/v1/core/block_pool.py` | `touch` |
| FlashAttention metadata | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata` |
| KV cache 写入 | `vllm/v1/attention/backends/flash_attn.py` | `do_kv_cache_update` |
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Gated DeltaNet state | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| GDN metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder` |

---

## 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Paged KV cache 的概念是把 key/value cache 分成 blocks，且 key/value cache 存在 separate blocks | vLLM Paged Attention docs | Paged Attention design doc | 官方文档确认 | 是，但该文档标注为历史文档 |
| vLLM Paged Attention 文档不再完整描述当前 vLLM 代码 | vLLM Paged Attention docs | Warning | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 Gated Attention 是 16 Q heads、2 KV heads、head dim 256、RoPE dim 64 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM v0.20.1 中 `KVCacheBlocks` 是 KVCacheManager 的 allocation result，是 Scheduler 和 KVCacheManager 之间的接口 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` | 源码直接确认 | 是 |
| `KVCacheManager.__init__` 接收 `kv_cache_config`、`max_model_len`、`hash_block_size`、`enable_caching` 等参数，并接入 block pool | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager.__init__` | 源码直接确认 | 是 |
| `KVCacheManager.allocate_slots` 给 request 的新增 tokens 分配 slots | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 源码直接确认 | 是 |
| `allocate_slots` 参数包含 `num_lookahead_tokens`，用于带 KV-cache 的 speculative decode proposer | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 源码直接确认 | 是 |
| `BlockPool` 管理 KVCacheBlocks，负责 allocate、free、cache blocks | `vllm/v1/core/block_pool.py` | `BlockPool` | 源码直接确认 | 是 |
| `BlockPool` 维护 free block queue 和 cached block hash map | `vllm/v1/core/block_pool.py` | `BlockPool.__init__` | 源码直接确认 | 是 |
| `BlockPool.get_new_blocks` 从 free block pool 取 blocks，并更新 ref count | `vllm/v1/core/block_pool.py` | `get_new_blocks` | 源码直接确认 | 是 |
| `BlockPool.free_blocks` 递减 ref count，并把可释放 blocks 加回 free queue | `vllm/v1/core/block_pool.py` | `free_blocks` | 源码直接确认 | 是 |
| FlashAttention metadata 中包含 `block_table` 与 `slot_mapping` | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionMetadata` 构造 | 源码直接确认 | 是 |
| FlashAttention backend 的 KV cache update 使用 `reshape_and_cache_flash` 和 `slot_mapping` 写入 K/V | `vllm/v1/attention/backends/flash_attn.py` | `do_kv_cache_update` | 源码直接确认 | 是 |
| Qwen3.6 的 `linear_attention` 和 `full_attention` 在 `Qwen3_5DecoderLayer` 中分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| `linear_attention` 使用 `GatedDeltaNetAttention`，`full_attention` 使用 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 继承 `MambaBase`，`mamba_type` 为 `gdn_attention`，state dtype/shape 由 calculator 计算 | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| Qwen3.6 的 GDN state 是否也被某些更高层 coordinator 以 KV cache group 形式统一抽象 | `vllm/v1/core/kv_cache_coordinator*`, `gdn_attn.py` | 待展开 | 待源码确认 | 否 |

---

## 源码阅读作业

按这个顺序读：

**1. 读 KVCacheBlocks**

```text
vllm/v1/core/kv_cache_manager.py
```

重点找：

```text
KVCacheBlocks
get_block_ids
get_unhashed_block_ids_all_groups
```

目标：理解 block allocation result 如何从 KVCacheManager 传给 Scheduler / backend。

**2. 读 allocate_slots**

```text
KVCacheManager.allocate_slots
```

重点关注：

```text
num_new_tokens
num_new_computed_tokens
new_computed_blocks
num_lookahead_tokens
num_external_computed_tokens
num_encoder_tokens
num_tokens_need_slot
num_blocks_to_allocate
```

目标：理解 vLLM 如何在一个 step 内决定要不要给请求分配更多 blocks。

**3. 读 BlockPool**

```text
vllm/v1/core/block_pool.py
```

重点找：

```text
BlockPool
get_new_blocks
touch
free_blocks
evict_blocks
cached_block_hash_to_block
free_block_queue
```

目标：理解 block 生命周期。

**4. 读 FlashAttention backend**

```text
vllm/v1/attention/backends/flash_attn.py
```

重点找：

```text
block_table
slot_mapping
do_kv_cache_update
reshape_and_cache_flash
```

目标：理解 block table / slot mapping 如何进入 attention backend。

**5. 读 Qwen3.6 边界**

```text
vllm/model_executor/models/qwen3_5.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
vllm/v1/attention/backends/gdn_attn.py
```

目标：确认 Gated Attention 的 K/V cache 和 Gated DeltaNet state 是两条不同路径。

---

## 实验验证作业

### 实验 1：观察 max_model_len 对 blocks 的影响

分别启动：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --language-model-only
```

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

记录：

```text
GPU memory usage
KV/cache capacity
num gpu blocks
max concurrency
TTFT
TPOT
```

目标：观察长上下文如何影响 cache 预算。

### 实验 2：制造不同长度请求，观察碎片与调度

准备三类请求：

```text
A: 1K prompt + 512 output
B: 32K prompt + 512 output
C: 128K prompt + 512 output
```

混合并发压测，观察：

```text
waiting requests
running requests
free blocks
preemptions
TTFT tail latency
```

目标：理解 PagedAttention 为什么对混合长度 workload 重要。

### 实验 3：prefix caching 命中

构造多个共享长前缀的请求：

```text
共同 system prompt / docs prefix: 16K tokens
不同 user query: 128 tokens
```

观察：

```text
prefix cache hit
TTFT
GPU blocks 使用情况
吞吐变化
```

目标：把 block hash / cached blocks 的概念和实际收益对应起来。

---

## 3 个自测题

**1. PagedAttention 为什么能降低 KV cache 碎片？**
因为它把长序列 K/V 拆成固定大小 blocks，请求逻辑上连续，但物理上可以使用分散的 free blocks。

**2. block table 的作用是什么？**
它把请求的逻辑 block id 映射到物理 KV cache block id，让 attention backend 能从正确的物理 block 读取 K/V。

**3. 为什么 Qwen3.6 不能只用 PagedAttention 一套模型解释所有缓存？**
因为 Qwen3.6 只有 10 个 Gated Attention 层适合普通 K/V cache 视角；另外 30 个 Gated DeltaNet 层走 Mamba/GDN-style state 路径。

---

## 下一章预告

第八章进入：

```text
Continuous Batching、Chunked Prefill、Prefix Caching
```

下一章会把第七章的 block 管理接到 scheduler 上，回答：

```text
1. 为什么 decode 阶段需要 continuous batching？
2. 为什么长 prompt prefill 会阻塞短请求？
3. chunked prefill 如何把大 prompt 切小，避免饿死 decode？
4. prefix caching 如何复用相同前缀的 blocks？
5. Qwen3.6 的 262K context、GDN state、MTP 会怎样影响这些调度策略？
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py)
- [vllm/vllm/v1/core/kv_cache_manager.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py)
