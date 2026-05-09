# LLM 推理与架构

**User:** Anonymous  
**Created:** 5/8/2026 18:23:31  
**Updated:** 5/9/2026 11:17:49  
**Exported:** 5/9/2026 11:19:05  
**Link:** [https://chatgpt.com/c/69fdb992-ca18-8321-8562-02b708eb1020](https://chatgpt.com/c/69fdb992-ca18-8321-8562-02b708eb1020)  

## Prompt:
2026/5/8 18:23:30

你是一位拥有 10 年经验的 LLM 推理系统架构师、CUDA/NVIDIA GPU 性能优化专家、vLLM 源码贡献者级别的严谨度讲解。
请固定以 vLLM v0.20.1 为分析版本,不要混用其他版本实现.在 NVIDIA GPU 上推理 Qwen/Qwen3.6-35B-A3B 为固定案例，带我从 0 系统学习 LLM 推理背后的根本原理。

事实校验来源分工：
1. vLLM GitHub release：验证 vLLM v0.20.1 的发布时间、release 类型、关键变更；
2. vLLM 官方 docs/recipes：验证部署命令、硬件建议、模型 serving 参数；
3. Qwen Hugging Face model card：验证模型结构、参数量、上下文长度、模型特性；
4. vLLM v0.20.1 tag 源码：验证源码路径、类名、函数名、kernel/backend 实现。

关键事实必须引用出处。  
同一段中多个事实来自同一来源时，可以在段末统一引用。  
源码路径、模型结构、硬件要求、部署命令必须引用；通用数学推导不需要引用。

请特别注意：不要把 Qwen/Qwen3.6-35B-A3B 简化成普通 Transformer。请先解释它与传统 Transformer Decoder 的共同点和差异，重点包括：
- Causal Language Model with Vision Encoder；
- Gated DeltaNet；
- Gated Attention；
- MoE；
- 256 experts；
- 8 routed experts + 1 shared expert；
- MTP；
- 262K 原生上下文；
- FP8 checkpoint。

课程结构请调整为：

第一轮：事实校验 + 模型结构差异 + 课程大纲  
第二轮：传统 LLM 推理总览  
第三轮：Qwen3.6 架构总览：Gated DeltaNet、Gated Attention、MoE、MTP  
第四轮：Token、Embedding、Logits、Sampling  
第五轮：Attention / Gated Attention / RoPE 数学  
第六轮：KV Cache 与长上下文显存问题  
第七轮：PagedAttention  
第八轮：Continuous Batching、Chunked Prefill、Prefix Caching  
第九轮：MoE：Router、Expert、Shared Expert、Top-k、Expert Dispatch  
第十轮：Tensor Parallel、Expert Parallel、NCCL 通信  
第十一轮：NVIDIA GPU Kernel：HBM、SM、Tensor Core、CUDA Kernel、MoE GEMM、Attention Backend  
第十二轮：vLLM Scheduler、Block Manager、Attention Backend 源码导读  
第十三轮：工程部署：FP8 单卡、BF16 多卡、Expert Parallel、降级配置  
第十四轮：压测、监控、排障  
第十五轮：总复盘与学习路线图

第一轮只输出：
1. 事实校验清单；
2. 传统 Transformer 与 Qwen3.6-35B-A3B 结构差异表；
3. 已确认 / 合理推断 / 待源码确认清单；
4. 15 轮课程总目录。

不要在第一轮展开每章详细目标。等我确认目录后，再输出每章学习目标。
第一轮不要正式展开第一章。等我说“开始第一章”后再进入正式课程。

每章默认按以下结构输出：
一、生活类比；
二、数学直觉；
三、张量形状；
四、计算流程；
五、显存布局；
六、vLLM 源码映射；
七、NVIDIA GPU 执行视角；
八、优化前 vs 优化后；
九、工程落地；
十、本章总结。

每章默认按十段结构展开，但请优先保证教学自然性。
如果某个小节不适合本章，请写：
“本节不强行展开，原因是……”
不要为了满足格式而编造数学、源码或 GPU 细节。
如果某章不适合某个小节，可以明确说明“不适用”，不要强行编造。

每章最后必须包含：
- 小白总结；
- 工程师总结；
- 关键公式；
- 关键源码入口；
- 源码阅读作业；
- 实验验证作业；
- 3 个自测题；
- 下一章预告。

凡涉及 vLLM v0.20.1 源码，请输出源码证据表：

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|

证据类型只能是：
- 源码直接确认；
- 官方文档确认；
- recipe 确认；
- 合理推断；
- 待源码确认。

不要根据旧版 vLLM 记忆直接回答 Scheduler、Block Manager、Attention Backend、MoE Kernel 的实现。必须以 v0.20.1 为准；如果没有读取源码，请标注待验证。

## Response:
2026/5/8 18:23:31

> **Browsing for vLLM v0.20.1 and Qwen3.6 model**
>
> [Releases · vllm-project/vllm - GitHub](https://github.com/vllm-project/vllm/releases)
> [vllm serve - vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/cli/serve/)
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Releases · vllm-project/vllm - GitHub | Release Alert](https://releasealert.dev/github/vllm-project/vllm)
> [vLLM](https://docs.vllm.ai/)
> [Model Details · ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)
> [vllm-project/vllm v0.20.1 on GitHub - NewReleases.io](https://newreleases.io/project/github/vllm-project/vllm/release/v0.20.1)
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://hf-p-cfw.fyan.top/Qwen/Qwen3.6-35B-A3B)
> [Previous Releases - vLLM](https://vllm.ai/releases)
>
> **Searching for official release dates on Github**
>
> [Releases · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases)
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vLLM](https://docs.vllm.ai/)
> [Previous vLLM Releases | vLLM](https://vllm.ai/releases)
>
> **Searching for relevant HF lines on deployment**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
>
> **Adjusting hardware section line timing**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
>
> **Opening Qwen configuration file on HuggingFace**
>
>
> **Searching for Hugging Face Qwen config details**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.6-35B-A3B开源了，本地部署教程 - 知乎](https://zhuanlan.zhihu.com/p/2028273911045866184)
> [Qwen3.6-35B-A3B-FP8 · Models](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8)
> [Qwen3.6 - 如何本地运行 | Unsloth Documentation](https://unsloth.ai/docs/zh/mo-xing/qwen3.6)
> [GitHub - QwenLM/Qwen3.6: Qwen3.6 is the large language model series ...](https://github.com/QwenLM/Qwen3.6)
> [Qwen3.6-35B-A3B 本地运行指南：Unsloth 快速上手-CSDN博客](https://blog.csdn.net/qq_30089667/article/details/160336103)
> [Qwen3.6-35B-A3B · Models](https://www.modelscope.ai/models/Qwen/Qwen3.6-35B-A3B)
> [Qwen3 - Hugging Face 文档](https://hugging-face.cn/docs/transformers/model_doc/qwen3)
>
> **Locating model card and VLLM serving info**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [千问3.6-35B-A3B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8)
>
> **Searching for Qwen HF model card FP8 checkpoint**
>
> [Qwen/Qwen3.6-35B-A3B-FP8 · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
> [Qwen3.6-35B-A3B-FP8 · Models](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8)
> [GitHub - QwenLM/Qwen3.6: Qwen3.6 is the large language model series ...](https://github.com/QwenLM/Qwen3.6)
> [Qwen3.6-35B-A3B-FP8 download | SourceForge.net](https://sourceforge.net/projects/qwen3-6-35b-a3b-fp8/)
> [Qwen3.6-35B-A3B 全面评测：阿里开源模型如何超越前沿级水平](https://www.cnblogs.com/sing1ee/p/19885253)
> [Qwen3.6-35B-A3B-FP8:可用于高效部署多模态对话与编码 ...](https://ai.gitcode.com/hf_mirrors/Qwen/Qwen3.6-35B-A3B-FP8)
>
> **Searching official vLLM recipes and Qwen3.5 docs**
>
> [qwen3_5 - vLLM](https://docs.vllm.ai/en/latest/api/vllm/transformers_utils/configs/qwen3_5/)
> [Qwen3.5 使用指南 - vLLM 方案 - vLLM 文档](https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [vLLM - Qwen](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)
> [从卡壳到丝滑：Qwen3模型在Xinference中VLLM启动全案解决](https://blog.csdn.net/gitblog_01425/article/details/150381048)
>
> **Searching for open recipe sources**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> **Verifying source paths and evidence table**
>
> I need to find the exact source paths/classes for vLLM v0.20.1 using GitHub. The vLLM docs don't seem to show this version yet, so I might open GitHub directly. The user requested an evidence table in the first round, so I’ll need to include that in section 3. I'll try searching for "qwen3_5" to verify the paths.
>
> **Searching vLLM v0.20.1 source for specific classes**
>
> [Releases · vllm-project/vllm - GitHub](https://github.com/vllm-project/vllm/releases)
> [Qwen3.5 使用指南 - vLLM 方案 - vLLM 文档](https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Qwen3.5-27B vllm claude-code本地化部署 - CSDN博客](https://blog.csdn.net/cwt0408/article/details/158622923)
> [vLLM - Qwen](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> **Verifying source code access**
>
> I need the source code from a specific tag, but web search isn't helping. Maybe I can use a container with Git, but I'm not sure if it has internet access. I could try a Git command to check remotely. If that doesn't work, I might need to check the raw source directly via a safe URL.
>
> **Searching for raw GitHub files via container**
>
> [Releases · vllm-project/vllm - GitHub](https://github.com/vllm-project/vllm/releases)
> [Previous Releases - vLLM](https://vllm.ai/releases)
> [vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/)
> [vllm: vLLM 是一个快速且易于使用的 LLM 推理和服务库 - Gitee](https://gitee.com/mirrors/vllm)
> [Releases · vllm-project/vllm - GitHub | Release Alert](https://releasealert.dev/github/vllm-project/vllm)
> [vLLM](https://docs.vllm.ai/)
> [Releases · vllm-project/vllm](https://www.githubclone.com/vllm-project/vllm/releases)
>
> **Accessing release tarball and source citation**
>
> To inspect the release, I need to download the tarball, and the container might have internet access for that.
>
> **Inspecting container download options and GitHub citations**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Accessing raw code**
>
> It seems that the open version doesn’t display the code. I might need to click on the raw version directly. For that, I’ll use the raw ID 78 to access it. That should help me retrieve the code I need directly.
>
> **Troubleshooting raw click ID display issue**
>
> [raw.githubusercontent.com](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Retrieving raw code and citations**
>
> The raw version seems minified into one line, probably due to some formatting, which may be making it harder to view. I’ll need enough citations for context. Also, I should find and open the raw version of ‘qwen3_next.py’ to check other related files and gather relevant information.
>
> **Clarifying raw minification and file requirements**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Accessing specific code**
>
> I need enough detail, so maybe I should click the raw version for the 'qwen3_next' file to view the specific code. It’s likely in one line, which should be enough to get what I need. I’ll focus on extracting the relevant information from there.
>
> **Searching for Qwen3_next code solution**
>
> [raw.githubusercontent.com](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Accessing specific file**
>
> I need to find the raw version of 'qwen3_next_mtp' — possibly a truncated or blank version. I could open it directly, focusing on extracting it by line. This should help me find the specific part I need without unnecessary extra information.
>
> **Investigating qwen3_next_mtp raw truncated blank issue**
>
> [raw.githubusercontent.com](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next_mtp.py)
>
> **Accessing scheduler source and metadata**
>
> I need to gather source lines for the scheduler and key-value cache. I might be able to get enough by opening the file with its metadata around the lines. If the scheduler open failed, it could be due to a different path. I’ll search for the correct path to check that.
>
> **Searching for scheduler and KV cache code references**
>
>
> **Verifying scheduler source and paths**
>
> It looks like I need to check the GitHub URL directly for the scheduler, since there was an internal error. I'll try a different path, like “core/sched/scheduler.py.” I might also need to explore the GitHub tree to ensure the relevant files are in place. For now, I’ll mark the scheduler source as pending verification since the user is focused on verifying vLLM v0.20.1 sources. I’ll check paths like v1/core/scheduler.py and kv_cache_manager.py, and validate their existence via metadata.
>
> **Verifying GitHub paths for scheduler source files**
>
> [raw.githubusercontent.com](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/v1/core/kv_cache_manager.py)
>
> **Considering scheduler path and GitHub access**
>
> I’m thinking the scheduler path might have been moved to vllm/v1/core/sched/scheduler.py. Maybe I could use the GitHub API or even GitHub's tree view to explore this. If it’s helpful, I could also download a tarball using wget in a container for internal checks.
>
> **Downloading and listing specific files from GitHub**
>
>
> **Searching for scheduler path via GitHub**
>
> I need to find the scheduler, possibly at vllm/v1/core/sched/scheduler.py. There's no internet access in the container, so I’ll search through the web instead. Maybe I can use GitHub’s tree view for version v0.20.1 and look in the vllm/v1/core folder to find the path.
>
> **Searching GitHub for vllm scheduler path**
>
> [vllm/vllm/v1/core at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/core)
>
> **Opening scheduler directory**
>
> I need to open the scheduler directory to move forward. It sounds like an essential step to locate the relevant files for the task, so I’ll be accessing that next. I’m focusing on finding the right location for the scheduler, which will help me proceed effectively.
>
> **Searching for open scheduling directories**
>
> [vllm/vllm/v1/core/sched at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/core/sched)
>
> **Accessing raw scheduler info**
>
> I need to click and access the raw scheduler details. I'm aiming to get the most basic, unprocessed data so I can examine it more closely. This will help me move forward and ensure everything is in place before I proceed with the next steps.
>
> **Searching for click scheduler raw data**
>
> [vllm/vllm/v1/core/sched/scheduler.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Accessing raw data**
>
> I think I need to click through to get the raw data. This will allow me to focus on the unprocessed information, which should give me the insight I need. From there, I'll be able to proceed carefully and make sure I don’t overlook anything important.
>
> **Searching for raw click data**
>
> [raw.githubusercontent.com](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Checking attention backends**
>
> I think I need the list of attention backends, possibly within a file path. It might help to open the tree at `vllm/v1/attention/backends` to pinpoint it. That should give me the structure I need to proceed further with the task I'm working through.
>
> **Searching for attention backend file paths**
>
> [vllm/vllm/v1/attention/backends at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends)
>
> **Clarifying source citations and model specifics**
>
> I should collect evidence from sources and cite all important facts, making sure to use citations in Chinese. It's key to note that the "Qwen/Qwen3.6-35B-A3B FP8 checkpoint" is based on a model card that doesn't directly say FP8, but links to it in the recipe. It’s better to open the FP8 model on HF for more clarity.
>
> **Citing sources for FP8 checkpoint clarification**
>
> [Qwen/Qwen3.6-35B-A3B-FP8 · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
>
> **Clarifying structure and citations requirements**
>
> The output needs to be concise but complete, with the sections requested, and in Chinese. I'll include a "已确认/合理推断/待源码确认" list, along with a source evidence table. For hardware references, "硬件建议" will note that while the vLLM recipe mentions TP=8 GPUs, it doesn’t specify validated hardware models. The Qwen3.6 model card mentions TP 8 GPUs. I'll also create a "事实校验清单” with relevant details from sources.
>
> Thought for 2m 0s

我会先按你指定的四类来源做事实校验；如果某个版本、模型卡或 tag 找不到，我会明确标为“待确认/未找到”，不会用旧版记忆补齐。

## 1. 事实校验清单

| 校验项 | 已核验结论 | 状态 |
|---|---|---|
| vLLM 分析版本 | 固定为 **vLLM v0.20.1**。GitHub Release 页面显示 v0.20.1，发布时间为 **2026-05-04 10:36**，标记为 Latest。([GitHub](https://github.com/vllm-project/vllm/releases)) | 已确认 |
| vLLM v0.20.1 release 类型 | vLLM 官方发布流程说明：patch release 用于新模型、关键性能/功能/安全修复等；v0.20.1 release note 明确称其为基于 v0.20.0 的 **patch release**。([GitHub](https://github.com/vllm-project/vllm/blob/main/RELEASE.md)) ([GitHub](https://github.com/vllm-project/vllm/releases)) | 已确认 |
| vLLM v0.20.1 关键变更 | v0.20.1 主要聚焦 **DeepSeek V4 稳定性与性能改进**，包括 DeepSeek V4 Base 支持、多流 pre-attention GEMM、FlashInfer all-to-all 支持、若干 TopK / RoPE / CUDA graph / memory pool 修复等。([GitHub](https://github.com/vllm-project/vllm/releases)) | 已确认 |
| Qwen/Qwen3.6-35B-A3B 类型 | 模型卡写明：**Causal Language Model with Vision Encoder**，不是纯文本普通 Decoder-only Transformer。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| Qwen3.6-35B-A3B 参数量 | 总参数 **35B**，每 token 激活 **3B**。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| Qwen3.6-35B-A3B 主干结构 | hidden size 2048、40 层、layout 为 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| Gated DeltaNet | Gated DeltaNet：V 侧 32 个 linear attention heads，QK 侧 16 个 heads，head dim 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| Gated Attention | Gated Attention：Q heads 16、KV heads 2、head dim 256、RoPE dim 64。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| MoE | 256 experts；每 token 激活 **8 routed experts + 1 shared expert**；expert intermediate dim 512。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| MTP | 模型卡写明 **MTP: trained with multi-steps**；vLLM/Qwen serving 示例也给出 MTP speculative decoding 命令。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| 上下文长度 | 原生上下文 **262,144 tokens**，可扩展到 **1,010,000 tokens**；OOM 时可降低 context window，但模型卡建议至少保持 128K 以保留复杂任务/思考能力。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| FP8 checkpoint | Hugging Face 上存在 **Qwen/Qwen3.6-35B-A3B-FP8**；模型卡称其包含 FP8-quantized weights/config，采用 fine-grained FP8 quantization，block size 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)) | 已确认 |
| vLLM 官方/recipe 部署命令 | vLLM recipe 给出 Qwen3.6 命令：`vllm serve Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3`；Qwen 模型卡也给出标准、Tool Call、MTP、Text-Only vLLM serve 命令。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)) ([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) | 已确认 |
| NVIDIA GPU 固定案例 | 官方 recipe 的 Qwen3.6 部署示例使用 `--tensor-parallel-size 8`；recipe 的 NVIDIA 安装路径使用 `uv pip install -U vllm --torch-backend=auto`，并给出 NVIDIA Docker 示例。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)) | 已确认 |
| vLLM v0.20.1 Qwen 源码入口 | v0.20.1 tag 中存在 `vllm/model_executor/models/qwen3_5.py`，其 raw 源码包含 `Qwen3_5MoeForConditionalGeneration`、`Qwen3_5ForConditionalGeneration`，并接入 `Qwen3_VisionTransformer` 与 `Qwen3_5MoeForCausalLM`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_5.py)) | 已确认 |
| vLLM v0.20.1 Gated DeltaNet / Gated Attention / MoE 源码入口 | `qwen3_5.py` 引入 `GatedDeltaNetAttention`、`Qwen3NextAttention`、`Qwen3NextSparseMoeBlock`；`qwen3_next.py` 定义 `Qwen3NextSparseMoeBlock`、`Qwen3NextAttention`、`Qwen3NextDecoderLayer`。([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_5.py)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next.py)) | 已确认 |
| vLLM v0.20.1 MTP 源码入口 | `vllm/model_executor/models/qwen3_next_mtp.py` 定义 `Qwen3NextMultiTokenPredictor` 与 `Qwen3NextMTP`。([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next_mtp.py)) | 已确认 |
| vLLM v0.20.1 Scheduler / KV Cache / Attention Backend 源码入口 | Scheduler 位于 `vllm/v1/core/sched/scheduler.py`；KV cache manager 位于 `vllm/v1/core/kv_cache_manager.py`；attention backends 位于 `vllm/v1/attention/backends/`，目录中包含 `flash_attn.py`、`flashinfer.py`、`gdn_attn.py`、`linear_attn.py`、`triton_attn.py`、`turboquant_attn.py` 等。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/v1/core/kv_cache_manager.py)) ([GitHub](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends)) | 已确认 |

---

## 2. 传统 Transformer Decoder 与 Qwen3.6-35B-A3B 结构差异表

| 维度 | 传统 Transformer Decoder | Qwen/Qwen3.6-35B-A3B |
|---|---|---|
| 模型类型 | 通常是纯文本 causal decoder-only LM | **Causal Language Model with Vision Encoder**，支持文本、图像/视频等多模态输入路径。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| 主干层结构 | 多层重复：Self-Attention → FFN/MLP | `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，即每个大周期中 3 个线性注意力风格层 + 1 个 full attention 风格层。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| Attention 类型 | 标准 causal self-attention，通常所有层都是 full attention | 混合：**Gated DeltaNet** 与 **Gated Attention** 交替出现；不是纯标准 attention。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| Gated DeltaNet | 无 | 有。V 侧 32 heads、QK 侧 16 heads、head dim 128；vLLM v0.20.1 源码中通过 `GatedDeltaNetAttention` 映射。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_5.py)) |
| Gated Attention | 通常是普通 MHA/GQA/MQA attention，无额外输出 gate | 有。Q heads 16、KV heads 2、head dim 256、RoPE dim 64；vLLM `Qwen3NextAttention` 中包含 `attn_output_gate` 路径。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next.py)) |
| FFN / MLP | Dense FFN：每 token 经过同一组 FFN 权重 | **MoE**：每 token 路由到少量专家。模型卡确认 256 experts、8 routed + 1 shared expert。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| 激活参数 | Dense 模型通常每 token 激活全部层参数 | 总参数 35B，但每 token 约激活 3B。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| Shared Expert | 通常无 | 有 1 个 shared expert；vLLM `Qwen3NextSparseMoeBlock` 中存在 `shared_expert_gate` 与 `shared_expert` 路径。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next.py)) |
| MTP | 传统 decoder 推理通常一次预测 1 个 next token | 模型卡称 MTP trained with multi-steps；vLLM recipe 和 Qwen 模型卡给出 MTP speculative decoding 命令；vLLM v0.20.1 中有 `Qwen3NextMTP` 源码入口。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)) ([GitHub](https://github.com/vllm-project/vllm/raw/refs/tags/v0.20.1/vllm/model_executor/models/qwen3_next_mtp.py)) |
| 上下文 | 常见开源 decoder 原生 4K/8K/32K/128K 不等 | 原生 262K，可通过 RoPE/YaRN 扩展到约 1,010,000 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| 多模态显存影响 | 纯文本模型无 vision encoder 显存占用 | Qwen3.6 有 vision encoder；模型卡给出 `--language-model-only` 以跳过 vision encoder 和多模态 profiling，为 KV cache 释放显存。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| FP8 部署 | 取决于模型是否提供预量化权重 | 有官方 Hugging Face FP8 checkpoint：`Qwen/Qwen3.6-35B-A3B-FP8`，模型卡称为 fine-grained FP8 quantization，block size 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)) |

---

## 3. 已确认 / 合理推断 / 待源码确认清单

### 已确认

| 结论 | 文件路径 / 来源 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| vLLM v0.20.1 是 2026-05-04 发布的 patch release，重点是 DeepSeek V4 稳定性、性能与 bug fix | GitHub Release | v0.20.1 release note | 官方文档确认 | 是 |
| Qwen3.6-35B-A3B 是带 Vision Encoder 的 causal LM | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6-35B-A3B 是 35B total / 3B activated | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 主干是 `Gated DeltaNet → MoE` 与 `Gated Attention → MoE` 的混合结构 | Hugging Face model card | Hidden Layout | 官方文档确认 | 是 |
| Qwen3.6 MoE 为 256 experts，8 routed + 1 shared expert | Hugging Face model card | Mixture Of Experts | 官方文档确认 | 是 |
| Qwen3.6 原生 262,144 context，可扩展到 1,010,000 tokens | Hugging Face model card | Context Length | 官方文档确认 | 是 |
| Qwen3.6 有官方 FP8 checkpoint | Hugging Face FP8 model card | `Qwen/Qwen3.6-35B-A3B-FP8` | 官方文档确认 | 是 |
| vLLM recipe 推荐 Qwen3.6 使用 TP=8、max-model-len=262144、reasoning-parser=qwen3 | vLLM Recipes | `vllm serve ...` | recipe 确认 | 是 |
| Qwen3.6 vLLM text-only 模式可用 `--language-model-only` 跳过 vision encoder | Hugging Face model card | vLLM Serving command | 官方文档确认 | 是 |
| vLLM v0.20.1 中 Qwen3.5/3.6 兼容源码入口是 `qwen3_5.py`，MoE multimodal 类为 `Qwen3_5MoeForConditionalGeneration` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 Gated DeltaNet 映射到 `GatedDeltaNetAttention` | `vllm/model_executor/models/qwen3_5.py`, `qwen3_next.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 Gated Attention 映射到 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 MoE block 映射到 `Qwen3NextSparseMoeBlock` 与 `FusedMoE` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock`, `FusedMoE` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 MTP 入口是 `Qwen3NextMTP` | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP`, `Qwen3NextMultiTokenPredictor` | 源码直接确认 | 是 |
| vLLM v0.20.1 Scheduler 入口不在旧路径 `vllm/v1/core/scheduler.py`，而在 `vllm/v1/core/sched/scheduler.py` | `vllm/v1/core/sched/scheduler.py` | `Scheduler` | 源码直接确认 | 是 |
| vLLM v0.20.1 KV cache manager 入口是 `vllm/v1/core/kv_cache_manager.py` | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `KVCacheBlocks` | 源码直接确认 | 是 |
| vLLM v0.20.1 attention backend 目录是 `vllm/v1/attention/backends/` | `vllm/v1/attention/backends/` | `flash_attn.py`, `flashinfer.py`, `gdn_attn.py`, `linear_attn.py`, `triton_attn.py` 等 | 源码直接确认 | 是 |

### 合理推断

| 结论 | 依据 | 证据类型 | 是否已确认 |
|---|---|---|---|
| 后续课程中，“普通 Transformer Decoder”只能作为对照基线，不能作为 Qwen3.6 的完整解释模型 | Qwen3.6 模型卡明确给出 hybrid hidden layout、Vision Encoder、MoE、MTP | 合理推断 | 是 |
| 在 NVIDIA GPU 上部署 262K context 的 Qwen3.6-35B-A3B，KV cache 显存将成为比权重更敏感的约束之一 | 模型原生 262K context、vLLM/PagedAttention/KV cache 机制、Qwen 模型卡 OOM 降低 context window 建议 | 合理推断 | 是 |
| 单卡 FP8 案例更适合作为“可跑/降级/理解约束”的工程案例，而 262K 满上下文高吞吐更接近 8 GPU TP/EP 案例 | FP8 checkpoint 已确认；官方标准 serving 示例采用 TP=8 与 262K | 合理推断 | 是 |

### 待源码确认

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 在 vLLM v0.20.1 中的具体 config 字段如何从 HF `config.json` 映射到 `Qwen3_5MoeTextConfig` | `vllm/transformers_utils/configs/qwen3_5_moe.py` | `Qwen3_5MoeConfig`, `Qwen3_5MoeTextConfig` | 待源码确认 | 否 |
| Gated DeltaNet 的 CUDA/Triton 后端具体 kernel 调度路径 | `vllm/v1/attention/backends/gdn_attn.py`, `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | 待定 | 待源码确认 | 否 |
| MoE dispatch 在 v0.20.1 对 Qwen3.6 的实际 kernel 选择：CUTLASS / Triton / fused_moe / expert parallel 路径 | `vllm/model_executor/layers/fused_moe/` | `FusedMoE` 相关 | 待源码确认 | 否 |
| `--speculative-config '{"method":"qwen3_next_mtp"...}'` 与 recipe 中 `{"method":"mtp"...}` 在 v0.20.1 的解析兼容关系 | speculative decoding 配置解析源码 | 待定 | 待源码确认 | 否 |
| Attention backend 对 Qwen3.6 full attention 层的实际选择逻辑：FlashAttention / FlashInfer / Triton / fallback | `vllm/v1/attention/backends/registry.py` 及相关 backend | 待定 | 待源码确认 | 否 |
| Scheduler 中 continuous batching、chunked prefill、prefix caching 对 hybrid Mamba/GDN cache 的精确交互 | `vllm/v1/core/sched/scheduler.py`, `kv_cache_manager.py`, `mamba` cache 相关文件 | `Scheduler.schedule`, `KVCacheManager.allocate_slots` 等 | 待源码确认 | 否 |

---

## 4. 15 轮课程总目录

1. **事实校验 + 模型结构差异 + 课程大纲**  
2. **传统 LLM 推理总览**  
3. **Qwen3.6 架构总览：Gated DeltaNet、Gated Attention、MoE、MTP**  
4. **Token、Embedding、Logits、Sampling**  
5. **Attention / Gated Attention / RoPE 数学**  
6. **KV Cache 与长上下文显存问题**  
7. **PagedAttention**  
8. **Continuous Batching、Chunked Prefill、Prefix Caching**  
9. **MoE：Router、Expert、Shared Expert、Top-k、Expert Dispatch**  
10. **Tensor Parallel、Expert Parallel、NCCL 通信**  
11. **NVIDIA GPU Kernel：HBM、SM、Tensor Core、CUDA Kernel、MoE GEMM、Attention Backend**  
12. **vLLM Scheduler、Block Manager、Attention Backend 源码导读**  
13. **工程部署：FP8 单卡、BF16 多卡、Expert Parallel、降级配置**  
14. **压测、监控、排障**  
15. **总复盘与学习路线图**

---

**Sources:**

- [Releases · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases)
- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

## Prompt:
2026/5/8 18:27:48

第一章的内容已经输出, 继续开始第二章课程

## Response:
2026/5/9 09:49:39

> **Gathering sources for Chapter 2 on LLM reasoning**
>
> I need to fetch reliable web sources to verify facts and citations for Chapter 2, "Traditional LLM Reasoning Overview". I’ll need to gather evidence from sources, especially for anything about vLLM, backing with appropriate citations. I will look up the current documentation or source material for accuracy.
>
> **Searching for sources on vLLM and Qwen specifics**
>
> For Chapter 2, I’m focusing on the basics of traditional LLMs, but I need to clarify some key facts about Qwen (35B/3B, MoE, GDN/GA) and vLLM v0.20.1. I’ll look for the latest official docs on Qwen, model cards, and relevant info about PagedAttention to make sure I have accurate, cited data for those key facts.
>
> **Browsing for vLLM v0.20.1 and Qwen sources**
>
> [Releases · vllm-project/vllm - GitHub](https://github.com/vllm-project/vllm/releases)
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Releases · vllm-project/vllm - GitHub | Release Alert](https://releasealert.dev/github/vllm-project/vllm)
> [vLLM - Qwen](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)
> [Qwen3.6-35B-A3B · Models](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)
> [vllm-project/vllm v0.20.1 on GitHub - NewReleases.io](https://newreleases.io/project/github/vllm-project/vllm/release/v0.20.1)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
> [Previous Releases - vLLM](https://vllm.ai/releases)
> [Qwen3 使用指南 - vLLM 方案 - vLLM 文档](https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3.html)
>
> **Searching for relevant open source lines**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Releases · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases)
>
> **Considering accessing vLLM code**
>
> I might need to look into the raw vLLM code to verify any technical details, particularly for the facts that need to be cited in Chapter 2. I’ll check if the code is available publicly and whether it'll provide the clarity I'm looking for on how vLLM works.
>
> **Searching for raw open vLLM code**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Clarifying source snippet for citation**
>
> It seems like the web source may be compressed due to long lines or format issues. I need to search for exact snippets to get the most accurate, properly formatted quotations. I'll ensure I find clear sections for citation without weird line breaks or formatting problems.
>
> **Searching for citations on web raw compression issues**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Searching for PagedAttention and continuous batching docs**
>
> I might need documentation on PagedAttention and continuous batching to ensure I have the right technical details. I'll search for these terms to find the most relevant resources. This will help me get a clearer understanding for accurate inclusion in the work.
>
> **Searching for PagedAttention continuous batching documentation**
>
> [vLLM Continuous Batching 和KVCache - 向着朝阳 - 博客园](https://www.cnblogs.com/aibi1/p/19489019)
> [Offline Inference - vLLM](https://docs.vllm.ai/en/latest/serving/offline_inference/)
> [vLLM 原理深度解析 (PagedAttention , Continuous Batching等 ...](https://blog.csdn.net/m0_38097087/article/details/149334128)
> [vllm/docs at main · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/tree/main/docs)
> [详解vLLM PagedAttention优化大模型推理KV Cache内存的 ...](https://developer.aliyun.com/article/1664805)
> [vLLM: PagedAttention & Continuous Batching - 知乎](https://zhuanlan.zhihu.com/p/1912879297791767692)
>
> **Searching for official PagedAttention documentation**
>
> To ensure I get accurate details, I’ll search for the official documentation or design papers about PagedAttention. I'll look for terms like "vLLM PagedAttention paper official" to find the most authoritative and detailed resources available, so I can include that in the work.
>
> **Searching for PagedAttention documentation and paper**
>
> [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/)
> [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
> [Efficient Memory Management for Large Language Model Serving with ...](https://www.cs.princeton.edu/~ravian/COS597_F24/papers/vllm.pdf)
> [详解vLLM PagedAttention优化大模型推理KV Cache内存的 ...](https://developer.aliyun.com/article/1664805)
>
> **Checking latest v0.20.1 documentation**
>
> I might need to check the latest documentation for v0.20.1 to ensure I have the most up-to-date information. I'll look for the main docs, focusing on the newest version, to make sure all facts are current and accurate. This will help me precisely include the right details.
>
> **Checking for version 0.20.1 documentation**
>
> [vLLM](https://docs.vllm.ai/)
> [[2309.06180] Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
>
> **Planning response for Chapter 2**
>
> I need to focus on delivering a conceptual answer for Chapter 2, centering on the "attention" formula while avoiding unnecessary widgets. It seems like the user wants details, but without the full complexity, so I'll use LaTeX for formulas. I'll provide evidence and cite sources where necessary, highlighting Qwen3.6 specifics only for contrast.
>
> Thought for 37s

第二章我会把“传统 decoder-only LLM 推理”作为骨架讲清楚，同时不断标注哪些地方对 Qwen3.6 只是类比、哪些地方会在后续章节被替换成 Gated DeltaNet / Gated Attention / MoE / MTP。

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

# 小白总结

LLM 推理就是“读题 + 接龙”。

读题阶段叫 **prefill**，模型把 prompt 全部看完，并存下以后要用的 KV cache。接龙阶段叫 **decode**，模型每次生成一个 token，再把它接回上下文继续生成。

传统 Transformer 可以帮助我们理解基本流程，但 Qwen3.6-35B-A3B 更复杂：它不是普通文本 decoder，而是带 Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP 的 hybrid 模型。

---

# 工程师总结

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

# 关键公式

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

# 关键源码入口

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

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. 为什么 LLM 推理要分 prefill 和 decode？**  
提示：prompt 阶段可以并行处理多个历史 token；decode 阶段每一步依赖上一步生成的 token。

**2. 为什么长上下文场景下，权重不是唯一显存瓶颈？**  
提示：KV cache 会随 sequence length 增长，262K context 会显著放大这个问题。

**3. 为什么不能把 Qwen3.6-35B-A3B 当成普通 Transformer Decoder？**  
提示：它有 Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP，并且 MoE 是 256 experts、8 routed + 1 shared expert。

---

# 下一章预告

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

## Prompt:
2026/5/9 09:52:11

继续第三章

## Response:
2026/5/9 09:52:13

> **Browsing for updated model card and source references**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Searching for raw sources on GitHub exclusions**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Inspecting code lines on GitHub around line 1300**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Searching for class details on forward and attention gating**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Searching for Qwen model card citation lines**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> **Searching for Open Qwen model lines and FP8 info**
>
> [Qwen/Qwen3.6-35B-A3B-FP8 · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
>
> **Planning the response with citations and structure**
>
> I need to provide a detailed, robust answer while avoiding over-explaining. Citations are important, especially for factual claims, and I'll include a source evidence table. I'll mention "vLLM v0.20.1" and GitHub-based code facts. To keep it focused, I should also balance detail with readability, highlighting key aspects like the Qwen3.6 architecture.
>
> **Clarifying math and architecture details**
>
> I should avoid diving into exact DeltaNet math unless it's confirmed. It might be useful to clarify that some abstractions are for understanding, not exact code representations. I'll include the key formulas and possibly a source evidence table. I’ll also detail the architecture, mentioning the layer layout and referencing sources like HF and vLLM.
>
> **Answering educational query with citations and formulas**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Finalizing structured response with citations**
>
> I need to structure my response with citations properly, avoiding raw URLs. Citations should be placed in paragraphs, possibly at the end, and maybe in a "依据" column for tables. I'll mention that not all details are included. I must adhere to the "关键事实必须引用" guideline while organizing everything clearly. I'll make sure to cite internet sources and specifics like the vLLM recipe and HF model card.
>
> Thought for 32s

第三章会从“结构地图”切入：先把 Qwen3.6 的层级、路径和传统 Transformer 的对应关系说清，再把 Gated DeltaNet、Gated Attention、MoE、MTP 分别落到张量、显存和 vLLM 源码入口。

# 第三章：Qwen3.6 架构总览  
## Gated DeltaNet、Gated Attention、MoE、MTP

本章目标：把第二章的“传统 LLM 推理骨架”替换成 **Qwen/Qwen3.6-35B-A3B 的真实结构地图**。

先给一句总括：

> Qwen3.6-35B-A3B 不是“40 层普通 self-attention + MLP”的 Transformer Decoder，而是一个 **带 Vision Encoder 的 causal LM**，语言主干是 **Gated DeltaNet / Gated Attention / MoE / MTP** 组合结构。

模型卡确认：它是 **Causal Language Model with Vision Encoder**，总参数 35B、每 token 激活 3B，hidden size 2048，40 层，padded token embedding 和 LM output 都是 248,320；hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`；MoE 是 256 experts，每 token 激活 8 routed experts + 1 shared expert；MTP trained with multi-steps；原生上下文 262,144 tokens，可扩展到 1,010,000 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

传统 Transformer Decoder 像一个固定流程的工厂：

```text
每一层：
  全局读一遍历史上下文
  然后过一个固定的 MLP 加工站
```

Qwen3.6 更像一个混合型专家工厂：

```text
输入：
  文本 token
  或图像 / 视频经过 Vision Encoder 后形成的视觉 embedding

语言主干：
  快速记忆工位：Gated DeltaNet
  专家加工站：MoE
  快速记忆工位：Gated DeltaNet
  专家加工站：MoE
  快速记忆工位：Gated DeltaNet
  专家加工站：MoE
  全局注意力工位：Gated Attention
  专家加工站：MoE

上述结构重复 10 次
```

所以它不是“每层都完整翻阅历史记录”。更准确的直觉是：

| 模块 | 类比 | 作用 |
|---|---|---|
| Vision Encoder | 图像/视频前台接待员 | 把视觉输入转成语言模型可消费的 embedding |
| Gated DeltaNet | 快速滚动笔记 | 用状态化/线性注意力风格处理长序列信息 |
| Gated Attention | 全局复盘会议 | 用 full attention 看历史上下文 |
| MoE | 256 人专家团 | 每个 token 只找 8 个 routed experts + 1 个 shared expert |
| MTP | 草稿预测员 | 辅助提前预测多个未来 token，用于 speculative decoding |

vLLM recipe 也把 Qwen3.5/Qwen3.6 描述为 multimodal MoE models，并指出它们具有 gated delta networks 架构；同一 recipe 给出的 Qwen3.6 serving 命令使用 `--tensor-parallel-size 8`、`--max-model-len 262144` 和 `--reasoning-parser qwen3`，还给出 MTP speculative decoding 配置。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 二、数学直觉

### 1. 传统 Decoder 层的抽象

传统 Transformer Decoder 可以粗略写成：

$$
h_{\ell+1}
=
h_{\ell}
+
\text{MLP}(
  h_{\ell}
  +
  \text{Attention}(h_{\ell})
)
$$

更常见的 pre-norm 写法是：

$$
u_{\ell}=h_{\ell}+\text{Attention}(\text{Norm}(h_{\ell}))
$$

$$
h_{\ell+1}=u_{\ell}+\text{MLP}(\text{Norm}(u_{\ell}))
$$

这只是传统模型骨架，不是 Qwen3.6 的完整结构。

### 2. Qwen3.6 的层级抽象

Qwen3.6 的一个 4 层小周期可以抽象为：

```text
Layer A: Gated DeltaNet  → MoE
Layer B: Gated DeltaNet  → MoE
Layer C: Gated DeltaNet  → MoE
Layer D: Gated Attention → MoE
```

这个小周期重复 10 次，得到 40 层。模型卡明确给出这个 hidden layout。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Gated DeltaNet 的直觉

传统 full attention 的核心是：

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

它要显式地把当前 query 和历史所有 key 做交互。

Gated DeltaNet 可以先理解成“状态化的线性注意力风格模块”：它不应该被简化成普通 attention。源码中 `GatedDeltaNetAttention` 属于 `PluggableLayer`，并继承 `MambaBase`；它有 `mamba_type = "gdn_attention"`，状态 dtype/shape 通过 `MambaStateDtypeCalculator.gated_delta_net_state_dtype` 和 `MambaStateShapeCalculator.gated_delta_net_state_shape` 计算。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

本章只做架构总览，不强行展开 Gated DeltaNet 的完整递推公式；原因是它的具体 kernel、state layout、chunked computation 和 speculative decoding 交互需要到第六、十一、十二章结合源码和显存布局一起讲。

### 4. Gated Attention 的直觉

Gated Attention 可以理解为 full attention 加了输出门控。vLLM v0.20.1 的 `Qwen3NextAttention` 源码中会读取 `config.num_attention_heads`、`config.num_key_value_heads`、`config.head_dim`，并有 `attn_output_gate = getattr(config, "attn_output_gate", True)`；这说明它不是一个只含普通 QKV 投影的最简 attention 模块。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

抽象成公式可以写成：

$$
a_t = \text{Attention}(q_t, K_{\le t}, V_{\le t})
$$

$$
y_t = g_t \odot a_t
$$

其中 $g_t$ 是门控信号，$\odot$ 是逐元素乘法。这个公式是教学抽象，不代表 vLLM 源码中的逐行实现。

### 5. MoE 的直觉

传统 MLP 是：

$$
y = \text{MLP}(h)
$$

MoE 是：

$$
y =
\sum_{e \in \text{TopK}(r(h))}
w_e \cdot E_e(h)
+
E_{\text{shared}}(h)
$$

其中 $r(h)$ 是 router，$E_e$ 是第 $e$ 个 expert，$w_e$ 是路由权重。Qwen3.6 的模型卡确认有 256 experts，每 token 激活 8 routed experts + 1 shared expert，expert intermediate dimension 为 512。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

vLLM v0.20.1 中，MoE block 对应 `Qwen3NextSparseMoeBlock`；源码中它读取 `config.num_experts` 作为 routed experts 数，并使用 expert parallel group 信息，如 `get_ep_group()`、`ep_rank`、`ep_size`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 6. MTP 的直觉

传统 autoregressive decoding 是：

```text
主模型每次预测 1 个 token
接受后再预测下一个
```

MTP / speculative decoding 的直觉是：

```text
辅助头或辅助路径先草拟多个未来 token
主模型验证其中可以接受的 token
一次推进多个位置
```

Qwen3.6 模型卡写明 MTP trained with multi-steps；vLLM recipe 给出 Qwen3.6 的 MTP speculative decoding 命令，使用 `--speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'`。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 三、张量形状

为了统一后续章节，我们用 vLLM serving 中常见的 flattened token 视角：

| 符号 | 含义 |
|---|---|
| $N$ | 当前 batch 内被合并后的 token 总数 |
| $B$ | 请求数 / sequence 数 |
| $T$ | 单个请求的序列长度 |
| $H$ | hidden size |
| $V$ | vocab size |
| $E$ | expert 数 |
| $K$ | 每 token routed experts 数 |

对 Qwen3.6-35B-A3B：

```text
H = 2048
V = 248320 padded
layers = 40
experts = 256
routed experts per token = 8
shared experts per token = 1
native context = 262144
```

这些值来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 1. 文本输入

```text
input_ids:       [N]
hidden_states:   [N, 2048]
logits:          [N, 248320]
```

vLLM 中很多模型 forward 会把 batch 内 token flatten 后处理，所以工程上经常看到 `[num_tokens, hidden_size]`，而不是教学里常见的 `[B, T, H]`。

### 2. Vision Encoder 输入

多模态输入时：

```text
image / video
→ Vision Encoder
→ multimodal_embeddings
→ merge into text embeddings
→ language model
```

vLLM v0.20.1 的 `qwen3_5.py` 中，Qwen3.5/3.6 多模态类会构造 `Qwen3_VisionTransformer`，并构造 `Qwen3_5MoeForCausalLM` 作为 language model；`embed_input_ids` 会先得到文本 embedding，再在存在 multimodal embeddings 时调用 `_merge_multimodal_embeddings` 合并。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 3. Gated DeltaNet 形状

模型卡给出 Gated DeltaNet 的头部配置：

```text
V-side linear attention heads: 32
QK-side linear attention heads: 16
head dim: 128
```

这说明它的 QK 和 V 侧 head 组织并不等同于普通 MHA 的 “Q/K/V heads 完全对齐” 形式。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

vLLM 源码中，`GatedDeltaNetAttention` 读取 `linear_num_value_heads`、`linear_num_key_heads`、`linear_key_head_dim`、`linear_value_head_dim`，并计算 `key_dim` 等内部尺寸。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

### 4. Gated Attention 形状

模型卡给出 Gated Attention 的配置：

```text
Q heads: 16
KV heads: 2
head dim: 256
RoPE dim: 64
```

这更接近 GQA：query heads 多，KV heads 少。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

传统 attention 的直觉形状是：

```text
Q: [N, num_q_heads, head_dim]
K: [N, num_kv_heads, head_dim]
V: [N, num_kv_heads, head_dim]
```

对 Qwen3.6 的 Gated Attention：

```text
Q: [N, 16, 256]   before tensor parallel partition
K: [N, 2, 256]    before tensor parallel partition
V: [N, 2, 256]    before tensor parallel partition
```

源码中 `Qwen3NextAttention` 会根据 tensor parallel size 切分 `total_num_heads` 和 `total_num_kv_heads`，如果 KV heads 少于 TP size，则会复制 KV heads；如果 KV heads 大于等于 TP size，则跨 TP 分区。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 5. MoE 形状

MoE 的 router logits 可以理解为：

```text
hidden_states:   [N, 2048]
router_logits:   [N, 256]
topk_indices:    [N, 8]
expert_outputs:  [N, 8, 2048]
shared_output:   [N, 2048]
final_output:    [N, 2048]
```

vLLM `Qwen3NextSparseMoeBlock` 的 forward 片段显示：router logits 形状注释为 `(num_tokens, n_experts)`，随后调用 `self.experts(hidden_states=hidden_states, router_logits=router_logits)`；在 sequence parallel 场景下还会做 `tensor_model_parallel_all_gather`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 四、计算流程

Qwen3.6 推理流程可以分成两条路径：多模态前处理路径和语言模型主干路径。

### 1. 多模态路径

```text
image / video
→ Qwen3_VisionTransformer
→ multimodal embeddings
→ merge with text embeddings
→ language model
```

vLLM v0.20.1 中，`Qwen3_5MoeForConditionalGeneration` 注册了 `Qwen3VLMultiModalProcessor`，构造 `Qwen3_VisionTransformer`，再构造 `Qwen3_5MoeForCausalLM`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 2. 语言主干路径

```text
input embeddings
→ 40 layers
→ final norm
→ lm_head
→ logits
→ sampling
```

但 40 层不是同构的普通 Transformer。Qwen3.6 的真实周期是：

```text
repeat 10 times:
  Gated DeltaNet  → MoE
  Gated DeltaNet  → MoE
  Gated DeltaNet  → MoE
  Gated Attention → MoE
```

模型卡确认了这个 hidden layout。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. vLLM 层选择路径

vLLM v0.20.1 中，`Qwen3_5DecoderLayer` 根据 `layer_type` 做分支：

```text
if layer_type == "linear_attention":
    use GatedDeltaNetAttention
elif layer_type == "full_attention":
    use Qwen3NextAttention
else:
    raise ValueError
```

同一个 decoder layer 之后再根据 config model type 选择 MoE 或 dense MLP；对于 `qwen3_5_moe_text`，使用 `Qwen3NextSparseMoeBlock`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

这就是本章最重要的源码映射：**Gated DeltaNet / Gated Attention 是 attention 路径分支，MoE 是每层后面的 MLP 替代物。**

---

## 五、显存布局

Qwen3.6 的显存不能只按传统 Transformer 看。

### 1. 权重显存

35B 总参数意味着 BF16 权重粗略是 70GB 量级；FP8 checkpoint 会显著降低权重占用。Qwen3.6 官方 FP8 模型卡说明：该仓库包含 FP8-quantized weights/config，采用 fine-grained FP8 quantization，block size 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

### 2. KV Cache

Gated Attention 层仍然需要 full attention 风格的 KV cache。因为它有 Q heads、KV heads、head dim 和 RoPE dim，decode 时需要读取历史 K/V。

### 3. Gated DeltaNet / Mamba-style state

Gated DeltaNet 不应简单归入普通 KV cache。vLLM 源码把它作为 `MambaBase` 风格的模块处理，并通过 mamba state dtype/shape calculator 计算状态。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py))

这意味着后续讲第六章“KV Cache 与长上下文显存问题”时，我们必须区分：

```text
full attention KV cache
Gated DeltaNet / Mamba-like state cache
MoE expert weights
MTP speculative token 额外占用
```

### 4. Vision Encoder 显存

多模态服务还要加载 vision encoder。vLLM recipe 对 Qwen3.5 的 text-only throughput 配置说明：`--language-model-only` 可以跳过 vision encoder，把更多显存留给 KV cache；对于 Qwen3.6，这个思路同样是部署规划中必须考虑的方向，但具体命令和可用性要以 v0.20.1 实测为准。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 六、vLLM 源码映射

本章的核心源码地图如下。

| 架构概念 | vLLM v0.20.1 源码入口 | 说明 |
|---|---|---|
| Qwen3.6 multimodal MoE 模型 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` |
| Vision Encoder | `vllm/model_executor/models/qwen3_5.py` | 构造 `Qwen3_VisionTransformer` |
| Language Model | `vllm/model_executor/models/qwen3_5.py` | 构造 `Qwen3_5MoeForCausalLM` |
| Hybrid decoder layer | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Full / Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Sparse MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| MTP | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP`, `Qwen3NextMultiTokenPredictor` |
| GDN attention backend metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder` 等后续再展开 |

特别注意：虽然文件名里是 `qwen3_5.py`，但 vLLM v0.20.1 的该文件包含 Qwen3.5/Qwen3.6 兼容路径。源码中 `Qwen3_5MoeForConditionalGeneration` 构造 `Qwen3_VisionTransformer` 和 `Qwen3_5MoeForCausalLM`，并通过 `set_moe_parameters()` 收集 MoE 层信息。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

---

## 七、NVIDIA GPU 执行视角

从 GPU 看，Qwen3.6 不是一条简单的 dense GEMM 流水线，而是四类计算混在一起：

| 模块 | GPU 计算特征 | 主要瓶颈 |
|---|---|---|
| Gated DeltaNet | 状态化/线性注意力风格计算，包含投影、卷积/状态更新、门控 | kernel 选择、状态读写、chunk 组织 |
| Gated Attention | QKV projection + full attention + output gate | KV cache 读带宽、attention backend |
| MoE | router + token dispatch + expert GEMM + combine | expert dispatch、GEMM 分组、通信 |
| MTP | 额外 draft token 路径 + 主模型验证 | 接受率、额外 KV/state 占用、吞吐/延迟权衡 |

MTP 并不是免费午餐。vLLM recipe 的配置说明指出，MTP-1 可降低 per-token latency，但在高并发下会降低 text throughput，因为 speculative tokens 会消耗 KV cache capacity，降低有效 batch size；`num_speculative_tokens` 可调，但接受率和吞吐存在权衡。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

对 NVIDIA GPU 优化来说，Qwen3.6 的难点是：

```text
不是只优化 attention
也不是只优化 GEMM
而是要同时优化：
  long context memory
  MoE expert dispatch
  tensor/expert parallel communication
  GDN state cache
  full attention KV cache
  MTP speculative path
```

---

## 八、优化前 vs 优化后

### 优化前：把它当普通 Transformer 跑

错误心智模型：

```text
40 layers × (Attention + MLP)
```

会导致几个误判：

| 误判 | 后果 |
|---|---|
| 以为每层都有完整 KV cache | 高估或误判 Gated DeltaNet 层缓存结构 |
| 以为 MLP 是 dense | 忽略 router、expert dispatch、shared expert |
| 以为 FP8 只影响权重 | 忽略 KV/state cache 和 MoE dispatch 的瓶颈 |
| 以为 MTP 一定提升吞吐 | 忽略 speculative tokens 消耗缓存容量 |
| 以为多模态只是输入格式变化 | 忽略 vision encoder 显存和吞吐影响 |

### 优化后：按真实模块分治

正确心智模型：

```text
Vision Encoder
+
10 × (
  3 × (Gated DeltaNet → Sparse MoE)
  1 × (Gated Attention → Sparse MoE)
)
+
MTP optional path
```

对应优化策略：

| 模块 | 优化方向 |
|---|---|
| Vision Encoder | text-only 时考虑跳过；多模态时考虑 encoder data parallel |
| Gated DeltaNet | 关注 mamba/GDN state、chunk、backend |
| Gated Attention | 关注 KV cache、attention backend、RoPE |
| MoE | 关注 expert parallel、router top-k、dispatch、expert GEMM |
| MTP | 关注接受率、TPOT、KV/state 额外占用 |
| 262K context | 关注 max model len、KV capacity、prefix caching、chunked prefill |

---

## 九、工程落地

如果你要在 NVIDIA GPU 上部署 Qwen3.6-35B-A3B，第三章阶段先建立以下落地判断。

### 1. 最小正确启动命令

vLLM recipe 给出的 Qwen3.6 基础命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

带 MTP speculative decoding 的命令是：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

这两个命令均来自 vLLM Qwen3.5/Qwen3.6 recipe。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 2. FP8 单卡不是“完整等价配置”

Qwen3.6 有官方 FP8 checkpoint，权重量化方式是 fine-grained FP8、block size 128。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8))

但工程上要区分：

```text
能加载权重
≠ 能跑 262K 上下文
≠ 能高并发服务
≠ 能多模态 + 长上下文 + MTP 同时打开
```

FP8 主要降低权重和部分 GEMM 压力；长上下文下，full attention KV cache、GDN state、MTP speculative tokens 和 workspace 仍然可能成为显存瓶颈。

### 3. text-only 与 multimodal 要分开规划

Qwen3.6 是带 Vision Encoder 的 causal LM。多模态模式下，要考虑 vision encoder 显存、预处理缓存和视觉 token 数；text-only 模式下，应尽量把显存留给 language model 和 KV/state cache。vLLM recipe 对 Qwen3.5 text-only 说明了 `--language-model-only` 可跳过 vision encoder 并释放 KV cache 显存，这给 Qwen3.6 的部署规划提供了明确方向。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 4. MTP 要按 workload 开关

低并发、追求 TPOT 时，可以试 MTP；高并发、追求总吞吐时，MTP 可能因为额外 speculative tokens 占用 cache 而降低有效 batch size。这个权衡在 vLLM recipe 的配置提示中已有明确说明。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 十、本章总结

Qwen3.6-35B-A3B 的核心结构可以压缩成一句话：

```text
多模态输入通过 Vision Encoder 进入语言模型；
语言模型不是普通 Transformer，而是 10 个周期的
3 × (Gated DeltaNet → MoE) + 1 × (Gated Attention → MoE)；
推理时还可以使用 MTP speculative decoding。
```

传统 Transformer 是本课程的“入门参照物”，但从第三章开始，我们要把真实服务中的问题拆成四条主线：

```text
1. Gated DeltaNet：长上下文状态化计算
2. Gated Attention：full attention + KV cache + RoPE
3. MoE：router + experts + dispatch + parallelism
4. MTP：draft tokens + verification + latency/throughput trade-off
```

---

# 小白总结

Qwen3.6 不是一个普通“读上下文然后回答”的模型。

它更像一个混合系统：

```text
有看图/视频的前端
有快速记忆模块 Gated DeltaNet
有全局注意力模块 Gated Attention
有 256 个专家组成的 MoE
还有提前打草稿的 MTP
```

所以后面讲显存、速度、部署、源码时，不能只用“普通 Transformer”解释。

---

# 工程师总结

工程上，Qwen3.6 的复杂性来自四个方向叠加：

```text
Hybrid attention:
  Gated DeltaNet + Gated Attention

Sparse computation:
  256 experts, top-8 routed + shared expert

Long context:
  native 262K context

Speculative decoding:
  MTP trained with multi-steps
```

在 vLLM v0.20.1 中，关键源码入口已经确认：

```text
qwen3_5.py:
  Qwen3_5DecoderLayer
  Qwen3_5MoeForConditionalGeneration
  Qwen3_5MoeForCausalLM

qwen3_next.py:
  Qwen3NextAttention
  Qwen3NextSparseMoeBlock

gdn_linear_attn.py:
  GatedDeltaNetAttention

qwen3_next_mtp.py:
  Qwen3NextMTP
```

---

# 关键公式

**1. 传统 attention**

$$
\text{Attention}(Q,K,V)
=
\text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

**2. Gated Attention 教学抽象**

$$
a_t = \text{Attention}(q_t, K_{\le t}, V_{\le t})
$$

$$
y_t = g_t \odot a_t
$$

**3. MoE 路由抽象**

$$
r = \text{Router}(h)
$$

$$
\mathcal{E}_{topk} = \text{TopK}(r, k=8)
$$

$$
y =
\sum_{e \in \mathcal{E}_{topk}}
w_e E_e(h)
+
E_{\text{shared}}(h)
$$

**4. Qwen3.6 层级结构**

$$
10 \times
\left[
3 \times
(\text{GatedDeltaNet} \rightarrow \text{MoE})
+
1 \times
(\text{GatedAttention} \rightarrow \text{MoE})
\right]
$$

**5. MTP 推理直觉**

$$
\text{draft tokens}
=
(\hat{x}_{t+1}, \hat{x}_{t+2}, \ldots, \hat{x}_{t+k})
$$

$$
\text{accepted prefix}
\subseteq
\text{draft tokens}
$$

---

# 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 多模态 MoE 总入口 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` |
| Vision Encoder | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer` |
| Language Model | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForCausalLM` |
| Hybrid decoder layer | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Sparse MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| MTP | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP` |
| GDN backend metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder` 等，后续确认展开 |

---

# 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 是带 Vision Encoder 的 causal LM，35B total / 3B activated，40 层，hidden size 2048 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MoE 是 256 experts，8 routed + 1 shared | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MTP trained with multi-steps | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM recipe 给出 Qwen3.6 TP=8、262144 max model len、qwen3 reasoning parser 命令 | vLLM Qwen3.5/Qwen3.6 recipe | `vllm serve Qwen/Qwen3.6-35B-A3B ...` | recipe 确认 | 是 |
| vLLM recipe 给出 Qwen3.6 MTP speculative decoding 配置 | vLLM Qwen3.5/Qwen3.6 recipe | `--speculative-config '{"method": "mtp", ...}'` | recipe 确认 | 是 |
| vLLM v0.20.1 中 Qwen3.6 MoE multimodal 总入口继承 multimodal conditional generation 与 MoE protocol | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 multimodal 路径构造 `Qwen3_VisionTransformer` | `vllm/model_executor/models/qwen3_5.py` | `self.visual = Qwen3_VisionTransformer(...)` | 源码直接确认 | 是 |
| vLLM v0.20.1 中 language model 路径构造 `Qwen3_5MoeForCausalLM` | `vllm/model_executor/models/qwen3_5.py` | `self.language_model = Qwen3_5MoeForCausalLM(...)` | 源码直接确认 | 是 |
| `Qwen3_5DecoderLayer` 根据 `layer_type` 选择 Gated DeltaNet 或 Qwen3NextAttention | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer.__init__` | 源码直接确认 | 是 |
| linear attention 路径使用 `GatedDeltaNetAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.linear_attn = GatedDeltaNetAttention(...)` | 源码直接确认 | 是 |
| full attention 路径使用 `Qwen3NextAttention` | `vllm/model_executor/models/qwen3_5.py` | `self.self_attn = Qwen3NextAttention(...)` | 源码直接确认 | 是 |
| MoE text config 下使用 `Qwen3NextSparseMoeBlock` | `vllm/model_executor/models/qwen3_5.py` | `self.mlp = Qwen3NextSparseMoeBlock(...)` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 读取 `config.num_experts` 并使用 expert parallel group | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock.__init__` | 源码直接确认 | 是 |
| `Qwen3NextAttention` 读取 Q heads、KV heads、head dim，并有 `attn_output_gate` | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention.__init__` | 源码直接确认 | 是 |
| `GatedDeltaNetAttention` 注册为 pluggable layer，并返回 `mamba_type = "gdn_attention"` | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | 源码直接确认 | 是 |
| vLLM v0.20.1 中存在 Qwen3Next MTP 模型入口 | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP` | 源码直接确认 | 是 |
| `Qwen3NextMTP` 当前不支持 `mamba_cache_mode="all"`，提示使用 `--mamba-cache-mode=align` | `vllm/model_executor/models/qwen3_next_mtp.py` | `Qwen3NextMTP.__init__` | 源码直接确认 | 是 |
| GDN backend metadata 具体如何构造 block table、spec decode metadata | `vllm/v1/attention/backends/gdn_attn.py` | `build(...)` | 待源码确认 | 否 |
| MoE kernel 最终选择 CUTLASS / Triton / fused_moe 的条件 | `vllm/model_executor/layers/fused_moe/` | 待定 | 待源码确认 | 否 |

---

# 源码阅读作业

按下面顺序读，不要求一次读懂 kernel。

**第一步：读模型总入口**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5MoeForConditionalGeneration
Qwen3_5MoeForCausalLM
Qwen3_5DecoderLayer
embed_input_ids
set_moe_parameters
```

目标：确认 Vision Encoder、language model、MoE 参数收集是如何串起来的。

**第二步：读 hybrid layer 分支**

```text
Qwen3_5DecoderLayer.__init__
```

目标：确认：

```text
linear_attention → GatedDeltaNetAttention
full_attention   → Qwen3NextAttention
qwen3_5_moe_text → Qwen3NextSparseMoeBlock
```

**第三步：读 MoE block**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextSparseMoeBlock
self.gate
self.experts
router_logits
```

**第四步：读 MTP 入口**

```text
vllm/model_executor/models/qwen3_next_mtp.py
```

重点找：

```text
Qwen3NextMTP
Qwen3NextMultiTokenPredictor
mamba_cache_mode
compute_logits
```

---

# 实验验证作业

实验重点不是追求性能，而是验证“结构差异”。

### 实验 1：查看 config

下载或查看 HF config，确认：

```text
hidden_size
num_hidden_layers
num_experts
num_experts_per_tok
linear_num_key_heads
linear_num_value_heads
num_attention_heads
num_key_value_heads
head_dim
```

目标：把模型卡上的结构参数和 config 字段对应起来。

### 实验 2：启动基础 serving

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

记录：

```text
启动日志中的 model config
GPU memory usage
KV cache capacity
是否加载 vision encoder
是否出现 GDN / Mamba cache 相关日志
```

### 实验 3：启动 MTP 对比

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

对比：

```text
TTFT
TPOT
tokens/s
显存占用
可承载并发数
```

结论不要只看单请求延迟；还要看高并发下吞吐是否下降。

---

# 3 个自测题

**1. 为什么 Qwen3.6-35B-A3B 不能被称为普通 Transformer Decoder？**  
答案要包含：Vision Encoder、Gated DeltaNet、Gated Attention、MoE、MTP。

**2. Qwen3.6 的 40 层 hidden layout 如何组织？**  
答案应能写出：

```text
10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))
```

**3. 为什么 MTP 可能降低高并发吞吐？**  
提示：speculative tokens 会消耗 KV/state cache capacity，降低有效 batch size；接受率也会影响收益。

---

# 下一章预告

第四章进入：

```text
Token、Embedding、Logits、Sampling
```

我们会从最基础的 token id 开始，讲清楚：

```text
文本如何变成 input_ids
input_ids 如何变成 hidden_states
hidden_states 如何变成 logits
logits 如何经过 temperature / top-k / top-p 变成 next token
```

并且会特别注意 Qwen3.6 的两个差异：

```text
1. 多模态 embedding 如何合并进语言序列
2. vocab / lm_head 的 248320 padded 输出维度如何影响 logits 与 sampling
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
- [vllm/vllm/model_executor/layers/mamba/gdn_linear_attn.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/mamba/gdn_linear_attn.py)

## Prompt:
2026/5/9 10:12:39

继续第四章

## Response:
2026/5/9 10:12:41

> **Deciding on citation and sources approach**
>
> The user's asking for a continuation of chapter 4 in Chinese. Since facts need citations, I'll probably need to use web sources for verification. Citations should be from real sources, like web-accessible references. I also wonder if I should look into relevant source files, maybe even for math equations, but prefetched math may not be needed.
>
> **Considering course and source needs**
>
> The user is asking for a course, and it seems formulas might be important. I could include a math widget, but it’s not mandatory if the equations aren’t central. I’ll also need to verify sources, like the Qwen model card and relevant code files, and make sure I have the correct paths before searching.
>
> **Browsing for sources and citations from web references**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> **Clarifying source and code details**
>
> I might need to be more specific with the source, especially regarding lines around embedding and the `compute_logits` function. I should look at the relevant sections carefully to ensure I get all the details right before moving forward with the task.
>
> **Searching code for lines around embed, compute_logits**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Reviewing _get_logits and sampler**
>
> I might need to open up more lines of code for the `_get_logits` function and the sampler to fully understand how they work. It’s important to get a clear view of these sections before moving forward. I'll review the code thoroughly to make sure everything is clear.
>
> **Investigating _get_logits and sampler adjustments**
>
> [vllm/vllm/model_executor/layers/logits_processor.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py)
>
> **Troubleshooting sampler error**
>
> It seems like there might be an internal error in the `sampler.py` file, possibly due to a path change in vLLM v0.20.1. I’ll need to check if the file path has been updated and perform a search to confirm the exact issue. I’ll look into it more carefully.
>
> **Searching for vLLM sampling error solutions**
>
> [vllm/vllm/sampling_params.py at main · vllm-project/vllm](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)
> [vllm参数详细解释_vllm samplingparams-CSDN博客](https://blog.csdn.net/m0_37617067/article/details/142499463)
> [推理参数 | vLLM 中文站](https://vllm.hyper.ai/docs/api/inference_params/)
> [vllm 源码分析 (1) sampler (1) - 知乎](https://zhuanlan.zhihu.com/p/1931352112924722260)
> [sampling_params - vLLM](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
> [vllm的SamplingParams参数-腾讯云开发者社区-腾讯云](https://cloud.tencent.com/developer/article/2387116)
> [深入理解VLLM的SamplingParams参数](https://cloud.baidu.com/article/3262475)
>
> **Checking v0.20.1 sampling parameters**
>
> I think I might need to check the `sampling_params` in v0.20.1 to ensure they're accessible or open, as it could be key to resolving the issue. It's probably a good idea to verify these parameters to see if they’re causing any problems in the model's sampling process.
>
> **Searching for v0.20.1 sampling_params open details**
>
> [vllm/vllm/sampling_params.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py)
>
> **Verifying sampler path**
>
> I think I might need to check the sampler path, possibly `v1/sample/sampler.py`. It’s important to ensure the correct path is open and accessible in case it’s affecting the process. I'll go check that out and make sure everything is lined up properly.
>
> **Searching for sampler path in v1/sample folder**
>
> [vllm/vllm/v1/sample/sampler.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/sample/sampler.py)
>
> **Considering raw data source**
>
> I’m thinking I might need to open the raw data since the HTML lines aren't quite working as expected. Maybe raw URLs will be more effective here. I'll check to see if that approach works better and get things moving smoothly. Maybe that’s the key!
>
> **Parsing raw URLs for better results**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py)
>
> **Opening raw data for more details**
>
> I'm wondering if the data was collapsed due to minification, but maybe it's already enough. I think I should check out the `sampling_params` details carefully
>
> **Investigating raw data collapse and sampling parameters**
>
> [vllm/vllm/sampling_params.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py)
>
> **Considering class documentation lines**
>
> I think I may need the class documentation starting from line 2393 onward. That part might have the specific details I’m looking for. I'll review that section to see if it provides the clarity or info I need for this task. Now to focus on that!
>
> **Fetching class documentation from line 2393 onward**
>
> [vllm/vllm/sampling_params.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py)
>
> **Organizing chapter and citations**
>
> I need to structure my answer carefully, ensuring I include citations from the web without raw URLs in the main content, but I'll show sources where needed. The user’s request involves several sections, including Token, Embedding, Logits, Sampling, as well as "critical formulas" and "self-tests." I’ll ensure citations are accurate and avoid including them in code blocks. I'll also reference specific paths and models like Qwen, sticking to external documentation for sources.
>
> Thought for 34s

# 第四章：Token、Embedding、Logits、Sampling

本章从推理链路中最“入口”和最“出口”的部分讲起：

```text
文本 / 多模态输入
→ token ids / multimodal embeddings
→ embedding
→ Qwen3.6 language model
→ hidden states
→ lm_head
→ logits
→ logits processor
→ sampling
→ next token
→ detokenize
```

对传统 LLM 来说，这一章主要讲 token、embedding、logits、softmax、采样。对 **Qwen/Qwen3.6-35B-A3B** 来说，还必须额外注意两点：第一，它是 **Causal Language Model with Vision Encoder**，因此输入不只有纯文本 token；第二，它的 token embedding 和 LM output 都是 **248,320 padded**，hidden size 是 **2048**，所以 logits 维度非常大，sampling 本身也会成为真实 serving 中不可忽略的计算环节。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

可以把 LLM 看成一个“只认识编号、不认识文字”的工厂。

用户输入：

```text
帮我解释 KV Cache
```

模型内部不会直接处理这些汉字，而是先把它们变成 token id：

```text
[1532, 37046, 912, 128xxx, ...]
```

每个 token id 再去查一张“向量字典表”：

```text
1532   → 一个 2048 维向量
37046  → 一个 2048 维向量
912    → 一个 2048 维向量
```

这张字典表就是 **embedding table**。

模型经过 40 层 Qwen3.6 hybrid block 后，最后会输出一个 hidden state。这个 hidden state 再通过 **LM head** 映射成对整个词表的打分：

```text
“的”     8.7
“是”     7.9
“缓存”   7.4
“。”     6.1
...
```

这些分数叫 **logits**。logits 不是概率，softmax 后才是概率。sampling 再根据 temperature、top-k、top-p 等规则，从概率分布中选出下一个 token。

---

## 二、数学直觉

### 1. Token id 是离散符号

token id 本身没有连续数学意义。比如：

```text
token id = 100
token id = 101
```

不代表 101 比 100 “语义更大”。它只是查表索引。

### 2. Embedding 是查表

设 vocabulary size 是 $V$，hidden size 是 $H$，embedding table 是：

$$
E \in \mathbb{R}^{V \times H}
$$

某个 token id 为 $i$，它的 embedding 是：

$$
x_i = E[i]
$$

对 Qwen3.6-35B-A3B，模型卡给出 hidden dimension 为 2048，token embedding 为 248,320 padded，因此可以把 embedding table 理解成接近：

$$
E \in \mathbb{R}^{248320 \times 2048}
$$

注意这里的 248,320 是 padded vocabulary size，不一定等于 tokenizer 原始词表语义上的“自然词数”。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. LM head 把 hidden state 投回词表空间

最后一层 hidden state：

$$
h_t \in \mathbb{R}^{H}
$$

LM head 权重：

$$
W_{\text{lm}} \in \mathbb{R}^{V \times H}
$$

logits：

$$
z_t = W_{\text{lm}} h_t
$$

所以：

$$
z_t \in \mathbb{R}^{V}
$$

对 Qwen3.6，单个 token 的 logits 可以理解成：

$$
z_t \in \mathbb{R}^{248320}
$$

vLLM v0.20.1 的 `Qwen3_5ForCausalLMBase` 在最后一个 pipeline rank 上，如果没有 tie word embedding，会创建 `ParallelLMHead(config.vocab_size, config.hidden_size)`；随后 `compute_logits` 调用 `self.logits_processor(self.lm_head, hidden_states)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 4. Softmax 把 logits 变概率

$$
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}
$$

如果加入 temperature：

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_{j=1}^{V}\exp(z_j / \tau)}
$$

其中：

| temperature | 效果 |
|---:|---|
| 接近 0 | 更接近 greedy，选最大 logit |
| 1.0 | 原始分布 |
| 大于 1 | 分布更平，随机性更强 |

vLLM v0.20.1 的 `SamplingParams` 文档说明：temperature 控制随机性，较低更确定，较高更随机，0 表示 greedy sampling；top-p 控制考虑的累计概率，top-k 控制考虑的最高分 token 数量。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

---

## 三、张量形状

### 1. 传统教学形状

普通教学里常写：

```text
input_ids:      [B, T]
embeddings:     [B, T, H]
hidden_states:  [B, T, H]
logits:         [B, T, V]
next_logits:    [B, V]
```

其中：

| 符号 | 含义 |
|---|---|
| $B$ | batch size |
| $T$ | sequence length |
| $H$ | hidden size |
| $V$ | vocab size |

### 2. vLLM serving 里的 flattened 形状

vLLM 推理时更常见的是 flatten 后的 token 视角：

```text
input_ids:      [num_tokens]
positions:      [num_tokens]
hidden_states:  [num_tokens, hidden_size]
logits:         [num_tokens, vocab_size]
```

`qwen3_5.py` 中 forward docstring 明确说 `input_ids` 是 flattened / concatenated input ids，`positions` 也是 flattened / concatenated position ids；如果启用 mrope，positions 形状可能是 `(3, seq_len)`，否则是 `(seq_len,)`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 3. 固定案例 Qwen3.6 的核心形状

```text
hidden_size = 2048
padded vocab / embedding size = 248320
LM output = 248320 padded
num_layers = 40
```

所以单步 decode 时，可以粗略理解成：

```text
hidden_states: [N, 2048]
logits:        [N, 248320]
sampled_ids:   [N]
```

这里 $N$ 是当前 batch 内要采样的 token 数，通常接近当前正在 decode 的请求数；如果是 prefill logprobs 或 prompt scoring，$N$ 也可能包含 prompt token。Qwen3.6 的这些模型尺寸来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 4. 多模态输入时的形状

Qwen3.6 是带 Vision Encoder 的 causal LM。vLLM v0.20.1 的 `Qwen3_5MoeForConditionalGeneration` 会构造 `Qwen3_VisionTransformer`，文本 embedding 先由 `self.language_model.embed_input_ids` 得到；如果存在 `multimodal_embeddings`，则调用 `_merge_multimodal_embeddings` 合并到输入 embedding 中。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

因此多模态时可以理解成：

```text
text input_ids
→ text embeddings: [N_text, 2048]

image / video
→ vision encoder
→ multimodal embeddings: [N_mm, 2048]

merge
→ input embeddings: [N_text + N_mm, 2048]
```

这里的核心是：**视觉内容最终也必须变成 language model 可消费的 hidden-size 向量**。但具体视觉 token 数量取决于图像/视频处理配置，本章不强行展开；原因是视觉 tokenization、grid、mrope 会在多模态部署章节中更自然。

---

## 四、计算流程

### 1. 输入到 embedding

```text
用户文本
→ tokenizer
→ input_ids
→ embedding table lookup
→ inputs_embeds
```

源码映射：

```text
Qwen3_5MoeForConditionalGeneration.embed_input_ids
→ _embed_text_input_ids
→ language_model.embed_input_ids
→ possible _merge_multimodal_embeddings
```

vLLM v0.20.1 源码确认：`embed_input_ids` 先调用 `_embed_text_input_ids`，使用 `self.language_model.embed_input_ids` 生成文本 embedding；若没有 multimodal embeddings，直接返回文本 embedding；否则再调用 `_merge_multimodal_embeddings`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

### 2. embedding 到 hidden states

```text
inputs_embeds
→ Qwen3.6 40 层语言主干
→ hidden_states
```

这里的 40 层不是普通 Transformer，而是：

```text
10 × (
  3 × (Gated DeltaNet → MoE)
  1 × (Gated Attention → MoE)
)
```

这一点来自 Qwen3.6 模型卡，后续第五章会进入 attention / gated attention / RoPE，第九章会进入 MoE。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. hidden states 到 logits

```text
hidden_states
→ lm_head
→ logits
→ logits processor
```

vLLM v0.20.1 源码中，`Qwen3_5ForCausalLMBase.forward` 调用 `self.model(...)` 得到 hidden states，`compute_logits` 再调用 `self.logits_processor(self.lm_head, hidden_states)`。`LogitsProcessor` 的职责说明包括：从 hidden states 收集 logits、必要时 scale logits、应用 logits processors。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

`LogitsProcessor._get_logits` 会调用 `lm_head.quant_method.apply(...)` 得到 logits，然后对 tensor parallel 的 logits 做 gather / all-gather，并去掉 vocab padding。也就是说，Qwen3.6 的 padded LM output 虽然是模型结构事实，但 vLLM 的 logits processor 有“remove paddings in vocab”的路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py))

### 4. logits 到 next token

vLLM v1 sampler 的 docstring 把采样流程写得很清楚：先可选计算 logprobs，把 logits 转成 float32，应用 allowed token ids、bad words、logit processors、penalties，然后采样；采样内部会处理 greedy、temperature、argmax-invariant processors、top-k/top-p，最后返回 `SamplerOutput`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

简化流程：

```text
logits
→ allowed token mask / bad words mask
→ repetition / frequency / presence penalties
→ temperature
→ top-k / top-p
→ softmax
→ sample token id
```

---

## 五、显存布局

### 1. Embedding table

Qwen3.6 的 token embedding 是 248,320 padded，hidden size 是 2048。按 BF16 粗略估算：

$$
248320 \times 2048 \times 2 \approx 1.02 \text{ GB}
$$

这只是 token embedding 一项的粗估，不包括 LM head 是否 tied、MoE expert 权重、attention/GDN 参数、vision encoder、KV cache、runtime workspace 等。Qwen3.6 的 embedding 和 LM output 维度来自模型卡；是否 tie word embeddings 则由 vLLM 源码中的 `config.tie_word_embeddings` 分支决定。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 2. Logits tensor

logits 的显存通常是临时显存，但 vocabulary 很大时，它仍然昂贵。

例如：

```text
N = 128 个当前要采样的 token
V = 248320
dtype = FP32 for sampling
```

那么 logits 大小粗略为：

$$
128 \times 248320 \times 4 \approx 127 \text{ MB}
$$

这解释了为什么 sampler 会关注 dtype、in-place 修改、top-k/top-p kernel、logprobs 是否返回等细节。vLLM v1 sampler 源码中明确会将 logits 转成 float32，并且 sampling / logprobs 过程会根据请求需求保留或处理 logits。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

### 3. Logprobs 会显著增加输出和计算负担

当用户请求 `logprobs` 或 `prompt_logprobs` 时，vLLM 需要保留更多概率信息。`SamplingParams` 文档说明：`logprobs` 会返回每个输出 token 的若干最可能 token 及所选 token 的 log probability；设置为 `-1` 时返回全部 vocab log probabilities。`prompt_logprobs` 对 prompt token 也有类似含义。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

工程结论：  
**不要在高吞吐线上默认打开全量 logprobs。** 对 Qwen3.6 这种 248,320 padded output 维度的模型，全量 logprobs 的内存、带宽和序列化成本都很高。

### 4. 多模态 embedding 的显存

多模态时，vision encoder 产生的 embedding 会合并进语言模型输入序列。vLLM 源码确认了 `_merge_multimodal_embeddings` 路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py))

工程结论：  
**图像/视频不是“免费 prompt 文本”**。它们会增加输入 token/embedding 数量，影响 prefill、position、KV/state cache，以及 batch 调度。

---

## 六、vLLM 源码映射

本章源码地图如下：

| 概念 | vLLM v0.20.1 路径 | 类/函数 | 作用 |
|---|---|---|---|
| 文本 embedding | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.embed_input_ids` | 调用底层 model 的 embedding |
| 多模态 embedding 合并 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration.embed_input_ids` | 文本 embedding 与 multimodal embeddings 合并 |
| Vision Encoder | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer` | 图像/视频 encoder |
| LM head | `vllm/model_executor/models/qwen3_5.py` | `ParallelLMHead` | hidden state 到 vocab logits |
| logits processor | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor` | gather logits、scale、去 padding |
| sampling params | `vllm/sampling_params.py` | `SamplingParams` | temperature、top-p、top-k、penalty、stop 等 |
| sampler | `vllm/v1/sample/sampler.py` | `Sampler` | logits 到 sampled token |
| top-k/top-p | `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` | top-k/top-p 过滤与随机采样 |

尤其要注意两个源码细节。

第一，`LogitsProcessor._get_logits` 会先通过 LM head 的 quant method 得到 logits，再做 tensor parallel gather / all-gather，最后去掉 vocab padding。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py))

第二，vLLM v1 sampler 的流程不是“直接 softmax 然后采样”这么简单，而是包含 allowed token mask、bad words、logits processors、penalties、temperature、top-k/top-p、logprobs gathering 等步骤。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

---

## 七、NVIDIA GPU 执行视角

### 1. Embedding lookup

embedding lookup 本质是按 token id 从 embedding table 读向量：

```text
input_ids: [N]
embedding table: [V, H]
output: [N, H]
```

它不是 Tensor Core 友好的大 GEMM，而更像不连续内存读取。batch 越乱、token id 越分散，越依赖 HBM 读取效率。

### 2. LM head GEMM

hidden states 到 logits 是一个大矩阵乘：

```text
hidden_states: [N, 2048]
lm_head:       [248320, 2048]
logits:        [N, 248320]
```

这一步非常适合 Tensor Core，但输出矩阵很大。对 Qwen3.6 而言，vocab 输出维度 248,320 padded，意味着 LM head 和 logits 都不是小开销。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. Tensor Parallel 下的 logits gather

当使用 tensor parallel 时，LM head 通常按 vocabulary 或 hidden dimension 分片；vLLM 的 `LogitsProcessor` 源码中存在 `_gather_logits`，会根据平台选择 `tensor_model_parallel_all_gather` 或 `tensor_model_parallel_gather`。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/logits_processor.py))

工程含义：

```text
多卡并行不仅切模型计算
还要在 logits / sampling 前后处理跨卡结果
```

如果返回 logprobs 或 full logits，这部分通信和内存压力会更明显。

### 4. Sampling kernel

sampling 看似只是“选一个 token”，但真实步骤包括：

```text
logits float32
penalty
mask
temperature
top-k / top-p
softmax
random sample
```

vLLM v0.20.1 的 `TopKTopPSampler` 源码说明该模块执行可选 top-k/top-p filtering，然后 weighted random sampling；在 CUDA 平台上，FlashInfer sampling 是可选优化，需要设置 `VLLM_USE_FLASHINFER_SAMPLER=1` 才启用，否则走 native 路径。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/ops/topk_topp_sampler.py))

---

## 八、优化前 vs 优化后

### 优化前：朴素思路

```text
每步 decode:
  hidden → logits
  logits → softmax
  softmax → sample
```

问题：

| 问题 | 后果 |
|---|---|
| 每步都生成完整 logits | vocab 大时显存和带宽开销高 |
| 默认返回 logprobs | 输出序列化和 GPU/CPU 传输压力大 |
| top-p 需要排序 | 大 vocab 下排序成本高 |
| temperature 很低但仍走随机路径 | 浪费 sampling 计算 |
| 多卡 logits gather 不受控 | TP 下通信变重 |

### 优化后：工程化采样

```text
只在需要的位置计算 logits
尽量避免 full logprobs
temperature=0 时走 greedy
top-k/top-p 走专用路径
必要时使用 optimized sampler
```

vLLM v1 sampler 源码中，当所有请求都是 greedy 时，会直接返回 greedy sampled tokens；随机采样路径才应用 temperature、argmax-invariant processors、top-k/top-p 等步骤。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py))

### 对 Qwen3.6 的特殊提醒

Qwen3.6 输出维度大，logits 和 sampling 的成本更值得关注。它不是只在 attention/MoE 上复杂，**输出侧同样很重**：

```text
hidden_size = 2048
vocab padded = 248320
logits per token = 248320 scores
```

这些尺寸来自 Qwen3.6 模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 九、工程落地

### 1. 常用 API 参数理解

OpenAI-compatible 请求里常见参数：

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 512,
  "logprobs": 5,
  "stop": ["</answer>"]
}
```

在 vLLM v0.20.1 中，这些最终会映射到 `SamplingParams` 一类参数。`SamplingParams` 定义了 `temperature`、`top_p`、`top_k`、`min_p`、`seed`、`stop`、`stop_token_ids`、`ignore_eos`、`max_tokens`、`logprobs`、`prompt_logprobs`、`detokenize`、`skip_special_tokens`、`allowed_token_ids`、`bad_words` 等字段。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

### 2. 推荐初始采样配置

对工程压测，建议先用稳定配置：

```json
{
  "temperature": 0,
  "max_tokens": 512
}
```

这样可以先排除随机采样带来的波动，观察：

```text
TTFT
TPOT
decode tokens/s
GPU utilization
KV/state cache capacity
```

再逐步加入：

```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20
}
```

### 3. 什么时候不要打开 logprobs

以下场景不建议默认打开 logprobs：

```text
高并发 chat serving
长输出生成
多卡 TP 部署
Qwen3.6 这种大 vocab 输出模型
流式响应且用户不需要概率
```

因为 logprobs 会让系统保留和传输更多 vocab 维度的信息；`SamplingParams` 明确支持 `logprobs=-1` 返回全部 vocab log probabilities，但这在大 vocab 模型上代价很高。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/sampling_params.py))

### 4. 多模态输入要先区分 text-only 与 image/video

Qwen3.6 是带 Vision Encoder 的 causal LM；vLLM 源码中也确实有 `Qwen3_VisionTransformer` 和 multimodal embedding merge 路径。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

工程上建议压测分三组：

```text
纯文本短 prompt
纯文本长 prompt
图像/视频 + 文本 prompt
```

不要把三者混在一起看平均值，否则你会分不清瓶颈来自 tokenizer、vision encoder、prefill、KV/state cache，还是 sampling。

---

## 十、本章总结

这一章建立了 LLM 推理的“入口/出口”闭环：

```text
text / multimodal input
→ token ids / multimodal embeddings
→ embeddings
→ hidden states
→ logits
→ sampling
→ next token
```

对传统 LLM，这就是最基础的 token-to-token 流程。对 Qwen3.6，要记住四个特殊点：

```text
1. 它是 Causal Language Model with Vision Encoder
2. hidden size 是 2048
3. token embedding 和 LM output 是 248320 padded
4. logits / sampling 在大 vocab 下不是免费操作
```

这些结构事实来自 Qwen3.6 模型卡；vLLM v0.20.1 源码则确认了 `embed_input_ids`、`ParallelLMHead`、`LogitsProcessor`、`Sampler`、`TopKTopPSampler` 等入口。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

# 小白总结

模型不直接看文字，而是看 token id。

```text
文字 → token id → embedding 向量 → 模型计算 → logits → 采样 → 新 token
```

logits 是“每个候选 token 的分数”，不是概率。softmax 后才是概率。temperature、top-p、top-k 会改变模型从这些概率里选下一个 token 的方式。

对 Qwen3.6 来说，词表输出很大：每个位置要对 248,320 个 padded vocab 位置打分，所以输出侧也很重。

---

# 工程师总结

工程视角下，本章最重要的链路是：

```text
Qwen3_5MoeForConditionalGeneration.embed_input_ids
→ Qwen3_5ForCausalLMBase.forward
→ Qwen3_5ForCausalLMBase.compute_logits
→ LogitsProcessor
→ Sampler
→ TopKTopPSampler
```

关键性能点：

```text
embedding lookup: HBM read
lm_head: large GEMM
logits gather: TP communication
sampling: FP32 logits + penalty + top-k/top-p + softmax/random
logprobs: potentially huge memory/output cost
```

对 Qwen3.6-35B-A3B，attention、Gated DeltaNet、MoE 固然重要，但输出侧的 `248320` padded vocab 也会直接影响 logits、sampling、logprobs 和 TP gather 成本。

---

# 关键公式

**1. Embedding lookup**

$$
x_i = E[i]
$$

**2. Hidden state 到 logits**

$$
z_t = W_{\text{lm}} h_t
$$

**3. Softmax**

$$
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
$$

**4. Temperature softmax**

$$
p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}
$$

**5. Greedy decoding**

$$
x_{t+1} = \arg\max_i z_i
$$

**6. Top-k**

$$
\mathcal{S}_k = \text{TopK}(z, k)
$$

只在 $\mathcal{S}_k$ 内采样。

**7. Top-p / nucleus sampling**

$$
\mathcal{S}_p =
\min \left\{
S:
\sum_{i \in S} p_i \ge p
\right\}
$$

只在累计概率达到阈值 $p$ 的候选集合中采样。

---

# 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 模型入口 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` |
| 文本 embedding | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.embed_input_ids` |
| 多模态 embedding 合并 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration.embed_input_ids` |
| LM head | `vllm/model_executor/models/qwen3_5.py` | `ParallelLMHead` |
| logits 计算 | `vllm/model_executor/models/qwen3_5.py` | `compute_logits` |
| logits processor | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor.forward`, `_get_logits`, `_gather_logits` |
| sampling 参数 | `vllm/sampling_params.py` | `SamplingParams` |
| sampler | `vllm/v1/sample/sampler.py` | `Sampler.forward`, `Sampler.sample` |
| top-k/top-p sampler | `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` |

---

# 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6-35B-A3B 的 hidden size 是 2048，token embedding 和 LM output 是 248320 padded | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6-35B-A3B 是 Causal Language Model with Vision Encoder | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM v0.20.1 中 Qwen3.6 兼容入口使用 `Qwen3_5MoeForConditionalGeneration` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration` | 源码直接确认 | 是 |
| Qwen3.6 多模态路径会构造 `Qwen3_VisionTransformer` | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_VisionTransformer` | 源码直接确认 | 是 |
| 文本 embedding 由 language model 的 `embed_input_ids` 产生 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5MoeForConditionalGeneration.embed_input_ids` | 源码直接确认 | 是 |
| 存在 multimodal embeddings 时会调用 `_merge_multimodal_embeddings` 合并 | `vllm/model_executor/models/qwen3_5.py` | `_merge_multimodal_embeddings` | 源码直接确认 | 是 |
| vLLM v0.20.1 在最后 PP rank 上创建 `ParallelLMHead(config.vocab_size, config.hidden_size)`，或在 tie embeddings 时复用 embedding | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase.__init__` | 源码直接确认 | 是 |
| Qwen3.6 logits 计算调用 `self.logits_processor(self.lm_head, hidden_states)` | `vllm/model_executor/models/qwen3_5.py` | `compute_logits` | 源码直接确认 | 是 |
| `LogitsProcessor` 会从 hidden states 获取 logits、必要时 scale，并应用 logits processors | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor` | 源码直接确认 | 是 |
| `LogitsProcessor._get_logits` 会调用 LM head quant method、gather TP logits、去掉 vocab padding | `vllm/model_executor/layers/logits_processor.py` | `_get_logits`, `_gather_logits` | 源码直接确认 | 是 |
| `SamplingParams` 包含 temperature、top-p、top-k、min-p、stop、logprobs 等采样参数 | `vllm/sampling_params.py` | `SamplingParams` | 源码直接确认 | 是 |
| vLLM v1 `Sampler` 的流程包括 logits float32、mask、logits processors、penalties、greedy/random、temperature、top-k/top-p、logprobs gathering | `vllm/v1/sample/sampler.py` | `Sampler.forward`, `Sampler.sample` | 源码直接确认 | 是 |
| `TopKTopPSampler` 执行 top-k/top-p filtering 和 weighted random sampling，并可选 FlashInfer CUDA 采样路径 | `vllm/v1/sample/ops/topk_topp_sampler.py` | `TopKTopPSampler` | 源码直接确认 | 是 |
| tokenizer 的具体 Qwen3.6 special tokens、chat template、image/video placeholder token 细节 | HF tokenizer files / processor config | 待具体文件核验 | 待源码确认 | 否 |

---

# 源码阅读作业

按这个顺序读：

**1. 读 embedding 与 multimodal merge**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
Qwen3_5MoeForConditionalGeneration.embed_input_ids
_embed_text_input_ids
_merge_multimodal_embeddings
```

目标：理解 text embedding 和 multimodal embedding 如何合并。

**2. 读 LM head 与 logits**

```text
vllm/model_executor/models/qwen3_5.py
```

重点找：

```text
ParallelLMHead
LogitsProcessor(config.vocab_size)
compute_logits
```

目标：理解 hidden states 到 logits 的路径。

**3. 读 logits processor**

```text
vllm/model_executor/layers/logits_processor.py
```

重点找：

```text
LogitsProcessor.forward
_get_logits
_gather_logits
```

目标：理解 TP gather 和 vocab padding removal。

**4. 读 sampler**

```text
vllm/v1/sample/sampler.py
vllm/v1/sample/ops/topk_topp_sampler.py
vllm/sampling_params.py
```

重点找：

```text
SamplingParams
Sampler.forward
Sampler.sample
TopKTopPSampler
```

目标：把 temperature、top-p、top-k、penalty 和源码步骤对应起来。

---

# 实验验证作业

### 实验 1：temperature=0 vs temperature=0.7

同一个 prompt，分别请求：

```json
{
  "temperature": 0,
  "max_tokens": 128
}
```

```json
{
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 128
}
```

观察：

```text
输出是否稳定
TPOT 是否变化
tokens/s 是否变化
```

### 实验 2：logprobs 开关对性能的影响

分别请求：

```json
{
  "temperature": 0,
  "max_tokens": 128
}
```

```json
{
  "temperature": 0,
  "max_tokens": 128,
  "logprobs": 5
}
```

如果系统支持，再谨慎测试：

```json
{
  "temperature": 0,
  "max_tokens": 16,
  "logprobs": -1
}
```

记录：

```text
延迟
输出 payload 大小
GPU memory
CPU serialization 时间
```

### 实验 3：多模态 vs text-only

对 Qwen3.6 分别压测：

```text
纯文本 prompt
图像 + 文本 prompt
视频 + 文本 prompt
```

记录：

```text
TTFT
prefill 时间
GPU memory
vision encoder 是否成为瓶颈
```

---

# 3 个自测题

**1. logits 和 probability 有什么区别？**  
logits 是未归一化分数；probability 是 softmax 后的概率。

**2. 为什么 Qwen3.6 的 logits / sampling 成本不能忽略？**  
因为它的 LM output 是 248,320 padded，每个待采样 token 都要面对非常大的 vocab 维度。

**3. temperature=0 时为什么通常可以走 greedy？**  
因为 temperature 接近 0 时，分布趋向于最大 logit token；vLLM `SamplingParams` 也把 0 temperature 视为 greedy sampling。

---

# 下一章预告

第五章进入：

```text
Attention / Gated Attention / RoPE 数学
```

我们会从传统 attention 的：

```text
Q, K, V
softmax(QK^T / sqrt(d))V
causal mask
KV cache
```

讲到 Qwen3.6 的：

```text
Gated Attention
Q heads = 16
KV heads = 2
head dim = 256
RoPE dim = 64
```

并特别说明：为什么 Qwen3.6 的 full attention 层只是 40 层中的一部分，不能把所有层都当普通 attention。

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/sample/sampler.py)

## Prompt:
2026/5/9 10:23:09

继续第五章

## Response:
2026/5/9 10:23:10

> **Browsing for up-to-date vLLM sources and citations**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Finding exact lines matching specific patterns**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Searching for Open Qwen3_5 data or models**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Searching for rotary embedding source code**
>
> [rotary_embedding - vLLM](https://docs.vllm.ai/en/v0.12.0/api/vllm/model_executor/layers/rotary_embedding/)
> [vllm/vllm/model_executor/layers/rotary_embedding/__init__.py ... - GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/__init__.py)
> [【vLLM-模型特性适配】vllm-ascend开发之Rope适配 - CSDN博客](https://blog.csdn.net/qq_36396435/article/details/156431974)
> [从源代码编译构建vLLM并解决常见编译问题本文详细记录了从 ...](https://juejin.cn/post/7470462338192179234)
> [Positional Embeddings | xiaojingbao/vllm | DeepWiki](https://deepwiki.com/xiaojingbao/vllm/3.2-positional-embeddings)
> [ROPE 原理学习与Megatron-LM代码实现 - 知乎](https://zhuanlan.zhihu.com/p/1895076081607889981)
> [vllm/vllm/model_executor/layers/rotary_embedding.py · vl/vllm ...](https://gitcode.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py)
> [大模型-qwen3 RoPE旋转位置编码详细解读-69 - jack ...](https://www.cnblogs.com/cavalier-chen/p/18937176)
> [基本模型 - vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/contributing/model/basic.html)
> [详解大模型旋转位置编码RoPE原理与ChatGLM代码实现 ...](https://developer.aliyun.com/article/1507811)
>
> **Searching for GitHub __init__ raw v0.20.1**
>
> [vllm/vllm/model_executor/layers/rotary_embedding/__init__.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/rotary_embedding/__init__.py)
>
> **Finding open lines around code in GitHub**
>
> [vllm/vllm/model_executor/layers/rotary_embedding/__init__.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/rotary_embedding/__init__.py)
>
> **Searching for vLLM attention class source code**
>
> [vllm/vllm/v1/attention/backends at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends)
>
> **Drafting concise chapter structure with citations**
>
> I'm focusing on providing the requested chapter outline—10 sections, summaries, and proper citations. Since math is central to this chapter, I might include a math-focused widget, using the RoPE formula directly after the equation. I'll make sure to cite only web-sourced factual info and avoid raw URLs, keeping everything clear and concise.
>
> Thought for 29s

第五章会把传统 Attention 数学讲清楚，但会严格限制在 Qwen3.6 的 full-attention 层上；Gated DeltaNet 不是普通 Attention，这章只做边界说明，不强行套公式。

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

# 小白总结

Attention 就是“当前词去历史里找重点”。

```text
Q = 我想找什么
K = 历史信息的标签
V = 历史信息的内容
```

RoPE 是把“位置信息”旋进 Q 和 K 里，让模型知道谁在前、谁在后。  
Qwen3.6 的 Gated Attention 还多了一个 gate，像一个阀门，决定 attention 输出的哪些维度应该通过、哪些维度应该被压低。

---

# 工程师总结

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

# 关键公式

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

# 关键源码入口

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

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. 为什么 attention 要除以 $\sqrt{d}$？**  
因为 $q \cdot k$ 的尺度会随 head dim 增大而增大，除以 $\sqrt{d}$ 可以稳定 softmax 输入尺度。

**2. Qwen3.6 的 Gated Attention 和普通 Attention 至少有哪三个差异？**  
答案应包含：GQA，RoPE 只作用于部分维度，attention output 经过 sigmoid gate 逐元素调制。

**3. 为什么不能按 40 层 full attention 来估算 Qwen3.6 的 KV cache？**  
因为 Qwen3.6 的 40 层中按模型卡布局只有 10 个 Gated Attention 层，另外 30 层是 Gated DeltaNet，后者不是普通 full-attention KV cache 路径。

---

# 下一章预告

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

## Prompt:
2026/5/9 10:26:30

继续第六章

## Response:
2026/5/9 10:26:34

> **Searching for vLLM v0.20.1 related sources**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vllm/vllm/model_executor/models/qwen3_5.py at main - GitHub](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5.py)
> [VLLM V1 KVCacheManager组件 - 知乎](https://zhuanlan.zhihu.com/p/1906034492776388150)
> [vLLM Paged Attention - vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/design/kernel/paged_attention.html)
> [Model Details · ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.6-35B-A3B: Agentic Coding Power, Now Open to All](https://qwen.ai/blog?id=qwen3.6-35b-a3b)
> [vLLM - Qwen](https://qwen.readthedocs.io/zh-cn/latest/deployment/vllm.html)
> [深入解析vLLM的PagedAttention技术：解决大模型KVCache ...](https://blog.csdn.net/2401_85343303/article/details/152799210)
>
> **Searching for official sources information**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
> [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
>
> **Finding precise lines for KV manager and QWEN files**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
> [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
>
> **Exploring open lines around Qwen attention**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
> [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
>
> **Preparing citations and sources for chapter 6 structure**
>
> I need to finalize chapter 6 in Chinese, ensuring proper citation usage, including web sources, and considering historical context. For v0.20.1, I’ll mention the code and reference the GitHub KV manager. I’ll possibly reference KVCacheBlocks, but won’t go overboard. I should also ensure the concluding sections include evidence tables.
>
> Thought for 22s

第六章会从“显存账本”讲起：先只算 Gated Attention 的 KV cache，再说明为什么 Gated DeltaNet 的 state 不能混进普通 KV cache 公式里。

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

# 小白总结

KV Cache 就像把读过的书签和笔记存下来。  
没有它，每写一个新字都要重新读完整篇文章；有了它，只要查之前保存的笔记。

但文章越长，笔记越多。Qwen3.6 支持 262K 原生上下文，所以这些“笔记”会非常大。更复杂的是，Qwen3.6 只有一部分层是普通 K/V 笔记，另一部分 Gated DeltaNet 层用的是另一种状态笔记，不能混在一起算。

---

# 工程师总结

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

# 关键公式

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

# 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Qwen3.6 layer 分流 | `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5DecoderLayer` |
| Full attention / KV cache 接入 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention`, `Attention` |
| Gated DeltaNet state | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager` |
| cache slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` |
| Paged KV cache 概念 | vLLM docs | Paged Attention design doc |

---

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. KV Cache 为什么能加速 decode？**  
因为历史 token 的 K/V 已经保存，decode 每步只需要计算新 token 的 Q/K/V，并读取历史 K/V，而不是反复重算所有历史 token 的 K/V。

**2. 为什么不能把 Qwen3.6 按 40 层 full attention 来估算 KV cache？**  
因为 Qwen3.6 的 40 层中，按模型卡 layout 只有 10 个 Gated Attention 层；另外 30 个是 Gated DeltaNet，要按 GDN/Mamba-style state 单独估算。

**3. 为什么 FP8 权重不能彻底解决长上下文显存问题？**  
因为 FP8 主要降低权重显存，长上下文的 KV cache、GDN state、MTP speculative cache、多模态 buffer 和 runtime workspace 仍会随上下文和并发增长。

---

# 下一章预告

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

## Prompt:
2026/5/9 10:28:59

继续第七章

## Response:
2026/5/9 10:29:01

> **Verifying sources and continuing with course**
>
> I need to confirm that I have the right sources and citations for this chapter. While I have some previous results, they don’t seem to include verified web citations. I'll check the latest docs to make sure I have reliable information, especially details around vLLM.
>
> **Browsing for up-to-date vLLM documentation and sources**
>
> [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
> [vllm/vllm/v1/core/kv_cache_manager.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py)
>
> **Reviewing relevant documentation**
>
> I need to carefully review the lines in the documentation related to important concepts. It seems there's some critical information I'm missing or need to confirm. I will open up the necessary documents and inspect them before moving forward to make sure I have all the details.
>
> **Inspecting important lines for documentation concepts**
>
> [Paged Attention - vLLM](https://docs.vllm.ai/en/latest/design/paged_attention.html)
> [vllm/vllm/v1/core/kv_cache_manager.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py)
>
> **Verifying kv manager details and sources**
>
> I need more specifics on the kv manager initialization, block pool details, and the hybrid kv manager. It seems I might need the exact source instead of something like GitHub HTML, which might not have enough relevant lines. I’ll open the raw source for better details and verification.
>
> **Searching for raw sources on KV manager details**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py)
>
> **Handling raw data access**
>
> It seems like the raw data I’m opening is a bit hard to read due to minification. It's not giving me enough clarity. I might need to get the exact lines directly from a more accessible format. Let me see if I can open the source in a more readable way.
>
> **Browsing for raw, open lines in minified code**
>
> [vllm/vllm/v1/core/kv_cache_manager.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py)
>
> **Preparing chapter 7 and gathering sources**
>
> I need to create Chapter 7 in Chinese with citations, possibly including more sources regarding the qwen3_5 split and model card. I’ll need reliable web sources for these citations. It looks like I should review the qwen model card and the qwen source, along with any related documents.
>
> **Finding sources for Qwen 3.5 model and citations**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Reviewing qwen model card and source lines**
>
> I need to find specific lines in the qwen model card for citation, especially those referencing layers. I should carefully review the qwen source to ensure I capture the right lines. These will help provide solid backing for the chapter and maintain accurate references.
>
> **Searching for Qwen model card and source lines**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Crafting chapter with citations and sources**
>
> I need to gather the necessary sources for the hybrid KV cache manager design and possibly the latest docs, though the version tag might not be essential. I'll focus on ensuring all required subsections are present in the chapter, with citations for facts. It’s important to include “Paged Attention docs” and provide the citations in proper format. I might also mention Qwen3.6 boundaries and GDN metadata—both will need citations. Also, I’ll consider if using a math widget for the KV block formula is appropriate.
>
> **Deciding on including math widget and citations**
>
> The math widget could help if the formula is central to the discussion, but it may not be necessary since it doesn't support custom formulas like the one for PagedAttention. I'll focus on citing the vLLM docs, noting the PagedAttention design is historical. Including the BlockPool lines and defining evidence types will be essential, like "源码直接确认" or "官方文档确认". I'll also render citations as web links, avoiding raw markdown URLs. For "继续第七章", no further clarification needed.
>
> Thought for 27s

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

# 小白总结

KV Cache 像模型的“历史笔记”。  
PagedAttention 就是把这本笔记拆成一页一页的小纸片，而不是放在一整本连续的大本子里。

这样做的好处是：  
有人用完几页，就能把这几页还回去；新请求可以拿这些空页继续写，不需要找一整段连续空间。

---

# 工程师总结

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

# 关键公式

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

# 关键源码入口

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

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. PagedAttention 为什么能降低 KV cache 碎片？**  
因为它把长序列 K/V 拆成固定大小 blocks，请求逻辑上连续，但物理上可以使用分散的 free blocks。

**2. block table 的作用是什么？**  
它把请求的逻辑 block id 映射到物理 KV cache block id，让 attention backend 能从正确的物理 block 读取 K/V。

**3. 为什么 Qwen3.6 不能只用 PagedAttention 一套模型解释所有缓存？**  
因为 Qwen3.6 只有 10 个 Gated Attention 层适合普通 K/V cache 视角；另外 30 个 Gated DeltaNet 层走 Mamba/GDN-style state 路径。

---

# 下一章预告

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

## Prompt:
2026/5/9 10:36:09

继续第八章

## Response:
2026/5/9 10:36:11

> **Looking up vLLM 0.20.1 and relevant sources**
>
> I'll need to verify and cite precise information related to vLLM v0.20.1. Since user query involves source code, it seems like browsing could help. I'll search for relevant official docs and source files, aiming for accurate citations. I’ll make sure to confirm details through trusted web sources as needed.
>
> **Searching for vLLM v0.20.1 source and documentation**
>
> [vLLM 调度器详解——Continuous Batching 核心机制 - 知乎](https://zhuanlan.zhihu.com/p/2021170878503815058)
> [vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/)
>
> **Choosing the right documentation source**
>
> I’m considering whether to reference docs from a Chinese mirror or the official one. To be safe, it seems better to use the official site (docs.vllm.ai), as it should be the most up-to-date and reliable. I'll keep this in mind while browsing for information.
>
> **Searching official vLLM documentation for configuration details**
>
> [Automatic Prefix Caching - vLLM](https://docs.vllm.ai/en/latest/design/prefix_caching/)
>
> **Finding v0.20.1 exact source**
>
> I need to pinpoint the exact source for vLLM v0.20.1. The latest documentation might not reflect this specific version, but it should work for general docs and recipes. Since the user needs exact v0.20.1 for certain files, GitHub’s open-source version could be the right place to verify specifics.
>
> **Finding v0.20.1 source and documentation**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Handling potential formatting issue**
>
> It looks like the line numbers might not align properly due to compression. I’m thinking that opening the file at specific line number ranges could help me identify the issue. I’ll focus on that approach to make sure I’m looking at the right parts of the source.
>
> **Opening lines within specified ranges for analysis**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Finding citation sources and handling minified code**
>
> I need to cite the scheduler lines, and it seems like the raw line numbers are a bit off due to compression, though I can still use them for citation purposes. I’ll rely on web sources and maybe even GitHub for specific details. I'll also need to find the source for the Qwen model card.
>
> **Searching for open source citations and references**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> **Accessing source lines for recipe commands**
>
> I think I’ll need to open the exact lines that correspond to the recipe commands. This should help me pinpoint the relevant details. I'll go ahead and check for these specific lines to ensure I'm pulling the right information for the user.
>
> **Looking for specific recipe command lines**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> **Finalizing long chapter answer with citations**
>
> I think I have enough information for the long chapter answer. I’ll make sure to cite plenty of facts, and I don't think I need more web sources. However, it might be helpful to cite metrics, especially for observability. I may need to sift through the source and open specific lines related to metrics.
>
> **Citing observability metrics and open standards**
>
> [Metrics - vLLM](https://docs.vllm.ai/en/latest/design/metrics/)
>
> **Finalizing chapter framework and citations**
>
> I’m working on finalizing the chapter: it should be a cohesive 10 sections plus a summary. I'll include ample citations but ensure it doesn't get too verbose. Following the structure is key, especially including a source evidence table. I'll mention "下一章预告" and "第八章." I’ll avoid forbidden phrases and cite web sources after facts without raw URLs. I also need to discuss the Continuous batching process, vLLM scheduler, chunked prefill, prefix caching, and Qwen model structure.
>
> Thought for 29s

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

$$
\Delta_i =
N^{\text{target}}_i - N^{\text{computed}}_i
$$

其中：

| 符号 | 含义 |
|---|---|
| $N^{\text{computed}}_i$ | 请求 $i$ 已经算过的 token 数 |
| $N^{\text{target}}_i$ | 请求 $i$ 当前应该追上的 token 数，包含 prompt、output、spec tokens |
| $\Delta_i$ | 当前请求还需要调度的 token 数 |

每个 scheduler step 有一个 token budget：

$$
\sum_i \Delta_i^{\text{scheduled}} \leq \text{max\_num\_scheduled\_tokens}
$$

vLLM v0.20.1 源码中，`self.max_num_scheduled_tokens` 来自 `scheduler_config.max_num_scheduled_tokens`，如果未设置则使用 `scheduler_config.max_num_batched_tokens`；`schedule()` 中用 `token_budget = self.max_num_scheduled_tokens` 控制本轮能安排多少 token。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 2. Continuous batching 的直觉公式

传统静态 batch：

$$
\text{batch} = \{\text{同一时刻开始的一组请求}\}
$$

Continuous batching：

$$
\text{batch}_t =
\{\text{本轮能推进的 running requests}\}
\cup
\{\text{本轮能接纳的 waiting requests}\}
$$

也就是说，batch 不是固定的。每个 GPU step 都可以加入新请求、继续旧请求、暂停或抢占请求。

### 3. Chunked prefill 的直觉公式

如果一个 prompt 还剩 $P$ 个 token 没 prefill，而本轮剩余 token budget 是 $B$，则本轮只安排：

$$
\Delta_{\text{prefill}} = \min(P, B)
$$

如果配置了 `long_prefill_token_threshold`，源码还会进一步限制单次调度的 token 数：

$$
\Delta_{\text{prefill}}
=
\min(P, B, \text{long\_prefill\_token\_threshold})
$$

vLLM v0.20.1 Scheduler 源码在 running 与 waiting 调度路径中都检查 `long_prefill_token_threshold`，当阈值大于 0 且小于 `num_new_tokens` 时，会把 `num_new_tokens` 截断到该阈值。([raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 4. Prefix caching 的直觉公式

如果新请求 prompt 是：

```text
prefix + suffix
```

且 prefix 对应的 KV blocks 已经在 cache 中，那么实际需要重新 prefill 的 token 数从：

$$
T_{\text{prompt}}
$$

变成：

$$
T_{\text{prompt}} - T_{\text{cached}}
$$

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

$$
\text{cache usage}
=
\sum_i
\text{active tokens}_i
\times
\text{cache bytes per token}
$$

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

# 小白总结

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

# 工程师总结

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

# 关键公式

**1. 请求剩余计算量**

$$
\Delta_i =
N^{\text{target}}_i
-
N^{\text{computed}}_i
$$

**2. 每轮 token budget**

$$
\sum_i \Delta_i^{\text{scheduled}}
\leq
\text{max\_num\_scheduled\_tokens}
$$

**3. Chunked prefill**

$$
\Delta_{\text{prefill}}
=
\min(
P_{\text{remaining}},
\text{token\_budget},
\text{long\_prefill\_token\_threshold}
)
$$

**4. Prefix caching 节省量**

$$
T_{\text{recompute}}
=
T_{\text{prompt}}
-
T_{\text{cached}}
$$

**5. KV cache 活跃占用粗略关系**

$$
\text{cache usage}
\propto
\sum_i
\text{active tokens}_i
\times
\text{cache bytes per token}
$$

---

# 关键源码入口

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

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. vLLM v0.20.1 Scheduler 为什么说没有严格的 prefill phase / decode phase？**  
因为它用统一的 `num_computed_tokens` 追赶 `num_tokens_with_spec` 模型来调度请求；prefill、decode、prefix caching、spec decode 都可以表示成“还差多少 token 没算”。

**2. Chunked prefill 为什么能改善 decode 延迟？**  
因为长 prompt 不再一次性占满整个 GPU step，而是被切成 chunk；decode 请求可以优先被调度，并和 prefill chunk 混合执行。

**3. Prefix caching 为什么只缓存 full blocks？**  
因为 vLLM 的 prefix cache 以 KV cache block 为基本单位，full block 可以稳定 hash、复用和管理；不完整 block 还会继续追加 tokens，不适合作为稳定 cache key。

---

# 下一章预告

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

## Prompt:
2026/5/9 10:38:50

继续第九章

## Response:
2026/5/9 10:38:52

> **Browsing for vLLM v0.20.1 sources and documentation**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Testing raw link behavior for GitHub files**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Looking for source lines for qwen3_5 decoder**
>
> [vllm/vllm/model_executor/layers/fused_moe/layer.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py)
>
> **Searching for FusedMoE forward function in GitHub**
>
> [vllm/vllm/model_executor/layers/fused_moe/layer.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/layers/fused_moe/layer.py)
>
> **Determining required citations and sources**
>
> I think I need a final answer, and I’m considering whether I need to include a FusedMoE call. I might need to check the documentation for the FusedMoE class, especially top_k. I should make sure to cite sources for all relevant info, especially considering I may need to find details about Qwen3_5’s decoder layer MLP.
>
> **Searching for FusedMoE and Qwen3_5 decoder info**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Addressing open source issue**
>
> The open source is proving tricky, likely due to some weird lines in the HTML. But I think I can find a way around it. It’ll take a bit more digging, but hopefully, I can still find a useful path forward despite this small hurdle.
>
> **Helping identify issues in open source HTML line**
>
> [vllm/vllm/model_executor/models/qwen3_5.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_5.py)
>
> **Planning final structure with citations**
>
> I’m organizing the final structure carefully. I’ll use citations from the web and include a source evidence table once. I don’t need raw direct quotes but will cite important sections, like lines from the Qwen model card and FusedMoE class documentation. I’ll add formulas and ensure the document covers all required sections, with a focus on clarity and citations after each paragraph. Bullet points and tables are okay where needed. I’ll also highlight FusedMoE and router top-k, keeping pending items marked clearly.
>
> Thought for 28s

第九章会聚焦 MoE 的真实推理路径：router 怎么选专家、top-8 routed + shared expert 如何合并，以及为什么 expert dispatch 会成为 GPU 和多卡通信的核心瓶颈。

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
\mathcal{E}_{topk}(h)=\operatorname{TopK}(r, k=8)
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
r = \operatorname{Router}(h)
$$

$$
\mathcal{E}_{top8} = \operatorname{TopK}(r, 8)
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

# 小白总结

MoE 就是“不是所有 token 都找同一个老师”。

Qwen3.6 有 256 个专家。每个 token 会先让 router 判断该找哪些专家，然后只找 8 个 routed experts，再额外找 1 个 shared expert。这样模型可以有很多总参数，但每个 token 只激活一部分参数。

---

# 工程师总结

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

# 关键公式

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
\operatorname{TopK}(r, 8)
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

# 关键源码入口

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

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. Qwen3.6 的 MoE 中，router 输出的形状是什么？**  
如果当前 batch 有 $N$ 个 token，router logits 是 `[N, 256]`，因为 Qwen3.6 有 256 个 routed experts。

**2. 为什么 MoE 不等于“每个 token 跑 256 个 MLP”？**  
因为每个 token 只选择 top-8 routed experts，再额外经过 1 个 shared expert；没有把 256 个 routed experts 全部计算一遍。

**3. 为什么 MoE 的 GPU 优化难点不只是 GEMM？**  
因为还要做 router top-k、token dispatch、expert 分组、负载均衡、expert output combine，以及多卡 expert parallel 下的跨 GPU 通信。

---

# 下一章预告

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

## Prompt:
2026/5/9 10:47:38

继续第十章

## Response:
2026/5/9 10:47:48

> **Browsing sources for vLLM v0.20.1 documentation**
>
> [彻底搞懂MoE的EP并行（基于vLLM） - 知乎](https://zhuanlan.zhihu.com/p/1911059432953061899)
> [专家并行部署 - vLLM - vLLM 文档](https://docs.vllm.com.cn/en/latest/serving/expert_parallel_deployment/)
>
> **Searching official docs for vLLM expert parallel deployment**
>
> [Expert Parallel Deployment - vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
>
> **Providing source citations and path references**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Analyzing input for potential corrections and suggestions**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Investigating raw file line number issue with GitHub**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/distributed/parallel_state.py)
>
> **Searching for sources on distributed group functions**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/distributed/parallel_state.py)
>
> **Searching for Open Qwen model card and recipe**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
>
> **Clarifying version discrepancies for chapter 10**
>
> For chapter 10, I need to focus on providing correct sections, summaries, formulas, code tables, and citations. While the user wants documentation for v0.20.1, the latest docs might differ. I'll use the official vLLM v0.20.1 source for implementation but will note that deployment docs may vary based on version. I'll also cite relevant Qwen model details, using the Hugging Face model card and recipes where appropriate.
>
> Thought for 36s

第十章会把单卡里的 “QKV / MoE / LM head” 扩展成多卡执行图，重点区分 Tensor Parallel 切张量、Expert Parallel 切专家，以及它们各自触发的 NCCL 通信。

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

$$
Y = XW
$$

假设：

$$
X \in \mathbb{R}^{N \times H}
$$

$$
W \in \mathbb{R}^{H \times O}
$$

#### Column Parallel

把 $W$ 的输出维度切开：

$$
W = [W_1, W_2, \ldots, W_p]
$$

每张 GPU 算：

$$
Y_i = XW_i
$$

最后：

$$
Y = [Y_1, Y_2, \ldots, Y_p]
$$

这通常需要 **all-gather**，除非后续计算可以直接消费 shard 后的输出。vLLM v0.20.1 的 `ColumnParallelLinear` 源码注释明确写道，矩阵 $A$ 沿第二维并行化为 `[A_1, ..., A_p]`；如果 `gather_output=True`，forward 会对 partition 输出做 `tensor_model_parallel_all_gather`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/layers/linear.py))

#### Row Parallel

把 $W$ 的输入维度切开，同时输入 $X$ 也按 hidden 维度切开：

$$
X = [X_1, X_2, \ldots, X_p]
$$

$$
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_p
\end{bmatrix}
$$

每张 GPU 算 partial output：

$$
Y_i = X_iW_i
$$

最终：

$$
Y = \sum_{i=1}^{p}Y_i
$$

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

$$
y =
\sum_{e \in \operatorname{TopK}(r(h), 8)}
w_eE_e(h)
+
E_{\text{shared}}(h)
$$

Tensor Parallel 会把同一个 expert 的矩阵切到多 GPU；Expert Parallel 则会把不同 experts 放到不同 GPU。

如果有 256 experts，EP size = 8，最简单的线性放置是：

$$
\text{experts per rank} = 256 / 8 = 32
$$

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

# 小白总结

多 GPU 推理不是简单“多几张卡就快几倍”。

Tensor Parallel 是“大家一起做同一个大矩阵乘”；Expert Parallel 是“不同专家住在不同 GPU，token 要被送到对应专家那里”；NCCL 通信就是 GPU 之间传结果、拼结果、求和的高速物流系统。

Qwen3.6 有 256 个 experts，每个 token 只找 8 个 routed experts + 1 个 shared expert，所以 MoE 很适合 expert parallel，但也会带来 token dispatch 和 all-to-all 通信。

---

# 工程师总结

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

# 关键公式

**1. Column Parallel**

$$
W = [W_1, W_2, \ldots, W_p]
$$

$$
Y_i = XW_i
$$

$$
Y = [Y_1, Y_2, \ldots, Y_p]
$$

**2. Row Parallel**

$$
X = [X_1, X_2, \ldots, X_p]
$$

$$
W =
\begin{bmatrix}
W_1 \\
W_2 \\
\vdots \\
W_p
\end{bmatrix}
$$

$$
Y = \sum_{i=1}^{p}X_iW_i
$$

**3. Expert Parallel 均分**

$$
\text{experts per rank}
=
\frac{\text{num experts}}{\text{EP size}}
$$

Qwen3.6 简单均分例子：

$$
256 / 8 = 32
$$

**4. MoE routed output**

$$
y_{\text{routed}}
=
\sum_{e \in \operatorname{TopK}(r(h), 8)}
w_eE_e(h)
$$

**5. Shared expert 合并**

$$
y =
y_{\text{routed}} + E_{\text{shared}}(h)
$$

**6. 通信量直觉**

$$
\text{EP dispatch traffic}
\propto
N
\times
k
\times
H
\times
\text{bytes}
$$

其中 $N$ 是 token 数，$k=8$，$H=2048$。

---

# 关键源码入口

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

# 源码证据表

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

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. ColumnParallelLinear 和 RowParallelLinear 的通信差异是什么？**  
Column parallel 沿输出维切权重，必要时 all-gather 拼输出；Row parallel 沿输入维切权重，每张卡产生 partial output，必要时 all-reduce 求和。

**2. 为什么 Qwen3.6 在 TP=8 时不能简单认为 KV cache 除以 8？**  
因为 Qwen3.6 Gated Attention 的 KV heads 只有 2；vLLM 源码在 KV heads 少于 TP size 时会复制 KV heads across tensor parallel GPUs。

**3. Expert Parallel 为什么会引入 all-to-all？**  
因为 token 的 top-8 experts 可能分布在不同 GPU 上，hidden state 必须被 dispatch 到 expert 所在 rank，expert 输出也要回传合并。

---

# 下一章预告

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

## Prompt:
2026/5/9 10:59:24

继续第十一章

## Response:
2026/5/9 10:59:27

> **Browsing for sources and verifying facts**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Finding relevant code lines with patterns**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Looking for NVIDIA documentation**
>
> I think I might need the NVIDIA documentation for something. I'll search for it and see if I can find the details I need. There's a chance I could find helpful resources directly from their official site or other trusted sources. Let's check it out.
>
> **Searching for NVIDIA CUDA documentation**
>
> [CUDA Programming Guide - NVIDIA Documentation Hub](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)
> [CUDA 编程使用 Tensor core 详解 - 知乎](https://zhuanlan.zhihu.com/p/706494789)
> [CUDA 编程手册系列第四章: 硬件的实现 - NVIDIA 技术博客](https://developer.nvidia.cn/blog/cuda-hardware-cn/)
> [【CUDA进阶】Tensor Core实战教程（下）_cuda tensor ...](https://blog.csdn.net/qq_40672115/article/details/149479598)
> [NVIDIA Tensor Core Programming - Lei Mao's Log Book](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
>
> **Looking for NVIDIA tech blog and docs**
>
> I need to find some open technical documentation about NVIDIA’s GPUs, maybe something that's not in a PDF format but rather a blog post. I'll look for official developer resources, possibly in Chinese. Once I have the specifics, I can provide facts like the details about NVIDIA GPU's SM units or other key aspects.
>
> **Searching for NVIDIA technical blog posts in Chinese**
>
> [CUDA 编程手册系列第四章: 硬件的实现 - NVIDIA 技术博客](https://developer.nvidia.cn/blog/cuda-hardware-cn/)
>
> **Searching for official Tensor Core documentation**
>
> I need to find official documentation about Tensor Cores, ideally from NVIDIA’s architecture docs. PDFs maybe aren’t the best, but I can still cite specific sources from my search results. I’ll look for official materials that directly explain Tensor Cores and the relevant architecture.
>
> **Searching official NVIDIA documentation for Tensor Cores**
>
> [Train With Mixed Precision - NVIDIA Docs](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
> [Nvidia Tensor Core-WMMA API编程入门 - 知乎](https://zhuanlan.zhihu.com/p/620766588)
> [Accurate Models of NVIDIA Tensor Cores - arXiv.org](https://arxiv.org/pdf/2512.07004)
> [NVIDIA Tensor Core Programming - Lei Mao's Log Book](https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/)
> [GPU Microarch 学习笔记【3】Tensor Core - CSDN博客](https://blog.csdn.net/hit_shaoqi/article/details/134498937)
> [Nvidia Tensor Core初探 - 简书](https://www.jianshu.com/p/09391f94681b)
> [深入理解混合精度训练：从 Tensor Core 到 CUDA 编程 ...](https://www.cnblogs.com/megengine/p/15698175.html)
>
> **Looking for vLLM attention backends directory**
>
> I need to find a source that provides information on the directory of vLLM attention backends. A clear, open tree structure could be useful. I'll search for any relevant directories or lists from official resources to get the most accurate and up-to-date details.
>
> **Searching for vLLM attention backends directory sources**
>
> [vllm/vllm/v1/attention/backends at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/tree/v0.20.1/vllm/v1/attention/backends)
>
> **Finalizing and formatting citations**
>
> I need to prepare the final version of this work. It should be long but concise, citing every factual source, but not every repeated one. I should include "源码证据表" and consider mentioning "不强行展开" for unsupported sections. Citations should be spaced out, focusing on key facts. For vLLM and other docs, I’ll cite appropriately. If there's a recipe source with deployment lines, I should mention that too, but focus on the details without overloading the text.
>
> **Creating Final Report with Citations and Sources**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> Thought for 28s

# 第十一章：NVIDIA GPU Kernel  
## HBM、SM、Tensor Core、CUDA Kernel、MoE GEMM、Attention Backend

本章把前面学过的 **Attention / Gated DeltaNet / MoE / MTP / Scheduler** 放到 NVIDIA GPU 执行视角下重新理解。

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

NVIDIA 官方 CUDA 编程资料说明，NVIDIA GPU 架构围绕一组可扩展的多线程 **SM** 构建；CUDA kernel grid 的线程块会被分发到可用 SM 上执行，一个 SM 可以并发执行多个线程块。资料还说明，SM 以 32 个线程组成的 **warp** 为单位创建、管理、调度和执行线程。([developer.nvidia.cn](https://developer.nvidia.cn/blog/cuda-hardware-cn/))

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

$$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}
{\text{Bytes moved from/to memory}}
$$

如果一个 kernel 每读 1 byte 能做很多 FLOPs，它更可能是 **compute-bound**；如果读很多数据但每个数据只算很少，它更可能是 **memory-bound**。

### 2. Prefill 为什么更容易 compute-bound

Prefill 一次处理很多 prompt tokens。以线性层为例：

$$
Y = XW
$$

其中：

```text
X: [N, H]
W: [H, O]
Y: [N, O]
```

当 $N$ 很大时，GEMM 很大，Tensor Core 更容易吃满。NVIDIA 性能文档说明，Tensor Cores 是为加速矩阵乘加操作而引入的硬件单元，用于机器学习和科学计算中的矩阵乘累加；NVIDIA 的矩阵乘性能文档也强调，Tensor Cores 用于最大化 tensor multiply / matrix multiply 的速度。([docs.nvidia.com](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)) ([docs.nvidia.com](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html))

### 3. Decode 为什么更容易 memory-bound

Decode 每步每个请求通常只新增 1 个 token。GEMM 的 $N$ 变小，但 attention 要读历史 KV cache：

$$
o_t =
\operatorname{softmax}
\left(
\frac{q_t K_{\le t}^{\top}}{\sqrt{d}}
\right)
V_{\le t}
$$

当上下文是 262K，当前 query 很小，历史 K/V 很大。于是 decode 的瓶颈经常变成：

```text
从 HBM 读取历史 K/V
而不是 Tensor Core 算不够快
```

对 Qwen3.6，这个判断还要拆开：只有 10 个 Gated Attention 层走普通 K/V attention；另外 30 个 Gated DeltaNet 层走 GDN / Mamba-style state 路径。模型卡给出了 40 层 layout，vLLM v0.20.1 源码中 `Qwen3NextAttention` 负责 full-attention 层，`GatedDeltaNetAttention` 负责 GDN 路径。([huggingface.co](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) ([github.com](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

### 4. MoE 为什么不是普通 GEMM

MoE expert 的数学上像很多 MLP：

$$
E_e(h)=W_{2,e}\phi(W_{1,e}h)
$$

但执行上不是一个大 dense GEMM，而是：

$$
\text{router}
\rightarrow
\text{top-k}
\rightarrow
\text{dispatch}
\rightarrow
\text{expert GEMM}
\rightarrow
\text{combine}
$$

Qwen3.6 每 token 选 8 个 routed experts，再走 1 个 shared expert。也就是说，本轮如果有 $N$ 个 tokens，routed expert 的 token-expert 对数量是：

$$
N \times 8
$$

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

NVIDIA CUDA 资料说明，SM 会同时执行大量线程，使用 SIMT 架构；每个 SM 有寄存器和并行数据缓存/共享内存，kernel 的 occupancy 会受到寄存器和共享内存使用量影响。([developer.nvidia.cn](https://developer.nvidia.cn/blog/cuda-hardware-cn/))

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

CUDA kernel 会被拆成 thread blocks，blocks 分发到 SM 上；SM 内部再以 warp 为单位调度执行。warp 是 32 个线程组成的执行单元；当 warp 内线程走不同分支时，会降低效率。([developer.nvidia.cn](https://developer.nvidia.cn/blog/cuda-hardware-cn/))

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

# 小白总结

GPU 不是“一个很快的计算器”，而是很多小车间组成的工厂。

有些任务像大矩阵乘，Tensor Core 很擅长；有些任务像查很长的历史笔记，主要卡在从显存搬数据；MoE 则像把 token 分发给不同专家，难点在分发、合并和负载均衡。

Qwen3.6 的推理快不快，不只看模型参数量，也要看每一步 GPU kernel 是否吃满 Tensor Core、是否被 HBM 卡住、是否被 MoE dispatch 或多卡通信拖慢。

---

# 工程师总结

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

# 关键公式

**1. 算术强度**

$$
\text{Arithmetic Intensity}
=
\frac{\text{FLOPs}}
{\text{Bytes moved}}
$$

**2. GEMM FLOPs 粗估**

$$
\text{FLOPs}
\approx
2M N K
$$

对：

$$
C_{M \times N} = A_{M \times K} B_{K \times N}
$$

**3. Attention decode 读取量直觉**

$$
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
$$

**4. Qwen3.6 Gated Attention KV 每 token 粗估**

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

**5. MoE routed token-expert 对数量**

$$
N_{\text{token-expert}}
=
N_{\text{tokens}}
\times
8
$$

**6. MoE dispatch traffic 粗估**

$$
\text{traffic}
\propto
N
\times
k
\times
H
\times
B_{\text{dtype}}
$$

其中 Qwen3.6 中：

$$
k=8,\quad H=2048
$$

---

# 关键源码入口

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

# 源码证据表

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
| Qwen3.6 在你的 NVIDIA GPU 上实际选择 FlashAttention、FlashInfer、Triton 或其他 attention backend | 运行日志 / backend registry | 待运行确认 | 待源码确认 | 否 |
| Qwen3.6 MoE 在你的 GPU 上实际选择的 runner/kernel/backend | `fused_moe/runner` 与运行日志 | 待运行确认 | 待源码确认 | 否 |
| GDN kernel 在特定硬件上的 bottleneck 是 HBM、SM occupancy 还是 state layout | Nsight / profiling | 实验验证 | 否 |

---

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. 为什么 prefill 通常比 decode 更容易吃满 Tensor Core？**  
因为 prefill 一次处理多个 prompt token，GEMM 的 $N$ 更大，矩阵乘更规则；decode 每步新增 token 少，更多时间花在读取历史 KV/GDN state。

**2. 为什么 MoE kernel 优化不等于只优化 GEMM？**  
因为 MoE 还包含 router、top-k、token dispatch、expert grouping、expert output combine，以及 EP 下的 all-to-all 通信。

**3. 为什么不能看到 `vllm/v1/attention/backends/flashinfer.py` 就断言一定用了 FlashInfer？**  
因为 backend 选择依赖 GPU、dtype、KV cache dtype、模型结构、配置、环境变量和运行时条件；源码目录只能证明 backend 存在，实际选择要看运行日志和 profiler。

---

# 下一章预告

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

## Prompt:
2026/5/9 11:02:01

继续第十二章

## Response:
2026/5/9 11:02:05

> **Searching web for v0.20.1 GitHub source files**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Opening line ranges for relevant sections**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/kv_cache_manager.py)
>
> **Assisting with GDN line visibility and formatting**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/attention/backends/gdn_attn.py)
>
> **Searching for Qwen models and detailed sources**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Gathering and citing resources for chapter 12**
>
> I need to work on finalizing chapter 12. A web search may be enough to gather relevant details. Citations should be part of the paragraphs, not raw URLs. I'll use sources to ensure I'm giving a comprehensive, but concise answer. I also have to include references for specific tools and components mentioned.
>
> **Generating structured chapter with citations and references**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> Thought for 22s

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

$$
\Delta_i =
N^{\text{with\_spec}}_i
-
N^{\text{computed}}_i
$$

其中：

```text
N_computed      = 已经真正算完的 token 数
N_with_spec     = prompt tokens + output tokens + speculative tokens
Delta           = 当前 request 还需要追赶的 token 数
```

每个 scheduler step 有 token budget：

$$
\sum_i \Delta^{\text{scheduled}}_i
\le
\text{max\_num\_scheduled\_tokens}
$$

源码中 `max_num_scheduled_tokens` 来自 scheduler config；若没有单独设置，则 fallback 到 `max_num_batched_tokens`。`schedule()` 每轮用 `token_budget = self.max_num_scheduled_tokens` 控制本轮最多安排多少 token。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

对 KV cache，核心映射是：

$$
\text{Request tokens}
\rightarrow
\text{logical blocks}
\rightarrow
\text{physical KVCacheBlock}
\rightarrow
\text{block\_table / slot\_mapping}
\rightarrow
\text{Attention backend}
$$

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

# 小白总结

vLLM 像一个调度系统。

Scheduler 决定“这一轮谁上 GPU、算多少 token”；KVCacheManager 决定“这些 token 的历史笔记放到哪些 block”；BlockPool 管理这些 block 的借出、归还和复用；Attention Backend 拿到 block table 和 slot mapping 后，真正去读写 KV cache 并执行 attention。

Qwen3.6 更复杂，因为它不是普通 Transformer：一部分层走 Gated Attention 的 KV cache，一部分层走 Gated DeltaNet 的 state，所有层还接 MoE。

---

# 工程师总结

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

# 关键公式

**1. Scheduler token gap**

$$
\Delta_i =
N^{\text{with\_spec}}_i
-
N^{\text{computed}}_i
$$

**2. 本轮 token budget**

$$
\sum_i \Delta^{\text{scheduled}}_i
\le
\text{max\_num\_scheduled\_tokens}
$$

**3. Prefix cache recompute tokens**

$$
T_{\text{recompute}}
=
T_{\text{request}}
-
T_{\text{cached}}
$$

**4. Block id 映射**

$$
\text{physical\_block}
=
\text{block\_table}[\text{request}, \text{logical\_block}]
$$

**5. Slot mapping**

$$
\text{cache\_slot}
=
\text{slot\_mapping}[\text{new\_token\_index}]
$$

**6. Qwen3.6 cache 拆账**

$$
\text{Cache/State}
=
\text{GatedAttention KV}
+
\text{GatedDeltaNet State}
+
\text{Speculative Lookahead}
+
\text{Encoder/Multimodal Cache}
$$

---

# 关键源码入口

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

# 源码证据表

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
| 当前机器上实际选择 FlashAttention、FlashInfer、Triton 或其他 backend | 运行日志 / profiler | 待运行确认 | 待源码确认 | 否 |
| 当前机器上 GDN prefill backend 实际为 FlashInfer 还是 Triton/FLA | 运行日志 / `gdn_prefill_backend` | 待运行确认 | 待源码确认 | 否 |
| Qwen3.6 MTP 在 v0.20.1 中 `method="mtp"` 与 `method="qwen3_next_mtp"` 的完整解析差异 | speculative decoding config 源码 | 待展开 | 待源码确认 | 否 |

---

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. vLLM v0.20.1 Scheduler 为什么不直接说 prefill phase / decode phase？**  
因为源码用 `num_computed_tokens` 追赶 `num_tokens_with_spec` 的统一模型，prefill、decode、prefix caching、speculative decoding 都可以表达成“还差多少 token 没算”。

**2. `block_table` 和 `slot_mapping` 的区别是什么？**  
`block_table` 用来从请求的逻辑 block 找到物理 KV block；`slot_mapping` 用来把本轮新 token 的 K/V 写到对应 cache slot。

**3. 为什么 Qwen3.6 的 GDN 层不能直接套 FlashAttention KV cache 心智？**  
因为 GDN backend 是 SSM/Mamba-style path，metadata 包含 state indices、chunk indices、chunk offsets、initial state 等；它不是普通 `QKᵀ → softmax → V` 的 paged K/V attention。

---

# 下一章预告

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

## Prompt:
2026/5/9 11:08:00

继续第十三章

## Response:
2026/5/9 11:08:05

> **Browsing for vLLM and Qwen model documentation**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Qwen3-Next 使用指南 - vLLM 方案 - vLLM 文档](https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
>
> **Searching for open official sources**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Finding specific relevant lines in source code**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
> [Expert Parallel Deployment - vLLM](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
>
> **Searching for FP8 graphics card information**
>
> [Qwen/Qwen3.6-35B-A3B-FP8 · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8)
>
> **Finalizing Chapter 13 with structure and citations**
>
> Ok, I need a detailed Chapter 13, with a structured overview and citations. I'll focus on key topics like FP8, BF16 multi-card setups, and Expert Parallel configurations. I'll ensure accuracy on single-card FP8 limitations, emphasize the reduced weight in FP8 checkpoints, and keep in mind memory context sizes. I'll include formulas and cite all relevant documentation.
>
> Thought for 22s

第十三章会从“可跑、可稳、可降级”三个层次讲部署，不会把 FP8 简化成万能答案；我会把命令、硬件/显存判断、TP/EP/MTP/text-only 的取舍分开。

# 第十三章：工程部署：FP8 单卡、BF16 多卡、Expert Parallel、降级配置

本章把前 12 章的原理落成部署决策。核心目标不是“给一条万能命令”，而是建立一套判断框架：

```text
1. 先能启动
2. 再能稳定跑目标上下文
3. 再能承载目标并发
4. 最后再追求吞吐、延迟和成本最优
```

对 **Qwen/Qwen3.6-35B-A3B**，部署时必须始终记住：它是带 Vision Encoder 的 causal LM，35B total / 3B activated，hidden size 2048，40 层，结构是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，MoE 为 256 experts、8 routed + 1 shared expert，原生上下文 262,144 tokens，可扩展到 1,010,000 tokens。官方模型卡还明确提示：如果遇到 OOM，可以降低 context window，但建议至少保持 128K 以保留复杂任务能力。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

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

---

## 二、数学直觉

### 1. 权重显存

粗略地说：

$$
\text{weight memory}
\approx
\text{num parameters}
\times
\text{bytes per parameter}
$$

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

其中：

```text
10 = Gated Attention 层数
第一个 2 = K 和 V
第二个 2 = KV heads
256 = head dim
最后 2 = BF16/FP16 bytes
```

262,144 tokens 时，仅单序列 Gated Attention KV cache 粗略为：

$$
262144 \times 20480 \approx 5.37\ \text{GB}
$$

这还没有算 30 个 Gated DeltaNet 层的 state、MTP lookahead、MoE runtime buffer、block metadata、多模态、workspace、TP 下 KV heads 复制等。Qwen3.6 的层布局、Gated Attention 参数和 262K context 均来自模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

### 3. 并发放大

如果平均每个请求占用 $T$ 个 active tokens，并发数是 $B$，则仅 Gated Attention KV cache 就近似按：

$$
B
\times
T
\times
20480
$$

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

$$
\text{收益}
=
\text{accepted speculative tokens 带来的 TPOT 降低}
-
\text{额外 compute/cache/state/调度成本}
$$

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

# 小白总结

FP8 就像把模型文件压缩了，能省很多空间；但模型推理还需要草稿纸、历史笔记、临时工作台。Qwen3.6 的历史上下文很长，还有专家系统和视觉编码器，所以不能只看“模型文件能不能放进显卡”。

最稳妥的部署方法是：先多卡跑通 262K，再一步步打开 text-only、prefix cache、MTP、EP；单卡 FP8 更适合小上下文或降级服务，不要一开始就承诺满血 262K 高并发。

---

# 工程师总结

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

# 关键公式

**1. 权重显存粗估**

$$
\text{weight memory}
\approx
\text{num parameters}
\times
\text{bytes per parameter}
$$

**2. Qwen3.6 Gated Attention KV 每 token 粗估**

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

**3. 单序列 262K Gated Attention KV 粗估**

$$
262144
\times
20480
\approx
5.37\ \text{GB}
$$

**4. 并发 cache 粗估**

$$
\text{KV bytes}
\propto
B
\times
T
\times
20480
$$

**5. MoE token-expert 对数量**

$$
N_{\text{token-expert}}
=
N_{\text{tokens}}
\times
8
$$

**6. MTP 收益判断**

$$
\text{Net gain}
=
\text{accepted tokens gain}
-
\text{extra compute/cache/state cost}
$$

---

# 关键源码入口

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

# 源码证据表

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
| 单卡 FP8 是否能承载 262K、高并发、多模态和 MTP | 运行环境 / 显存 / profiler | 实验验证 | 否 |
| 当前机器上 MTP `method="mtp"` 与 `method="qwen3_next_mtp"` 的解析差异 | speculative config 源码 / 启动日志 | 待源码确认 | 否 |
| 当前机器上 EP all-to-all backend 是否优于纯 TP | runtime benchmark | 实验验证 | 否 |

---

# 源码阅读作业

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

# 实验验证作业

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

# 3 个自测题

**1. FP8 checkpoint 为什么不能保证单卡 262K 高并发？**  
因为 FP8 主要降低权重显存；262K 下的 KV cache、Gated DeltaNet state、MTP lookahead、多模态 buffer、MoE dispatch buffer 和 runtime workspace 仍然会消耗大量显存。

**2. 为什么 text-only 部署能释放显存？**  
因为 Qwen3.6 是带 Vision Encoder 的 causal LM；`--language-model-only` 会跳过 vision encoder 和 multimodal profiling，把更多内存留给 KV cache / state。

**3. Expert Parallel 为什么需要实测？**  
因为 EP 会把 experts 分到多个 GPU，可能降低 expert 权重压力并改善局部性，但也引入 all-to-all、expert imbalance、EPLB 显存代价和通信瓶颈；是否更快取决于硬件拓扑和 workload。

---

# 下一章预告

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

## Prompt:
2026/5/9 11:10:19

继续第十四章

## Response:
2026/5/9 11:10:21

> **Browsing web for up-to-date citations and model documentation**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Metrics - vLLM](https://docs.vllm.ai/en/latest/design/metrics/)
> [vllm/vllm/v1/core/sched/scheduler.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Searching for open line ranges metrics names**
>
> [Metrics - vLLM](https://docs.vllm.ai/en/latest/design/metrics/)
>
> **Searching for prefix cache metrics and Qwen recipes**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [vllm/vllm/v1/core/sched/scheduler.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Summarizing and citing Chapter 14 content and evidence**
>
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py)
>
> **Structuring the answer with citations**
>
> I need to focus on the Qwen structure, metrics, and optimization strategies like preemption. I'll cite the relevant sections while keeping it concise. Each section will include necessary formulas and commands, with sources cited properly instead of just raw URLs. The final answer will be split into 10 sections, with a conclusion at the end. I’ll also compile a source evidence table for clarity.
>
> **Crafting structure, optimization, and citations for Qwen model**
>
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
>
> Thought for 26s

第十四章会把“为什么慢/为什么 OOM/为什么吞吐不稳”拆成可观测指标和排障路径：先定义压测矩阵，再把问题定位到 Scheduler、KV/GDN state、MoE、Attention backend、NCCL 或 sampling。

# 第十四章：压测、监控、排障

本章目标：把 Qwen/Qwen3.6-35B-A3B 在 vLLM v0.20.1 上的部署问题，拆成可以测、可以看、可以定位的工程闭环。

前面章节讲了很多结构和源码，这一章只关心一句话：

```text
出现慢、抖、OOM、吞吐不达标时，怎么判断瓶颈在哪里？
```

固定案例仍然是 **Qwen/Qwen3.6-35B-A3B on NVIDIA GPU with vLLM v0.20.1**。模型结构事实先固定：Qwen3.6-35B-A3B 是带 Vision Encoder 的 causal LM，35B total / 3B activated，hidden size 2048，40 层，hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，MoE 为 256 experts、8 routed + 1 shared expert，原生上下文 262,144 tokens。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 一、生活类比

压测像体检，监控像戴手环，排障像看病。

如果一个人说“我不舒服”，医生不会直接开药，而是先问：

```text
是心跳快？
是血压高？
是走路喘？
是吃饭后疼？
是睡觉时发作？
```

LLM serving 也是一样。用户说“慢”，可能是：

```text
TTFT 高：首 token 慢，常见于 prefill / queue / prefix cache miss
TPOT 高：后续 token 慢，常见于 decode / KV cache / GDN state / MoE
吞吐低：整体 tokens/s 不够，常见于 batch 不足 / kernel 碎 / 通信重
OOM：可能发生在启动、prefill、decode、MTP、多模态任一阶段
抖动大：可能是 preemption、chunked prefill、NCCL、MoE imbalance
```

所以工程上不要问“vLLM 慢怎么办”，要问：

```text
慢的是 TTFT，还是 TPOT？
OOM 发生在启动、prefill，还是 decode？
GPU 是算力满了，还是 HBM 满了？
KV cache usage 是否接近 1？
waiting queue 是否堆积？
MoE experts 是否严重不均？
MTP accepted tokens 是否足够？
```

vLLM V1 暴露 Prometheus `/metrics`，指标前缀为 `vllm:`，包括 running request 数、KV cache 使用率、prefix cache queries/hits、prompt/generation token 总数等；请求级指标还包括 TTFT、inter-token latency、prefill time、decode time 等。([vLLM](https://docs.vllm.ai/en/latest/design/metrics/))

---

## 二、数学直觉

压测不是随便发请求，而是构造输入变量矩阵。

核心变量：

```text
prompt_len
output_len
concurrency
max_model_len
max_num_batched_tokens
MTP on/off
text-only vs multimodal
prefix shared ratio
logprobs on/off
```

### 1. TTFT

TTFT 可以粗略拆成：

$$
\text{TTFT}
=
T_{\text{queue}}
+
T_{\text{prefill}}
+
T_{\text{first\_sample}}
$$

如果 TTFT 高，要优先看：

```text
request_queue_time
request_prefill_time
prefix cache hit rate
waiting requests
chunked prefill 是否过小或过大
```

### 2. TPOT / ITL

TPOT 可以粗略理解为：

$$
\text{TPOT}
\approx
\frac{T_{\text{decode total}}}{N_{\text{generated tokens}}}
$$

如果 TPOT 高，要优先看：

```text
decode attention backend
KV/GDN state 读取
MoE expert dispatch
NCCL all-reduce / all-to-all
sampling / logits processor
```

vLLM 指标中 `vllm:inter_token_latency_seconds` 对应 inter-token latency，也就是工程上常说的 TPOT / ITL；`vllm:time_to_first_token_seconds` 对应 TTFT。([vLLM](https://docs.vllm.ai/en/latest/design/metrics/))

### 3. Throughput

吞吐通常用：

$$
\text{throughput}
=
\frac{\text{prompt tokens} + \text{generation tokens}}
{\text{wall time}}
$$

但生产里更常看：

```text
prefill tokens/s
decode tokens/s
output tokens/s
requests/s
```

Qwen3.6 是 MoE + GDN + Gated Attention 混合架构，prefill 与 decode 的瓶颈可能完全不同。prefill 更可能受 GEMM / MoE expert GEMM / attention compute 影响；decode 更可能受 KV/GDN state 读取、MoE dispatch、NCCL 和 sampler 影响。

### 4. KV cache 压力

KV cache 使用率接近 1 时，系统可能 preempt 或排队。vLLM 文档说明，当 KV cache 空间不足以处理所有 batched requests 时，vLLM 可以 preempt requests 以释放 KV cache；频繁 preemption 会影响端到端延迟，官方建议方向包括提高 `gpu_memory_utilization`、降低 `max_num_seqs` 或 `max_num_batched_tokens`、增加 TP 或 PP。([vLLM](https://docs.vllm.ai/en/latest/configuration/optimization/))

---

## 三、张量形状

压测时要把请求分成几种“形状”。

### 1. 短 prompt 长输出

```text
prompt_len = 1K
output_len = 4K
```

主要测试 decode：

```text
KV/GDN state 读取
MoE dispatch
TPOT
MTP 是否有效
sampling/logits 开销
```

### 2. 长 prompt 短输出

```text
prompt_len = 128K
output_len = 128
```

主要测试 prefill：

```text
chunked prefill
TTFT
prefix caching
GDN prefill state
prefill attention / MoE kernel
```

### 3. 长 prompt 长输出

```text
prompt_len = 128K / 262K
output_len = 4K / 16K
```

主要测试极限容量：

```text
KV cache
GDN state
scheduler preemption
long-context decode
NCCL / backend stability
```

### 4. 共享 prefix workload

```text
shared_prefix = 16K / 64K
unique_suffix = 128
output_len = 512
```

主要测试 prefix caching。vLLM metrics 暴露 `vllm:prefix_cache_queries` 和 `vllm:prefix_cache_hits`，可以直接观察命中情况。([vLLM](https://docs.vllm.ai/en/latest/design/metrics/))

### 5. 多模态 workload

```text
image/video + text prompt
```

主要测试：

```text
Vision Encoder
multimodal processor
encoder budget
media embedding size
prefill time
显存峰值
```

vLLM Qwen recipe 对多模态 workload 建议使用 `--mm-encoder-tp-mode data` 做 data-parallel vision encoding，并使用 `--mm-processor-cache-type shm` 高效缓存和传输预处理后的多模态输入。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 四、计算流程

压测流程建议按“从干净基线到复杂组合”推进。

### 1. 建立最小基线

先跑官方 Qwen3.6 基础命令：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

vLLM recipe 给出的 Qwen3.6 基础命令就是 TP=8、`--max-model-len 262144`、`--reasoning-parser qwen3`。同一 recipe 还给出 MTP speculative decoding 配置 `{"method": "mtp", "num_speculative_tokens": 2}`。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 2. 固定变量，逐个改变

不要一次性改变所有参数。建议顺序：

```text
A. 固定 text-only，MTP off，prefix cache off/on 分开测
B. 改 prompt_len
C. 改 output_len
D. 改 concurrency
E. 改 max_num_batched_tokens
F. 打开 MTP
G. 打开多模态
H. 测 EP / EPLB
```

### 3. 每组实验至少记录

```text
TTFT p50 / p90 / p99
TPOT p50 / p90 / p99
e2e latency p50 / p90 / p99
prompt tokens/s
output tokens/s
requests/s
GPU memory
HBM bandwidth
GPU utilization
NVLink / PCIe / IB bandwidth
kv_cache_usage_perc
num_requests_running
num_requests_waiting
prefix cache hit rate
preemption count
```

vLLM 的 Grafana dashboard 指标集特别强调 e2e request latency、prompt/generation tokens、ITL、TTFT、running/waiting request 数、KV cache usage、queue time、prefill time 和 decode time。([vLLM](https://docs.vllm.ai/en/latest/design/metrics/))

### 4. 压测结果要按瓶颈分类

```text
TTFT 高：
  看 queue / prefill / prefix miss / chunked prefill

TPOT 高：
  看 decode / KV / GDN / MoE / sampler / NCCL

吞吐低：
  看 batch size / GPU utilization / kernel 碎片 / communication

OOM：
  看发生阶段和显存账本

抖动：
  看 preemption / waiting queue / long prompt / EP all-to-all
```

---

## 五、显存布局

Qwen3.6 的显存排障一定要拆账，不能只看 `nvidia-smi` 总占用。

### 1. 启动阶段 OOM

常见原因：

```text
权重太大
Vision Encoder 加载
profiling / CUDA graph / workspace
KV/GDN state 预留过大
max_model_len 太大
```

优先处理：

```text
使用 FP8 checkpoint
加 --language-model-only
降低 max_model_len
降低 gpu_memory_utilization 之外的其他峰值项
减少并行/多模态组合复杂度
```

### 2. Prefill 阶段 OOM

常见原因：

```text
prompt 太长
chunked prefill 太大
多模态 embedding 太多
prefill activation / MoE buffer 峰值高
```

优先处理：

```text
降低 max_num_batched_tokens
开启/调小 chunked prefill
限制单请求 prompt_len
启用 prefix caching
多模态服务单独分池
```

vLLM optimization 文档说明，chunked prefill 会把大 prefill 切小并与 decode requests batch 到一起；V1 中可通过 `max_num_batched_tokens` 调节性能，较小值通常改善 ITL，较大值通常改善 TTFT。([vLLM](https://docs.vllm.ai/en/latest/configuration/optimization/))

### 3. Decode 阶段 OOM

常见原因：

```text
并发请求多
输出太长
KV cache / GDN state 随生成增长
MTP lookahead tokens 占额外 slots
prefix cached blocks 挤占空间
```

优先处理：

```text
降低 max_num_seqs
降低 max_tokens
降低 max_model_len
关闭 MTP 或降低 num_speculative_tokens
限制长上下文并发
观察 preemption
```

### 4. MTP 后 OOM

常见原因：

```text
speculative tokens 额外占 cache/state
num_speculative_tokens 太大
高并发下有效 batch size 下降
```

vLLM v0.20.1 Scheduler 源码中，speculative config 会设置 `num_spec_tokens` 和 `num_lookahead_tokens`；Scheduler 在分配 running request 的 slots 时会把 `num_lookahead_tokens` 传入 `kv_cache_manager.allocate_slots()`。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

### 5. GDN state 相关 OOM 或 miss

Qwen3.6 有 30 个 Gated DeltaNet 层，不是普通 KV cache。vLLM v0.20.1 的 Scheduler 在 hybrid/Mamba cache align 模式下会调用 `_mamba_block_aligned_split()`；源码注释说明，为了启用 Mamba state 的 block-aligned caching，prefill 阶段 `num_new_tokens` 必须是 block size 的倍数，不足一个 block 的 state 不缓存。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

---

## 六、vLLM 源码映射

| 排障对象 | 文件路径 | 类/函数 | 看什么 |
|---|---|---|---|
| 调度与等待 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | running / waiting / token_budget / preemption |
| MTP lookahead | `vllm/v1/core/sched/scheduler.py` | `num_spec_tokens`, `num_lookahead_tokens` | speculative tokens 是否增加 cache 预算 |
| GDN block align | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` | GDN/Mamba state caching 是否被 block 对齐限制 |
| KV slot 分配 | `vllm/v1/core/kv_cache_manager.py` | `allocate_slots` | 新 token / lookahead / encoder tokens 是否能分配 slots |
| Prefix cache | `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks` | prefix 命中多少 full blocks |
| GDN metadata | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionMetadataBuilder` | state indices、chunk indices、spec/non-spec state |
| Qwen3.6 hybrid layer | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextDecoderLayer` | linear_attention vs full_attention 分流 |
| Qwen3.6 MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | router、FusedMoE、shared expert |
| 指标 | vLLM docs | `/metrics` | TTFT、ITL、KV usage、prefix hits、running/waiting |

Scheduler 源码明确引入 `KVCacheManager`、`KVCacheBlocks`、`PrefixCacheStats`、`SpecDecodingStats`、`RoutedExpertsReader` 等对象，并维护 `waiting`、`skipped_waiting`、`running` 队列；这说明压测时看到的 queue、prefix cache、MTP、MoE routed experts 都不是外部猜测，而是 vLLM v0.20.1 Scheduler 的显式关注点。([GitHub](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py))

---

## 七、NVIDIA GPU 执行视角

### 1. Prefill 慢

可能瓶颈：

```text
大 GEMM
Gated Attention prefill
GDN prefill chunk kernel
MoE expert GEMM
Vision Encoder
```

观察：

```text
Tensor Core utilization
SM occupancy
prefill kernels
MoE kernels
vision encoder kernels
```

### 2. Decode 慢

可能瓶颈：

```text
KV cache 读 HBM
GDN state 读写
MoE dispatch 小 batch
NCCL all-reduce / all-to-all
sampling / logits
```

观察：

```text
HBM bandwidth
attention backend kernel time
GDN kernel time
FusedMoE kernel time
NCCL timeline
sampler / logits processor time
```

### 3. GPU 利用率低

可能原因：

```text
concurrency 太低
decode batch 太小
max_num_batched_tokens 太小
请求长度分布太碎
MTP acceptance 低
MoE experts token 数太分散
CPU/tokenizer/processor 成为瓶颈
```

处理：

```text
增加并发
调大 max_num_batched_tokens
打开 prefix caching
分离多模态预处理
检查 tokenizer/processor CPU
```

### 4. GPU 利用率高但吞吐低

可能原因：

```text
大量时间花在 HBM 读写
NCCL 通信重
MoE dispatch 重
sampler/logprobs 重
preemption/recompute
```

处理：

```text
看 HBM bandwidth
看 NVLink/PCIe/IB
关 logprobs
降低长上下文并发
检查 EP all-to-all
检查 preemption
```

---

## 八、优化前 vs 优化后

### 优化前：只看 tokens/s

错误：

```text
tokens/s 不够 → 盲目调 tensor_parallel_size
OOM → 盲目换 FP8
TTFT 高 → 盲目打开 MTP
TPOT 高 → 盲目打开 prefix caching
```

这些动作可能完全不对应瓶颈。

### 优化后：按症状定位

| 症状 | 第一优先指标 | 第二优先指标 | 常见动作 |
|---|---|---|---|
| TTFT 高 | queue time / prefill time | prefix hit rate | prefix caching、chunked prefill、调 batch token |
| TPOT 高 | inter-token latency | decode time / HBM | 降上下文并发、看 backend、MTP 对照 |
| OOM 启动 | GPU memory at load | model/vision/cache | FP8、text-only、降 max_model_len |
| OOM prefill | prompt len / prefill time | activation/workspace | 限 prompt、chunked prefill |
| OOM decode | KV usage | output len / concurrency | 限并发、限 max_tokens、关 MTP |
| 吞吐低 | GPU util / tokens/s | running/waiting | 调并发、batch token、EP |
| 抖动大 | preemption / waiting | NCCL / expert imbalance | 降 cache 压力、EPLB、分池 |
| logprobs 慢 | sampler/logits time | payload size | 禁止 `logprobs=-1` |

### Qwen3.6 的特殊排障边界

```text
Gated Attention 慢：
  看 attention backend、KV cache、block table、slot mapping

Gated DeltaNet 慢：
  看 GDN backend、chunk indices、mamba_cache_mode、state cache

MoE 慢：
  看 router、top-k、expert token imbalance、FusedMoE、EP all-to-all

MTP 不收益：
  看 accepted speculative tokens、lookahead cache、并发下有效 batch size

多模态慢：
  看 Vision Encoder、processor cache、media embedding size、encoder budget
```

---

## 九、工程落地

### 1. 压测矩阵

建议至少跑这个矩阵：

| Case | prompt_len | output_len | concurrency | 目的 |
|---|---:|---:|---:|---|
| A | 1K | 512 | 1 / 8 / 32 / 128 | 基础 decode |
| B | 1K | 4096 | 1 / 8 / 32 | 长输出 decode |
| C | 32K | 512 | 1 / 8 / 32 | 中长 prefill |
| D | 128K | 512 | 1 / 4 / 8 | 长 prefill |
| E | 262K | 128 | 1 / 2 / 4 | 极限上下文 |
| F | shared 16K prefix + 128 suffix | 512 | 8 / 32 | prefix caching |
| G | image + text | 512 | 1 / 8 | 多模态 |
| H | 1K | 4096 | 8 / 32 | MTP off/on |

### 2. 推荐启动基线

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

text-only 对照：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

MTP 对照：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

Qwen3.6 的 recipe 给出 TP=8、262144 max model len 的基础命令和 MTP speculative decoding 命令；同一 recipe 还提示，MTP-1 在低并发低延迟场景可降低 TPOT，但高并发下会因 speculative tokens 消耗 KV cache capacity 而降低 throughput。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 3. 监控面板最小集合

生产面板至少包含：

```text
TTFT p50/p90/p99
TPOT p50/p90/p99
E2E latency p50/p90/p99
prompt tokens/s
generation tokens/s
request queue time
request prefill time
request decode time
num_requests_running
num_requests_waiting
kv_cache_usage_perc
prefix_cache_queries
prefix_cache_hits
request_prompt_tokens
request_generation_tokens
GPU memory
GPU HBM bandwidth
GPU SM utilization
NVLink / PCIe / IB bandwidth
```

vLLM metrics 文档列出的 V1 metrics 包括 `vllm:num_requests_running`、`vllm:kv_cache_usage_perc`、`vllm:prefix_cache_queries`、`vllm:prefix_cache_hits`、`vllm:time_to_first_token_seconds`、`vllm:inter_token_latency_seconds`、`vllm:request_prefill_time_seconds` 和 `vllm:request_decode_time_seconds`。([vLLM](https://docs.vllm.ai/en/latest/design/metrics/))

### 4. 排障 Runbook

**Runbook A：TTFT 高**

```text
1. 看 num_requests_waiting
2. 看 request_queue_time_seconds
3. 看 request_prefill_time_seconds
4. 看 prefix_cache_hits / prefix_cache_queries
5. 看 prompt_len 分布
6. 调 max_num_batched_tokens
7. 检查 chunked prefill
```

**Runbook B：TPOT 高**

```text
1. 看 inter_token_latency_seconds
2. 看 request_decode_time_seconds
3. 看 context length 分布
4. 看 KV cache usage
5. profiler 看 attention/GDN/MoE/sampler
6. 对照 MTP off/on
7. 检查 NCCL timeline
```

**Runbook C：频繁 preemption**

```text
1. 看 kv_cache_usage_perc 是否接近 1
2. 看 max_model_len 是否过大
3. 看 max_num_seqs / concurrency
4. 看 output_len 是否过长
5. 关 MTP 或降低 speculative tokens
6. 提高 gpu_memory_utilization 或增加 TP/PP
```

vLLM 官方优化文档说明，KV cache 空间不足时会发生 preemption/recompute，并建议提高 `gpu_memory_utilization`、降低并发相关参数或增加并行度，但也提醒增加 TP 可能带来同步开销。([vLLM](https://docs.vllm.ai/en/latest/configuration/optimization/))

**Runbook D：MoE 慢**

```text
1. 打开 routed experts 统计或 profiler
2. 看 expert hit count
3. 看 per-rank GPU utilization
4. 看 all-to-all 时间
5. 比较 EP off/on
6. 如果 expert imbalance 明显，测试 EPLB
```

**Runbook E：多模态慢**

```text
1. 单独测 text-only
2. 单独测 image/video
3. 看 vision encoder time
4. 看 processor cache
5. 使用 mm processor cache / encoder data parallel
6. text-only 和 multimodal 分服务池
```

vLLM Qwen recipe 对多模态 workload 明确给出 `--mm-encoder-tp-mode data` 和 `--mm-processor-cache-type shm` 的建议。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 十、本章总结

压测、监控、排障的核心不是“跑一个平均 tokens/s”，而是建立完整因果链：

```text
Workload shape
→ Scheduler 行为
→ KV/GDN state 压力
→ Attention backend
→ MoE dispatch / expert GEMM
→ NCCL 通信
→ logits / sampler
→ TTFT / TPOT / throughput
```

Qwen3.6 的排障尤其要拆开：

```text
Gated Attention:
  KV cache / attention backend / context length

Gated DeltaNet:
  GDN state / mamba cache mode / chunk metadata

MoE:
  router / top-k / expert dispatch / expert imbalance

MTP:
  acceptance rate / lookahead cache / high-concurrency throughput

Vision Encoder:
  multimodal preprocessing / encoder cache / media embeddings
```

---

# 小白总结

不要只问“模型慢不慢”。

要分别看：

```text
第一个字出来慢不慢？--TTFT
后面每个字慢不慢？--TPOT
总共每秒吐多少字？--throughput
显存是不是快满了？--KV cache usage
请求是不是在排队？--waiting requests
```

Qwen3.6 很复杂，有长上下文、有专家、有 Gated DeltaNet、有多模态、有 MTP。排障时要像医生看报告一样，先看指标，再定位问题，不要盲目调参数。

---

# 工程师总结

生产排障优先级：

```text
1. Workload 分型：
   prompt_len / output_len / concurrency / multimodal / prefix reuse

2. Request-level metrics：
   TTFT / TPOT / e2e / queue / prefill / decode

3. Engine-level metrics：
   running / waiting / kv_cache_usage / prefix hit rate

4. GPU-level metrics：
   SM / HBM / Tensor Core / NCCL / memory

5. Module-level profiler：
   attention backend / GDN / FusedMoE / LM head / sampler

6. 配置对照：
   text-only vs multimodal
   MTP off/on
   prefix cache off/on
   EP off/on
   max_model_len 262K/128K/64K
```

最重要的工程纪律：**每次只改变一个变量**。否则你不知道性能变化来自哪里。

---

# 关键公式

**1. TTFT 拆解**

$$
\text{TTFT}
=
T_{\text{queue}}
+
T_{\text{prefill}}
+
T_{\text{first\_sample}}
$$

**2. TPOT**

$$
\text{TPOT}
=
\frac{T_{\text{decode total}}}
{N_{\text{generated tokens}}}
$$

**3. Throughput**

$$
\text{throughput}
=
\frac{N_{\text{prompt tokens}} + N_{\text{generation tokens}}}
{T_{\text{wall}}}
$$

**4. Prefix cache hit rate**

$$
\text{hit rate}
=
\frac{\text{prefix\_cache\_hits}}
{\text{prefix\_cache\_queries}}
$$

**5. KV cache pressure**

$$
\text{KV pressure}
\propto
\sum_i
T_i
\times
\text{cache bytes per token}
$$

**6. Qwen3.6 Gated Attention KV 每 token 粗估**

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

**7. MoE routed token-expert 对数量**

$$
N_{\text{token-expert}}
=
N_{\text{tokens}}
\times
8
$$

---

# 关键源码入口

| 模块 | 文件路径 | 类/函数 |
|---|---|---|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` |
| Running / waiting 队列 | `vllm/v1/core/sched/scheduler.py` | `self.running`, `self.waiting`, `self.skipped_waiting` |
| Preemption | `vllm/v1/core/sched/scheduler.py` | `_preempt_request`, `preempted_reqs` |
| MTP lookahead | `vllm/v1/core/sched/scheduler.py` | `num_spec_tokens`, `num_lookahead_tokens` |
| GDN/Mamba align | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `allocate_slots`, `get_computed_blocks` |
| GDN backend | `vllm/v1/attention/backends/gdn_attn.py` | `GDNAttentionBackend`, `GDNAttentionMetadataBuilder` |
| Qwen3.6 hybrid layer | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextDecoderLayer` |
| Qwen3.6 Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Qwen3.6 MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| vLLM metrics | vLLM docs | `/metrics`, `vllm:*` |

---

# 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| Qwen3.6 是带 Vision Encoder 的 causal LM，35B total / 3B activated，40 层，262K context | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MoE 是 256 experts，8 routed + 1 shared | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 recipe 使用 TP=8、max model len 262144、reasoning parser qwen3 | vLLM Qwen recipe | Running Qwen3.6 | recipe 确认 | 是 |
| Qwen3.6 recipe 给出 MTP speculative decoding 配置 `{"method": "mtp", "num_speculative_tokens": 2}` | vLLM Qwen recipe | MTP command | recipe 确认 | 是 |
| vLLM V1 暴露 Prometheus `/metrics`，指标以 `vllm:` 为前缀 | vLLM metrics docs | Metrics Overview | 官方文档确认 | 是 |
| vLLM 指标包含 running requests、KV cache usage、prefix cache queries/hits | vLLM metrics docs | V1 metrics | 官方文档确认 | 是 |
| vLLM 指标包含 TTFT、inter-token latency、prefill time、decode time | vLLM metrics docs | Request metrics | 官方文档确认 | 是 |
| vLLM V1 chunked prefill 会把大 prefill 切小并与 decode requests batch；decode 优先 | vLLM optimization docs | Chunked Prefill | 官方文档确认 | 是 |
| `max_num_batched_tokens` 较小倾向改善 ITL，较大倾向改善 TTFT | vLLM optimization docs | Performance tuning with chunked prefill | 官方文档确认 | 是 |
| KV cache 空间不足时 vLLM 会 preempt requests 并 recompute | vLLM optimization docs | Preemption | 官方文档确认 | 是 |
| Scheduler 维护 running/waiting 队列，使用 token budget 调度请求 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 中 speculative config 会设置 `num_spec_tokens` / `num_lookahead_tokens` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.__init__` | 源码直接确认 | 是 |
| Scheduler 会把 `num_lookahead_tokens` 传入 `allocate_slots` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 对 Mamba/GDN align mode 使用 `_mamba_block_aligned_split` | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` | 源码直接确认 | 是 |
| Scheduler 引入 `RoutedExpertsReader`，说明 routed experts 可作为观测/返回路径的一部分 | `vllm/v1/core/sched/scheduler.py` | imports / init path | 源码直接确认 | 是 |
| GDN backend / Qwen3Next / KV manager 的具体运行瓶颈需要 profiler 验证 | Nsight / runtime logs | N/A | 合理推断 | 否 |
| 当前机器上实际 attention backend、GDN backend、MoE runner | 运行日志 / profiler | N/A | 待源码确认 | 否 |

---

# 源码阅读作业

**1. 读 Scheduler 排障入口**

```text
vllm/v1/core/sched/scheduler.py
```

重点找：

```text
self.running
self.waiting
token_budget
num_lookahead_tokens
_mamba_block_aligned_split
preempted_reqs
RoutedExpertsReader
```

目标：把 waiting/running、MTP、GDN align、preemption 和 MoE routed experts 观测关联起来。

**2. 读 KV cache 排障入口**

```text
vllm/v1/core/kv_cache_manager.py
```

重点找：

```text
KVCacheManager
get_computed_blocks
allocate_slots
usage
```

目标：理解 prefix hit、slot allocation、KV usage 如何影响 TTFT / OOM / preemption。

**3. 读 GDN backend**

```text
vllm/v1/attention/backends/gdn_attn.py
```

重点找：

```text
GDNAttentionMetadata
GDNAttentionMetadataBuilder
chunk_indices
chunk_offsets
spec_state_indices
non_spec_state_indices
```

目标：理解 GDN state 不是普通 KV cache。

**4. 读 Qwen3Next MoE**

```text
vllm/model_executor/models/qwen3_next.py
```

重点找：

```text
Qwen3NextSparseMoeBlock
router_logits
FusedMoE
shared_expert
```

目标：理解 MoE 慢时应该看 router、expert dispatch 和 shared expert。

**5. 读 metrics 文档**

重点整理这些指标：

```text
vllm:time_to_first_token_seconds
vllm:inter_token_latency_seconds
vllm:request_prefill_time_seconds
vllm:request_decode_time_seconds
vllm:num_requests_running
vllm:num_requests_waiting
vllm:kv_cache_usage_perc
vllm:prefix_cache_queries
vllm:prefix_cache_hits
```

---

# 实验验证作业

### 实验 1：基础压测矩阵

跑：

```text
prompt_len: 1K / 32K / 128K / 262K
output_len: 128 / 512 / 4096
concurrency: 1 / 8 / 32 / 128
```

每组记录：

```text
TTFT p50/p90/p99
TPOT p50/p90/p99
tokens/s
kv_cache_usage_perc
num_requests_running
num_requests_waiting
GPU utilization
HBM bandwidth
```

### 实验 2：prefix caching

构造：

```text
shared_prefix = 16K or 64K
different_suffix = 128
output_len = 512
```

观察：

```text
prefix_cache_queries
prefix_cache_hits
TTFT
request_prefill_time
GPU blocks usage
```

### 实验 3：MTP off/on

对照：

```text
MTP off
MTP on with num_speculative_tokens = 2
```

观察：

```text
accepted speculative tokens
TPOT
throughput
kv_cache_usage_perc
preemption count
```

### 实验 4：多模态分池

对照：

```text
text-only
image + text
video + text
```

观察：

```text
vision encoder time
request_prefill_time
GPU memory
processor cache
TTFT
```

### 实验 5：MoE expert imbalance

采集：

```text
每层 expert hit count
per-rank expert tokens
FusedMoE kernel time
all-to-all time
GPU utilization per rank
```

目标：判断是否需要 EP / EPLB。

---

# 3 个自测题

**1. TTFT 高和 TPOT 高分别优先排查什么？**  
TTFT 高优先看 queue、prefill、prefix cache、chunked prefill；TPOT 高优先看 decode、KV/GDN state、MoE、NCCL、sampler。

**2. 为什么 KV cache usage 接近 1 时容易出现延迟抖动？**  
因为 KV cache 空间不足时，vLLM 可能 preempt requests 并在之后 recompute；这会增加端到端延迟并造成抖动。

**3. 为什么 Qwen3.6 的压测必须区分 text-only 和 multimodal？**  
因为 Qwen3.6 带 Vision Encoder，多模态请求会引入 vision encoder、processor cache、media embedding、encoder budget 等额外路径；把它和纯文本请求混在一起看平均值会掩盖瓶颈。

---

# 下一章预告

第十五章进入：

```text
总复盘与学习路线图
```

最后一章会把整门课收束成三张地图：

```text
1. 原理地图：
   token → embedding → GDN / attention / MoE → logits → sampling

2. 系统地图：
   scheduler → KV/GDN state → backend → GPU kernel → metrics

3. 工程路线图：
   从小模型复现，到 Qwen3.6 生产部署，再到源码贡献级阅读路径
```

---

**Sources:**

- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Metrics - vLLM](https://docs.vllm.ai/en/latest/design/metrics/)
- [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/v1/core/sched/scheduler.py)

## Prompt:
2026/5/9 11:12:23

继续第十五章

## Response:
2026/5/9 11:12:27

> **Considering citation needs for continuation**
>
> The user wants a continuation based on previous facts, and since those sources were already cited, I wonder if I need additional, up-to-date citations. While I could reference old ones, it might be best to browse and confirm new details before continuing. I'll look for any required updates.
>
> **Searching for model information and citations**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Releases · vllm-project/vllm - GitHub](https://github.com/vllm-project/vllm/releases)
> [Qwen3-Next 使用指南 - vLLM 方案 - vLLM 文档](https://docs.vllm.com.cn/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
> [Qwen3.6-35B-A3B-FP8 · Models](https://www.modelscope.cn/models/Qwen/Qwen3.6-35B-A3B-FP8)
> [Previous Releases - vLLM](https://vllm.ai/releases)
>
> **Considering citations for chapter 15 answer**
>
> I need to provide an answer for chapter 15 with proper citations. I think we have enough sources, but I'm wondering if I should cite the vLLM release. I might need to open the relevant page and check metrics, then include that in the citations. Let's make sure to capture the raw data too.
>
> **Searching for citations and source references for chapter 15**
>
> [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
> [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
> [Releases · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases)
> [raw.githubusercontent.com](https://raw.githubusercontent.com/vllm-project/vllm/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Navigating source accessibility and citations**
>
> It seems like I might be running into a 404 error or a problem with opening the source lines. If the raw source isn't accessible, I might need to use previous statements instead. A source evidence table could help, and if it's still inaccessible, citing pages or opening a GitHub blob might work.
>
> **Seeking raw open lines and source evidence**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Finding specific lines for qwen3_next functions**
>
> I might need to locate specific lines for the qwen3_next functions. To do this, I'll have to carefully find and isolate the relevant parts. Once I have those, it'll make it easier to proceed with and reference the necessary details without going off track.
>
> **Finding specific lines for qwen3_next functions**
>
> [vllm/vllm/model_executor/models/qwen3_next.py at v0.20.1 · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py)
>
> **Preparing chapter 15 with required sections and citations**
>
> I need to prepare the final chapter with all the essential sections like comparisons and summaries, as well as specific items like key formulas, source code entries, and reading assignments. I’ll need to include citations, especially for facts, such as the v0.20.1 release on May 4, 2026. Since web sources are used, relevant sections will be cited as whole sources, though not every paragraph needs to be individually cited.
>
> Thought for 25s

# 第十五章：总复盘与学习路线图

本章把 15 轮课程收束成三张地图：

```text
1. 原理地图：
   token → embedding → Gated DeltaNet / Gated Attention / MoE → logits → sampling

2. 系统地图：
   request → scheduler → KV/GDN state → backend → GPU kernel → metrics

3. 工程路线图：
   小模型复现 → Qwen3.6 部署 → vLLM 源码导读 → profiler 优化 → 源码贡献
```

版本边界仍然固定：**vLLM v0.20.1**。GitHub release 页面显示 v0.20.1 发布于 **2026-05-04**，是基于 v0.20.0 的 patch release，重点包括 DeepSeek V4 稳定性/性能改进和若干 bug fixes；本课程所有 vLLM 源码判断都只按 v0.20.1 tag 对齐，不混用旧版实现。([GitHub](https://github.com/vllm-project/vllm/releases))

---

## 一、生活类比

把整门课想成从“会开车”到“会造车”的过程。

最开始你只知道：

```text
输入一句话
模型输出一句话
```

现在你应该能把它拆成：

```text
文字如何变成 token
token 如何变成 embedding
embedding 如何经过 40 层 hybrid block
Gated DeltaNet 如何不同于普通 attention
Gated Attention 如何读 KV cache
MoE 如何路由到 8 routed experts + 1 shared expert
MTP 如何草拟未来 tokens
vLLM Scheduler 如何决定这一轮算谁
KV cache / GDN state 如何占显存
NVIDIA GPU 如何执行 GEMM、attention、MoE、sampling
```

对 Qwen3.6，这个拆解特别重要。它不是普通 Transformer Decoder，而是带 Vision Encoder 的 causal LM；语言主干是 40 层 hybrid layout：`10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，并且有 35B total / 3B activated、256 experts、8 routed + 1 shared expert、262K 原生上下文、MTP trained with multi-steps。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

---

## 二、数学直觉

整门课的数学主线可以压缩成四层。

第一层是自回归语言模型：

$$
P(x_{1:T})=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

第二层是 token 到 logits：

$$
x_i = E[i]
$$

$$
z_t = W_{\text{lm}}h_t
$$

$$
p_t = \operatorname{softmax}(z_t)
$$

第三层是传统 attention 参照系：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + M
\right)V
$$

第四层是 Qwen3.6 的真实替换：

$$
10 \times
\left[
3 \times
(\text{GatedDeltaNet}\rightarrow \text{MoE})
+
1 \times
(\text{GatedAttention}\rightarrow \text{MoE})
\right]
$$

MoE 的核心公式是：

$$
r = \operatorname{Router}(h)
$$

$$
\mathcal{E}_{top8} = \operatorname{TopK}(r, 8)
$$

$$
y =
\sum_{e \in \mathcal{E}_{top8}} w_eE_e(h)
+
E_{\text{shared}}(h)
$$

这就是从“普通 LLM”到“Qwen3.6 hybrid MoE 推理系统”的核心数学跃迁。

---

## 三、张量形状

你现在应该能从形状上直接看出性能风险。

Qwen3.6 的关键形状是：

```text
hidden_size = 2048
layers = 40
Gated DeltaNet layers = 30
Gated Attention layers = 10

Gated Attention:
  Q heads = 16
  KV heads = 2
  head dim = 256
  RoPE dim = 64

MoE:
  experts = 256
  routed experts per token = 8
  shared experts per token = 1
  expert intermediate dim = 512

LM output:
  vocab padded = 248320

Context:
  native = 262144 tokens
```

这些结构参数来自 Qwen3.6 模型卡。([Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B))

由这些形状能直接推出几个工程判断：

```text
1. logits 很大：
   [N, 248320]，logprobs 和 sampling 不能忽略。

2. Gated Attention KV cache 不能按 40 层 full attention 算：
   只对 10 个 Gated Attention 层套普通 KV cache 公式。

3. Gated DeltaNet 需要单独按 state / chunk / Mamba-like 路径理解。

4. MoE 不是一个 MLP：
   router_logits [N, 256]
   topk_indices [N, 8]
   expert dispatch / combine 是核心开销。

5. TP=8 不代表所有显存都除以 8：
   Qwen3.6 KV heads 只有 2，KV heads 少于 TP size 时存在复制逻辑。
```

vLLM v0.20.1 的 `Qwen3NextAttention` 源码入口明确读取 `config.num_attention_heads`、`config.num_key_value_heads`、TP size 等，并在 attention 模块中处理 TP head 切分；`Qwen3NextSparseMoeBlock` 是 Qwen3.6 MoE block 的源码入口。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/model_executor/models/qwen3_next.py))

---

## 四、计算流程

完整推理链路现在可以写成：

```text
1. Request enters vLLM server

2. Tokenizer / processor:
   text / image / video → token ids / multimodal embeddings

3. Scheduler:
   running requests first
   waiting requests second
   token budget
   prefix cache
   chunked prefill
   MTP lookahead
   encoder budget

4. KVCacheManager / BlockPool:
   allocate slots
   reuse prefix blocks
   assign KV blocks
   manage paged cache

5. Model forward:
   embedding
   → 40 hybrid layers:
       Gated DeltaNet or Gated Attention
       → MoE
   → final norm
   → lm_head

6. Backend execution:
   Attention backend
   GDN backend
   FusedMoE backend
   NCCL collectives

7. Logits / sampling:
   logits processor
   penalties / masks
   temperature / top-k / top-p
   sampled token

8. Decode loop:
   append token
   update cache/state
   repeat
```

vLLM recipe 给出的 Qwen3.6 serving 基线是 `vllm serve Qwen/Qwen3.6-35B-A3B --tensor-parallel-size 8 --max-model-len 262144 --reasoning-parser qwen3`；同一 recipe 还给出 MTP speculative decoding 配置 `{"method": "mtp", "num_speculative_tokens": 2}`。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

---

## 五、显存布局

最终显存账本应该长这样：

```text
Weights:
  embedding
  Gated DeltaNet weights
  Gated Attention weights
  MoE expert weights
  shared expert
  LM head
  Vision Encoder, if multimodal

Attention KV cache:
  only 10 Gated Attention layers

GDN state:
  30 Gated DeltaNet layers
  Mamba/GDN-style state
  chunk metadata

MoE runtime:
  router logits
  top-k indices / weights
  expert dispatch buffers
  expert GEMM workspace
  combine buffers

Scheduler / cache metadata:
  block table
  slot mapping
  KVCacheBlocks
  prefix cache hashes

MTP:
  speculative tokens
  lookahead cache/state budget

Runtime:
  CUDA graph memory
  backend workspace
  NCCL buffers
  sampler/logits buffers
```

最重要的显存公式仍然是：

$$
\text{GatedAttention KV bytes/token}
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

这只覆盖 10 个 Gated Attention 层的 BF16/FP16 K/V cache，不覆盖 Gated DeltaNet state、MoE runtime、MTP、多模态和 workspace。

---

## 六、vLLM 源码映射

第十五章的源码总图如下。

| 层级 | 文件路径 | 核心类/函数 | 你应该理解什么 |
|---|---|---|---|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | token budget、running/waiting、chunked prefill、spec tokens |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `KVCacheBlocks`, `allocate_slots` | slots、blocks、prefix cache hit |
| Block pool | `vllm/v1/core/block_pool.py` | `BlockPool` | free blocks、cached blocks、eviction |
| Qwen3.6 hybrid layer | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextDecoderLayer` | linear_attention vs full_attention |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | QKV、RoPE、Attention backend、TP heads |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` | Mamba/GDN state、chunked delta rule |
| MoE | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | router、shared expert、FusedMoE |
| Attention backend | `vllm/v1/attention/backends/` | FlashAttention / FlashInfer / Triton / GDN backend | KV layout、slot mapping、block table |
| Fused MoE | `vllm/model_executor/layers/fused_moe/` | `FusedMoE` | top-k routing、expert map、EP/EPLB |
| Distributed | `vllm/distributed/parallel_state.py` | `GroupCoordinator` | all-reduce、all-gather、reduce-scatter |

Scheduler 的关键源码注释是：vLLM v0.20.1 不把请求硬分为 prefill phase / decode phase，而是让 `num_computed_tokens` 追赶 `num_tokens_with_spec`，这个抽象覆盖 chunked prefill、prefix caching、speculative decoding 等路径。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/sched/scheduler.py))

`KVCacheBlocks` 的源码注释明确说它是 `KVCacheManager` 的 allocation result，是 Scheduler 和 KVCacheManager 之间的接口，用来隐藏 KVCacheManager 内部数据结构。([GitHub](https://github.com/vllm-project/vllm/blob/v0.20.1/vllm/v1/core/kv_cache_manager.py))

---

## 七、NVIDIA GPU 执行视角

现在你应该用 GPU 视角理解每个瓶颈。

```text
Prefill:
  token 多
  GEMM 大
  Tensor Core 更容易吃满
  但 activation / MoE buffer / attention workspace 峰值高

Decode:
  token 少
  每步读历史 KV / GDN state
  HBM bandwidth 更关键
  batch 太小时 SM 吃不满

Gated Attention:
  QKV GEMM
  RoPE
  KV cache update
  paged KV read
  attention backend
  O projection

Gated DeltaNet:
  state update
  chunk metadata
  causal conv / delta rule
  Mamba/GDN cache alignment

MoE:
  router GEMM
  top-k
  expert dispatch
  grouped expert GEMM
  combine
  optional all-to-all under EP

LM head / sampling:
  [N, 2048] → [N, 248320]
  logits processor
  top-k/top-p
  logprobs cost
```

第十一章之后你应该不再说“GPU 利用率低就调大 batch”这么粗糙，而是先问：低的是 Tensor Core、SM occupancy，还是 HBM 带宽？瓶颈在 attention、GDN、MoE、NCCL，还是 sampling？

---

## 八、优化前 vs 优化后

课程开始前的错误心智：

```text
Qwen3.6 = 一个大 Transformer
vLLM = 一个 batch server
FP8 = 省一半显存
TP=8 = 所有东西除以 8
MTP = 必然更快
MoE = 只跑少量专家所以一定省
```

课程结束后的正确心智：

```text
Qwen3.6 =
  Vision Encoder
  + 30 Gated DeltaNet layers
  + 10 Gated Attention layers
  + 40 MoE blocks
  + MTP capability
  + 262K context

vLLM =
  Scheduler
  + KVCacheManager
  + BlockPool
  + Attention/GDN/MoE backends
  + distributed collectives
  + metrics/profiling surface

FP8 =
  primarily weight pressure relief,
  not a full solution for long-context cache/state.

TP =
  tensor sharding with all-reduce/all-gather,
  not automatic 8× speedup.

EP =
  expert placement + token dispatch,
  not communication-free.

MTP =
  latency/throughput trade-off,
  not universally beneficial.
```

---

## 九、工程落地

最终推荐的学习与落地路线分四阶段。

### 阶段 1：小模型复现基本推理

目标：理解 token、prefill、decode、KV cache、sampling。

```text
模型：
  Qwen3-8B 或其他小模型

实验：
  short prompt / long output
  long prompt / short output
  temperature=0 vs temperature>0
  logprobs off/on
```

你要能解释：

```text
TTFT 为什么变
TPOT 为什么变
max_tokens 为什么影响 decode 显存
logprobs 为什么慢
```

### 阶段 2：用 Qwen3.6 跑通部署基线

目标：跑通官方推荐形态。

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

这条命令来自 vLLM Qwen3.5/Qwen3.6 recipe。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

然后建立 text-only 对照：

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only
```

注意：`--language-model-only` 在 recipe 中被用于 text throughput 场景，因为它跳过 vision encoder，把更多显存留给 KV cache；多模态场景则应单独压测 vision encoder 和 processor cache。([vLLM](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html))

### 阶段 3：系统压测与排障

目标：建立自己的压测矩阵。

```text
prompt_len:
  1K / 32K / 128K / 262K

output_len:
  128 / 512 / 4096

concurrency:
  1 / 8 / 32 / 128

features:
  prefix cache off/on
  MTP off/on
  EP off/on
  text-only vs multimodal
```

必须记录：

```text
TTFT
TPOT / ITL
tokens/s
running requests
waiting requests
kv_cache_usage_perc
prefix cache hit rate
GPU SM / HBM
NCCL timeline
MoE expert distribution
attention backend
GDN backend
sampler/logprobs time
```

### 阶段 4：源码贡献级阅读

源码阅读顺序：

```text
1. scheduler.py
2. kv_cache_manager.py
3. block_pool.py
4. qwen3_next.py
5. gdn_linear_attn.py
6. gdn_attn.py
7. flash_attn.py / flashinfer.py / triton_attn.py
8. fused_moe/
9. parallel_state.py
10. sampler.py / logits_processor.py
```

你的目标不是背代码，而是能回答：

```text
一个 request 在哪一行被调度？
prefix cache hit 在哪里发生？
slots 在哪里分配？
block_table 在哪里进入 backend？
GDN state metadata 在哪里构造？
MoE router logits 在哪里产生？
FusedMoE 在哪里接管 expert dispatch？
NCCL collective 从哪里触发？
```

---

## 十、本章总结

整门课的最终抽象是：

```text
LLM 推理不是一个 forward()。
它是：
  概率模型
  + 张量程序
  + 显存管理系统
  + GPU kernel pipeline
  + 分布式通信系统
  + 在线调度系统
  + 工程观测系统
```

对 Qwen3.6-35B-A3B，更准确地说：

```text
Qwen3.6 推理 =
  token / multimodal embedding
  + Gated DeltaNet state update
  + Gated Attention paged KV attention
  + MoE router / expert dispatch / shared expert
  + MTP speculative path
  + vLLM Scheduler / KVCacheManager / BlockPool
  + NVIDIA GPU GEMM / attention / MoE / NCCL kernels
  + metrics / profiler / deployment runbook
```

你已经从“会调用模型”走到了“能解释模型为什么慢、为什么 OOM、为什么吞吐不稳，以及该读哪段源码”的阶段。

---

# 小白总结

这门课最后要记住三件事。

第一，模型不是直接“读文字”，而是：

```text
文字 → token → embedding → 模型层 → logits → sampling
```

第二，Qwen3.6 不是普通 Transformer：

```text
它有 Vision Encoder
它有 Gated DeltaNet
它有 Gated Attention
它有 MoE
它有 MTP
它有 262K 长上下文
```

第三，vLLM 不是只负责“跑模型”：

```text
它负责调度请求
管理 KV cache 和 GDN state
动态 batching
prefix caching
chunked prefill
选择 backend
连接 GPU kernel
暴露监控指标
```

---

# 工程师总结

工程上最重要的 10 条原则：

```text
1. 不把 Qwen3.6 简化成普通 Transformer。

2. 不按 40 层 full attention 估算 KV cache。

3. 不把 FP8 当成长上下文显存万能解。

4. 不把 TP=8 理解成所有显存和延迟都除以 8。

5. 不默认 MTP 在高并发下一定提升吞吐。

6. 不默认 EP 一定优于 TP；必须看 all-to-all 和 expert imbalance。

7. 不默认源码目录存在某 backend 就说明运行时选中了它。

8. 不在高吞吐服务默认打开 logprobs=-1。

9. 不混合 text-only 和 multimodal workload 做平均压测。

10. 不凭感觉排障；用 TTFT、TPOT、KV usage、prefix hit、GPU profiler 定位。
```

---

# 关键公式

**1. 自回归分解**

$$
P(x_{1:T})=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

**2. LM head**

$$
z_t = W_{\text{lm}}h_t
$$

**3. Attention**

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d}} + M
\right)V
$$

**4. MoE**

$$
y =
\sum_{e \in \operatorname{TopK}(r(h), 8)}
w_eE_e(h)
+
E_{\text{shared}}(h)
$$

**5. Qwen3.6 layer layout**

$$
10 \times
\left[
3 \times
(\text{GatedDeltaNet}\rightarrow \text{MoE})
+
1 \times
(\text{GatedAttention}\rightarrow \text{MoE})
\right]
$$

**6. Gated Attention KV cache 粗估**

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

**7. Scheduler token gap**

$$
\Delta_i =
N^{\text{with\_spec}}_i
-
N^{\text{computed}}_i
$$

**8. 本轮 token budget**

$$
\sum_i \Delta_i^{\text{scheduled}}
\le
\text{max\_num\_scheduled\_tokens}
$$

**9. TTFT**

$$
\text{TTFT}
=
T_{\text{queue}}
+
T_{\text{prefill}}
+
T_{\text{first\_sample}}
$$

**10. MoE token-expert 对数量**

$$
N_{\text{token-expert}}
=
N_{\text{tokens}}
\times
8
$$

---

# 关键源码入口

| 主题 | 文件路径 | 入口 |
|---|---|---|
| Scheduler | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` |
| KV cache manager | `vllm/v1/core/kv_cache_manager.py` | `KVCacheManager`, `KVCacheBlocks`, `allocate_slots` |
| Block pool | `vllm/v1/core/block_pool.py` | `BlockPool` |
| Qwen3.6 model | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextForCausalLM`, `Qwen3NextDecoderLayer` |
| Gated Attention | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` |
| Gated DeltaNet | `vllm/model_executor/layers/mamba/gdn_linear_attn.py` | `GatedDeltaNetAttention` |
| MoE block | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` |
| Fused MoE | `vllm/model_executor/layers/fused_moe/layer.py` | `FusedMoE` |
| Attention backends | `vllm/v1/attention/backends/` | `flash_attn.py`, `flashinfer.py`, `triton_attn.py`, `gdn_attn.py` |
| Logits | `vllm/model_executor/layers/logits_processor.py` | `LogitsProcessor` |
| Sampling | `vllm/v1/sample/sampler.py` | `Sampler` |
| Distributed | `vllm/distributed/parallel_state.py` | `GroupCoordinator` |

---

# 源码证据表

| 结论 | 文件路径 | 类/函数 | 证据类型 | 是否已确认 |
|---|---|---|---|---|
| vLLM v0.20.1 是 2026-05-04 发布的 patch release | GitHub release page | v0.20.1 release notes | 官方文档确认 | 是 |
| Qwen3.6 是带 Vision Encoder 的 causal LM，35B total / 3B activated，hidden size 2048，40 层，262K context | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 hidden layout 是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))` | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 Gated Attention 是 16 Q heads、2 KV heads、head dim 256、RoPE dim 64 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| Qwen3.6 MoE 是 256 experts、8 routed + 1 shared、expert intermediate dim 512 | Hugging Face model card | Model Overview | 官方文档确认 | 是 |
| vLLM recipe 给出 Qwen3.6 TP=8、max model len 262144、reasoning parser qwen3 的部署命令 | vLLM Qwen recipe | Running Qwen3.6 | recipe 确认 | 是 |
| vLLM recipe 给出 Qwen3.6 MTP speculative decoding 配置 | vLLM Qwen recipe | MTP command | recipe 确认 | 是 |
| `Qwen3NextAttention` 是 Qwen3Next full-attention 源码入口 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextAttention` | 源码直接确认 | 是 |
| `Qwen3NextSparseMoeBlock` 是 Qwen3Next sparse MoE 源码入口 | `vllm/model_executor/models/qwen3_next.py` | `Qwen3NextSparseMoeBlock` | 源码直接确认 | 是 |
| Scheduler 不硬分 prefill/decode phase，而是用 `num_computed_tokens` 追赶 `num_tokens_with_spec` | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| Scheduler 先调度 running requests，再调度 waiting requests | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 源码直接确认 | 是 |
| `_mamba_block_aligned_split` 用于 Mamba/GDN state 的 block-aligned caching | `vllm/v1/core/sched/scheduler.py` | `_mamba_block_aligned_split` | 源码直接确认 | 是 |
| `KVCacheBlocks` 是 KVCacheManager 的 allocation result，是 Scheduler 和 KVCacheManager 之间的接口 | `vllm/v1/core/kv_cache_manager.py` | `KVCacheBlocks` | 源码直接确认 | 是 |
| Qwen3.6 在具体机器上的实际 attention backend | `vllm/v1/attention/backends/` + runtime logs | backend selection | 待源码确认 | 否 |
| Qwen3.6 在具体机器上的实际 MoE runner / all-to-all backend | `vllm/model_executor/layers/fused_moe/` + runtime logs | backend selection | 待源码确认 | 否 |
| 单卡 FP8 是否能满足 262K 高并发 SLA | runtime benchmark | N/A | 合理推断 | 否 |

---

# 源码阅读作业

最终源码阅读作业分 5 周。

**第 1 周：Scheduler / cache**

```text
vllm/v1/core/sched/scheduler.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/block_pool.py
```

目标：

```text
看懂 request 如何被调度
看懂 slots 如何分配
看懂 prefix blocks 如何复用
看懂 preemption 为什么发生
```

**第 2 周：Qwen3.6 model path**

```text
vllm/model_executor/models/qwen3_next.py
vllm/model_executor/layers/mamba/gdn_linear_attn.py
```

目标：

```text
看懂 Qwen3NextDecoderLayer
看懂 Qwen3NextAttention
看懂 GatedDeltaNetAttention
看懂 Qwen3NextSparseMoeBlock
```

**第 3 周：Attention / GDN backend**

```text
vllm/v1/attention/backends/flash_attn.py
vllm/v1/attention/backends/flashinfer.py
vllm/v1/attention/backends/triton_attn.py
vllm/v1/attention/backends/gdn_attn.py
```

目标：

```text
看懂 block_table
看懂 slot_mapping
看懂 KV cache shape
看懂 GDN metadata
看懂 chunk_indices / state_indices
```

**第 4 周：MoE / EP**

```text
vllm/model_executor/layers/fused_moe/
vllm/distributed/parallel_state.py
```

目标：

```text
看懂 router
看懂 top-k
看懂 expert map
看懂 FusedMoE
看懂 EP / EPLB
看懂 all-reduce / all-gather / reduce-scatter
```

**第 5 周：Profiler + patch**

目标：

```text
用 Nsight Systems 找一个瓶颈
用 Nsight Compute 深挖一个 kernel
读对应源码
写一个最小 patch 或最小 benchmark
提交 issue / PR / 内部分析报告
```

---

# 实验验证作业

最终实验路线：

**实验 1：小模型复现推理生命周期**

```text
短 prompt + 长输出
长 prompt + 短输出
temperature=0 vs 0.7
logprobs off/on
```

输出报告：

```text
TTFT / TPOT / tokens/s
prefill vs decode 区别
logprobs 成本
```

**实验 2：Qwen3.6 262K 基线**

```bash
vllm serve Qwen/Qwen3.6-35B-A3B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --reasoning-parser qwen3
```

输出报告：

```text
启动显存
KV/GDN state capacity
backend selection
单请求 262K TTFT
低并发 TPOT
```

**实验 3：text-only vs multimodal**

```text
同样 prompt 长度：
  text-only
  image + text
  video + text
```

输出报告：

```text
Vision Encoder cost
processor cache cost
prefill time
显存差异
```

**实验 4：prefix caching**

```text
shared prefix = 16K / 64K
different suffix = 128
```

输出报告：

```text
prefix_cache_queries
prefix_cache_hits
TTFT improvement
cache usage
```

**实验 5：MTP off/on**

```text
MTP off
MTP on, num_speculative_tokens=2
```

输出报告：

```text
accepted speculative tokens
TPOT
tokens/s
KV/GDN state usage
high concurrency throughput
```

**实验 6：MoE / EP**

```text
EP off
EP on
EPLB on
```

输出报告：

```text
expert hit count
per-rank utilization
all-to-all time
FusedMoE kernel time
tokens/s
```

---

# 3 个自测题

**1. 为什么 Qwen3.6 不能按普通 40 层 Transformer Decoder 估算 KV cache？**  
因为 Qwen3.6 的 40 层是 `10 × (3 × (Gated DeltaNet → MoE) → 1 × (Gated Attention → MoE))`，只有 10 个 Gated Attention 层适合普通 K/V cache 公式；另外 30 个 Gated DeltaNet 层走 state / chunk / Mamba-like 路径。

**2. vLLM v0.20.1 Scheduler 的核心抽象是什么？**  
不是硬分 prefill / decode phase，而是让每个 request 的 `num_computed_tokens` 追赶 `num_tokens_with_spec`；这统一覆盖 chunked prefill、prefix caching、speculative decoding 等路径。

**3. 部署 Qwen3.6 时，为什么 FP8、TP、EP、MTP 都不能单独作为万能优化？**  
FP8 主要缓解权重显存，不解决全部 KV/GDN state；TP 会引入 all-reduce/all-gather，且 KV heads 少时可能复制；EP 会引入 expert dispatch/all-to-all；MTP 会增加 lookahead cache/state，收益取决于接受率和并发。

---

# 下一步路线图

课程正文到这里结束。后续学习路线建议按这个顺序推进：

```text
1. 做完第十五章 6 个实验。
2. 读完第十五章源码阅读作业。
3. 选一个真实瓶颈：
   TTFT、TPOT、MoE、GDN、NCCL、sampling 任一即可。
4. 用 profiler 证明瓶颈。
5. 写一份包含源码证据、指标、复现实验、优化建议的技术报告。
6. 再尝试做 vLLM patch 或贡献 issue / PR。
```

真正达到“LLM 推理系统架构师 + CUDA/NVIDIA GPU 性能优化 + vLLM 源码贡献者级别严谨度”的标志，不是能背概念，而是能完成这条闭环：

```text
现象
→ 指标
→ 源码
→ kernel
→ 实验
→ 修复或优化
→ 复测
```

---

**Sources:**

- [Releases · vllm-project/vllm · GitHub](https://github.com/vllm-project/vllm/releases)
- [Qwen/Qwen3.6-35B-A3B · Hugging Face](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- [Qwen3.5 & Qwen3.6 Usage Guide - vLLM Recipes](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)