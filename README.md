# LLM 推理与架构：从 Qwen3.6 到 vLLM Serving 实战

这是一本文向工程落地的 LLM 推理系统技术书，固定以 **Qwen/Qwen3.6-35B-A3B**、**vLLM v0.20.1** 和 **NVIDIA GPU** 为案例，系统讲清楚从 token、attention、KV cache、PagedAttention、continuous batching、MoE、并行通信、GPU kernel 到部署压测的完整链路。

本书不是泛泛复述 Transformer。Qwen3.6-35B-A3B 是带 Vision Encoder 的 causal LM，语言主干包含 Gated DeltaNet、Gated Attention、MoE 和 MTP；因此本书把传统 decoder-only LLM 当作基线，再逐步替换成 Qwen3.6 的真实 hybrid serving 心智模型。

## 这本书解决什么问题

很多 LLM 推理材料停留在两端：一端讲抽象数学公式，另一端给部署命令。本书关注中间最容易断层的部分：

- 为什么长上下文会迅速吃光显存。
- 为什么 KV cache、PagedAttention 和 block manager 是 serving 的核心。
- 为什么 MoE 不是“只激活 3B 参数”这么简单。
- 为什么 Tensor Parallel、Expert Parallel、NCCL 和 GPU kernel 会决定真实吞吐。
- 如何把模型卡、vLLM 源码、部署命令、监控指标和 profiler 结果串成一条工程链路。

## 适合读者

- 正在部署或调优 LLM 服务的工程师。
- 想从应用层进入推理系统、vLLM、GPU serving 的后端/平台工程师。
- 需要理解 Qwen3.6、MoE、长上下文、speculative decoding 和多卡部署的技术负责人。
- 已经懂基础 Transformer，但想建立生产级 serving 心智模型的读者。

## 不适合读者

- 只想学习 prompt engineering 或应用层 API 调用的读者。
- 完全不想接触张量形状、显存、源码路径和 GPU 概念的读者。
- 需要覆盖所有模型家族和所有 serving 框架的横向综述读者。

## 阅读路径

- **快速建立全局图**：第 1、2、3、15 章。
- **理解推理计算链路**：第 4、5、6、7、8 章。
- **理解 Qwen3.6 的差异化架构**：第 3、5、9、10、11 章。
- **准备上线部署和排障**：第 12、13、14 章。
- **做技术审稿或商业发布复核**：阅读 `publishing/release-checklist.md` 和 `publishing/review-prompts.md`。

## 版本边界

本书固定边界如下：

- 模型：`Qwen/Qwen3.6-35B-A3B`
- Serving 框架：`vLLM v0.20.1`
- 硬件视角：NVIDIA GPU
- 重点主题：KV cache、PagedAttention、continuous batching、chunked prefill、prefix caching、MoE、TP/EP、NCCL、GPU kernel、部署、压测、监控、排障

当书中涉及 vLLM 源码路径、部署命令或模型结构时，会尽量绑定官方文档、模型卡或 v0.20.1 tag 源码证据。

## 目录

请从 [SUMMARY.md](SUMMARY.md) 进入 GitBook 目录。

## 原始材料

原始 ChatGPT 导出文件保留在 `raw/ChatGPT-LLM 推理与架构.md`，发布稿以 `chapters/` 下章节为准。
