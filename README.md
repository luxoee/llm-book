# LLM 推理与架构：从 Qwen3.6 到 vLLM Serving 实战

这是一本文向工程落地的 LLM 推理系统技术书，固定以 **Qwen/Qwen3.6-35B-A3B**、**vLLM v0.20.1** 和 **NVIDIA GPU** 为案例，系统讲清楚从 token、attention、KV cache、PagedAttention、continuous batching、MoE、并行通信、GPU kernel 到部署压测的完整链路。

它更像一本“推理系统案例书”，而不是横向百科：所有概念都会落回一个固定问题——如何把 Qwen3.6 这样带 Gated DeltaNet、Gated Attention、MoE、MTP、262K context 和 FP8 checkpoint 的模型，放进 vLLM serving 的真实工程链路里分析。

本书不是泛泛复述 Transformer。Qwen3.6-35B-A3B 是带 Vision Encoder 的 causal LM，语言主干包含 Gated DeltaNet、Gated Attention、MoE 和 MTP；因此本书把传统 decoder-only LLM 当作基线，再逐步替换成 Qwen3.6 的真实 hybrid serving 心智模型。

## 这本书解决什么问题

很多 LLM 推理材料停留在两端：一端讲抽象数学公式，另一端给部署命令。本书关注中间最容易断层的部分：

- 为什么长上下文会迅速吃光显存。
- 为什么 KV cache、PagedAttention 和 block manager 是 serving 的核心。
- 为什么 MoE 不是“只激活 3B 参数”这么简单。
- 为什么 Tensor Parallel、Expert Parallel、NCCL 和 GPU kernel 会决定真实吞吐。
- 如何把模型卡、vLLM 源码、部署命令、监控指标和 profiler 结果串成一条工程链路。

## 你将获得什么

读完本书，你应该能形成一套可复用的 LLM serving 判断框架：

- 从模型卡判断结构、上下文长度、专家数量、MTP、FP8 对部署的影响。
- 粗估 KV cache、Gated DeltaNet state、MoE runtime buffer 和多卡通信对显存/吞吐的压力。
- 解释 PagedAttention、continuous batching、chunked prefill、prefix caching 为什么影响 TTFT、TPOT 和并发。
- 区分单卡 FP8、BF16 多卡、text-only、多模态、Expert Parallel 等部署形态的适用边界。
- 设计压测矩阵、监控面板和排障 runbook，而不是只看一个 tokens/s 数字。

## 适合读者

- 正在部署或调优 LLM 服务的工程师。
- 想从应用层进入推理系统、vLLM、GPU serving 的后端/平台工程师。
- 需要理解 Qwen3.6、MoE、长上下文、speculative decoding 和多卡部署的技术负责人。
- 已经懂基础 Transformer，但想建立生产级 serving 心智模型的读者。

## 不适合读者

- 只想学习 prompt engineering 或应用层 API 调用的读者。
- 完全不想接触张量形状、显存、源码路径和 GPU 概念的读者。
- 需要覆盖所有模型家族和所有 serving 框架的横向综述读者。

## 这本书不承诺什么

- 不承诺某一张 GPU、某一种拓扑或某个云厂商实例上的固定性能数字。
- 不把 FP8、MTP、Expert Parallel 或 prefix caching 描述成万能开关。
- 不替代你自己的线上压测、容量评估、监控和 profiler 分析。
- 不覆盖所有模型家族和所有 serving 框架，而是用一个深度案例建立可迁移的判断方法。

## 阅读路径

- **架构学习路径**：第 1、2、3、4、5 章，先建立从传统 decoder-only baseline 到 Qwen3.6 hybrid 架构的心智模型。
- **显存与调度路径**：第 6、7、8、12 章，重点理解 KV cache、PagedAttention、scheduler、block manager 和 attention backend。
- **性能与多卡路径**：第 9、10、11 章，聚焦 MoE、Tensor Parallel、Expert Parallel、NCCL 和 GPU kernel。
- **上线部署路径**：第 13、14 章，直接进入部署决策、压测矩阵、监控指标和排障 runbook。
- **复盘行动路径**：第 15 章，把源码阅读、实验验证和后续学习路线整理成执行计划。

## 版本边界

本书固定边界如下：

- 模型：`Qwen/Qwen3.6-35B-A3B`
- Serving 框架：`vLLM v0.20.1`
- 硬件视角：NVIDIA GPU
- 重点主题：KV cache、PagedAttention、continuous batching、chunked prefill、prefix caching、MoE、TP/EP、NCCL、GPU kernel、部署、压测、监控、排障

当书中涉及 vLLM 源码路径、部署命令或模型结构时，会尽量绑定官方文档、模型卡或 v0.20.1 tag 源码证据。

## 目录

请从 [SUMMARY.md](SUMMARY.md) 进入 GitBook 目录。

## 质量说明

本书把结论分为“已确认事实”“合理推断”和“待运行验证”。涉及具体硬件、backend 选择和性能数据时，请以你的运行日志、监控指标和 profiler 结果为准。
