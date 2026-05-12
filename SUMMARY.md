# Summary

## 开始阅读

- [首页](README.md)

## 第一部分：导读与推理基线

- [第一章：导读、事实边界与全书路线图](chapters/01-导读-事实边界-全书路线图.md)
- [第二章：传统 LLM 推理总览](chapters/02-传统-LLM-推理总览.md)

## 第二部分：Qwen3.6 模型计算链路

- [第三章：Qwen3.6 架构总览：Gated DeltaNet、Gated Attention、MoE、MTP](chapters/03-Qwen3.6-架构总览.md)
- [第四章：Token、Embedding、Logits、Sampling](chapters/04-Token-Embedding-Logits-Sampling.md)
- [第五章：Attention / Gated Attention / RoPE 数学](chapters/05-Attention-Gated-Attention-RoPE-数学.md)

## 第三部分：显存、缓存与调度

- [第六章：KV Cache 与长上下文显存问题](chapters/06-KV-Cache-与长上下文显存问题.md)
- [第七章：PagedAttention](chapters/07-PagedAttention.md)
- [第八章：Continuous Batching、Chunked Prefill、Prefix Caching](chapters/08-Continuous-Batching-Chunked-Prefill-Prefix-Caching.md)

## 第四部分：MoE、并行与 GPU 执行

- [第九章：MoE：Router、Expert、Shared Expert、Top-k、Expert Dispatch](chapters/09-MoE-Router-Expert-Shared-Expert-Top-k-Expert-Dispatch.md)
- [第十章：Tensor Parallel、Expert Parallel、NCCL 通信](chapters/10-Tensor-Parallel-Expert-Parallel-NCCL-通信.md)
- [第十一章：NVIDIA GPU Kernel：HBM、SM、Tensor Core、CUDA Kernel、MoE GEMM、Attention Backend](chapters/11-NVIDIA-GPU-Kernel.md)

## 第五部分：源码、部署、压测与路线图

- [第十二章：vLLM Scheduler、Block Manager、Attention Backend 源码导读](chapters/12-vLLM-Scheduler-Block-Manager-Attention-Backend-源码导读.md)
- [第十三章：工程部署：FP8 单卡、BF16 多卡、Expert Parallel、降级配置](chapters/13-工程部署-FP8-单卡-BF16-多卡-Expert-Parallel-降级配置.md)
- [第十四章：压测、监控、排障](chapters/14-压测-监控-排障.md)
- [第十五章：总复盘与学习路线图](chapters/15-总复盘与学习路线图.md)
