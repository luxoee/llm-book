# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a Markdown book/course about LLM inference and serving architecture, written primarily in Chinese. The material focuses on Qwen/Qwen3.6-35B-A3B, vLLM v0.20.1, KV cache, PagedAttention, batching, MoE, GPU kernels, deployment, benchmarking, monitoring, and troubleshooting.

## Common commands

There is no package manifest, Makefile, test runner, build step, or configured linter in this repository. Treat validation as Markdown/content review unless tooling is added later.

Useful inspection commands:

```bash
# List all book chapters and source material
find book raw -maxdepth 1 -type f -name '*.md' | sort

# Check the top-level chapter headings
grep -R '^# ' book raw --include='*.md'

# Check chapter sizes
wc -l book/*.md raw/*.md
```

If a future change adds tooling, update this section with the actual build, lint, test, and single-test commands rather than inventing generic ones.

## Content structure

- `raw/ChatGPT-LLM 推理与架构.md` is the original long export/source document.
- `book/README.md` is the table of contents for the split book.
- `book/01-...md` through `book/15-...md` are the curated chapter files.

The split chapters are the primary edited output. Use the raw file as source/reference material when checking continuity or recovering context, but avoid editing `raw/` unless the user explicitly asks to update the source export too.

## Big-picture architecture of the book

The course is organized as a 15-chapter progression:

1. Establish verified facts, model differences, and the course outline.
2. Introduce traditional decoder-only LLM inference as the baseline mental model.
3. Explain Qwen3.6 architecture: Gated DeltaNet, Gated Attention, MoE, and MTP.
4. Cover tokenization, embeddings, logits, and sampling.
5. Cover Attention, Gated Attention, and RoPE math.
6. Explain KV cache and long-context memory pressure.
7. Explain PagedAttention.
8. Explain continuous batching, chunked prefill, and prefix caching.
9. Explain MoE routing, experts, shared experts, top-k, and expert dispatch.
10. Explain tensor parallelism, expert parallelism, and NCCL communication.
11. Explain NVIDIA GPU kernel concepts relevant to serving.
12. Walk through vLLM scheduler, block manager, and attention backend source paths.
13. Cover deployment cases: FP8 single GPU, BF16 multi-GPU, expert parallel, and degraded configs.
14. Cover benchmarking, monitoring, and troubleshooting.
15. Recap and provide a learning roadmap.

Most chapters follow a repeated teaching pattern: analogy, mathematical intuition, tensor shapes, compute flow, memory layout, source-code mapping/evidence, operational pitfalls, summary, formulas/source entries, exercises, self-check questions, and next-chapter preview. Preserve that structure when expanding or revising chapters unless the user asks for a different style.

## Content conventions

- Write in Chinese by default, matching the existing chapter style.
- Preserve the fixed case study boundaries unless updating facts deliberately: Qwen/Qwen3.6-35B-A3B and vLLM v0.20.1.
- Keep factual claims tied to verifiable sources when the surrounding chapter uses citations.
- Distinguish confirmed facts, reasonable inferences, and items needing source confirmation; chapter 1 already uses this pattern.
- When discussing vLLM internals, prefer the v0.20.1 paths used in the book, such as `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/block_pool.py`, and `vllm/v1/attention/backends/`.
- Do not silently replace the Qwen3.6 hybrid architecture with a generic Transformer explanation. Traditional decoder-only LLMs are used as a baseline, not as the full explanation of this model.

## Editing guidance

- Make surgical Markdown edits; do not reformat unrelated chapters.
- If updating the table of contents, keep `book/README.md` synchronized with the chapter filenames and headings.
- If splitting or moving material, preserve links and chapter numbering.
- Prefer editing the split chapter files in `book/`; mention if the raw export also appears stale rather than automatically rewriting it.
