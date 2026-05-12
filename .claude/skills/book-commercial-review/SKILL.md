# Book Commercial Review

Use this skill when reviewing this repository's Markdown book for commercial publication, GitBook readiness, or technical book quality.

## Scope

This skill is for the book about LLM inference and serving architecture, focused on Qwen/Qwen3.6-35B-A3B, vLLM v0.20.1, and NVIDIA GPU serving.

## Workflow

1. Read `README.md` and `SUMMARY.md` to understand the market positioning and GitBook structure.
2. Read `publishing/release-checklist.md` for the release criteria.
3. Inspect `chapters/` for chapter distribution, heading hierarchy, source evidence, and reader flow.
4. Use `publishing/review-prompts.md` when a deeper review perspective is needed.
5. Treat `raw/ChatGPT-LLM 推理与架构.md` as source material only; do not edit it unless explicitly asked.

## Review dimensions

- Commercial positioning: target reader, promise, differentiation, market pain.
- Technical trust: source-backed claims, vLLM v0.20.1 boundaries, Qwen3.6 architecture accuracy.
- Reading experience: chapter progression, transitions, examples, exercises, self-check questions.
- GitBook readiness: `README.md`, `SUMMARY.md`, link validity, Markdown rendering, absence of chat artifacts.
- Engineering usefulness: deployment commands, metrics, troubleshooting paths, profiler-oriented thinking.

## Output format

When asked to review, produce:

1. Overall verdict: ready / almost ready / not ready.
2. Top 5 blockers.
3. Top 5 improvements.
4. Commercialization score from 1 to 10.
5. Technical credibility score from 1 to 10.
6. GitBook readiness score from 1 to 10.
7. Concrete file-level action list.
