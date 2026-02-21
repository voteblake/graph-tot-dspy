# Independent Review Notes: Graph ToT + Agent Implementation

Date: 2026-02-21

## Scope
- Reviewed the implementation under `src/graph_tot/` as a candidate reference implementation of Graph Tree-of-Thought (ToT) + agent-based graph retrieval.
- Compared behavior against the paper-derived algorithm description embedded in repository docs/comments.

## Network verification status (re-tried)
- Re-attempted direct paper access at `https://arxiv.org/html/2502.13247v3` in this runtime.
- Access is still blocked with `Tunnel connection failed: 403 Forbidden` from both `curl` and Python `urllib`.
- Also attempted browser-tool fallback; Chromium failed to launch in the tool container (SIGSEGV), so no browser-based retrieval was possible.

## High-level assessment
- Algorithmic fidelity: **medium-high** for the core loop (k-branch generation, voting, beam prune, optional rounds, no final synthesis).
- DSPy quality: **good baseline, not exemplary**. The decomposition into modules/signatures is clear, but several DSPy-idiomatic improvements are still missing (e.g., stronger structured evaluator inputs/outputs, better compile-time task design for branch scoring stability).
- Python/library quality: **good overall readability**, but there are several publishability gaps (API boundaries, packaging metadata, reproducibility defaults, stricter typing/contracts).

## Key implementation strengths
- Clear modular split: environment/tools vs. agent vs. evaluator vs. orchestrator.
- Explicit support for two voting modes.
- Retry handling for rate limits.
- Concurrency path has tests and deterministic branch ordering semantics.

## Main risks / deltas to address before "reference-quality" claims
1. Branch history retention bug: solver only returns the most recent scored round in `all_branches`, not all rounds.
2. Selection-vote candidate payload truncates reasoning traces but not with explicit token accounting.
3. FAISS cache key does not encode graph identity, so changing graphs with same embedding model can reuse stale cached embeddings.
4. Public API lacks formal `py.typed`/typing guarantees and stricter export/versioning policy.
5. Error signaling in tools is string-based, which is pragmatic for ReAct but weak for library consumers.
