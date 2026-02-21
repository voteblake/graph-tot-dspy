# Issue: Make `selection_vote` candidate payload token-safe and robust

## Problem statement
`TreeOfThoughtEvaluator._selection_vote()` builds one large `candidates` string and truncates each trace by character count (`b.trace[:2000]`) without explicit token budgeting or graceful degradation. This can still exceed model limits and destabilize selection quality.

## Why this matters
- Long traces can push selection prompts near/over context limits.
- Char-based truncation is a weak proxy for token usage.
- Failure modes are silent quality degradation rather than explicit control.

## Current behavior (code pointers)
- `src/graph_tot/dspy_modules.py`, `_selection_vote()`:
  - concatenates all branches into one string
  - per-branch trace truncation by chars (`[:2000]`)
  - no dynamic sizing based on number of candidates or model token budget

## Expected behavior
Selection vote should deterministically fit within a configured budget and preserve the most informative trace content.

## Proposed implementation
1. Introduce configurable budget knobs on evaluator/solver:
   - `max_selection_prompt_tokens` (or equivalent)
   - `max_trace_tokens_per_candidate` fallback cap.
2. Switch from char truncation to token-aware truncation (via available tokenizer utility / model-compatible estimate).
3. Use progressive summarization strategy when over budget:
   - first trim observations,
   - then trim intermediate steps,
   - always preserve final answer + last reasoning step.
4. Add warning logs when truncation occurs and include final prompt size estimate.
5. Add optional deterministic compact candidate format (JSON-like schema) to reduce verbosity.

## Acceptance criteria
- [ ] Selection prompt assembly never exceeds configured budget in tests.
- [ ] Candidate serialization is deterministic for stable reproducibility.
- [ ] Logs indicate when and how much truncation happened.
- [ ] No regression in current API defaults if no budget args are supplied.

## Suggested tests
- Unit tests for payload builder with synthetic long traces.
- Verify token budget enforcement across `k in {2,3,5}`.
- Snapshot test for serialized candidate format stability.

## Out of scope
- Replacing selection-vote with a separate reranker model.
