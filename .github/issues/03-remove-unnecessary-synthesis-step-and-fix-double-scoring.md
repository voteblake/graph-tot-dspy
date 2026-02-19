# Remove unnecessary synthesis LLM call and fix double-scoring on single-round runs

## Summary

The solver's `forward()` method has two wasteful patterns: (1) a `FinalAnswerSignature` synthesis step that adds an LLM call not present in the paper, and (2) a double-scoring bug where branches are evaluated twice on single-round runs.

## Issue 1: Unnecessary FinalAnswerSignature

### Current behavior

After selecting the best branch, `GraphToTSolver.forward()` makes an additional LLM call through `self.synthesizer` (`dspy_modules.py:351-354`):

```python
synth = self.synthesizer(
    question=question,
    best_trace=f"Reasoning trace:\n{best.trace}\n\nCandidate answer: {best.answer}",
)
```

This uses `FinalAnswerSignature` (`dspy_modules.py:79-90`) to "synthesize a final, concise answer" from the best trace.

### Paper's approach

The paper takes the best branch's answer directly. There is no synthesis step — the answer from the highest-scoring trace is the output.

### Why this matters

The extra LLM call:
- Adds latency and cost (one additional API call per question)
- Can distort answers: the synthesizer might paraphrase, hallucinate details, or lose specificity from the original trace
- Is especially problematic for GRBench where gold answers are terse entity lists — synthesis tends to produce prose that scores worse on ROUGE-L

## Issue 2: Double scoring on single-round runs

### Current behavior

When `max_rounds=1` (the default), the flow through `forward()` is:

1. Generate k branches (line 306)
2. Enter loop with `round_idx=1` (line 308)
3. Score and prune branches — **first scoring** (lines 310-315)
4. `round_idx == max_rounds` is True, so `break` (lines 324-326)
5. "Final scoring pass" — **second scoring** (lines 340-344)

Branches are scored **twice**: once in the loop body and once in the "final scoring pass" after the loop. With `temperature > 0` on the LLM (which is required for branch diversity), the two scoring passes can produce different scores, meaning the final ranking may not match the intermediate ranking.

### Impact

- Doubles the LLM calls for evaluation (k extra calls for score_vote, 1 extra for selection_vote)
- Can produce inconsistent rankings between the two passes

## Suggested approach

1. Remove `FinalAnswerSignature` and `self.synthesizer` entirely. Return `best.answer` directly as the final answer.
2. Restructure the scoring flow so that single-round runs only score once. One approach: always generate, then score, then the loop only handles multi-round expansion. Or track whether scoring has already happened.

## Files to modify

- `src/graph_tot/dspy_modules.py` — `GraphToTSolver.forward()` (lines 294-361), `FinalAnswerSignature` (lines 79-90), `__init__` synthesizer declaration (line 292)
