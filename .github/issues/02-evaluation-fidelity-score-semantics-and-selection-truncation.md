# Fix evaluation mode fidelity: score vote semantics and selection vote truncation

## Summary

Both evaluation modes (`score_vote` and `selection_vote`) deviate from the paper in ways that affect scoring quality. The score vote uses a different scoring formulation than the paper, and the selection vote truncates reasoning traces to 300 characters, losing most of the evidence the evaluator needs.

## Issue 1: Score vote semantics

### Current behavior

`ScoreVoteSignature` (`dspy_modules.py:47-62`) asks the LLM to assess quality on three criteria — "logical soundness of graph traversal steps, completeness and specificity of the answer, direct relevance to the question" — and return a float from 0 to 1.

### Paper's formulation

The paper defines the value function as:

```
V(p_theta, S)(s) = P[s = s*]
```

This is explicitly a **probability estimate** — "what is the probability that this state leads to the correct solution?" — not a rubric-based quality judgment. The distinction matters: a rubric score could rate two traces as 0.8 and 0.7 where the probability formulation would rate them 0.9 and 0.2 (one is almost certainly right, the other is almost certainly wrong).

### Suggested fix

Revise `ScoreVoteSignature`'s docstring and field descriptions to ask for a probability estimate of correctness rather than a multi-criteria quality rubric. Something like: "Estimate the probability that this reasoning trace arrives at the correct answer."

## Issue 2: Selection vote trace truncation

### Current behavior

In `_selection_vote` (`dspy_modules.py:224-227`):

```python
candidates_text = "\n\n".join(
    f"[{i}] Answer: {b.get('answer', '')}\n"
    f"    Reasoning: {b.get('trace', '')[:300]}..."
    for i, b in enumerate(branches)
)
```

Traces are **truncated to 300 characters**. A typical agent trace with ~5 graph traversal steps (retrieve, check feature, check wrong edge, check correct edge, finish) produces traces of 1000-2000+ characters. At 300 chars, the evaluator sees only the first 1-2 steps.

### Paper's description

The paper's Selection method presents full candidate traces: "given a set of candidate solutions, the LLM is asked to choose the best one." No truncation is described.

### Suggested fix

Remove the hard 300-char truncation. If context window limits are a concern, truncate to a much larger limit (e.g., 2000 chars) or summarize rather than hard-cut. Consider whether the truncation limit should be configurable via a parameter.

## Files to modify

- `src/graph_tot/dspy_modules.py` — `ScoreVoteSignature` (lines 47-62) and `_selection_vote` (lines 223-244)
