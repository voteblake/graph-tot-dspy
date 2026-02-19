# Implement thought-level ToT branching to match paper's Algorithm 1

## Summary

The current implementation branches at the **complete ReAct trace level** rather than at the **individual thought level** as described in the paper (arXiv:2502.13247). This is the most significant algorithmic deviation from the paper and fundamentally changes the search dynamics.

## Current behavior

`GraphToTSolver` generates `k` fully independent ReAct agent traces (each running the complete thought-action-observation loop to termination), then scores and prunes the finished traces. Each branch is an atomic, complete trajectory.

The relevant code path:
- `dspy_modules.py:367-380` — `_generate_branches()` runs `self.agent(question=..., context=...)` k times, each producing a full trace
- `dspy_modules.py:304-306` — Round 0 seeds k independent branches
- `dspy_modules.py:308-337` — Subsequent rounds score complete traces, prune to b survivors, then generate k *new* complete traces from each survivor

The CLAUDE.md already acknowledges this: "ToT = complete-trace branching: `dspy.ReAct` manages a single trace internally; ToT branches happen at the level of full ReAct runs."

## Paper's algorithm

The paper describes branching at each reasoning depth `i`. At each step, the LLM generates `k` candidate next-thoughts from a given partial state, and states are pruned to top-b before expanding further:

```
z_{i+1}^{(j)} ~ p_theta(z_{i+1} | s_i),  j = 1,...,k
```

Where `s_i = [question, z_1, z_2, ..., z_i, graph_info]` is a **partial** reasoning chain. Branching is fine-grained: each individual thought step gets k alternatives. The system can abandon a bad line of reasoning after one or two thoughts rather than committing to an entire trajectory.

## Why this matters

- **Search efficiency**: The paper's approach explores up to k^d partial paths across d depths but prunes aggressively at each level. The current implementation explores only k complete paths per round. Recovery from a bad early thought is impossible.
- **Diversity mechanism**: The paper gets diversity from different thought choices at branching points. The implementation relies entirely on temperature sampling for diversity, which is acknowledged as fragile (the "all k branches identical" issue).
- **Multi-round context loss**: In multi-round mode (`dspy_modules.py:331-336`), each survivor generates k entirely new traces from scratch, using only the survivor's answer (truncated to 800 chars) as a context hint. The graph traversal work from the prior round is lost — new agents must re-explore from zero. The paper's approach preserves the full partial state.

## Root cause

`dspy.ReAct` manages its internal thought-action-observation loop as a black box. There is no API to pause mid-trace, branch, score a partial state, and resume. The current design is a pragmatic workaround for this framework limitation.

## Implementation hints

A faithful implementation would need to either:
1. Rewrite the ReAct loop from scratch (not using `dspy.ReAct`) to expose mid-trace branching points — after each thought, generate k alternatives, score them, prune, then continue
2. Use `dspy.ReAct` but instrument it to yield intermediate states, or use the trajectory dict to reconstruct partial states for branching
3. Implement a custom `dspy.Module` that interleaves single-step agent calls with ToT branching/pruning logic

The key challenge is that at each branch point you need to: generate k candidate next-thoughts, optionally call a tool for each, score the partial states, prune to b, and continue. This requires access to the LLM call *within* the ReAct loop, not just the final output.

Consider whether DSPy's `Adapter` or `Module` hooks can intercept mid-trace states.
