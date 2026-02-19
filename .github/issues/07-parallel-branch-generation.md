# Add parallel branch generation for k independent agent traces

## Summary

`GraphToTSolver._generate_branches()` runs k independent agent traces **sequentially**. Since branches are independent (they don't share state), they can run concurrently. With k=3 and max_iters=12, sequential execution means up to 36 serial LLM calls just for initial branch generation. Parallelizing this would reduce wall-clock time by roughly k×.

## Current behavior

In `dspy_modules.py:367-380`:

```python
def _generate_branches(self, question: str, contexts: list[str]) -> list[Branch]:
    branches: list[Branch] = []
    for context in contexts:
        pred = self.agent(question=question, context=context)
        trace = self.agent.get_trajectory_text(pred)
        answer = getattr(pred, "answer", "") or "No answer produced."
        branches.append(Branch(...))
    return branches
```

This is a simple sequential loop. Each agent call involves multiple LLM round-trips (one per ReAct iteration), so the total latency for k=3 branches is 3x the single-branch latency.

The same pattern appears in multi-round expansion (`dspy_modules.py:331-336`) where each survivor generates k new branches sequentially.

## Why sequential?

DSPy's `dspy.ReAct` and `dspy.Predict` are synchronous by default. DSPy does not natively support parallel forward passes.

## Suggested approaches

1. **`concurrent.futures.ThreadPoolExecutor`**: Wrap each `self.agent(...)` call in a thread. LLM API calls are I/O-bound (network latency), so threading is appropriate and safe as long as the agent instances don't share mutable state. Currently there's a single `self.agent` instance — need to verify it's thread-safe or create k instances.

2. **`asyncio` with DSPy's async support**: Check if `dspy.ReAct` supports async execution (DSPy >= 3.x may have `asyncify` or native async). If so, use `asyncio.gather()` for concurrent branches.

3. **`dspy.Parallel`** or similar: Check if DSPy provides a built-in parallel execution primitive for running multiple module calls concurrently.

## Thread safety consideration

The current design uses a single `GraphToTAgent` instance called k times. If `dspy.ReAct` maintains internal state between calls (e.g., in its trajectory buffer), parallel calls to the same instance could corrupt that state. The safe approach is to either:
- Create k separate `GraphToTAgent` instances (which also benefits DSPy compilation — see the compilation issue)
- Or verify that `dspy.ReAct.__call__` is stateless/thread-safe

The `GraphEnvironment` tools (`retrieve_node`, `node_feature`, etc.) are read-only over immutable data structures, so they should be thread-safe.

## Impact

For a typical run with k=3, b=1, max_rounds=1:
- Current: 3 sequential agent traces = ~3x single-trace latency
- Parallel: 3 concurrent agent traces = ~1x single-trace latency

For k=3, b=2, max_rounds=2: the improvement is even larger since expansion also parallelizes.

## Files to modify

- `src/graph_tot/dspy_modules.py` — `_generate_branches()` (lines 367-380) and possibly `GraphToTSolver.__init__()` if creating multiple agent instances
