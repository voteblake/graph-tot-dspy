# Adopt DSPy idioms: declarative signatures, typed forward(), and assertions

## Summary

Several patterns in the codebase work but are non-idiomatic for DSPy, which means they won't play well with DSPy's compiler, tracing infrastructure, and optimization tools. This issue covers three related areas: signature docstring style, forward() return types, and use of dspy.Assert/Suggest.

## Area 1: Verbose prompt-engineering docstrings on signatures

### Current state

Signatures like `GraphQASignature` (`dspy_modules.py:27-32`) have detailed behavioral instructions in their docstrings:

```python
class GraphQASignature(dspy.Signature):
    """You are a medical knowledge graph expert. Use the available graph tools
    to answer questions about diseases, compounds, genes, symptoms, and their
    relationships. Always start by using retrieve_node to locate relevant
    entities in the graph, then explore their features and neighbors to build
    a complete, factual answer. Be precise and cite specific node IDs when
    possible."""
```

This reads like a hand-crafted system prompt ("Always start by...", "Be precise and cite...").

### DSPy philosophy

DSPy's design principle is "programming, not prompting." Signatures should be **declarative task descriptions**, not behavioral instructions. The optimizer fills in behavioral guidance through few-shot examples and (with MIPROv2) auto-generated instructions. Over-specifying behavior in the docstring makes compilation less effective because the optimizer can't easily override hardcoded instructions.

### Suggested approach

Shorten signature docstrings to describe *what* the task is, not *how* to do it. For example:

```python
class GraphQASignature(dspy.Signature):
    """Answer a medical question using knowledge graph exploration tools."""
```

The behavioral details ("always start with retrieve_node", "cite specific node IDs") should emerge from few-shot demonstrations added during compilation, not from the docstring.

## Area 2: TreeOfThoughtEvaluator uses list[dict] instead of DSPy types

### Current state

`TreeOfThoughtEvaluator.forward()` (`dspy_modules.py:189`) takes `branches: list[dict]` and returns `list[dict]`. The `Branch` dataclass is outside DSPy's type system. The evaluator ↔ solver interface is:

```
Solver → [Branch.as_dict() for each branch] → Evaluator → [scored dicts] → Solver._dicts_to_branches()
```

This dict-based round-trip is fragile (see the answer-text collision issue) and opaque to DSPy's tracing.

### DSPy convention

`forward()` methods should accept signature-compatible inputs and return `dspy.Prediction`. This allows DSPy's optimization infrastructure to introspect the evaluator's behavior and trace the full pipeline.

### Suggested approach

Either:
- Have the evaluator accept and return `Branch` objects directly (simplest, avoids the dict round-trip entirely)
- Or define evaluation as a DSPy signature where inputs/outputs are properly typed

## Area 3: Manual try/except instead of dspy.Assert / dspy.Suggest

### Current state

Error handling uses manual `try/except` blocks throughout:
- `GraphToTAgent.forward()` line 144: catches all exceptions, returns fallback prediction
- `_score_vote` line 217: catches scoring failures, defaults to 0.0
- `_selection_vote` line 236: catches selection failures, defaults to index 0

### DSPy mechanisms

`dspy.Assert` and `dspy.Suggest` are DSPy's runtime constraint enforcement tools. They integrate with the compiler — failed assertions can trigger automatic retry with backtracking, and the compiler can learn to avoid patterns that trigger assertions.

### Suggested uses

- `dspy.Assert(0.0 <= score <= 1.0, "Score must be between 0 and 1")` in the score voter
- `dspy.Assert(0 <= best_index < len(branches), "Selection index must be valid")` in the selection voter
- `dspy.Suggest(len(answer) > 0, "Answer should not be empty")` in the agent

## Minor: get_trajectory_text could be a staticmethod

`get_trajectory_text` (`dspy_modules.py:148-164`) doesn't use `self.react` or any instance state — it only reads from the prediction argument. It could be a `@staticmethod` or a standalone utility function.

## Files to modify

- `src/graph_tot/dspy_modules.py` — all four signatures, `TreeOfThoughtEvaluator.forward()`, error handling blocks, `get_trajectory_text`
