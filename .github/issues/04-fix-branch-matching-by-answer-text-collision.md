# Fix _dicts_to_branches answer-text matching causing silent branch collisions

## Summary

`GraphToTSolver._dicts_to_branches()` matches scored dicts back to original `Branch` objects using answer text as the key. When two branches produce the same answer (which happens frequently at low temperature or with simple questions), the dict silently drops one branch, causing incorrect trace-to-score associations.

## Current behavior

In `dspy_modules.py:390`:

```python
orig_by_answer = {b.answer: b for b in original_branches}
```

If branches `[B0, B1, B2]` produce answers `["aspirin", "aspirin", "ibuprofen"]`, the dict becomes `{"aspirin": B1, "ibuprofen": B2}` — `B0` is silently overwritten. When the evaluator scores what was originally B0's trace, the result gets associated with B1's trace instead.

The CLAUDE.md acknowledges a symptom: "Only 1 branch shown despite k=3: `_dicts_to_branches` matches by answer text — if branches produce identical answers they merge."

## Impact

- **Wrong trace-to-score pairing**: A score earned by one reasoning trace gets attributed to a different trace that happened to produce the same answer
- **Silent data loss**: Branches are dropped without warning
- **Incorrect beam selection**: In multi-round mode, the "survivor" branches may carry the wrong trace, meaning the context passed to the next round is from the wrong reasoning path

## Suggested fix

Replace answer-text matching with a stable identifier. Options:

1. **Index-based matching**: The evaluator receives branches in a known order and returns them in scored order. Track the original index through the scoring pipeline (add an `"index"` key to the branch dict before sending to evaluator, preserve it through scoring, use it to match back).

2. **Unique branch ID**: Add a `branch_id` field (e.g., UUID or incrementing counter) to the `Branch` dataclass and use it as the matching key.

3. **Skip round-trip entirely**: Instead of converting Branch -> dict -> scoring -> dict -> Branch, have the evaluator work directly with Branch objects, just adding scores in-place.

Option 3 is the cleanest and also connects to the broader issue of using proper DSPy types (see related issue on DSPy type system alignment).

## Files to modify

- `src/graph_tot/dspy_modules.py` — `_dicts_to_branches()` (lines 382-407) and the `Branch` dataclass (lines 98-114)
