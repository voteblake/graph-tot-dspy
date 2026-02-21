# Issue: Preserve full branch history across all ToT rounds in `GraphToTSolver`

## Problem statement
`GraphToTSolver.forward()` currently overwrites `all_scored` on each round (`all_scored = current_beam`), so the returned `all_branches` only contains the final round’s scored branches instead of the full multi-round branch history.

## Why this matters
- Debuggability: impossible to inspect why earlier-round survivors were selected/eliminated.
- Evaluation: post-hoc analysis of branch quality across rounds is incomplete.
- Reproducibility: observability promised by the API/docstring is misleading for multi-round runs.

## Current behavior (code pointers)
- `src/graph_tot/dspy_modules.py` inside `GraphToTSolver.forward()`:
  - initializes `all_scored: list[Branch] = []`
  - reassigns `all_scored = current_beam` each round
  - returns `all_branches=[b.as_dict() for b in all_scored]`

## Expected behavior
Return branch history for *all* scored branches across rounds, preserving enough metadata to reconstruct round-level evolution.

## Proposed implementation
1. Replace round-overwrite with accumulation:
   - append each round’s scored branches into a persistent collection.
2. Add round metadata per branch entry (e.g., `round_idx`, `rank`, maybe `is_survivor`).
3. Keep existing `round_log` but align it with accumulated `all_branches`.
4. Ensure backward compatibility by preserving existing fields (`answer`, `trace`, `score`, `parent_context`) while adding new ones.

## Acceptance criteria
- [ ] For `max_rounds=1`, behavior is unchanged except optional additive metadata.
- [ ] For `max_rounds>1`, `all_branches` length equals sum of branches scored each round.
- [ ] Round index and survivor status can be reconstructed from returned artifacts.
- [ ] Unit tests verify accumulation and metadata correctness for multi-round search.

## Suggested tests
- Add/extend tests in `tests/` with a mocked evaluator to produce deterministic scores.
- Case A: `k=3, b=1, max_rounds=2` should return 3 + 3 = 6 scored branches.
- Case B: `k=3, b=2, max_rounds=2` should return 3 + 6 = 9 scored branches.
- Assert round metadata and survivor markers are correct.

## Out of scope
- Changing beam-search semantics.
- Introducing new ranking models.
