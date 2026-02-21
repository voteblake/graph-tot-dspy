# Issue: Add structured error semantics for graph tools while preserving ReAct compatibility

## Problem statement
Graph tool methods currently return plain strings for both success and errors (e.g., `"Error: ..."`). This is workable for ReAct prompting but weak for programmatic library consumers and test assertions.

## Why this matters
- Consumers cannot reliably distinguish data vs error without string parsing.
- Error handling is brittle and localization/format changes become breaking.
- Reduces composability for non-ReAct use cases.

## Current behavior (code pointers)
- `src/graph_tot/graph_env.py` tool methods (`retrieve_node`, `node_feature`, `neighbour_check`, `node_degree`) return `str` only, including error text.

## Expected behavior
Expose structured error information for programmatic callers, while keeping ReAct-facing behavior simple.

## Proposed implementation
1. Introduce internal structured result model, e.g.:
   - `{ok: bool, data: ..., error_code: ..., message: ...}`
   - or typed dataclasses.
2. Keep current ReAct tool signatures returning strings by adding thin adapters:
   - structured core methods for library callers,
   - string formatter wrappers for ReAct tools.
3. Define stable error codes (`NODE_NOT_FOUND`, `FEATURE_NOT_FOUND`, etc.).
4. Add helper APIs for consumers to call structured variants directly.
5. Update docs with both usage modes (ReAct-string and structured-programmatic).

## Acceptance criteria
- [ ] Structured APIs exist for all four tool operations.
- [ ] Existing ReAct integration remains backward compatible.
- [ ] Error codes are tested and documented.
- [ ] No regressions in current CLI/demo behavior.

## Suggested tests
- Unit tests for structured methods success/error branches.
- Compatibility tests confirming ReAct tool wrappers still return expected strings.

## Out of scope
- Changing DSPy `ReAct` tool interface itself.
