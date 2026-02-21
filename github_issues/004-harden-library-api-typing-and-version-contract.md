# Issue: Harden public library API contract (typing, exports, versioning)

## Problem statement
The package exports useful modules but lacks strong library contract artifacts (e.g., explicit typed package marker, clearer semver policy hooks, and tighter public API boundaries for long-term consumption).

## Why this matters
- Downstream users need stable import paths and predictable change policy.
- Type-checker friendliness improves integration in larger codebases.
- Publishing confidence requires clear public/private boundaries.

## Current behavior (code pointers)
- Public exports are defined in `src/graph_tot/__init__.py`.
- Packaging metadata is in `pyproject.toml`.
- No explicit `py.typed` marker is present.

## Expected behavior
A library-grade API contract with explicit typed distribution and documented compatibility boundaries.

## Proposed implementation
1. Add `py.typed` marker and include it in package data for wheel/sdist.
2. Audit and document public API surface:
   - keep stable symbols in `__all__`,
   - discourage imports from internal module internals.
3. Add minimal typing hardening:
   - stronger return/arg annotations for exported functions,
   - optional Protocols/types for graph tool interfaces.
4. Add versioning guidance in README/CONTRIBUTING for API-breaking changes.
5. Validate built artifacts include typing marker and intended files.

## Acceptance criteria
- [ ] `py.typed` is shipped in wheel/sdist.
- [ ] Public API list is explicit and documented.
- [ ] Type-checker smoke check passes on sample consumer snippet.
- [ ] Release notes policy mentions API compatibility expectations.

## Suggested tests/checks
- Build package and inspect wheel contents for `py.typed`.
- Run a lightweight static type check on sample usage script.

## Out of scope
- Full mypy strict migration of entire repository.
