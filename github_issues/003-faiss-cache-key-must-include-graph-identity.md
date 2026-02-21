# Issue: Prevent stale FAISS cache reuse by keying cache on graph identity

## Problem statement
FAISS embedding cache filename currently depends only on embedding model name. If graph content/path changes while using the same embedding model, stale embeddings may be loaded incorrectly.

## Why this matters
- Incorrect retrieval results due to embedding/document mismatch.
- Hard-to-detect silent data corruption in experiments.
- Breaks reproducibility when switching datasets or graph revisions.

## Current behavior (code pointers)
- `src/graph_tot/graph_env.py`, `_build_or_load_faiss_index()`:
  - cache path: `faiss_{embedding_model}.pkl`
  - no graph fingerprint/version component.

## Expected behavior
Cache artifacts should be uniquely tied to both embedding model and graph identity.

## Proposed implementation
1. Build graph identity fingerprint (at least one of):
   - hash of graph file bytes (preferred),
   - hash of `(graph_path, file_size, mtime)` fallback.
2. Include fingerprint in cache filename (or sidecar metadata).
3. Validate cache shape/doc count consistency before use.
4. Optional migration path:
   - detect legacy cache naming and rebuild automatically.
5. Improve logging to show model + graph fingerprint for transparency.

## Acceptance criteria
- [ ] Changing graph file while keeping model name triggers cache miss/rebuild.
- [ ] Same graph+model reuses cache.
- [ ] Mismatch in embedding count vs `doc_lookup` causes safe rebuild.
- [ ] Tests cover both cache-hit and cache-invalidated flows.

## Suggested tests
- Use temporary graph files with small synthetic graphs.
- Build cache for graph A, then initialize with graph B and confirm rebuild.
- Corrupt cache length and assert rebuild path is taken.

## Out of scope
- Moving from pickle to external vector DB.
