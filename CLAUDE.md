# CLAUDE.md — Graph ToT DSPy

## Project
DSPy implementation of Graph Tree-of-Thought + Agent-based retrieval (arXiv:2502.13247).
GRBench Healthcare benchmark. Anthropic Claude backend via DSPy.

## Commands
```bash
uv sync                                              # install / sync deps
uv run python main.py --demo                         # single Q (shows download instructions if no graph)
uv run python main.py --demo --k 3 --b 1 --max-iters 12   # demo after graph downloaded
uv run python main.py --level easy --max-samples 20  # batch eval subset
uv run python main.py --max-samples 0                # full 270-question eval
```

## Architecture
```
main.py                  Click CLI; configures DSPy with dspy.LM("anthropic/...")
src/graph_tot/
  graph_env.py           GraphEnvironment: loads graph JSON/pkl, builds FAISS index,
                         exposes 4 bound-method tools for dspy.ReAct
  dspy_modules.py        4 DSPy Signatures + GraphToTAgent (ReAct) +
                         TreeOfThoughtEvaluator + GraphToTSolver (beam search)
  data_loader.py         load_grbench_qa() from HuggingFace PeterJinGo/GRBench;
                         find_graph_file() with graceful FileNotFoundError
  evaluate.py            RougeEvaluator (rouge-score), EvalReport with per-level breakdown
data/
  healthcare/            graph.json — download from Google Drive (see README)
  cache/                 FAISS index pickle — auto-built on first run (~6 min), reused after
```

## Key design decisions

**ToT = complete-trace branching**: `dspy.ReAct` manages a single trace internally;
ToT branches happen at the level of full ReAct runs. `GraphToTSolver` calls
`GraphToTAgent` k times independently, scores results, keeps top-b. Survivors'
answers are passed as `context` input to next-round agents.

**Tool docstrings matter**: `dspy.ReAct` builds its tool schema from function `__doc__`
and type annotations. All four tool methods have descriptive `Args:`/`Returns:` sections
and `-> str` annotations. Tools never raise exceptions — they return error strings so the
agent can self-correct.

**FAISS**: `normalize_embeddings=True` + `IndexFlatIP` = cosine similarity. Embeddings
are pickled to `data/cache/faiss_sentence-transformers_all-mpnet-base-v2.pkl`.

**Branch diversity**: requires `temperature >= 0.7` on the LM. At temperature=0 all k
branches are identical.

## Data
- QA pairs: HuggingFace `PeterJinGo/GRBench`, name="healthcare", split="test" (270 questions)
- Graph: Google Drive https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9
- Graph format: `{node_type: {node_id: {features: {...}, neighbors: {edge_type: [ids]}}}}`
- Edge types follow pattern `NodeType-verb-NodeType` e.g. `Compound-causes-Side Effect`

## Common issues

**Agent stops one step short**: increase `--max-iters` (12 recommended, default is 8).
The agent needs ~5 steps for a typical side-effect question:
retrieve → check feature (fails) → check wrong edge (fails, reads available types) → check correct edge → finish.

**All k branches identical**: temperature too low. Ensure `--temperature 0.7` or higher.

**Only 1 branch shown despite k=3**: `_dicts_to_branches` matches by answer text —
if branches produce identical answers they merge. Fixed by increasing max-iters for diversity.

**Symlinks warning on Windows**: cosmetic only. Set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`
in `.env` to suppress.

**ROUGE-L = 0 on prose answers**: expected when agent hallucinates instead of traversing.
Gold answers are terse entity lists; prose doesn't overlap. Fixing max-iters resolves this.

## Tuning levers (paper Table 4 correspondence)
| CLI flag | Paper param | Notes |
|----------|-------------|-------|
| `--k` | branching factor | 3 matches paper default |
| `--b` | beam width | 1 = greedy; paper tests 1,2,3 |
| `--max-rounds` | ToT depth | 1 = single-pass |
| `--eval-mode score_vote` | Score Vote | paper's primary method |
| `--eval-mode selection_vote` | Selection Vote | cheaper (1 LLM call vs k) |
| `--max-iters` | agent steps | paper uses up to 10; 12 safer |

## Dependencies (key)
- `dspy>=3.1.3` — ReAct, Predict, Signature, LM, configure
- `anthropic>=0.40.0` + `litellm>=1.50.0` — Claude backend
- `sentence-transformers>=3.0.0` — all-mpnet-base-v2 embeddings
- `faiss-cpu>=1.8.0` — IndexFlatIP similarity search
- `datasets>=2.20.0` — HuggingFace GRBench loader
- `rouge-score>=0.1.2` — ROUGE-L evaluation
- `click`, `rich`, `python-dotenv` — CLI/UX
