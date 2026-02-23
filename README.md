# Graph Tree-of-Thought · DSPy

A DSPy implementation of the **Graph Tree-of-Thought with Agent-based retrieval** approach from:

> **Grounding LLM Reasoning with Knowledge Graphs**
> arXiv:2502.13247 · February 2025
> https://arxiv.org/abs/2502.13247

The paper demonstrates that grounding LLM reasoning in structured knowledge graphs — via a ReAct agent that traverses graph edges and nodes — substantially outperforms ungrounded chain-of-thought on domain-specific QA benchmarks. The **Graph ToT + Agent** variant (this implementation) achieves the best results by combining tree-of-thought beam search over multiple agent traces with semantic node retrieval.

---

## What this implements

### The four graph tools (Section 3.2 of the paper)
A `dspy.ReAct` agent is given exactly four tools to navigate the knowledge graph:

| Tool | Description |
|------|-------------|
| `retrieve_node(keyword)` | Semantic search for graph nodes via FAISS + MPNet-v2 |
| `node_feature(node_id, feature)` | Read a specific attribute of a node |
| `neighbour_check(node_id, edge_type)` | List neighbors via a named edge type |
| `node_degree(node_id, edge_type)` | Count neighbors of a given type |

### Tree-of-Thought beam search (Section 3.3)
Rather than a single reasoning trace, the solver runs `k` independent ReAct agents in parallel (branching), scores each trace with an LLM evaluator, keeps the top `b` (beam pruning), and optionally repeats for multiple rounds. Two evaluation modes match the paper:
- **Score Vote** — each branch is scored 0–1 independently
- **Selection Vote** — a single LLM call picks the best branch by index

### Benchmark
- **GRBench Healthcare** — 270 questions over the Hetionet biomedical knowledge graph (~47K nodes, ~11 edge types: Disease, Compound, Gene, Symptom, Side Effect, ...)
- QA data loaded automatically from HuggingFace: [`PeterJinGo/GRBench`](https://huggingface.co/datasets/PeterJinGo/GRBench)
- Graph data downloaded separately from Google Drive (see setup below)
- Evaluated with **ROUGE-L F1**, per difficulty level (easy / medium / hard)

---

## Setup

### 1. Prerequisites
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager
- An [Anthropic API key](https://console.anthropic.com/)

### 2. Install dependencies
```bash
uv sync
```

### 3. Configure API keys
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...
```
`HF_TOKEN` is optional but recommended — without it, HuggingFace dataset downloads are rate-limited. After the first download, the dataset is cached locally.

### 4. Download the graph data
The healthcare knowledge graph must be downloaded manually:

1. Visit: https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9
2. Download the `healthcare` folder
3. Place the graph file at `./data/healthcare/graph.json`

On first run, a FAISS index is built over all 47K node embeddings (~6–7 minutes on CPU) and cached to `./data/cache/` — subsequent runs load it instantly.

---

## Usage

```bash
# Single demo question with verbose trace output
uv run python main.py --demo --k 3 --b 1 --max-iters 12

# Evaluate 20 easy questions
uv run python main.py --level easy --max-samples 20 --k 3 --b 1

# Full 270-question benchmark evaluation
uv run python main.py --max-samples 0 --k 3 --b 1 --output results/healthcare_eval.json

# Two-round beam search (k=3 branches, b=2 survivors, 2 rounds)
uv run python main.py --demo --k 3 --b 2 --max-rounds 2 --max-iters 12

# Selection vote instead of score vote
uv run python main.py --demo --eval-mode selection_vote
```

### Key options

| Option | Default | Description |
|--------|---------|-------------|
| `--k` | 3 | Branching factor — independent agent traces per round |
| `--b` | 1 | Beam width — traces kept after scoring |
| `--max-rounds` | 1 | Beam search rounds (1 = single-pass ToT) |
| `--max-iters` | 8 | Max ReAct steps per trace (12+ recommended) |
| `--eval-mode` | `score_vote` | `score_vote` or `selection_vote` |
| `--model` | `claude-haiku-4-5-20251001` | Anthropic model ID |
| `--temperature` | 0.7 | Sampling temperature (≥0.7 for branch diversity) |
| `--level` | all | Filter by difficulty: `easy`, `medium`, `hard` |
| `--max-samples` | 5 | Number of questions (0 = all 270) |
| `--demo` | off | Single question with full trace output |
| `--compiled` | none | Path to compiled solver state (from `optimize.py`) |

---

## Prompt optimization / compilation

DSPy's core value is automatic prompt optimization. The `optimize.py` script compiles the solver's internal prompts by bootstrapping few-shot demonstrations from training examples and (with MIPROv2) optimizing instruction text. The compiled state is saved to disk and loaded at inference time via `--compiled`.

### Quick start

```bash
# Compile with BootstrapFewShot (fast, good for small data)
uv run python optimize.py --optimizer BootstrapFewShot --train-size 10

# Compile with MIPROv2 (stronger — Bayesian optimization of instructions + demos)
uv run python optimize.py --optimizer MIPROv2 --auto light --train-size 30

# Run inference with compiled prompts
uv run python main.py --demo --compiled ./compiled/solver.json
```

### How it works

DSPy optimizers inspect the solver's internal `Predict` / `ReAct` modules, run them on training examples, collect successful traces, and inject them as few-shot demonstrations into the prompts. The ROUGE-L metric (`rouge_metric`) scores each trace against gold answers.

| Optimizer | What it optimizes | When to use |
|-----------|-------------------|-------------|
| `BootstrapFewShot` | Few-shot demonstrations | Small data (~10 examples), quick iteration |
| `BootstrapFewShotWithRandomSearch` | Demos + random search over candidates | More data (50+), evaluates multiple programs |
| `MIPROv2` | Instructions + demos jointly | Best quality; Bayesian optimization |

### Optimization options

| Option | Default | Description |
|--------|---------|-------------|
| `--optimizer` | `BootstrapFewShot` | Optimizer to use |
| `--train-size` | 20 | Training examples (rest used for validation) |
| `--auto` | `light` | MIPROv2 intensity: `light`, `medium`, `heavy` |
| `--max-bootstrapped-demos` | 4 | Max generated demos per predictor |
| `--max-labeled-demos` | 16 | Max trainset demos per predictor |
| `--save-path` | `./compiled/solver.json` | Where to save compiled state |
| `--eval-compiled` | on | Evaluate on validation set after compilation |

### Library usage

For users integrating these modules into their own codebase:

```python
import dspy
from graph_tot import (
    GraphEnvironment, GraphToTSolver,
    compile_solver, load_compiled_solver,
    rouge_metric, make_dspy_examples, train_val_split,
)

# 1. Configure DSPy and build solver as usual
dspy.configure(lm=dspy.LM("anthropic/claude-haiku-4-5-20251001", temperature=0.7))
env = GraphEnvironment(graph_path="./data/healthcare/graph.json", faiss_cache_dir="./data/cache")
solver = GraphToTSolver(graph_env=env, k=3, b=1, max_iters=12)

# 2. Prepare training data
from graph_tot.data_loader import load_grbench_qa
qa = load_grbench_qa(max_samples=50)
trainset, valset = train_val_split(qa)

# 3. Compile
compiled = compile_solver(
    solver, trainset,
    optimizer_class="BootstrapFewShot",
    save_path="./compiled/solver.json",
)

# 4. Later: load compiled state into a fresh solver
solver = GraphToTSolver(graph_env=env, k=3, b=1, max_iters=12)
load_compiled_solver(solver, "./compiled/solver.json")
result = solver(question="What compounds treat diabetes mellitus?")
```


### Bring your own graph backend

`GraphToTSolver` only needs a graph object that exposes the four tool methods (`retrieve_node`, `node_feature`, `neighbour_check`, `node_degree`) plus `get_tools()`.

This package now exposes:
- `GraphToolInterface` — protocol describing that tool contract
- `GraphNodeStore` — abstract loader for node data from any source
- `JsonPickleGraphStore` — default implementation for current GRBench JSON/pickle files

So you can integrate domain-specific backends such as Neo4j, RDF/OWL parsers, or custom services by implementing a `GraphNodeStore` (or directly implementing `GraphToolInterface`).

```python
from graph_tot import GraphEnvironment, GraphNodeStore

class Neo4jStore(GraphNodeStore):
    @property
    def identity(self) -> str:
        return "neo4j://my-db/v1"

    def iter_nodes(self):
        # yield (node_id, node_type, node_data={"features": ..., "neighbors": ...})
        ...

env = GraphEnvironment(
    graph_path=None,
    faiss_cache_dir="./data/cache",
    node_store=Neo4jStore(),
)
```

---

## Project structure

```
main.py             Click CLI for inference and evaluation
optimize.py         Click CLI for prompt compilation / optimization
src/graph_tot/
  graph_env.py      GraphEnvironment: graph loading, FAISS index, 4 tool methods
  dspy_modules.py   GraphToTAgent, TreeOfThoughtEvaluator, GraphToTSolver
  data_loader.py    load_grbench_qa(), make_dspy_examples(), train_val_split()
  evaluate.py       RougeEvaluator, EvalReport, rouge_metric (DSPy-compatible)
  optimize.py       compile_solver(), load_compiled_solver() — library API
data/
  healthcare/       graph.json (download manually — see Setup)
  cache/            FAISS index cache (auto-generated on first run)
compiled/           Compiled solver state JSON (generated by optimize.py)
results/            Evaluation JSON output
```

---

## How it works

```
Question
   │
   ├─ Branch 0: ReAct agent → Thought→Action→Obs loop (up to max_iters)
   ├─ Branch 1: ReAct agent → (independent, same question)
   └─ Branch k: ReAct agent → (independent)
         │
         ▼
   TreeOfThoughtEvaluator (score_vote or selection_vote)
         │
         ▼
   Top-b survivors  ──► (if max_rounds > 1: expand each survivor → repeat)
         │
         ▼
   Best branch answer → ROUGE-L score
```

Each ReAct trace reasons about which graph tools to call, self-corrects from error messages (e.g. wrong edge type names), and builds up a grounded answer from actual graph data rather than parametric knowledge.

---

## Citation

```bibtex
@misc{amayuelas2025groundingllmreasoningknowledge,
    title={Grounding LLM Reasoning with Knowledge Graphs},
    author={Alfonso Amayuelas and Joy Sain and Simerjot Kaur and Charese Smiley},
    year={2025},
    eprint={2502.13247},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2502.13247},
}
```
