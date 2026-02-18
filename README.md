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
| `--model` | `claude-3-5-haiku-20241022` | Anthropic model ID |
| `--temperature` | 0.7 | Sampling temperature (≥0.7 for branch diversity) |
| `--level` | all | Filter by difficulty: `easy`, `medium`, `hard` |
| `--max-samples` | 5 | Number of questions (0 = all 270) |
| `--demo` | off | Single question with full trace output |

---

## Project structure

```
src/graph_tot/
  graph_env.py      GraphEnvironment: graph loading, FAISS index, 4 tool methods
  dspy_modules.py   GraphToTAgent, TreeOfThoughtEvaluator, GraphToTSolver
  data_loader.py    load_grbench_qa() from HuggingFace, find_graph_file()
  evaluate.py       RougeEvaluator, EvalReport (ROUGE-L, per-difficulty breakdown)
main.py             Click CLI entry point
data/
  healthcare/       graph.json (download manually — see Setup)
  cache/            FAISS index cache (auto-generated on first run)
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
   FinalAnswerSynthesizer (DSPy Predict)
         │
         ▼
   Answer + ROUGE-L score
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
