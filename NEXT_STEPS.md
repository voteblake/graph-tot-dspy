# Next Steps

## 1. Fix max-iters (immediate)

The demo run showed the agent stopping one step short of the correct answer.
The gold answer (`Conjunctivitis, Sensitisation, Chemical burn, ...`) lives exactly
one more tool call away — `neighbour_check(DB00772, 'Compound-causes-Side Effect')`.

```bash
uv run python main.py --demo --k 3 --b 1 --max-iters 12
uv run python main.py --level easy --max-samples 20 --k 3 --b 1 --max-iters 12
```

Once ROUGE-L lifts above 0 on easy questions, run the full eval:
```bash
uv run python main.py --max-samples 0 --k 3 --b 1 --max-iters 12 \
  --output results/baseline_k3_b1_iters12.json
```

## 2. DSPy optimizer (MIPROv2)

DSPy's `MIPROv2` can automatically tune the four Signature prompts
(`GraphQASignature`, `ScoreVoteSignature`, `SelectionVoteSignature`,
`FinalAnswerSignature`) using a small held-out subset — without changing
any code. The tuned prompts are saved and reloaded for eval.

Rough sketch:
```python
import dspy
from src.graph_tot.dspy_modules import GraphToTSolver
from src.graph_tot.evaluate import RougeEvaluator
from src.graph_tot.data_loader import load_grbench_qa

# Small train split (GRBench is test-only; use first 30 easy questions)
trainset = [
    dspy.Example(question=qa.question, answer=qa.answer).with_inputs("question")
    for qa in load_grbench_qa(level_filter="easy", max_samples=30)
]

def rouge_metric(example, prediction, trace=None):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(example.answer, prediction.answer)["rougeL"].fmeasure

solver = GraphToTSolver(graph_env=..., k=3, b=1)
optimizer = dspy.MIPROv2(metric=rouge_metric, auto="light")
optimized = optimizer.compile(solver, trainset=trainset)
optimized.save("optimized_solver.json")
```

Paper Table 4 numbers were achieved without any prompt tuning, so this
should give a meaningful lift on top of the baseline.

## 3. Prime Intellect eval environment

**Open question to investigate:** could this be run inside a Prime eval environment?
https://docs.primeintellect.ai/tutorials-environments/evaluating

We already have everything Prime needs on paper:
- A defined task (GRBench Healthcare QA)
- A scalar metric (ROUGE-L F1, 0–1)
- A deterministic eval loop (`main.py --max-samples 0`)
- JSON result output (`results/evaluation.json`)

The key question is how DSPy's external API calls (Anthropic) interact with
Prime's sandboxed environment model — Prime evals typically assume the scoring
function is self-contained (local model inference), whereas here the "model"
is a remote API call. Worth investigating:

- Does Prime support evals that shell out to external inference APIs, or does
  it expect everything to run inside the container?
- If API calls aren't allowed, could the LLM backend be swapped to a locally
  served model via Ollama or vLLM inside the Prime environment? DSPy supports
  this via `dspy.LM("ollama/...")` — one config line change.
- Prime's eval environments provide beefy CPU/GPU compute, which would also
  solve the sequential runtime problem (parallelise across questions on many cores).
- The `evaluate.py` ROUGE-L output already writes structured JSON — likely
  compatible with whatever metric reporting Prime expects.

## 4. Library interface

The modules are importable but the `GraphEnvironment` constructor is a required
argument everywhere, making it awkward to use outside of `main.py`. Worth adding:

- A clean top-level `create_solver()` factory function in `src/graph_tot/__init__.py`
- A `GraphToTSolver.from_graph_file(path, ...)` classmethod
- Type stubs / docstrings on all public classes so IDEs can autocomplete properly
- Possibly a `pyproject.toml` `[project.scripts]` entry so `graph-tot` works as
  a standalone command after `uv tool install`

Example target API:
```python
from graph_tot import create_solver

solver = create_solver(
    graph_path="data/healthcare/graph.json",
    model="anthropic/claude-3-5-haiku-20241022",
    k=3, b=1,
)
result = solver(question="What are the side effects of Malathion?")
print(result.answer)
print(result.best_trace)
```
