# Add DSPy compilation and optimization support

## Summary

The codebase uses DSPy as a framework but does not leverage DSPy's core value proposition: **automatic prompt optimization through compilation**. None of the signatures have few-shot demonstrations, and there is no compilation step. Adding compilation support (e.g., `BootstrapFewShot`, `MIPROv2`) would likely improve answer quality significantly, as this is DSPy's primary differentiator over raw prompt engineering.

## Current state

The four signatures (`GraphQASignature`, `ScoreVoteSignature`, `SelectionVoteSignature`, `FinalAnswerSignature`) all use zero-shot prompting. The behavioral guidance is hardcoded in docstrings (see related issue on signature style). The modules are used in a hand-engineered orchestration pipeline with no optimization.

This means the program gets no benefit from DSPy that it couldn't get from calling the LLM directly via `litellm` or `anthropic`. The `dspy.ReAct` tool loop and `dspy.Predict` structured output parsing are convenient, but the optimization story is the reason to choose DSPy specifically.

## What DSPy compilation does

DSPy optimizers like `BootstrapFewShot` or `MIPROv2`:
1. Take a small set of labeled examples (trainset)
2. Run the program on those examples
3. Identify successful traces
4. Inject them as few-shot demonstrations into the signatures
5. Optionally optimize the instruction text (MIPROv2)

This typically improves accuracy substantially on structured tasks like graph QA.

## What's needed

### 1. Training data

A small set of (question, gold_answer) pairs for the compiler to bootstrap from. Options:
- Use a subset of GRBench QA pairs (e.g., 10-20 easy/medium questions) as training data, holding out the rest for evaluation
- Manually curate a small set of known-good (question, answer, trace) triples

### 2. Metric function

A function that takes a prediction and gold answer and returns a score. `RougeEvaluator.score_single()` already exists and can serve as this metric.

### 3. Compilation entry point

A new CLI command or flag (e.g., `--compile`) that:
1. Loads a trainset split
2. Instantiates the solver
3. Runs `BootstrapFewShot` (or `MIPROv2`) with the metric
4. Saves the compiled program to disk (e.g., `compiled_solver.json`)

### 4. Compiled program loading

The main inference path should check for and load a compiled program if available, using `module.load(path)`.

### 5. Module structure considerations

Currently `GraphToTSolver` creates a single `GraphToTAgent` instance and calls it k times. For compilation to give different branches different few-shot examples (which would improve diversity), you'd need k separate agent instances so DSPy can optimize their parameters independently. Consider whether this is worth the added complexity.

## Key signatures to optimize

In priority order:
1. **GraphQASignature** — the agent's core reasoning prompt. Few-shot examples of successful graph traversals would be very high-value.
2. **ScoreVoteSignature** — examples of good scoring calibration would improve beam search quality.
3. **SelectionVoteSignature** — examples of correct selection would improve the cheaper eval mode.

## Files to modify

- `src/graph_tot/dspy_modules.py` — module structure (possibly multiple agent instances)
- `main.py` — add `--compile` flag, compiled model loading
- New file: `src/graph_tot/compile.py` or similar for compilation logic
