"""
DSPy compilation and optimization utilities for Graph ToT modules.

Provides convenience functions for compiling (optimizing prompts of) a
GraphToTSolver using DSPy's built-in optimizers, and for loading compiled
state back into a solver instance.

Typical usage::

    from graph_tot import GraphToTSolver, compile_solver, rouge_metric
    from graph_tot.data_loader import load_grbench_qa, train_val_split

    qa = load_grbench_qa(max_samples=50)
    trainset, valset = train_val_split(qa)

    solver = GraphToTSolver(graph_env=env, k=3, b=1, max_iters=12)
    compiled = compile_solver(solver, trainset, save_path="./compiled/solver.json")
"""

import logging
from pathlib import Path

import dspy

from .evaluate import rouge_metric

logger = logging.getLogger(__name__)

OPTIMIZERS = {
    "BootstrapFewShot": "dspy.BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch": "dspy.BootstrapFewShotWithRandomSearch",
    "MIPROv2": "dspy.teleprompt.MIPROv2",
}


def _resolve_optimizer(name: str):
    """Import and return a DSPy optimizer class by short name."""
    if name == "BootstrapFewShot":
        return dspy.BootstrapFewShot
    elif name == "BootstrapFewShotWithRandomSearch":
        return dspy.BootstrapFewShotWithRandomSearch
    elif name == "MIPROv2":
        from dspy.teleprompt import MIPROv2
        return MIPROv2
    else:
        raise ValueError(
            f"Unknown optimizer: {name!r}. "
            f"Supported: {list(OPTIMIZERS.keys())}"
        )


def compile_solver(
    solver: dspy.Module,
    trainset: list[dspy.Example],
    optimizer_class: str = "BootstrapFewShot",
    metric=None,
    save_path: str | Path | None = None,
    **optimizer_kwargs,
) -> dspy.Module:
    """Compile a DSPy module using a prompt optimizer.

    Runs the chosen DSPy optimizer on the solver's internal Predict/ReAct
    modules, bootstrapping few-shot demonstrations and (for MIPROv2)
    optimizing instruction text.

    Args:
        solver:          A ``dspy.Module`` to optimize (typically ``GraphToTSolver``).
        trainset:        Training examples — list of ``dspy.Example`` with
                         ``.with_inputs("question")`` and an ``answer`` field.
        optimizer_class: One of ``"BootstrapFewShot"``,
                         ``"BootstrapFewShotWithRandomSearch"``, or ``"MIPROv2"``.
        metric:          Callable ``(gold, pred, trace=None) -> float``.
                         Defaults to :func:`~graph_tot.evaluate.rouge_metric`.
        save_path:       If provided, save compiled state to this path after
                         optimization.  Parent directories are created automatically.
        **optimizer_kwargs: Forwarded to the optimizer constructor (e.g.
                         ``max_bootstrapped_demos=4``, ``auto="light"``).

    Returns:
        The compiled (optimized) module.
    """
    if metric is None:
        metric = rouge_metric

    OptimizerClass = _resolve_optimizer(optimizer_class)
    optimizer = OptimizerClass(metric=metric, **optimizer_kwargs)

    logger.info(
        "Starting compilation with %s on %d training examples",
        optimizer_class, len(trainset),
    )

    compiled = optimizer.compile(student=solver, trainset=trainset)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        compiled.save(str(save_path))
        logger.info("Compiled state saved to %s", save_path)

    return compiled


def load_compiled_solver(
    solver: dspy.Module,
    path: str | Path,
) -> dspy.Module:
    """Load previously compiled state into a solver instance.

    The solver must have the same module structure as the one that was
    compiled (same ``k``, ``eval_mode``, etc.).

    Args:
        solver: A fresh ``dspy.Module`` instance (typically ``GraphToTSolver``).
        path:   Path to the saved compiled state (JSON or pickle).

    Returns:
        The solver with compiled state loaded (same object, mutated in place).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Compiled state not found: {path}")
    solver.load(path=str(path))
    logger.info("Loaded compiled state from %s", path)
    return solver
