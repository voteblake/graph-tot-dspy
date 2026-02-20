"""
Graph Tree-of-Thought prompt optimization — example CLI.

Demonstrates how to compile (optimize) a GraphToTSolver using DSPy's prompt
optimizers.  This is the compilation counterpart to main.py's inference demo.

Quick start:
  1. Set ANTHROPIC_API_KEY in .env or environment
  2. Ensure graph data is downloaded (see main.py --help)
  3. uv run python optimize.py --optimizer BootstrapFewShot --train-size 10
  4. uv run python main.py --demo --compiled ./compiled/solver.json
"""

import logging
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_GRAPH_DIR = "./data/healthcare"
DEFAULT_FAISS_CACHE = "./data/cache"
DEFAULT_SAVE_PATH = "./compiled/solver.json"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_dspy(model: str, temperature: float, max_tokens: int) -> None:
    """Configure DSPy with a language model."""
    import dspy

    load_dotenv()

    lm = dspy.LM(
        model=f"{model}",
        temperature=temperature,
        max_tokens=max_tokens,
        num_retries=8,
        cache=temperature == 0.0,
        cache_control_injection_points=[{
            "location": "message",
            "role": "system",
        }],
    )
    dspy.configure(lm=lm)
    console.print(f"[green]DSPy configured:[/green] {model} (temp={temperature})")


@click.command()
@click.option(
    "--graph-dir", default=DEFAULT_GRAPH_DIR, show_default=True,
    help="Directory containing downloaded GRBench healthcare graph file.",
)
@click.option(
    "--faiss-cache", default=DEFAULT_FAISS_CACHE, show_default=True,
    help="Directory for FAISS index cache.",
)
@click.option(
    "--model", default="anthropic/claude-haiku-4-5-20251001", show_default=True,
    help="Model ID for the task LM.",
)
@click.option(
    "--temperature", default=0.7, type=float, show_default=True,
    help="LLM sampling temperature. Use >=0.7 for branch diversity.",
)
@click.option(
    "--max-tokens", default=2048, type=int, show_default=True,
    help="Max tokens per LLM call.",
)
@click.option(
    "--k", default=3, type=int, show_default=True,
    help="ToT branching factor.",
)
@click.option(
    "--b", default=1, type=int, show_default=True,
    help="ToT beam width.",
)
@click.option(
    "--max-rounds", default=1, type=int, show_default=True,
    help="Number of ToT beam search rounds.",
)
@click.option(
    "--max-iters", default=8, type=int, show_default=True,
    help="Max ReAct iterations per agent branch.",
)
@click.option(
    "--eval-mode",
    default="score_vote",
    type=click.Choice(["score_vote", "selection_vote"]),
    show_default=True,
    help="Branch evaluation method.",
)
@click.option(
    "--optimizer",
    default="BootstrapFewShot",
    type=click.Choice(["BootstrapFewShot", "BootstrapFewShotWithRandomSearch", "MIPROv2"]),
    show_default=True,
    help="DSPy optimizer to use for compilation.",
)
@click.option(
    "--train-size", default=20, type=int, show_default=True,
    help="Number of QA pairs to use for training (remainder used for validation).",
)
@click.option(
    "--level",
    default=None,
    type=click.Choice(["easy", "medium", "hard"]),
    help="Filter to questions of a specific difficulty level.",
)
@click.option(
    "--save-path", default=DEFAULT_SAVE_PATH, show_default=True,
    help="Path to save compiled solver state.",
)
@click.option(
    "--auto",
    default="light",
    type=click.Choice(["light", "medium", "heavy"]),
    show_default=True,
    help="MIPROv2 optimization intensity (ignored for other optimizers).",
)
@click.option(
    "--max-bootstrapped-demos", default=4, type=int, show_default=True,
    help="Max bootstrapped (generated) demonstrations per predictor.",
)
@click.option(
    "--max-labeled-demos", default=16, type=int, show_default=True,
    help="Max labeled (trainset) demonstrations per predictor.",
)
@click.option(
    "--parallel/--no-parallel", default=True, show_default=True,
    help="Generate k branches in parallel using threads.",
)
@click.option(
    "--eval-compiled/--no-eval-compiled", default=True, show_default=True,
    help="Evaluate the compiled solver on the validation set after optimization.",
)
@click.option(
    "--verbose", is_flag=True, default=False,
    help="Enable debug-level logging.",
)
def main(
    graph_dir, faiss_cache, model, temperature, max_tokens,
    k, b, max_rounds, max_iters, eval_mode,
    optimizer, train_size, level, save_path,
    auto, max_bootstrapped_demos, max_labeled_demos,
    parallel, eval_compiled, verbose,
):
    """
    Compile (optimize) a GraphToTSolver using DSPy prompt optimizers.

    \b
    This optimizes the prompts inside the solver's ReAct agent and evaluator
    modules by bootstrapping few-shot demonstrations from training examples.
    The compiled state is saved to disk and can be loaded by main.py via
    the --compiled flag.

    \b
    Examples:
      uv run python optimize.py --optimizer BootstrapFewShot --train-size 10
      uv run python optimize.py --optimizer MIPROv2 --auto light --train-size 30
      uv run python main.py --demo --compiled ./compiled/solver.json
    """
    setup_logging(verbose)

    console.print(Panel.fit(
        "[bold magenta]Graph ToT Prompt Optimization[/bold magenta] · DSPy Compilation\n"
        "[dim]Optimize prompts for your domain graph and task[/dim]",
        border_style="magenta",
    ))

    # ------------------------------------------------------------------
    # Pre-flight: check graph data
    # ------------------------------------------------------------------
    from src.graph_tot.data_loader import (
        check_graph_available,
        find_graph_file,
        load_grbench_qa,
        train_val_split,
        DOWNLOAD_INSTRUCTIONS,
        GRAPH_DRIVE_URL,
    )

    if not check_graph_available(graph_dir):
        instructions = DOWNLOAD_INSTRUCTIONS.format(
            drive_url=GRAPH_DRIVE_URL,
            graph_dir=Path(graph_dir) / "graph.json",
        )
        console.print(Panel(
            instructions,
            title="[yellow]Graph Data Required[/yellow]",
            border_style="yellow",
        ))
        sys.exit(0)

    # ------------------------------------------------------------------
    # Configure DSPy
    # ------------------------------------------------------------------
    setup_dspy(model=model, temperature=temperature, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    # Load graph + build FAISS index
    # ------------------------------------------------------------------
    from src.graph_tot.graph_env import GraphEnvironment

    console.print("\n[bold]Loading knowledge graph...[/bold]")
    graph_file = find_graph_file(graph_dir)
    graph_env = GraphEnvironment(
        graph_path=graph_file,
        faiss_cache_dir=faiss_cache,
    )
    console.print(
        f"  Nodes indexed: [green]{len(graph_env.graph_index):,}[/green]"
    )

    # ------------------------------------------------------------------
    # Load QA pairs and split into train/val
    # ------------------------------------------------------------------
    console.print("\n[bold]Loading GRBench QA pairs...[/bold]")
    # Load enough data for both train and val
    qa_pairs = load_grbench_qa(
        domain="healthcare",
        max_samples=None,  # load all, then split
        level_filter=level,
    )
    console.print(f"  Total questions loaded: [green]{len(qa_pairs)}[/green]")

    # Split: use train_size for training, rest for validation
    train_frac = min(train_size / len(qa_pairs), 0.5) if qa_pairs else 0.2
    trainset, valset = train_val_split(qa_pairs, train_frac=train_frac)

    # Cap trainset to requested size
    if len(trainset) > train_size:
        trainset = trainset[:train_size]

    console.print(
        f"  Train: [green]{len(trainset)}[/green] examples, "
        f"Val: [green]{len(valset)}[/green] examples"
    )

    # ------------------------------------------------------------------
    # Build solver
    # ------------------------------------------------------------------
    from src.graph_tot.dspy_modules import GraphToTSolver

    solver = GraphToTSolver(
        graph_env=graph_env,
        k=k,
        b=b,
        max_rounds=max_rounds,
        max_iters=max_iters,
        eval_mode=eval_mode,
        parallel=parallel,
    )

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    from src.graph_tot.optimize import compile_solver

    optimizer_kwargs = {
        "max_bootstrapped_demos": max_bootstrapped_demos,
        "max_labeled_demos": max_labeled_demos,
    }
    if optimizer == "MIPROv2":
        optimizer_kwargs["auto"] = auto

    console.print(
        f"\n[bold]Compiling with {optimizer}...[/bold]\n"
        f"  max_bootstrapped_demos={max_bootstrapped_demos}, "
        f"max_labeled_demos={max_labeled_demos}"
    )
    if optimizer == "MIPROv2":
        console.print(f"  auto={auto}")
    console.print()

    compiled = compile_solver(
        solver=solver,
        trainset=trainset,
        optimizer_class=optimizer,
        save_path=save_path,
        **optimizer_kwargs,
    )

    console.print(f"\n[green]Compilation complete.[/green]")
    console.print(f"  Compiled state saved to: [bold]{save_path}[/bold]")

    # ------------------------------------------------------------------
    # Evaluate compiled solver on validation set
    # ------------------------------------------------------------------
    if eval_compiled and valset:
        console.print(
            f"\n[bold]Evaluating compiled solver on {len(valset)} validation examples...[/bold]\n"
        )
        predictions: list[str] = []
        from src.graph_tot.data_loader import QAPair

        # Convert dspy.Examples back to QAPairs for RougeEvaluator
        val_qa_pairs: list[QAPair] = []
        for i, ex in enumerate(valset):
            console.print(f"  [dim][{i + 1}/{len(valset)}][/dim] {ex.question[:90]}")
            pred = compiled(question=ex.question)
            predictions.append(pred.answer)
            val_qa_pairs.append(QAPair(
                qid=str(i),
                question=ex.question,
                answer=ex.answer,
                level="unknown",
            ))
            console.print(f"       -> {pred.answer[:80]}")

        from src.graph_tot.evaluate import RougeEvaluator

        evaluator = RougeEvaluator()
        report = evaluator.evaluate(val_qa_pairs, predictions)

        console.print()
        console.print(Panel(
            report.summary(),
            title="[green]Compiled Solver — Validation Results (ROUGE-L)[/green]",
            border_style="green",
        ))
    else:
        console.print(
            "\n[dim]Skipping validation evaluation "
            "(use --eval-compiled to enable).[/dim]"
        )

    console.print(
        f"\n[cyan]To use the compiled solver for inference:[/cyan]\n"
        f"  uv run python main.py --demo --compiled {save_path}\n"
    )


if __name__ == "__main__":
    main()
