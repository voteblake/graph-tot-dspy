"""
Graph Tree-of-Thought with Agent-based retrieval — CLI entry point.

Based on arXiv:2502.13247 "Grounding LLM Reasoning with Knowledge Graphs".
Uses GRBench Healthcare knowledge graph and Anthropic Claude via DSPy.

Quick start:
  1. Set ANTHROPIC_API_KEY in .env or environment
  2. Download graph data (see --help for instructions)
  3. uv run python main.py --demo
"""

import logging
import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

console = Console()

DEFAULT_GRAPH_DIR = "./data/healthcare"
DEFAULT_FAISS_CACHE = "./data/cache"
DEFAULT_OUTPUT = "./results/evaluation.json"


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def setup_dspy(model: str, temperature: float, max_tokens: int) -> None:
    """Configure DSPy with an Anthropic Claude language model."""
    import dspy

    load_dotenv()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print(
            "[red]Error: ANTHROPIC_API_KEY is not set.[/red]\n"
            "Add it to a .env file:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "or export it in your shell."
        )
        sys.exit(1)

    lm = dspy.LM(
        model=f"anthropic/{model}",
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=temperature == 0.0,  # disable cache when sampling for branch diversity
        cache_control_injection_points=[{
            # Allow for provider side prompt caching for ReAct prompts
            "location": "message",
            "role": "system",
        }],
    )
    dspy.configure(lm=lm)
    console.print(f"[green]DSPy configured:[/green] anthropic/{model} (temp={temperature})")


@click.command()
@click.option(
    "--graph-dir", default=DEFAULT_GRAPH_DIR, show_default=True,
    help="Directory containing downloaded GRBench healthcare graph file.",
)
@click.option(
    "--faiss-cache", default=DEFAULT_FAISS_CACHE, show_default=True,
    help="Directory for FAISS index cache (built once, reused on subsequent runs).",
)
@click.option(
    "--output", default=DEFAULT_OUTPUT, show_default=True,
    help="Path to save evaluation results JSON.",
)
@click.option(
    "--model", default="claude-3-5-haiku-20241022", show_default=True,
    help="Anthropic model ID (without 'anthropic/' prefix).",
)
@click.option(
    "--temperature", default=0.7, type=float, show_default=True,
    help="LLM sampling temperature. Use >=0.7 for branch diversity.",
)
@click.option(
    "--max-tokens", default=2048, type=int, show_default=True,
    help="Max tokens per LLM call. Scorer calls include full trajectories so "
         "needs to be higher than you might expect — 2048 is a safe default.",
)
@click.option(
    "--k", default=3, type=int, show_default=True,
    help="ToT branching factor: number of independent agent traces per round.",
)
@click.option(
    "--b", default=1, type=int, show_default=True,
    help="ToT beam width: number of branches kept after scoring.",
)
@click.option(
    "--max-rounds", default=1, type=int, show_default=True,
    help="Number of ToT beam search rounds (1 = single-pass).",
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
    help="Branch evaluation method: independent scoring or single selection call.",
)
@click.option(
    "--max-samples", default=5, type=int, show_default=True,
    help="Max questions to evaluate (0 = all 270 healthcare questions).",
)
@click.option(
    "--level",
    default=None,
    type=click.Choice(["easy", "medium", "hard"]),
    help="Filter to questions of a specific difficulty level.",
)
@click.option(
    "--demo", is_flag=True, default=False,
    help="Run a single demonstration question with verbose output.",
)
@click.option(
    "--parallel/--no-parallel", default=True, show_default=True,
    help="Generate k branches in parallel using threads (faster) or sequentially.",
)
@click.option(
    "--verbose", is_flag=True, default=False,
    help="Enable debug-level logging.",
)
def main(
    graph_dir, faiss_cache, output, model, temperature, max_tokens,
    k, b, max_rounds, max_iters, eval_mode,
    max_samples, level, demo, parallel, verbose,
):
    """
    Graph Tree-of-Thought QA over the GRBench Healthcare knowledge graph.

    \b
    The system implements the Graph ToT + Agent approach from arXiv:2502.13247:
      1. A ReAct agent explores the knowledge graph via 4 tools
      2. k independent traces are generated per round (branching)
      3. Branches are scored and pruned to beam width b
      4. The best trace is used to synthesise the final answer
    """
    setup_logging(verbose)

    console.print(Panel.fit(
        "[bold blue]Graph Tree-of-Thought[/bold blue] · DSPy · GRBench Healthcare\n"
        "[dim]arXiv:2502.13247 — Grounding LLM Reasoning with Knowledge Graphs[/dim]",
        border_style="blue",
    ))

    # ------------------------------------------------------------------
    # Pre-flight: check graph data before doing anything expensive
    # ------------------------------------------------------------------
    from src.graph_tot.data_loader import (
        check_graph_available,
        find_graph_file,
        load_grbench_qa,
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
        console.print(
            "\n[cyan]Once downloaded, re-run:[/cyan]\n"
            f"  uv run python main.py --demo --graph-dir {graph_dir}\n"
        )
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
    # Load QA pairs from HuggingFace
    # ------------------------------------------------------------------
    console.print("\n[bold]Loading GRBench QA pairs...[/bold]")
    n = None if max_samples == 0 else (1 if demo else max_samples)
    qa_pairs = load_grbench_qa(
        domain="healthcare",
        max_samples=n,
        level_filter=level,
    )
    console.print(f"  Questions loaded: [green]{len(qa_pairs)}[/green]")

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
    # Run inference
    # ------------------------------------------------------------------
    par_str = "parallel" if parallel else "sequential"
    console.print(
        f"\n[bold]Running Graph ToT inference[/bold] "
        f"(k={k}, b={b}, rounds={max_rounds}, eval={eval_mode}, {par_str})...\n"
    )

    predictions: list[str] = []
    for i, qa in enumerate(qa_pairs):
        console.print(f"  [dim][{i + 1}/{len(qa_pairs)}][/dim] {qa.question[:90]}")
        pred = solver(question=qa.question)
        predictions.append(pred.answer)

        if demo:
            console.print(f"\n  [bold]Predicted:[/bold] {pred.answer}")
            console.print(f"  [bold]Gold:     [/bold] {qa.answer}")
            console.print(f"  [bold]Difficulty:[/bold] {qa.level}")
            console.print(f"  [bold]Best branch score:[/bold] {pred.best_score:.3f}")

            # Show per-round progression for multi-round runs
            round_log = getattr(pred, "round_log", []) or []
            if len(round_log) > 1:
                console.print(f"\n  [bold]Round-by-round progression:[/bold]")
                for rl in round_log:
                    ctx = rl.get("context_from_previous")
                    ctx_str = f"  context=\"{ctx}...\"" if ctx else ""
                    console.print(
                        f"    Round {rl['round']}: "
                        f"{rl['num_branches']} branches, "
                        f"scores={[f'{s:.3f}' for s in rl['scores']]}"
                        f"{ctx_str}"
                    )
                    for ans in rl.get("survivor_answers", []):
                        console.print(f"      → survivor: {ans}")

            if pred.all_branches:
                console.print(f"\n  [bold]All {len(pred.all_branches)} branches (final round):[/bold]")
                for idx, br in enumerate(pred.all_branches):
                    console.print(
                        f"    [{idx}] score={br['score']:.3f}  "
                        f"answer={br['answer'][:80]}"
                    )

            if pred.best_trace:
                console.print(Panel(
                    pred.best_trace[:1500],
                    title="[cyan]Best Branch Trace[/cyan]",
                    border_style="cyan",
                ))
        else:
            console.print(f"       -> {pred.answer[:80]}")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    from src.graph_tot.evaluate import RougeEvaluator

    evaluator = RougeEvaluator()
    report = evaluator.evaluate(qa_pairs, predictions)

    console.print()
    console.print(Panel(
        report.summary(),
        title="[green]Evaluation Results (ROUGE-L)[/green]",
        border_style="green",
    ))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if not demo:
        report.save_json(output)
        console.print(f"\nFull results saved to: [bold]{output}[/bold]")


if __name__ == "__main__":
    main()
