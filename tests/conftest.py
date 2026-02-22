"""Shared fixtures and factories for Graph ToT unit tests."""

from unittest.mock import MagicMock

import dspy
import pytest

from src.graph_tot.dspy_modules import Branch, GraphToTAgent, GraphToTSolver


# ---------------------------------------------------------------------------
# Factories (module-level, usable without fixtures)
# ---------------------------------------------------------------------------


def make_prediction(answer: str = "test answer", trajectory: dict | None = None) -> dspy.Prediction:
    """Create a dspy.Prediction with an optional trajectory dict."""
    return dspy.Prediction(answer=answer, trajectory=trajectory or {})


def make_branch(
    answer: str = "Test answer",
    score: float = 0.0,
    trace: str = "",
    parent_context: str = "",
) -> Branch:
    """Factory for Branch objects — no LLM, no file I/O."""
    return Branch(
        answer=answer,
        trace=trace,
        prediction=make_prediction(answer),
        score=score,
        parent_context=parent_context,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_graph_env():
    """Minimal GraphEnvironment mock — no filesystem, no FAISS, no embeddings."""
    env = MagicMock()
    env.get_tools.return_value = []
    return env


@pytest.fixture
def make_solver(fake_graph_env):
    """
    Factory fixture returning a (solver, mock_agent) pair.

    The solver's self.agent is replaced with a MagicMock so no LLM calls
    are made.  parallel=False by default so beam search tests stay simple.
    """
    def _factory(k=3, b=1, max_rounds=1, parallel=False, eval_mode="score_vote"):
        solver = GraphToTSolver(
            graph_env=fake_graph_env,
            k=k,
            b=b,
            max_rounds=max_rounds,
            parallel=parallel,
            eval_mode=eval_mode,
        )
        mock_agent = MagicMock(spec=GraphToTAgent)
        mock_agent.get_trajectory_text.return_value = ""
        solver.agent = mock_agent
        return solver, mock_agent

    return _factory
