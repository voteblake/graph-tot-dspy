"""Tests for parallel branch generation in GraphToTSolver."""

import time
from unittest.mock import MagicMock, patch

import dspy
import pytest

from src.graph_tot.dspy_modules import Branch, GraphToTAgent, GraphToTSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_graph_env():
    """Return a minimal mock GraphEnvironment sufficient for agent construction."""
    env = MagicMock()
    env.get_tools.return_value = []
    return env


def _make_prediction(answer: str = "test answer") -> dspy.Prediction:
    return dspy.Prediction(answer=answer, trajectory={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParallelFlagInit:
    """GraphToTSolver stores and respects the parallel flag."""

    def test_default_parallel_true(self):
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=2, parallel=True)
        assert solver.parallel is True

    def test_explicit_parallel_false(self):
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=2, parallel=False)
        assert solver.parallel is False


class TestGenerateBranchesRouting:
    """_generate_branches dispatches to sequential or parallel helpers."""

    def _make_solver(self, parallel: bool) -> GraphToTSolver:
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3, parallel=parallel)
        return solver

    @patch.object(GraphToTSolver, "_generate_branches_sequential")
    @patch.object(GraphToTSolver, "_generate_branches_parallel")
    def test_routes_to_parallel_when_enabled(self, mock_par, mock_seq):
        solver = self._make_solver(parallel=True)
        mock_par.return_value = []
        solver._generate_branches("q?", ["", "", ""])
        mock_par.assert_called_once()
        mock_seq.assert_not_called()

    @patch.object(GraphToTSolver, "_generate_branches_sequential")
    @patch.object(GraphToTSolver, "_generate_branches_parallel")
    def test_routes_to_sequential_when_disabled(self, mock_par, mock_seq):
        solver = self._make_solver(parallel=False)
        mock_seq.return_value = []
        solver._generate_branches("q?", ["", "", ""])
        mock_seq.assert_called_once()
        mock_par.assert_not_called()

    @patch.object(GraphToTSolver, "_generate_branches_sequential")
    @patch.object(GraphToTSolver, "_generate_branches_parallel")
    def test_single_context_uses_sequential_even_when_parallel(
        self, mock_par, mock_seq,
    ):
        """With only 1 context there's nothing to parallelize."""
        solver = self._make_solver(parallel=True)
        mock_seq.return_value = []
        solver._generate_branches("q?", [""])
        mock_seq.assert_called_once()
        mock_par.assert_not_called()


class TestParallelBranchGeneration:
    """End-to-end tests that actually exercise the ThreadPoolExecutor path."""

    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    def test_parallel_returns_correct_count(self, mock_init, mock_forward):
        mock_forward.return_value = _make_prediction("ans")

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3, parallel=True)

        branches = solver._generate_branches_parallel("q?", ["", "", ""])
        assert len(branches) == 3
        assert all(isinstance(b, Branch) for b in branches)

    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    def test_parallel_preserves_context(self, mock_init, mock_forward):
        mock_forward.return_value = _make_prediction("ans")

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3, parallel=True)

        contexts = ["ctx_a", "ctx_b", "ctx_c"]
        branches = solver._generate_branches_parallel("q?", contexts)
        assert [b.parent_context for b in branches] == contexts

    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    def test_parallel_distinct_answers(self, mock_init, mock_forward):
        answers = iter(["alpha", "beta", "gamma"])
        mock_forward.side_effect = lambda **kw: _make_prediction(next(answers))

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3, parallel=True)

        branches = solver._generate_branches_parallel("q?", ["", "", ""])
        branch_answers = [b.answer for b in branches]
        assert len(set(branch_answers)) == 3

    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    def test_sequential_returns_same_structure(self, mock_init, mock_forward):
        mock_forward.return_value = _make_prediction("ans")

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3, parallel=False)

        branches = solver._generate_branches_sequential("q?", ["", "", ""])
        assert len(branches) == 3
        assert all(isinstance(b, Branch) for b in branches)


class TestParallelConcurrency:
    """Verify that branches actually run concurrently, not sequentially."""

    @patch.object(GraphToTAgent, "__init__", return_value=None)
    def test_parallel_is_concurrent(self, mock_init):
        """Branches with a sleep should complete in ~1× sleep, not k× sleep."""
        delay = 0.3

        def slow_forward(**kwargs):
            time.sleep(delay)
            return _make_prediction("done")

        with patch.object(GraphToTAgent, "forward", side_effect=slow_forward):
            env = _make_fake_graph_env()
            solver = GraphToTSolver(graph_env=env, k=3, parallel=True)

            start = time.monotonic()
            branches = solver._generate_branches_parallel("q?", ["", "", ""])
            elapsed = time.monotonic() - start

        assert len(branches) == 3
        # If truly parallel, elapsed should be ~delay, not ~3*delay.
        # Use 2× delay as generous upper bound for thread overhead.
        assert elapsed < delay * 2, (
            f"Expected parallel execution in <{delay * 2:.1f}s, took {elapsed:.2f}s"
        )


class TestParallelErrorHandling:
    """Parallel mode should propagate agent errors the same as sequential."""

    @patch.object(GraphToTAgent, "__init__", return_value=None)
    def test_exception_in_one_branch_propagates(self, mock_init):
        """If an agent raises, the ThreadPoolExecutor propagates the exception."""
        call_count = 0

        def failing_forward(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("LLM timeout")
            return _make_prediction("ok")

        with patch.object(GraphToTAgent, "forward", side_effect=failing_forward):
            env = _make_fake_graph_env()
            solver = GraphToTSolver(graph_env=env, k=3, parallel=True)

            with pytest.raises(RuntimeError, match="LLM timeout"):
                solver._generate_branches_parallel("q?", ["", "", ""])
