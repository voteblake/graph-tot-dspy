"""Tests for async branch generation in GraphToTSolver."""

import asyncio
import time
from unittest.mock import MagicMock, patch, AsyncMock

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


class TestAsyncModeInit:
    """GraphToTSolver stores and respects the async_mode flag."""

    def test_default_async_mode_false(self):
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=2)
        assert solver.async_mode is False

    def test_explicit_async_mode_true(self):
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=2, async_mode=True)
        assert solver.async_mode is True


class TestGenerateBranchesAsync:
    """Tests for _generate_branches_async method."""

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_async_returns_correct_count(self, mock_init, mock_forward):
        mock_forward.return_value = _make_prediction("ans")

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3)

        branches = await solver._generate_branches_async("q?", ["", "", ""])
        assert len(branches) == 3
        assert all(isinstance(b, Branch) for b in branches)

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_async_preserves_context(self, mock_init, mock_forward):
        mock_forward.return_value = _make_prediction("ans")

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3)

        contexts = ["ctx_a", "ctx_b", "ctx_c"]
        branches = await solver._generate_branches_async("q?", contexts)
        assert [b.parent_context for b in branches] == contexts

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "forward")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_async_distinct_answers(self, mock_init, mock_forward):
        answers = iter(["alpha", "beta", "gamma"])
        mock_forward.side_effect = lambda **kw: _make_prediction(next(answers))

        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3)

        branches = await solver._generate_branches_async("q?", ["", "", ""])
        branch_answers = [b.answer for b in branches]
        assert len(set(branch_answers)) == 3

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_async_uses_acall_when_available(self, mock_init):
        """When agent has acall method, it should be used for async calls."""
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3)

        # Create a mock agent with acall
        with patch("src.graph_tot.dspy_modules.GraphToTAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.acall = AsyncMock(return_value=_make_prediction("async ans"))
            mock_instance.get_trajectory_text.return_value = "trace"
            MockAgent.return_value = mock_instance

            branch = await solver._generate_single_branch_async("q?", "ctx")

            mock_instance.acall.assert_called_once_with(question="q?", context="ctx")
            assert branch.answer == "async ans"

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_async_falls_back_to_sync_without_acall(self, mock_init):
        """When agent lacks acall method, fall back to sync forward."""
        env = _make_fake_graph_env()
        solver = GraphToTSolver(graph_env=env, k=3)

        with patch("src.graph_tot.dspy_modules.GraphToTAgent") as MockAgent:
            mock_instance = MagicMock()
            # No acall attribute
            del mock_instance.acall
            mock_instance.return_value = _make_prediction("sync ans")
            mock_instance.get_trajectory_text.return_value = "trace"
            MockAgent.return_value = mock_instance

            branch = await solver._generate_single_branch_async("q?", "ctx")

            mock_instance.assert_called_once_with(question="q?", context="ctx")
            assert branch.answer == "sync ans"


class TestAsyncConcurrency:
    """Verify that async branches actually run concurrently."""

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_async_is_concurrent(self, mock_init):
        """Branches with a sleep should complete in ~1× sleep, not k× sleep."""
        delay = 0.3

        async def slow_forward(**kwargs):
            await asyncio.sleep(delay)
            return _make_prediction("done")

        with patch.object(GraphToTAgent, "forward"):
            with patch("src.graph_tot.dspy_modules.GraphToTAgent") as MockAgent:
                mock_instance = MagicMock()
                mock_instance.acall = AsyncMock(side_effect=slow_forward)
                mock_instance.get_trajectory_text.return_value = "trace"
                MockAgent.return_value = mock_instance

                env = _make_fake_graph_env()
                solver = GraphToTSolver(graph_env=env, k=3)

                start = time.monotonic()
                branches = await solver._generate_branches_async("q?", ["", "", ""])
                elapsed = time.monotonic() - start

        assert len(branches) == 3
        # If truly concurrent, elapsed should be ~delay, not ~3*delay.
        # Use 1.5× delay as generous upper bound.
        assert elapsed < delay * 1.5, (
            f"Expected async execution in <{delay * 1.5:.1f}s, took {elapsed:.2f}s"
        )


class TestForwardAsync:
    """Tests for the forward_async method."""

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_forward_async_returns_prediction(self, mock_init):
        """forward_async should return a dspy.Prediction with expected fields."""
        env = _make_fake_graph_env()

        with patch("src.graph_tot.dspy_modules.GraphToTAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.acall = AsyncMock(return_value=_make_prediction("test answer"))
            mock_instance.get_trajectory_text.return_value = "test trace"
            MockAgent.return_value = mock_instance

            solver = GraphToTSolver(graph_env=env, k=2, b=1, max_rounds=1)
            result = await solver.forward_async("test question?")

        assert isinstance(result, dspy.Prediction)
        assert hasattr(result, "answer")
        assert hasattr(result, "best_trace")
        assert hasattr(result, "best_score")
        assert hasattr(result, "all_branches")

    @pytest.mark.asyncio
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    async def test_forward_async_respects_k(self, mock_init):
        """forward_async should generate k branches."""
        env = _make_fake_graph_env()

        with patch("src.graph_tot.dspy_modules.GraphToTAgent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.acall = AsyncMock(return_value=_make_prediction("ans"))
            mock_instance.get_trajectory_text.return_value = "trace"
            MockAgent.return_value = mock_instance

            solver = GraphToTSolver(graph_env=env, k=5, b=1, max_rounds=1)
            result = await solver.forward_async("q?")

        # k=5 branches should be generated
        assert len(result.all_branches) == 5


class TestBackwardCompatibility:
    """Ensure sync methods still work after adding async support."""

    def test_sync_forward_still_works(self):
        """The sync forward() method should still work unchanged."""
        env = _make_fake_graph_env()

        with patch.object(GraphToTAgent, "forward") as mock_forward:
            mock_forward.return_value = _make_prediction("sync answer")

            solver = GraphToTSolver(graph_env=env, k=2, b=1, max_rounds=1, parallel=False)
            result = solver.forward("test question?")

        assert isinstance(result, dspy.Prediction)
        assert result.answer == "sync answer"

    def test_parallel_forward_still_works(self):
        """The sync forward() with parallel=True should still work."""
        env = _make_fake_graph_env()

        with patch.object(GraphToTAgent, "forward") as mock_forward:
            mock_forward.return_value = _make_prediction("parallel answer")

            solver = GraphToTSolver(graph_env=env, k=2, b=1, max_rounds=1, parallel=True)
            result = solver.forward("test question?")

        assert isinstance(result, dspy.Prediction)
        assert result.answer == "parallel answer"
