"""
Tests for rate-limit retry behavior in sequential and parallel branch generation.

Uses patch("src.graph_tot.dspy_modules.time.sleep") to avoid actual delays.
litellm.RateLimitError is constructed with the required keyword arguments.
"""

from unittest.mock import MagicMock, patch

import dspy
import litellm

from src.graph_tot.dspy_modules import GraphToTAgent, GraphToTSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rate_limit_error():
    return litellm.RateLimitError(
        message="Rate limit exceeded",
        llm_provider="anthropic",
        model="claude-sonnet-4-6",
    )


def _make_fake_env():
    env = MagicMock()
    env.get_tools.return_value = []
    return env


def _make_prediction(answer="retry_answer"):
    return dspy.Prediction(answer=answer, trajectory={})


# ---------------------------------------------------------------------------
# Sequential retry (_generate_branches_sequential)
# ---------------------------------------------------------------------------


class TestSequentialRateLimitRetry:

    @patch("src.graph_tot.dspy_modules.time.sleep")
    def test_succeeds_after_one_rate_limit_error(self, mock_sleep):
        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=False)
        solver.agent = MagicMock()
        solver.agent.side_effect = [_make_rate_limit_error(), _make_prediction("success")]
        solver.agent.get_trajectory_text = MagicMock(return_value="")

        branches = solver._generate_branches_sequential("Q?", [""])

        assert len(branches) == 1
        assert branches[0].answer == "success"

    @patch("src.graph_tot.dspy_modules.time.sleep")
    def test_exhausted_retries_returns_error_branch_not_raise(self, mock_sleep):
        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=False)
        solver.agent = MagicMock()
        solver.agent.side_effect = _make_rate_limit_error()

        branches = solver._generate_branches_sequential("Q?", [""])

        assert len(branches) == 1
        assert "Rate limit" in branches[0].answer

    @patch("src.graph_tot.dspy_modules.time.sleep")
    def test_first_retry_sleeps_60s(self, mock_sleep):
        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=False)
        solver.agent = MagicMock()
        solver.agent.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_prediction("ok"),
        ]
        solver.agent.get_trajectory_text = MagicMock(return_value="")

        solver._generate_branches_sequential("Q?", [""])

        # First sleep: attempt 0 → 60*(0+1) = 60s
        mock_sleep.assert_any_call(60)

    @patch("src.graph_tot.dspy_modules.time.sleep")
    def test_second_retry_sleeps_120s(self, mock_sleep):
        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=False)
        solver.agent = MagicMock()
        solver.agent.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_prediction("ok"),
        ]
        solver.agent.get_trajectory_text = MagicMock(return_value="")

        solver._generate_branches_sequential("Q?", [""])

        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [60, 120]


# ---------------------------------------------------------------------------
# Parallel retry (_generate_single_branch)
# ---------------------------------------------------------------------------


class TestParallelRateLimitRetry:

    @patch("src.graph_tot.dspy_modules.time.sleep")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    @patch.object(GraphToTAgent, "forward")
    def test_single_branch_succeeds_after_one_rate_limit(
        self, mock_forward, mock_init, mock_sleep
    ):
        mock_forward.side_effect = [_make_rate_limit_error(), _make_prediction("parallel_ok")]

        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=True)

        branch = solver._generate_single_branch("Q?", "ctx")

        assert branch.answer == "parallel_ok"

    @patch("src.graph_tot.dspy_modules.time.sleep")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    @patch.object(GraphToTAgent, "forward")
    def test_single_branch_exhausted_returns_error_branch(
        self, mock_forward, mock_init, mock_sleep
    ):
        mock_forward.side_effect = _make_rate_limit_error()

        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=True)

        branch = solver._generate_single_branch("Q?", "ctx")

        assert "Rate limit" in branch.answer

    @patch("src.graph_tot.dspy_modules.time.sleep")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    @patch.object(GraphToTAgent, "forward")
    def test_retry_sleep_sequence_60s_then_120s(
        self, mock_forward, mock_init, mock_sleep
    ):
        mock_forward.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_prediction("done"),
        ]

        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=True)
        solver._generate_single_branch("Q?", "")

        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [60, 120]

    @patch("src.graph_tot.dspy_modules.time.sleep")
    @patch.object(GraphToTAgent, "__init__", return_value=None)
    @patch.object(GraphToTAgent, "forward")
    def test_error_branch_preserves_parent_context(
        self, mock_forward, mock_init, mock_sleep
    ):
        mock_forward.side_effect = _make_rate_limit_error()

        env = _make_fake_env()
        solver = GraphToTSolver(graph_env=env, k=1, parallel=True)

        branch = solver._generate_single_branch("Q?", "my context")

        assert branch.parent_context == "my context"
