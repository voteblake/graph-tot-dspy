"""
Tests for TreeOfThoughtEvaluator: score_vote and selection_vote modes.

Both scoring paths must:
  - Return branches sorted descending by score
  - Clamp scores to [0, 1]
  - Handle edge cases (empty branches, single branch, out-of-range indices/scores)

No LLM calls: score_voter and selection_voter are replaced with MagicMocks.
"""

from unittest.mock import MagicMock

import dspy
import pytest

from src.graph_tot.dspy_modules import Branch, TreeOfThoughtEvaluator
from tests.conftest import make_branch, make_prediction


# ---------------------------------------------------------------------------
# score_vote mode
# ---------------------------------------------------------------------------


class TestScoreVoteMode:

    def _evaluator(self):
        ev = TreeOfThoughtEvaluator(mode="score_vote")
        return ev

    def test_branches_sorted_descending_by_score(self):
        ev = self._evaluator()
        raw_scores = [0.3, 0.9, 0.6]
        branches = [make_branch(f"a{i}") for i in range(3)]
        ev.score_voter = MagicMock(side_effect=[MagicMock(score=s) for s in raw_scores])

        result = ev.forward("Q?", branches)

        result_scores = [b.score for b in result]
        assert result_scores == sorted(raw_scores, reverse=True)

    def test_score_clamped_to_1_when_above_1(self):
        ev = self._evaluator()
        ev.score_voter = MagicMock(return_value=MagicMock(score=1.7))
        result = ev.forward("Q?", [make_branch("a")])
        assert result[0].score == pytest.approx(1.0)

    def test_score_clamped_to_0_when_below_0(self):
        ev = self._evaluator()
        ev.score_voter = MagicMock(return_value=MagicMock(score=-0.5))
        result = ev.forward("Q?", [make_branch("a")])
        assert result[0].score == pytest.approx(0.0)

    def test_score_at_boundary_0_not_clamped(self):
        ev = self._evaluator()
        ev.score_voter = MagicMock(return_value=MagicMock(score=0.0))
        result = ev.forward("Q?", [make_branch("a")])
        assert result[0].score == pytest.approx(0.0)

    def test_score_at_boundary_1_not_clamped(self):
        ev = self._evaluator()
        ev.score_voter = MagicMock(return_value=MagicMock(score=1.0))
        result = ev.forward("Q?", [make_branch("a")])
        assert result[0].score == pytest.approx(1.0)

    def test_empty_branches_returns_empty(self):
        ev = self._evaluator()
        assert ev.forward("Q?", []) == []

    def test_score_voter_called_once_per_branch(self):
        ev = self._evaluator()
        ev.score_voter = MagicMock(return_value=MagicMock(score=0.5))
        branches = [make_branch(f"a{i}") for i in range(4)]
        ev.forward("Q?", branches)
        assert ev.score_voter.call_count == 4

    def test_original_branch_answer_preserved(self):
        ev = self._evaluator()
        pred = make_prediction("original")
        branch = Branch(answer="original answer", trace="orig trace", prediction=pred)
        ev.score_voter = MagicMock(return_value=MagicMock(score=0.7))

        result = ev.forward("Q?", [branch])

        assert result[0].answer == "original answer"

    def test_original_branch_trace_preserved(self):
        ev = self._evaluator()
        pred = make_prediction("a")
        branch = Branch(answer="a", trace="original trace text", prediction=pred)
        ev.score_voter = MagicMock(return_value=MagicMock(score=0.7))

        result = ev.forward("Q?", [branch])

        assert result[0].trace == "original trace text"

    def test_original_branch_prediction_object_preserved(self):
        ev = self._evaluator()
        pred = make_prediction("a")
        branch = Branch(answer="a", trace="", prediction=pred)
        ev.score_voter = MagicMock(return_value=MagicMock(score=0.7))

        result = ev.forward("Q?", [branch])

        assert result[0].prediction is pred


# ---------------------------------------------------------------------------
# selection_vote mode
# ---------------------------------------------------------------------------


class TestSelectionVoteMode:

    def _evaluator(self):
        ev = TreeOfThoughtEvaluator(mode="selection_vote")
        return ev

    def test_winner_gets_score_1_others_get_0(self):
        ev = self._evaluator()
        branches = [make_branch(f"a{i}") for i in range(3)]
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=1))

        result = ev.forward("Q?", branches)

        winner = next(b for b in result if b.answer == "a1")
        losers = [b for b in result if b.answer != "a1"]
        assert winner.score == pytest.approx(1.0)
        assert all(b.score == pytest.approx(0.0) for b in losers)

    def test_winner_appears_first_after_sorting(self):
        ev = self._evaluator()
        branches = [make_branch(f"a{i}") for i in range(3)]
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=2))

        result = ev.forward("Q?", branches)

        assert result[0].score == pytest.approx(1.0)
        assert result[0].answer == "a2"

    def test_index_clamped_when_above_range(self):
        ev = self._evaluator()
        branches = [make_branch(f"a{i}") for i in range(3)]
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=99))

        result = ev.forward("Q?", branches)

        # Clamped to last valid index = 2
        assert result[0].score == pytest.approx(1.0)
        assert result[0].answer == "a2"

    def test_index_clamped_when_negative(self):
        ev = self._evaluator()
        branches = [make_branch(f"a{i}") for i in range(3)]
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=-5))

        result = ev.forward("Q?", branches)

        assert result[0].score == pytest.approx(1.0)
        assert result[0].answer == "a0"

    def test_selection_voter_called_exactly_once_regardless_of_branch_count(self):
        ev = self._evaluator()
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=0))
        branches = [make_branch(f"a{i}") for i in range(5)]
        ev.forward("Q?", branches)
        assert ev.selection_voter.call_count == 1

    def test_empty_branches_returns_empty(self):
        ev = self._evaluator()
        assert ev.forward("Q?", []) == []


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEvaluatorInit:

    def test_invalid_mode_raises_value_error(self):
        with pytest.raises(ValueError, match="mode must be"):
            TreeOfThoughtEvaluator(mode="bogus_mode")

    def test_score_vote_mode_accepted(self):
        ev = TreeOfThoughtEvaluator(mode="score_vote")
        assert ev.mode == "score_vote"

    def test_selection_vote_mode_accepted(self):
        ev = TreeOfThoughtEvaluator(mode="selection_vote")
        assert ev.mode == "selection_vote"

    def test_default_max_trace_chars_is_2000(self):
        ev = TreeOfThoughtEvaluator(mode="selection_vote")
        assert ev.max_trace_chars_per_candidate == 2000

    def test_custom_max_trace_chars_stored(self):
        ev = TreeOfThoughtEvaluator(mode="selection_vote", max_trace_chars_per_candidate=500)
        assert ev.max_trace_chars_per_candidate == 500


# ---------------------------------------------------------------------------
# selection_vote trace truncation
# ---------------------------------------------------------------------------


class TestSelectionVoteTraceTruncation:
    """_truncate_trace must take the last N chars and warn on truncation."""

    def _evaluator(self, limit: int = 2000):
        return TreeOfThoughtEvaluator(mode="selection_vote", max_trace_chars_per_candidate=limit)

    def test_short_trace_not_truncated(self):
        ev = self._evaluator(limit=100)
        trace = "short trace"
        assert ev._truncate_trace(trace) == trace

    def test_exact_limit_not_truncated(self):
        ev = self._evaluator(limit=10)
        trace = "a" * 10
        assert ev._truncate_trace(trace) == trace

    def test_long_trace_truncated_to_limit(self):
        ev = self._evaluator(limit=50)
        trace = "a" * 200
        result = ev._truncate_trace(trace)
        assert len(result) == 50

    def test_truncation_takes_last_chars(self):
        """Tail is preserved — the final answer lives at the end of the trace."""
        ev = self._evaluator(limit=10)
        trace = "BEGINNING" + "FINAL_TAIL"
        result = ev._truncate_trace(trace)
        assert result == "FINAL_TAIL"

    def test_warning_logged_on_truncation(self, caplog):
        import logging
        ev = self._evaluator(limit=5)
        long_trace = "x" * 100
        with caplog.at_level(logging.WARNING):
            ev._truncate_trace(long_trace)
        assert any("truncated" in record.message.lower() for record in caplog.records)

    def test_no_warning_when_no_truncation(self, caplog):
        import logging
        ev = self._evaluator(limit=200)
        short_trace = "hello"
        with caplog.at_level(logging.WARNING):
            ev._truncate_trace(short_trace)
        assert not any("truncated" in record.message.lower() for record in caplog.records)

    def test_selection_vote_uses_truncated_trace(self):
        """selection_voter candidates string must respect the char limit."""
        ev = self._evaluator(limit=10)
        long_trace = "START" * 100 + "THEEND"
        branch = make_branch(answer="ans", trace=long_trace)
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=0))

        ev.forward("Q?", [branch])

        call_kwargs = ev.selection_voter.call_args
        candidates_str = call_kwargs.kwargs.get("candidates") or call_kwargs.args[1] if call_kwargs.args else ""
        # The trace portion should be at most 10 chars
        assert "START" * 5 not in candidates_str

    def test_truncation_with_k2_branches(self, caplog):
        """Both branches get individually truncated."""
        import logging
        ev = self._evaluator(limit=5)
        b1 = make_branch(answer="a1", trace="x" * 50)
        b2 = make_branch(answer="a2", trace="y" * 50)
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=0))
        with caplog.at_level(logging.WARNING):
            ev.forward("Q?", [b1, b2])
        warnings = [r for r in caplog.records if "truncated" in r.message.lower()]
        assert len(warnings) == 2

    def test_truncation_with_k5_branches(self, caplog):
        """Five long branches each emit a truncation warning."""
        import logging
        ev = self._evaluator(limit=5)
        branches = [make_branch(answer=f"a{i}", trace="z" * 100) for i in range(5)]
        ev.selection_voter = MagicMock(return_value=MagicMock(best_index=0))
        with caplog.at_level(logging.WARNING):
            ev.forward("Q?", branches)
        warnings = [r for r in caplog.records if "truncated" in r.message.lower()]
        assert len(warnings) == 5


# ---------------------------------------------------------------------------
# GraphToTSolver passes max_trace_chars_per_candidate to evaluator
# ---------------------------------------------------------------------------


class TestSolverPassesBudgetToEvaluator:

    def test_default_budget_propagated(self):
        from unittest.mock import MagicMock
        from src.graph_tot.dspy_modules import GraphToTSolver
        env = MagicMock()
        env.get_tools.return_value = []
        solver = GraphToTSolver(graph_env=env)
        assert solver.evaluator.max_trace_chars_per_candidate == 2000

    def test_custom_budget_propagated(self):
        from unittest.mock import MagicMock
        from src.graph_tot.dspy_modules import GraphToTSolver
        env = MagicMock()
        env.get_tools.return_value = []
        solver = GraphToTSolver(graph_env=env, max_trace_chars_per_candidate=500)
        assert solver.evaluator.max_trace_chars_per_candidate == 500
