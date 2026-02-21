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
