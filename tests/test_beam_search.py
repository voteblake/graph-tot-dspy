"""
Tests for GraphToTSolver beam search algorithm invariants.

Corresponds to arXiv:2502.13247 Section 3. All tests patch
_generate_branches and evaluator.forward to inject Branch objects
with known scores — zero LLM calls.

All tests use parallel=False (sequential mode) to keep beam search logic
isolated; parallel dispatch is covered in test_parallel_branches.py.
"""

from unittest.mock import MagicMock, patch

import dspy
import pytest

from src.graph_tot.dspy_modules import Branch, GraphToTSolver
from tests.conftest import make_branch, make_prediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scored_branches(*score_answer_pairs):
    """Build a list of branches pre-sorted by score descending."""
    branches = [make_branch(answer=ans, score=sc) for sc, ans in score_answer_pairs]
    return branches


# ---------------------------------------------------------------------------
# Beam pruning: top-b branches selected
# ---------------------------------------------------------------------------


class TestBeamPruningInvariant:
    """Beam search must return the top-scoring branch's answer."""

    def test_b1_answer_is_top_scoring_branch(self, make_solver):
        solver, _ = make_solver(k=3, b=1)
        branches = _scored_branches((0.9, "best"), (0.5, "mid"), (0.1, "worst"))

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        assert result.answer == "best"

    def test_b2_answer_is_still_rank1_branch(self, make_solver):
        solver, _ = make_solver(k=3, b=2)
        branches = _scored_branches((0.9, "best"), (0.6, "second"), (0.1, "worst"))

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        assert result.answer == "best"

    def test_pruned_answer_not_in_result(self, make_solver):
        solver, _ = make_solver(k=3, b=1)
        branches = _scored_branches((0.9, "correct answer"), (0.1, "wrong answer"))

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        assert "wrong answer" not in result.answer


# ---------------------------------------------------------------------------
# Expansion: round 0 seeds with k empty contexts; each survivor generates k
# ---------------------------------------------------------------------------


class TestExpansionInvariant:
    """Branch generation call counts and context shapes must match the algorithm."""

    def test_round0_seeds_k_empty_contexts(self, make_solver):
        solver, _ = make_solver(k=3, b=1)
        branches = [make_branch(answer="a", score=0.5)]
        generate_calls = []

        def track(question, contexts):
            generate_calls.append(list(contexts))
            return branches

        with patch.object(solver, "_generate_branches", side_effect=track):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                solver.forward("Q?")

        assert generate_calls[0] == ["", "", ""]

    def test_single_survivor_generates_k_new_branches(self, make_solver):
        solver, _ = make_solver(k=3, b=1, max_rounds=2)
        survivor = make_branch(answer="survivor", score=0.9)
        filler = make_branch(answer="x", score=0.1)
        generate_calls = []

        def track(question, contexts):
            generate_calls.append(list(contexts))
            return [survivor, filler] if len(generate_calls) == 1 else [survivor]

        with patch.object(solver, "_generate_branches", side_effect=track):
            with patch.object(solver.evaluator, "forward", return_value=[survivor, filler]):
                solver.forward("Q?")

        # Second call (expansion of b=1 survivor) must use k=3 contexts
        assert len(generate_calls) == 2
        assert len(generate_calls[1]) == 3

    def test_two_survivors_each_call_generate_with_k_contexts(self, make_solver):
        solver, _ = make_solver(k=3, b=2, max_rounds=2)
        survivors = [make_branch(answer=f"s{i}", score=0.9 - i * 0.1) for i in range(2)]
        filler = make_branch(answer="loser", score=0.1)
        generate_calls = []

        def track(question, contexts):
            generate_calls.append(list(contexts))
            return survivors + [filler]

        with patch.object(solver, "_generate_branches", side_effect=track):
            with patch.object(solver.evaluator, "forward", return_value=survivors + [filler]):
                solver.forward("Q?")

        # Round 0 + 2 expansion calls (one per survivor)
        expansion_calls = generate_calls[1:]
        assert len(expansion_calls) == 2
        for ctx_list in expansion_calls:
            assert len(ctx_list) == 3  # k=3


# ---------------------------------------------------------------------------
# Context propagation: survivor answer (truncated to 800) passed to round 2
# ---------------------------------------------------------------------------


class TestContextPropagationInvariant:

    def test_survivor_answer_truncated_to_800_chars(self, make_solver):
        solver, _ = make_solver(k=2, b=1, max_rounds=2)
        long_answer = "x" * 1200
        survivor = make_branch(answer=long_answer, score=0.9)
        generate_calls = []

        def track(question, contexts):
            generate_calls.append(list(contexts))
            return [survivor]

        with patch.object(solver, "_generate_branches", side_effect=track):
            with patch.object(solver.evaluator, "forward", return_value=[survivor]):
                solver.forward("Q?")

        round1_contexts = generate_calls[1]
        assert all(ctx == long_answer[:800] for ctx in round1_contexts)
        assert all(len(ctx) == 800 for ctx in round1_contexts)

    def test_short_answer_passed_verbatim_as_context(self, make_solver):
        solver, _ = make_solver(k=2, b=1, max_rounds=2)
        short = "Metformin and Insulin"
        survivor = make_branch(answer=short, score=0.9)
        generate_calls = []

        def track(question, contexts):
            generate_calls.append(list(contexts))
            return [survivor]

        with patch.object(solver, "_generate_branches", side_effect=track):
            with patch.object(solver.evaluator, "forward", return_value=[survivor]):
                solver.forward("Q?")

        assert all(ctx == short for ctx in generate_calls[1])

    def test_round0_always_uses_empty_string_contexts(self, make_solver):
        solver, _ = make_solver(k=4, b=1)
        branch = make_branch(answer="a", score=0.5)
        generate_calls = []

        def track(question, contexts):
            generate_calls.append(list(contexts))
            return [branch]

        with patch.object(solver, "_generate_branches", side_effect=track):
            with patch.object(solver.evaluator, "forward", return_value=[branch]):
                solver.forward("Q?")

        assert generate_calls[0] == ["", "", "", ""]


# ---------------------------------------------------------------------------
# Single-round termination: evaluator and generate_branches called exactly once
# ---------------------------------------------------------------------------


class TestSingleRoundTermination:

    def test_evaluator_called_exactly_once_max_rounds_1(self, make_solver):
        solver, _ = make_solver(k=3, b=1, max_rounds=1)
        branches = [make_branch(answer="a", score=0.5)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches) as mock_eval:
                solver.forward("Q?")

        mock_eval.assert_called_once()

    def test_generate_branches_called_once_max_rounds_1(self, make_solver):
        solver, _ = make_solver(k=3, b=1, max_rounds=1)
        branches = [make_branch(answer="a", score=0.5)]

        with patch.object(solver, "_generate_branches", return_value=branches) as mock_gen:
            with patch.object(solver.evaluator, "forward", return_value=branches):
                solver.forward("Q?")

        mock_gen.assert_called_once()

    def test_generate_branches_called_twice_max_rounds_2_b1(self, make_solver):
        solver, _ = make_solver(k=3, b=1, max_rounds=2)
        branches = [make_branch(answer="a", score=0.9)]

        with patch.object(solver, "_generate_branches", return_value=branches) as mock_gen:
            with patch.object(solver.evaluator, "forward", return_value=branches):
                solver.forward("Q?")

        # Round 0 seed + 1 expansion = 2 total
        assert mock_gen.call_count == 2

    def test_evaluator_called_once_per_round(self, make_solver):
        solver, _ = make_solver(k=2, b=1, max_rounds=3)
        branches = [make_branch(answer="a", score=0.9)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches) as mock_eval:
                solver.forward("Q?")

        assert mock_eval.call_count == 3


# ---------------------------------------------------------------------------
# No synthesis: best branch answer/trace returned verbatim
# ---------------------------------------------------------------------------


class TestNoBestBranchSynthesis:
    """Result fields must exactly match the top-scoring branch — no modification."""

    def test_result_answer_equals_branch_answer_verbatim(self, make_solver):
        solver, _ = make_solver(k=2, b=1)
        expected = "Metformin, Insulin, Glipizide"
        best = make_branch(answer=expected, score=0.95)
        other = make_branch(answer="other", score=0.1)

        with patch.object(solver, "_generate_branches", return_value=[best, other]):
            with patch.object(solver.evaluator, "forward", return_value=[best, other]):
                result = solver.forward("Q?")

        assert result.answer == expected

    def test_result_best_trace_equals_branch_trace_verbatim(self, make_solver):
        solver, _ = make_solver(k=1, b=1)
        trace = "Thought 1: found node\nAction  1: retrieve_node()\nObs     1: Disease::D001"
        branch = make_branch(answer="ans", score=0.8, trace=trace)

        with patch.object(solver, "_generate_branches", return_value=[branch]):
            with patch.object(solver.evaluator, "forward", return_value=[branch]):
                result = solver.forward("Q?")

        assert result.best_trace == trace

    def test_empty_branches_returns_nonempty_fallback(self, make_solver):
        solver, _ = make_solver(k=1, b=1)

        with patch.object(solver, "_generate_branches", return_value=[]):
            with patch.object(solver.evaluator, "forward", return_value=[]):
                result = solver.forward("Q?")

        assert isinstance(result.answer, str)
        assert len(result.answer) > 0


# ---------------------------------------------------------------------------
# Output schema: all expected fields present with correct types
# ---------------------------------------------------------------------------


class TestForwardOutputSchema:

    def test_output_has_all_expected_fields(self, make_solver):
        solver, _ = make_solver(k=2, b=1)
        branch = make_branch(answer="ans", score=0.8)

        with patch.object(solver, "_generate_branches", return_value=[branch]):
            with patch.object(solver.evaluator, "forward", return_value=[branch]):
                result = solver.forward("Q?")

        assert hasattr(result, "answer")
        assert hasattr(result, "best_trace")
        assert hasattr(result, "best_score")
        assert hasattr(result, "all_branches")
        assert hasattr(result, "round_log")

    def test_all_branches_is_list_of_dicts(self, make_solver):
        solver, _ = make_solver(k=2, b=1)
        branch = make_branch(answer="ans", score=0.8)

        with patch.object(solver, "_generate_branches", return_value=[branch]):
            with patch.object(solver.evaluator, "forward", return_value=[branch]):
                result = solver.forward("Q?")

        assert isinstance(result.all_branches, list)
        for item in result.all_branches:
            assert isinstance(item, dict)

    def test_round_log_is_list(self, make_solver):
        solver, _ = make_solver(k=2, b=1)
        branch = make_branch(answer="ans", score=0.5)

        with patch.object(solver, "_generate_branches", return_value=[branch]):
            with patch.object(solver.evaluator, "forward", return_value=[branch]):
                result = solver.forward("Q?")

        assert isinstance(result.round_log, list)

    def test_best_score_matches_top_branch_score(self, make_solver):
        solver, _ = make_solver(k=2, b=1)
        best = make_branch(answer="best", score=0.87)
        other = make_branch(answer="other", score=0.3)

        with patch.object(solver, "_generate_branches", return_value=[best, other]):
            with patch.object(solver.evaluator, "forward", return_value=[best, other]):
                result = solver.forward("Q?")

        assert result.best_score == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# Branch history accumulation: all_branches spans all rounds
# ---------------------------------------------------------------------------


class TestBranchHistoryAccumulation:
    """all_branches must accumulate scored branches across every round."""

    def test_single_round_all_branches_equals_k(self, make_solver):
        """max_rounds=1, k=3 → all_branches has 3 entries."""
        solver, _ = make_solver(k=3, b=1, max_rounds=1)
        branches = [make_branch(answer=f"a{i}", score=0.9 - i * 0.1) for i in range(3)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        assert len(result.all_branches) == 3

    def test_case_a_k3_b1_max_rounds2_yields_6_branches(self, make_solver):
        """k=3, b=1, max_rounds=2 → round1: 3 scored + round2: 3 scored = 6 total."""
        solver, _ = make_solver(k=3, b=1, max_rounds=2)
        round1_branches = [make_branch(answer=f"r1_{i}", score=0.9 - i * 0.1) for i in range(3)]
        round2_branches = [make_branch(answer=f"r2_{i}", score=0.8 - i * 0.1) for i in range(3)]
        gen_call = [0]

        def gen_side(question, contexts):
            gen_call[0] += 1
            return round1_branches if gen_call[0] == 1 else round2_branches

        with patch.object(solver, "_generate_branches", side_effect=gen_side):
            with patch.object(solver.evaluator, "forward", side_effect=[round1_branches, round2_branches]):
                result = solver.forward("Q?")

        assert len(result.all_branches) == 6

    def test_case_b_k3_b2_max_rounds2_yields_9_branches(self, make_solver):
        """k=3, b=2, max_rounds=2 → round1: 3 scored + round2: 6 scored = 9 total."""
        solver, _ = make_solver(k=3, b=2, max_rounds=2)
        r1 = [make_branch(answer=f"r1_{i}", score=0.9 - i * 0.1) for i in range(3)]
        # 2 survivors each expand to k=3 → 6 branches in round 2
        r2 = [make_branch(answer=f"r2_{i}", score=0.8 - i * 0.1) for i in range(6)]
        gen_calls = []

        def gen_side(question, contexts):
            gen_calls.append(contexts)
            if len(gen_calls) == 1:
                return r1
            # Each expansion call returns 3 branches
            offset = (len(gen_calls) - 2) * 3
            return r2[offset: offset + 3]

        with patch.object(solver, "_generate_branches", side_effect=gen_side):
            with patch.object(solver.evaluator, "forward", side_effect=[r1, r2]):
                result = solver.forward("Q?")

        assert len(result.all_branches) == 9

    def test_round_idx_metadata_present(self, make_solver):
        """Each branch dict must contain round_idx."""
        solver, _ = make_solver(k=2, b=1, max_rounds=1)
        branches = [make_branch(answer="a", score=0.9), make_branch(answer="b", score=0.5)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        for d in result.all_branches:
            assert "round_idx" in d
            assert d["round_idx"] == 1

    def test_rank_metadata_present_and_ordered(self, make_solver):
        """rank field must reflect position in the scored list (0-based)."""
        solver, _ = make_solver(k=3, b=1, max_rounds=1)
        branches = [make_branch(answer=f"a{i}", score=0.9 - i * 0.1) for i in range(3)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        ranks = [d["rank"] for d in result.all_branches]
        assert ranks == [0, 1, 2]

    def test_is_survivor_marks_top_b_branches(self, make_solver):
        """is_survivor=True for rank < b, False otherwise."""
        solver, _ = make_solver(k=3, b=1, max_rounds=1)
        branches = [make_branch(answer=f"a{i}", score=0.9 - i * 0.1) for i in range(3)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        survivors = [d for d in result.all_branches if d["is_survivor"]]
        non_survivors = [d for d in result.all_branches if not d["is_survivor"]]
        assert len(survivors) == 1
        assert len(non_survivors) == 2
        assert survivors[0]["rank"] == 0

    def test_is_survivor_b2_marks_two_branches(self, make_solver):
        """b=2 → two branches with is_survivor=True."""
        solver, _ = make_solver(k=3, b=2, max_rounds=1)
        branches = [make_branch(answer=f"a{i}", score=0.9 - i * 0.1) for i in range(3)]

        with patch.object(solver, "_generate_branches", return_value=branches):
            with patch.object(solver.evaluator, "forward", return_value=branches):
                result = solver.forward("Q?")

        survivors = [d for d in result.all_branches if d["is_survivor"]]
        assert len(survivors) == 2

    def test_multi_round_round_idx_values(self, make_solver):
        """Branches from round 1 have round_idx=1, from round 2 have round_idx=2."""
        solver, _ = make_solver(k=2, b=1, max_rounds=2)
        r1 = [make_branch(answer=f"r1_{i}", score=0.9 - i * 0.2) for i in range(2)]
        r2 = [make_branch(answer=f"r2_{i}", score=0.8 - i * 0.2) for i in range(2)]
        gen_calls = [0]

        def gen_side(question, contexts):
            gen_calls[0] += 1
            return r1 if gen_calls[0] == 1 else r2

        with patch.object(solver, "_generate_branches", side_effect=gen_side):
            with patch.object(solver.evaluator, "forward", side_effect=[r1, r2]):
                result = solver.forward("Q?")

        round_indices = [d["round_idx"] for d in result.all_branches]
        assert round_indices == [1, 1, 2, 2]

    def test_existing_fields_preserved(self, make_solver):
        """Backward-compatible: answer, trace, score, parent_context still present."""
        solver, _ = make_solver(k=1, b=1, max_rounds=1)
        branch = make_branch(answer="my answer", score=0.75, trace="t", parent_context="ctx")

        with patch.object(solver, "_generate_branches", return_value=[branch]):
            with patch.object(solver.evaluator, "forward", return_value=[branch]):
                result = solver.forward("Q?")

        d = result.all_branches[0]
        assert d["answer"] == "my answer"
        assert d["score"] == pytest.approx(0.75)
        assert d["trace"] == "t"
        assert d["parent_context"] == "ctx"
