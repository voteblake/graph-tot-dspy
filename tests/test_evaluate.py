"""
Tests for RougeEvaluator and EvalReport.

RougeEvaluator wraps the rouge_score library (no LLM) and can be exercised
directly. EvalReport aggregation is pure arithmetic.
"""

import json

import dspy
import pytest

from src.graph_tot.data_loader import QAPair
from src.graph_tot.evaluate import EvalReport, EvalResult, RougeEvaluator, rouge_metric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(qid="1", level="easy", rouge_l=0.5):
    return EvalResult(
        qid=qid,
        question="Q?",
        gold_answer="gold",
        predicted_answer="pred",
        level=level,
        rouge_l=rouge_l,
    )


# ---------------------------------------------------------------------------
# RougeEvaluator
# ---------------------------------------------------------------------------


class TestRougeEvaluator:

    def test_identical_strings_score_1(self):
        ev = RougeEvaluator()
        assert ev.score_single("exact match", "exact match") == pytest.approx(1.0)

    def test_completely_different_strings_score_0(self):
        ev = RougeEvaluator()
        score = ev.score_single("apple", "elephant")
        assert score == pytest.approx(0.0)

    def test_partial_overlap_between_0_and_1(self):
        ev = RougeEvaluator()
        score = ev.score_single("Metformin and Insulin", "Metformin treats diabetes")
        assert 0.0 < score < 1.0

    def test_empty_prediction_scores_0(self):
        ev = RougeEvaluator()
        assert ev.score_single("", "some reference text") == pytest.approx(0.0)

    def test_evaluate_length_mismatch_raises_value_error(self):
        ev = RougeEvaluator()
        pairs = [QAPair("1", "Q?", "A", "easy")]
        with pytest.raises(ValueError, match="Length mismatch"):
            ev.evaluate(pairs, ["pred1", "pred2"])

    def test_evaluate_returns_eval_report(self):
        ev = RougeEvaluator()
        pairs = [QAPair("1", "Q?", "gold", "easy")]
        report = ev.evaluate(pairs, ["gold"])
        assert isinstance(report, EvalReport)
        assert len(report.results) == 1

    def test_evaluate_score_matches_score_single(self):
        ev = RougeEvaluator()
        pairs = [QAPair("1", "Q?", "gold answer", "easy")]
        pred = "gold answer"
        report = ev.evaluate(pairs, [pred])
        expected = ev.score_single(pred, "gold answer")
        assert report.results[0].rouge_l == pytest.approx(expected)


# ---------------------------------------------------------------------------
# EvalReport aggregation
# ---------------------------------------------------------------------------


class TestEvalReport:

    def test_overall_rouge_l_is_mean(self):
        report = EvalReport(results=[
            _result("1", "easy", 0.8),
            _result("2", "easy", 0.6),
            _result("3", "easy", 0.4),
        ])
        assert report.overall_rouge_l == pytest.approx(0.6)

    def test_overall_rouge_l_empty_returns_0(self):
        assert EvalReport().overall_rouge_l == pytest.approx(0.0)

    def test_rouge_l_by_level_groups_correctly(self):
        report = EvalReport(results=[
            _result("1", "easy", 0.8),
            _result("2", "easy", 0.6),
            _result("3", "hard", 0.2),
        ])
        by_level = report.rouge_l_by_level()
        assert by_level["easy"] == pytest.approx(0.7)
        assert by_level["hard"] == pytest.approx(0.2)

    def test_rouge_l_by_level_three_levels(self):
        report = EvalReport(results=[
            _result("1", "easy", 1.0),
            _result("2", "medium", 0.5),
            _result("3", "hard", 0.0),
        ])
        by_level = report.rouge_l_by_level()
        assert set(by_level.keys()) == {"easy", "medium", "hard"}

    def test_count_by_level(self):
        report = EvalReport(results=[
            _result("1", "easy", 0.9),
            _result("2", "easy", 0.8),
            _result("3", "hard", 0.3),
        ])
        counts = report.count_by_level()
        assert counts["easy"] == 2
        assert counts["hard"] == 1

    def test_count_by_level_empty_report(self):
        assert EvalReport().count_by_level() == {}

    def test_summary_contains_scores_and_level(self):
        report = EvalReport(results=[
            _result("1", "easy", 0.75),
            _result("2", "hard", 0.25),
        ])
        s = report.summary()
        # Verify overall score and both levels appear — the summary is the
        # primary human-readable output used after eval runs.
        assert "0.5000" in s  # overall mean of 0.75 and 0.25
        assert "easy" in s
        assert "hard" in s

    def test_save_json_writes_valid_json(self, tmp_path):
        report = EvalReport(results=[
            _result("1", "easy", 0.75),
            _result("2", "hard", 0.25),
        ])
        output = tmp_path / "results.json"
        report.save_json(output)

        with open(output) as f:
            data = json.load(f)

        assert "overall_rouge_l" in data
        assert "by_level" in data
        assert "count_by_level" in data
        assert "results" in data
        assert len(data["results"]) == 2

    def test_save_json_overall_value_correct(self, tmp_path):
        report = EvalReport(results=[
            _result("1", "easy", 0.8),
            _result("2", "easy", 0.4),
        ])
        output = tmp_path / "out.json"
        report.save_json(output)
        with open(output) as f:
            data = json.load(f)
        assert data["overall_rouge_l"] == pytest.approx(0.6)

    def test_save_json_creates_parent_directories(self, tmp_path):
        report = EvalReport(results=[_result()])
        deep_path = tmp_path / "nested" / "dir" / "results.json"
        report.save_json(deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# rouge_metric (DSPy-compatible wrapper)
# ---------------------------------------------------------------------------


class TestRougeMetricFunction:

    def test_exact_match_returns_1(self):
        gold = dspy.Example(answer="metformin")
        pred = dspy.Prediction(answer="metformin")
        assert rouge_metric(gold, pred) == pytest.approx(1.0)

    def test_no_overlap_returns_0(self):
        gold = dspy.Example(answer="metformin")
        pred = dspy.Prediction(answer="rhinoceros")
        assert rouge_metric(gold, pred) == pytest.approx(0.0)

    def test_missing_answer_field_returns_0(self):
        gold = dspy.Example(answer="x")
        pred = dspy.Prediction()  # no answer field
        score = rouge_metric(gold, pred)
        assert score == pytest.approx(0.0)
