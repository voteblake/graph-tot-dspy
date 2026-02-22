"""
Tests for the Branch dataclass.

Branch.as_dict() is part of the observability contract — its output
populates the all_branches field in GraphToTSolver.forward() results.
"""

import json

import pytest

from src.graph_tot.dspy_modules import Branch
from tests.conftest import make_prediction


class TestBranchDataclass:

    def test_defaults(self):
        b = Branch(answer="test", trace="t", prediction=make_prediction())
        assert b.score == 0.0
        assert b.parent_context == ""

    def test_as_dict_values_and_excluded_keys(self):
        """as_dict maps all four serializable fields and excludes prediction."""
        b = Branch(
            answer="Insulin",
            trace="multi-step trace",
            prediction=make_prediction("Insulin"),
            score=0.72,
            parent_context="prior context",
        )
        d = b.as_dict()
        assert d["answer"] == "Insulin"
        assert d["trace"] == "multi-step trace"
        assert d["score"] == pytest.approx(0.72)
        assert d["parent_context"] == "prior context"
        assert "prediction" not in d

    def test_as_dict_is_json_serializable(self):
        """Output must be JSON-serializable since it populates all_branches in solver output."""
        b = Branch(answer="a", trace="t", prediction=make_prediction(), score=0.5, parent_context="ctx")
        json.dumps(b.as_dict())  # must not raise
