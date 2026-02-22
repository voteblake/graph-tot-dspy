"""
Tests for the public library API contract (issue #18).

Verifies that:
  - py.typed PEP 561 marker exists in the package directory
  - __all__ is defined and non-empty
  - BranchDict is importable from the package
  - BranchDict has the expected keys
"""

import os


def test_py_typed_exists():
    """py.typed marker must be present in the installed package directory."""
    import graph_tot
    pkg_dir = os.path.dirname(graph_tot.__file__)
    marker = os.path.join(pkg_dir, "py.typed")
    assert os.path.exists(marker), f"py.typed not found at {marker}"


def test_all_defined():
    """Package must expose an explicit __all__ list."""
    import graph_tot
    assert hasattr(graph_tot, "__all__"), "__all__ not defined in graph_tot"
    assert isinstance(graph_tot.__all__, list)
    assert len(graph_tot.__all__) > 0


def test_all_contains_core_symbols():
    """__all__ must include all key public classes and functions."""
    import graph_tot
    expected = {
        "GraphEnvironment",
        "GraphToTSolver",
        "GraphToTAgent",
        "TreeOfThoughtEvaluator",
        "BranchDict",
        "RougeEvaluator",
        "EvalReport",
        "rouge_metric",
        "compile_solver",
        "load_compiled_solver",
    }
    missing = expected - set(graph_tot.__all__)
    assert not missing, f"Missing from __all__: {missing}"


def test_branch_dict_importable():
    """BranchDict must be importable from the top-level package."""
    from graph_tot import BranchDict
    assert BranchDict is not None


def test_branch_dict_has_expected_keys():
    """BranchDict TypedDict must declare the four Branch.as_dict() fields."""
    from graph_tot import BranchDict
    annotations = BranchDict.__annotations__
    for key in ("answer", "trace", "score", "parent_context"):
        assert key in annotations, f"BranchDict missing key: {key}"


def test_branch_as_dict_returns_branch_dict_compatible():
    """Branch.as_dict() output must be compatible with BranchDict structure."""
    from graph_tot import BranchDict
    from src.graph_tot.dspy_modules import Branch
    import dspy
    branch = Branch(
        answer="test",
        trace="trace text",
        prediction=dspy.Prediction(answer="test"),
        score=0.5,
        parent_context="ctx",
    )
    d = branch.as_dict()
    for key in BranchDict.__annotations__:
        assert key in d, f"Branch.as_dict() missing key: {key}"
