"""
Tests for GraphEnvironment tool methods.

GraphEnvironment.__init__ does filesystem and FAISS I/O, so we bypass it
entirely using object.__new__ + manual attribute injection. This lets us
test the pure dictionary-lookup logic of each tool method in isolation.
"""

import pytest

from src.graph_tot.graph_env import GraphEnvironment, HEALTHCARE_NODE_TEXT_KEYS


# ---------------------------------------------------------------------------
# Helper: create GraphEnvironment bypassing __init__
# ---------------------------------------------------------------------------


def _make_env(graph_index: dict = None):
    """Bypass __init__; inject graph_index and the other attributes tool methods read."""
    env = object.__new__(GraphEnvironment)
    env.graph_index = graph_index or {}
    env.doc_lookup = list((graph_index or {}).keys())
    env.doc_type = ["Unknown"] * len(env.doc_lookup)
    env.faiss_index = None
    env.embed_model = None
    env.top_k = 1
    return env


SAMPLE_GRAPH = {
    "Disease::D001": {
        "features": {"name": "Diabetes", "mesh_id": "D001"},
        "neighbors": {
            "Compound-treats-Disease": ["Compound::C001", "Compound::C002"],
            "Gene-associates-Disease": ["Gene::G001"],
        },
    },
    "Compound::C001": {
        "features": {"name": "Metformin", "pubchem_id": "4091"},
        "neighbors": {
            "Compound-treats-Disease": ["Disease::D001"],
        },
    },
    "Gene::G001": {
        "features": {"name": "PPARG", "description": "nuclear receptor"},
        "neighbors": {},
    },
}


# ---------------------------------------------------------------------------
# node_feature
# ---------------------------------------------------------------------------


class TestNodeFeature:

    def test_returns_feature_value(self):
        env = _make_env(SAMPLE_GRAPH)
        assert env.node_feature("Disease::D001", "name") == "Diabetes"

    def test_returns_second_feature_value(self):
        env = _make_env(SAMPLE_GRAPH)
        assert env.node_feature("Disease::D001", "mesh_id") == "D001"

    def test_unknown_node_returns_error_string(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.node_feature("FAKE::999", "name")
        assert result.startswith("Error:")
        assert "FAKE::999" in result

    def test_unknown_feature_returns_error_listing_available(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.node_feature("Disease::D001", "nonexistent")
        assert result.startswith("Error:")
        assert "nonexistent" in result
        assert "name" in result
        assert "mesh_id" in result

    def test_numeric_feature_coerced_to_string(self):
        graph = {"Node::1": {"features": {"count": 42}, "neighbors": {}}}
        env = _make_env(graph)
        result = env.node_feature("Node::1", "count")
        assert result == "42"
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# neighbour_check
# ---------------------------------------------------------------------------


class TestNeighbourCheck:

    def test_returns_neighbor_list_string(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.neighbour_check("Disease::D001", "Compound-treats-Disease")
        assert "Compound::C001" in result
        assert "Compound::C002" in result

    def test_unknown_node_returns_error(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.neighbour_check("FAKE::999", "Compound-treats-Disease")
        assert result.startswith("Error:")

    def test_unknown_neighbor_type_returns_error_listing_available(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.neighbour_check("Disease::D001", "NonExistent-edge")
        assert result.startswith("Error:")
        assert "NonExistent-edge" in result
        assert "Compound-treats-Disease" in result
        assert "Gene-associates-Disease" in result

    def test_truncates_list_beyond_20_with_count(self):
        neighbors = [f"Node::N{i}" for i in range(30)]
        graph = {"Hub::1": {"features": {}, "neighbors": {"type-A": neighbors}}}
        env = _make_env(graph)
        result = env.neighbour_check("Hub::1", "type-A")
        assert "10 more" in result
        assert "Node::N0" in result

    def test_exactly_20_neighbors_not_truncated(self):
        neighbors = [f"Node::N{i}" for i in range(20)]
        graph = {"Hub::1": {"features": {}, "neighbors": {"type-A": neighbors}}}
        env = _make_env(graph)
        result = env.neighbour_check("Hub::1", "type-A")
        assert "more" not in result

    def test_21_neighbors_triggers_truncation(self):
        neighbors = [f"Node::N{i}" for i in range(21)]
        graph = {"Hub::1": {"features": {}, "neighbors": {"type-A": neighbors}}}
        env = _make_env(graph)
        result = env.neighbour_check("Hub::1", "type-A")
        assert "1 more" in result


# ---------------------------------------------------------------------------
# node_degree
# ---------------------------------------------------------------------------


class TestNodeDegree:

    def test_returns_count_as_string(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.node_degree("Disease::D001", "Compound-treats-Disease")
        assert result == "2"
        assert isinstance(result, str)

    def test_unknown_node_returns_error(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.node_degree("FAKE::999", "any-type")
        assert result.startswith("Error:")

    def test_unknown_type_returns_error_listing_available(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.node_degree("Disease::D001", "fake-type")
        assert result.startswith("Error:")
        assert "Compound-treats-Disease" in result

    def test_zero_neighbors_returns_zero(self):
        graph = {"Node::1": {"features": {}, "neighbors": {"type-A": []}}}
        env = _make_env(graph)
        assert env.node_degree("Node::1", "type-A") == "0"

    def test_single_neighbor_returns_one(self):
        env = _make_env(SAMPLE_GRAPH)
        result = env.node_degree("Disease::D001", "Gene-associates-Disease")
        assert result == "1"


# ---------------------------------------------------------------------------
# _get_node_text
# ---------------------------------------------------------------------------


class TestGetNodeText:

    def test_gene_type_joins_name_and_description(self):
        env = _make_env()
        node_data = {"features": {"name": "PPARG", "description": "nuclear receptor"}}
        result = env._get_node_text("Gene", node_data)
        assert "PPARG" in result
        assert "nuclear receptor" in result

    def test_disease_type_uses_name_only(self):
        env = _make_env()
        # Disease maps to ["name"] only in HEALTHCARE_NODE_TEXT_KEYS
        node_data = {"features": {"name": "Diabetes", "description": "metabolic"}}
        result = env._get_node_text("Disease", node_data)
        assert "Diabetes" in result
        assert "metabolic" not in result

    def test_missing_key_in_features_skipped_gracefully(self):
        env = _make_env()
        # Gene expects name + description, but description is absent
        node_data = {"features": {"name": "BRCA1"}}
        result = env._get_node_text("Gene", node_data)
        assert "BRCA1" in result

    def test_unknown_node_type_falls_back_to_name_key(self):
        env = _make_env()
        node_data = {"features": {"name": "SomeName"}}
        result = env._get_node_text("UnknownType", node_data)
        assert "SomeName" in result

    def test_no_features_returns_node_type_string(self):
        env = _make_env()
        node_data = {"features": {}}
        result = env._get_node_text("Compound", node_data)
        assert result == "Compound"

    def test_all_healthcare_node_types_have_entries(self):
        """Every type in HEALTHCARE_NODE_TEXT_KEYS can be retrieved without error."""
        env = _make_env()
        for node_type, keys in HEALTHCARE_NODE_TEXT_KEYS.items():
            features = {k: f"value_{k}" for k in keys}
            node_data = {"features": features}
            result = env._get_node_text(node_type, node_data)
            assert isinstance(result, str)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# get_tools
# ---------------------------------------------------------------------------


class TestGetTools:

    def test_returns_exactly_4_callables(self):
        env = _make_env(SAMPLE_GRAPH)
        tools = env.get_tools()
        assert len(tools) == 4
        assert all(callable(t) for t in tools)

    def test_tool_names_match_expected_methods(self):
        env = _make_env(SAMPLE_GRAPH)
        tools = env.get_tools()
        names = {t.__name__ for t in tools}
        assert names == {"retrieve_node", "node_feature", "neighbour_check", "node_degree"}
