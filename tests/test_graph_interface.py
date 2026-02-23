"""Tests for pluggable graph interfaces and stores."""

from src.graph_tot.graph_env import GraphEnvironment, GraphNodeStore, GraphToolInterface


class InMemoryStore(GraphNodeStore):
    def __init__(self):
        self._nodes = [
            (
                "Disease::D001",
                "Disease",
                {
                    "features": {"name": "Diabetes", "mesh_id": "D001"},
                    "neighbors": {"Compound-treats-Disease": ["Compound::C001"]},
                },
            ),
            (
                "Compound::C001",
                "Compound",
                {
                    "features": {"name": "Metformin"},
                    "neighbors": {"Compound-treats-Disease": ["Disease::D001"]},
                },
            ),
        ]

    @property
    def identity(self) -> str:
        return "memory://demo-graph-v1"

    def iter_nodes(self):
        yield from self._nodes


class TestGraphBackendProtocol:
    def test_graph_environment_conforms_to_protocol(self):
        env = object.__new__(GraphEnvironment)
        assert isinstance(env, GraphToolInterface)


class TestCustomNodeStoreLoading:
    def test_load_graph_from_custom_store(self):
        env = object.__new__(GraphEnvironment)
        env.node_store = InMemoryStore()
        env.graph = {}
        env.graph_index = {}
        env.doc_lookup = []
        env.doc_type = []

        env._load_graph()

        assert "Disease::D001" in env.graph_index
        assert env.doc_lookup == ["Disease::D001", "Compound::C001"]
        assert env.doc_type == ["Disease", "Compound"]


class MinimalRemoteBackend:
    def retrieve_node(self, keyword: str) -> str:
        return f"Node ID: Remote::{keyword}"

    def node_feature(self, node_id: str, feature: str) -> str:
        return "value"

    def neighbour_check(self, node_id: str, neighbor_type: str) -> str:
        return "[]"

    def node_degree(self, node_id: str, neighbor_type: str) -> str:
        return "0"

    def get_tools(self):
        return [self.retrieve_node, self.node_feature, self.neighbour_check, self.node_degree]


class TestToolOnlyBackend:
    def test_minimal_remote_backend_conforms_to_protocol(self):
        assert isinstance(MinimalRemoteBackend(), GraphToolInterface)
