"""Tests for pluggable graph interfaces and stores."""

from src.graph_tot.graph_env import (
    GraphEnvironment,
    GraphNodeStore,
    GraphToolInterface,
    RdfOwlXmlGraphStore,
    _make_default_node_store,
)


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


class TestRdfOwlXmlGraphStore:
    def test_parses_rdf_xml_nodes_features_and_edges(self, tmp_path):
        rdf_xml = """<?xml version="1.0"?>
<rdf:RDF
    xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
    xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
    xmlns:ex="http://example.org/"
    xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://example.org#Disease">
    <rdfs:label>Disease</rdfs:label>
    <ex:relatedTo rdf:resource="http://example.org#Symptom"/>
  </owl:Class>
  <rdf:Description rdf:about="http://example.org#Symptom">
    <rdfs:label>Symptom</rdfs:label>
    <ex:description>Clinical finding</ex:description>
  </rdf:Description>
</rdf:RDF>
"""
        path = tmp_path / "toy.owl"
        path.write_text(rdf_xml, encoding="utf-8")

        store = RdfOwlXmlGraphStore(path)
        nodes = {(nid, ntype): data for nid, ntype, data in store.iter_nodes()}

        assert ("Disease", "Class") in nodes
        assert nodes[("Disease", "Class")]["features"]["name"] == "Disease"
        assert nodes[("Disease", "Class")]["neighbors"]["relatedTo"] == ["Symptom"]

        assert ("Symptom", "Resource") in nodes
        assert nodes[("Symptom", "Resource")]["features"]["description"] == "Clinical finding"


class TestDefaultStoreSelection:
    def test_selects_rdf_store_for_owl_extension(self, tmp_path):
        path = tmp_path / "toy.owl"
        path.write_text("<rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'/>", encoding="utf-8")
        store = _make_default_node_store(path)
        assert isinstance(store, RdfOwlXmlGraphStore)
