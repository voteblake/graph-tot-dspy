"""
Graph environment for GRBench knowledge graphs.

Loads a domain graph (JSON or pickle format from the Graph-CoT / GRBench
benchmark) and exposes four tool methods compatible with dspy.ReAct:

  - retrieve_node(keyword)           semantic search via FAISS + MPNet
  - node_feature(node_id, feature)   attribute lookup
  - neighbour_check(node_id, type)   list neighbors by edge type
  - node_degree(node_id, type)       count neighbors by edge type

The FAISS index is built once and cached to disk for fast subsequent loads.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Node types present in the GRBench Healthcare (Hetionet-based) graph,
# mapped to the feature key used as text for embedding.
HEALTHCARE_NODE_TEXT_KEYS: dict[str, list[str]] = {
    "Anatomy": ["name"],
    "Biological_Process": ["name"],
    "Cellular_Component": ["name"],
    "Compound": ["name"],
    "Disease": ["name"],
    "Gene": ["name", "description"],
    "Molecular_Function": ["name"],
    "Pathway": ["name"],
    "Pharmacologic_Class": ["name"],
    "Side_Effect": ["name"],
    "Symptom": ["name"],
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


class GraphEnvironment:
    """
    Wraps a GRBench knowledge graph and exposes four ReAct-compatible tool methods.

    The graph must be in one of these formats:
      - JSON: nested dict  {node_type: {node_id: {features: {...}, neighbors: {...}}}}
      - Pickle: same structure serialised with pickle

    A FAISS index over node-name embeddings is built on first use and cached.
    """

    def __init__(
        self,
        graph_path: str | Path,
        faiss_cache_dir: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 1,
    ) -> None:
        self.graph_path = Path(graph_path)
        self.faiss_cache_dir = Path(faiss_cache_dir)
        self.embedding_model_name = embedding_model
        self.top_k = top_k

        # Will be populated by _load_graph()
        self.graph: dict = {}
        self.graph_index: dict[str, dict] = {}  # node_id -> {features, neighbors}
        self.doc_lookup: list[str] = []          # position -> node_id
        self.doc_type: list[str] = []            # position -> node_type

        # Will be populated by _build_or_load_faiss_index()
        self.faiss_index = None
        self.embed_model = None

        self._load_graph()
        self._build_or_load_faiss_index()

    # ------------------------------------------------------------------
    # Graph loading
    # ------------------------------------------------------------------

    def _load_graph(self) -> None:
        """Load graph from JSON or pickle and build a flat node index."""
        if not self.graph_path.exists():
            raise FileNotFoundError(
                f"Graph file not found: {self.graph_path}\n"
                "Download the healthcare graph from:\n"
                "  https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9\n"
                f"Then place it at: {self.graph_path}"
            )

        suffix = self.graph_path.suffix.lower()
        if suffix == ".json":
            with open(self.graph_path, "r", encoding="utf-8") as f:
                self.graph = json.load(f)
        elif suffix in (".pkl", ".pickle"):
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
        else:
            # Try JSON first, then pickle
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    self.graph = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)

        seen: set[str] = set()
        for node_type, nodes in self.graph.items():
            if not isinstance(nodes, dict):
                continue
            for nid, node_data in nodes.items():
                if nid in seen:
                    logger.warning("Duplicate node ID %s in type %s; skipping", nid, node_type)
                    continue
                seen.add(nid)
                self.graph_index[nid] = node_data
                self.doc_lookup.append(nid)
                self.doc_type.append(node_type)

        logger.info(
            "Graph loaded: %d nodes across %d types from %s",
            len(self.graph_index),
            len(self.graph),
            self.graph_path,
        )

    # ------------------------------------------------------------------
    # FAISS index
    # ------------------------------------------------------------------

    def _get_node_text(self, node_type: str, node_data: dict) -> str:
        """Return the text string to embed for a node."""
        text_keys = HEALTHCARE_NODE_TEXT_KEYS.get(node_type, ["name"])
        features = node_data.get("features", {})
        parts = [str(features[k]) for k in text_keys if features.get(k)]
        return " ".join(parts) if parts else node_type

    def _build_or_load_faiss_index(self) -> None:
        """Build FAISS index or load from disk cache."""
        import faiss
        from sentence_transformers import SentenceTransformer

        self.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.embedding_model_name.replace("/", "_")
        cache_file = self.faiss_cache_dir / f"faiss_{safe_name}.pkl"

        if cache_file.exists():
            logger.info("Loading FAISS index from cache: %s", cache_file)
            with open(cache_file, "rb") as f:
                embeddings = pickle.load(f)
        else:
            logger.info(
                "Building FAISS index for %d nodes (first run — this takes a few minutes)...",
                len(self.doc_lookup),
            )
            self.embed_model = SentenceTransformer(self.embedding_model_name)
            texts = [
                self._get_node_text(ntype, self.graph_index[nid])
                for nid, ntype in zip(self.doc_lookup, self.doc_type)
            ]
            embeddings = self.embed_model.encode(
                texts,
                batch_size=512,
                show_progress_bar=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            with open(cache_file, "wb") as f:
                pickle.dump(embeddings, f)
            logger.info("FAISS cache saved: %s", cache_file)

        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)
        logger.info(
            "FAISS index ready: %d vectors, dim=%d", self.faiss_index.ntotal, dim
        )

        # Ensure embed model is loaded for query-time use
        if self.embed_model is None:
            from sentence_transformers import SentenceTransformer
            self.embed_model = SentenceTransformer(self.embedding_model_name)

    # ------------------------------------------------------------------
    # Tool methods (called by dspy.ReAct)
    # ------------------------------------------------------------------

    def retrieve_node(self, keyword: str) -> str:
        """Search the knowledge graph for the node most semantically similar to the keyword.

        Use this tool first to find the node ID for an entity mentioned in the question.
        Returns the node ID, its type, and primary features.

        Args:
            keyword: A natural-language entity name or description to search for
                     (e.g., 'diabetes mellitus', 'aspirin', 'BRCA1 gene').

        Returns:
            A string describing the best-matching node: its ID, type, and features.
        """
        if self.faiss_index is None or self.embed_model is None:
            return "Error: FAISS index not initialised."

        query_emb = self.embed_model.encode(
            [keyword],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)

        distances, indices = self.faiss_index.search(query_emb, self.top_k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self.doc_lookup):
                continue
            nid = self.doc_lookup[idx]
            ntype = self.doc_type[idx]
            features = self.graph_index[nid].get("features", {})
            results.append(f"Node ID: {nid} (type: {ntype}). Features: {features}")

        return results[0] if results else f"No node found matching: {keyword!r}"

    def node_feature(self, node_id: str, feature: str) -> str:
        """Retrieve a specific attribute of a graph node.

        Use this after retrieve_node to read a particular property of a node.

        Args:
            node_id: The node identifier returned by retrieve_node (e.g., 'Disease::DOID:9352').
            feature: The attribute name to retrieve (e.g., 'name', 'mesh_id', 'description').

        Returns:
            The feature value as a string, or an error listing available features.
        """
        if node_id not in self.graph_index:
            return f"Error: Node '{node_id}' not found in graph."
        features = self.graph_index[node_id].get("features", {})
        if feature not in features:
            return (
                f"Error: Feature '{feature}' not found for node '{node_id}'. "
                f"Available features: {list(features.keys())}"
            )
        return str(features[feature])

    def neighbour_check(self, node_id: str, neighbor_type: str) -> str:
        """List the neighbors of a graph node filtered by relationship type.

        Use this to explore which nodes are connected to a given node via a
        specific edge type (e.g., which compounds treat a disease).

        Args:
            node_id: The node identifier (e.g., 'Disease::DOID:9352').
            neighbor_type: The edge/relationship type to traverse
                           (e.g., 'Compound-treats-Disease', 'Gene-associates-Disease').

        Returns:
            A list of neighboring node IDs, or an error listing available neighbor types.
        """
        if node_id not in self.graph_index:
            return f"Error: Node '{node_id}' not found in graph."
        neighbors = self.graph_index[node_id].get("neighbors", {})
        if neighbor_type not in neighbors:
            return (
                f"Error: Neighbor type '{neighbor_type}' not found for '{node_id}'. "
                f"Available neighbor types: {list(neighbors.keys())}"
            )
        neighbor_list = neighbors[neighbor_type]
        if len(neighbor_list) > 20:
            return f"{neighbor_list[:20]} ... (and {len(neighbor_list) - 20} more)"
        return str(neighbor_list)

    def node_degree(self, node_id: str, neighbor_type: str) -> str:
        """Count the number of neighbors of a given relationship type for a node.

        Use this to understand the connectivity of a node (e.g., how many
        compounds treat a disease) without listing all neighbors.

        Args:
            node_id: The node identifier (e.g., 'Disease::DOID:9352').
            neighbor_type: The edge/relationship type to count
                           (e.g., 'Compound-treats-Disease').

        Returns:
            The integer count as a string, or an error listing available neighbor types.
        """
        if node_id not in self.graph_index:
            return f"Error: Node '{node_id}' not found in graph."
        neighbors = self.graph_index[node_id].get("neighbors", {})
        if neighbor_type not in neighbors:
            return (
                f"Error: Neighbor type '{neighbor_type}' not found for '{node_id}'. "
                f"Available neighbor types: {list(neighbors.keys())}"
            )
        return str(len(neighbors[neighbor_type]))

    def get_tools(self) -> list:
        """Return the list of bound tool methods for use with dspy.ReAct."""
        return [
            self.retrieve_node,
            self.node_feature,
            self.neighbour_check,
            self.node_degree,
        ]
