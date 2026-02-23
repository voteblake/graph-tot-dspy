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

import hashlib
import json
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

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


@dataclass
class ToolResult:
    """Structured result from a graph tool operation.

    Attributes:
        ok:         True if the operation succeeded, False on error.
        data:       The raw result data as a string (empty on error).
        error_code: A stable error code string from ErrorCode, or None on success.
        message:    Human-readable result suitable for ReAct prompting.
    """
    ok: bool
    data: str
    error_code: Optional[str]
    message: str


class ErrorCode:
    """Stable error code constants for graph tool operations."""
    NODE_NOT_FOUND = "NODE_NOT_FOUND"
    FEATURE_NOT_FOUND = "FEATURE_NOT_FOUND"
    EDGE_TYPE_NOT_FOUND = "EDGE_TYPE_NOT_FOUND"
    DEGREE_ERROR = "DEGREE_ERROR"
    INDEX_UNINITIALIZED = "INDEX_UNINITIALIZED"


@runtime_checkable
class GraphToolInterface(Protocol):
    """Interface for graph backends consumable by GraphToTAgent.

    Users can bring their own graph implementation (Neo4j, RDF/OWL, APIs, etc.)
    by implementing this protocol and passing the instance to GraphToTSolver.
    """

    def retrieve_node(self, keyword: str) -> str:
        ...

    def node_feature(self, node_id: str, feature: str) -> str:
        ...

    def neighbour_check(self, node_id: str, neighbor_type: str) -> str:
        ...

    def node_degree(self, node_id: str, neighbor_type: str) -> str:
        ...

    def get_tools(self) -> list:
        ...


class GraphNodeStore(ABC):
    """Abstract adapter for loading graph nodes from arbitrary sources.

    Implementations can read from local files, databases, or remote services.
    """

    @property
    @abstractmethod
    def identity(self) -> str:
        """Stable identity used for cache keying (path, URI, version, etc.)."""

    @abstractmethod
    def iter_nodes(self) -> Iterable[tuple[str, str, dict[str, Any]]]:
        """Yield (node_id, node_type, node_data) tuples for all graph nodes."""


class JsonPickleGraphStore(GraphNodeStore):
    """GraphNodeStore implementation for the current GRBench JSON/pickle schema."""

    def __init__(self, graph_path: str | Path) -> None:
        self.graph_path = Path(graph_path)
        self.graph: dict[str, Any] = {}
        self._load()

    @property
    def identity(self) -> str:
        return str(self.graph_path)

    def _load(self) -> None:
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
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    self.graph = json.load(f)
            except (UnicodeDecodeError, json.JSONDecodeError):
                with open(self.graph_path, "rb") as f:
                    self.graph = pickle.load(f)

    def iter_nodes(self) -> Iterable[tuple[str, str, dict[str, Any]]]:
        for node_type, nodes in self.graph.items():
            if not isinstance(nodes, dict):
                continue
            for node_id, node_data in nodes.items():
                yield node_id, node_type, node_data


def _graph_fingerprint(graph_path: str) -> str:
    """Return a short fingerprint of a graph file for use in cache filenames.

    Combines the path, file size, and modification time into an 8-char hex
    digest.  This is fast (no file reading) and detects both content changes
    and file replacement with different size/mtime.  Returns ``"unknown"`` if
    the file cannot be stat-ed.
    """
    try:
        size = os.path.getsize(graph_path)
        mtime = os.path.getmtime(graph_path)
        key = f"{graph_path}:{size}:{mtime}"
        return hashlib.md5(key.encode()).hexdigest()[:8]
    except OSError:
        return "unknown"


class GraphEnvironment:
    """
    Wraps a graph backend and exposes four ReAct-compatible tool methods.

    By default, it uses JsonPickleGraphStore for the GRBench JSON/pickle schema.
    You can pass a custom GraphNodeStore to load nodes from other sources
    (e.g., Neo4j, RDF/OWL parsers, or remote APIs).

    A FAISS index over node-name embeddings is built on first use and cached.
    """

    def __init__(
        self,
        graph_path: str | Path | None,
        faiss_cache_dir: str | Path,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = 1,
        node_store: Optional[GraphNodeStore] = None,
    ) -> None:
        if graph_path is None and node_store is None:
            raise ValueError("graph_path is required when node_store is not provided")
        self.graph_path = Path(graph_path) if graph_path is not None else Path(node_store.identity)
        self.faiss_cache_dir = Path(faiss_cache_dir)
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.node_store = node_store or JsonPickleGraphStore(self.graph_path)

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
        """Load graph from the configured GraphNodeStore and build flat indexes."""
        seen: set[str] = set()
        self.graph = getattr(self.node_store, "graph", {})
        for nid, node_type, node_data in self.node_store.iter_nodes():
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
            self.node_store.identity,
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
        """Build FAISS index or load from disk cache.

        The cache filename encodes both the embedding model name and a
        fingerprint of the graph file (path + size + mtime), so that switching
        graphs or updating the graph file automatically triggers a rebuild
        rather than silently reusing stale embeddings.
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        self.faiss_cache_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.embedding_model_name.replace("/", "_")
        fingerprint_source = getattr(getattr(self, "node_store", None), "identity", str(self.graph_path))
        fingerprint = _graph_fingerprint(fingerprint_source)
        cache_file = self.faiss_cache_dir / f"faiss_{safe_name}_{fingerprint}.pkl"
        logger.info(
            "FAISS cache key: model=%s graph_fingerprint=%s path=%s",
            self.embedding_model_name, fingerprint, cache_file,
        )

        embeddings = None
        if cache_file.exists():
            logger.info("Loading FAISS index from cache: %s", cache_file)
            with open(cache_file, "rb") as f:
                loaded = pickle.load(f)
            if len(loaded) != len(self.doc_lookup):
                logger.warning(
                    "FAISS cache mismatch: cache has %d embeddings but graph has %d nodes; "
                    "rebuilding cache at %s",
                    len(loaded), len(self.doc_lookup), cache_file,
                )
            else:
                embeddings = loaded

        if embeddings is None:
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

    # ------------------------------------------------------------------
    # Structured tool APIs (for programmatic / library consumers)
    # ------------------------------------------------------------------

    def retrieve_node_structured(self, keyword: str) -> ToolResult:
        """Structured variant of retrieve_node; returns a ToolResult object."""
        if self.faiss_index is None or self.embed_model is None:
            msg = "Error: FAISS index not initialised."
            return ToolResult(ok=False, data="", error_code=ErrorCode.INDEX_UNINITIALIZED, message=msg)

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

        if results:
            return ToolResult(ok=True, data=results[0], error_code=None, message=results[0])
        msg = f"No node found matching: {keyword!r}"
        return ToolResult(ok=False, data="", error_code=ErrorCode.NODE_NOT_FOUND, message=msg)

    def node_feature_structured(self, node_id: str, feature: str) -> ToolResult:
        """Structured variant of node_feature; returns a ToolResult object."""
        if node_id not in self.graph_index:
            msg = f"Error: Node '{node_id}' not found in graph."
            return ToolResult(ok=False, data="", error_code=ErrorCode.NODE_NOT_FOUND, message=msg)
        features = self.graph_index[node_id].get("features", {})
        if feature not in features:
            msg = (
                f"Error: Feature '{feature}' not found for node '{node_id}'. "
                f"Available features: {list(features.keys())}"
            )
            return ToolResult(ok=False, data="", error_code=ErrorCode.FEATURE_NOT_FOUND, message=msg)
        value = str(features[feature])
        return ToolResult(ok=True, data=value, error_code=None, message=value)

    def neighbour_check_structured(self, node_id: str, neighbor_type: str) -> ToolResult:
        """Structured variant of neighbour_check; returns a ToolResult object."""
        if node_id not in self.graph_index:
            msg = f"Error: Node '{node_id}' not found in graph."
            return ToolResult(ok=False, data="", error_code=ErrorCode.NODE_NOT_FOUND, message=msg)
        neighbors = self.graph_index[node_id].get("neighbors", {})
        if neighbor_type not in neighbors:
            msg = (
                f"Error: Neighbor type '{neighbor_type}' not found for '{node_id}'. "
                f"Available neighbor types: {list(neighbors.keys())}"
            )
            return ToolResult(ok=False, data="", error_code=ErrorCode.EDGE_TYPE_NOT_FOUND, message=msg)
        neighbor_list = neighbors[neighbor_type]
        if len(neighbor_list) > 20:
            value = f"{neighbor_list[:20]} ... (and {len(neighbor_list) - 20} more)"
        else:
            value = str(neighbor_list)
        return ToolResult(ok=True, data=value, error_code=None, message=value)

    def node_degree_structured(self, node_id: str, neighbor_type: str) -> ToolResult:
        """Structured variant of node_degree; returns a ToolResult object."""
        if node_id not in self.graph_index:
            msg = f"Error: Node '{node_id}' not found in graph."
            return ToolResult(ok=False, data="", error_code=ErrorCode.NODE_NOT_FOUND, message=msg)
        neighbors = self.graph_index[node_id].get("neighbors", {})
        if neighbor_type not in neighbors:
            msg = (
                f"Error: Neighbor type '{neighbor_type}' not found for '{node_id}'. "
                f"Available neighbor types: {list(neighbors.keys())}"
            )
            return ToolResult(ok=False, data="", error_code=ErrorCode.EDGE_TYPE_NOT_FOUND, message=msg)
        value = str(len(neighbors[neighbor_type]))
        return ToolResult(ok=True, data=value, error_code=None, message=value)

    # ------------------------------------------------------------------
    # ReAct-compatible string tool methods (delegate to structured variants)
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
        return self.retrieve_node_structured(keyword).message

    def node_feature(self, node_id: str, feature: str) -> str:
        """Retrieve a specific attribute of a graph node.

        Use this after retrieve_node to read a particular property of a node.

        Args:
            node_id: The node identifier returned by retrieve_node (e.g., 'Disease::DOID:9352').
            feature: The attribute name to retrieve (e.g., 'name', 'mesh_id', 'description').

        Returns:
            The feature value as a string, or an error listing available features.
        """
        return self.node_feature_structured(node_id, feature).message

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
        return self.neighbour_check_structured(node_id, neighbor_type).message

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
        return self.node_degree_structured(node_id, neighbor_type).message

    def get_tools(self) -> list:
        """Return the list of bound tool methods for use with dspy.ReAct."""
        return [
            self.retrieve_node,
            self.node_feature,
            self.neighbour_check,
            self.node_degree,
        ]
