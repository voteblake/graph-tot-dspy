"""
Tests for FAISS cache filename keying on graph identity.

Verifies that _graph_fingerprint produces stable, distinct hashes and that
the cache path includes both model name and fingerprint.  No actual graph
data, sentence-transformers, or FAISS binaries are required.
"""

from unittest.mock import patch, MagicMock


from src.graph_tot.graph_env import _graph_fingerprint


# ---------------------------------------------------------------------------
# _graph_fingerprint: stability and distinctness
# ---------------------------------------------------------------------------


class TestGraphFingerprint:

    def test_returns_string(self):
        with patch("os.path.getsize", return_value=1000), \
             patch("os.path.getmtime", return_value=1700000000.0):
            result = _graph_fingerprint("/some/graph.json")
        assert isinstance(result, str)

    def test_returns_8_hex_chars(self):
        with patch("os.path.getsize", return_value=1000), \
             patch("os.path.getmtime", return_value=1700000000.0):
            result = _graph_fingerprint("/some/graph.json")
        assert len(result) == 8
        assert all(c in "0123456789abcdef" for c in result)

    def test_consistent_for_same_inputs(self):
        with patch("os.path.getsize", return_value=5000), \
             patch("os.path.getmtime", return_value=1234567890.0):
            r1 = _graph_fingerprint("/data/graph.json")
            r2 = _graph_fingerprint("/data/graph.json")
        assert r1 == r2

    def test_different_size_gives_different_fingerprint(self):
        with patch("os.path.getsize", return_value=1000), \
             patch("os.path.getmtime", return_value=1700000000.0):
            fp_a = _graph_fingerprint("/data/graph.json")
        with patch("os.path.getsize", return_value=9999), \
             patch("os.path.getmtime", return_value=1700000000.0):
            fp_b = _graph_fingerprint("/data/graph.json")
        assert fp_a != fp_b

    def test_different_mtime_gives_different_fingerprint(self):
        with patch("os.path.getsize", return_value=1000), \
             patch("os.path.getmtime", return_value=1700000000.0):
            fp_a = _graph_fingerprint("/data/graph.json")
        with patch("os.path.getsize", return_value=1000), \
             patch("os.path.getmtime", return_value=1700000001.0):
            fp_b = _graph_fingerprint("/data/graph.json")
        assert fp_a != fp_b

    def test_different_path_gives_different_fingerprint(self):
        with patch("os.path.getsize", return_value=1000), \
             patch("os.path.getmtime", return_value=1700000000.0):
            fp_a = _graph_fingerprint("/data/graph_a.json")
            fp_b = _graph_fingerprint("/data/graph_b.json")
        assert fp_a != fp_b

    def test_oserror_returns_unknown(self):
        with patch("os.path.getsize", side_effect=OSError("not found")):
            result = _graph_fingerprint("/nonexistent/path.json")
        assert result == "unknown"

    def test_oserror_on_mtime_returns_unknown(self):
        with patch("os.path.getsize", return_value=100), \
             patch("os.path.getmtime", side_effect=OSError("permission denied")):
            result = _graph_fingerprint("/some/path.json")
        assert result == "unknown"


# ---------------------------------------------------------------------------
# Cache filename includes fingerprint
# ---------------------------------------------------------------------------


class TestCacheFilenameIncludesFingerprint:
    """The FAISS cache path must embed both the model slug and graph fingerprint."""

    def _make_env_with_mocked_init(self, fingerprint="abc12345"):
        """Return a GraphEnvironment instance with __init__ bypassed."""
        from src.graph_tot.graph_env import GraphEnvironment
        from pathlib import Path
        env = object.__new__(GraphEnvironment)
        env.graph_path = Path("/fake/graph.json")
        env.faiss_cache_dir = Path("/fake/cache")
        env.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        env.doc_lookup = ["node1", "node2", "node3"]
        env.doc_type = ["Disease", "Compound", "Gene"]
        env.graph_index = {
            "node1": {"features": {"name": "Diabetes"}, "neighbors": {}},
            "node2": {"features": {"name": "Aspirin"}, "neighbors": {}},
            "node3": {"features": {"name": "BRCA1"}, "neighbors": {}},
        }
        env.faiss_index = None
        env.embed_model = None
        return env, fingerprint

    def _run_build_with_mocks(self, env, fingerprint, cache_exists=False, cached_embeddings=None):
        """Run _build_or_load_faiss_index with all heavy deps fully mocked via sys.modules."""
        import sys
        import numpy as np

        fake_embeddings = np.zeros((3, 4), dtype=np.float32)
        fake_index = MagicMock()
        fake_index.ntotal = 3
        mock_st_model = MagicMock()
        mock_st_model.encode.return_value = fake_embeddings
        mock_st_cls = MagicMock(return_value=mock_st_model)
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = mock_st_cls
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = fake_index

        saved_paths = []

        def mock_open(path, mode="r", **kwargs):
            saved_paths.append(str(path))
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            return m

        load_return = cached_embeddings if cached_embeddings is not None else fake_embeddings

        with patch.dict(sys.modules, {"faiss": mock_faiss, "sentence_transformers": mock_st_module}), \
             patch("src.graph_tot.graph_env._graph_fingerprint", return_value=fingerprint), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=cache_exists), \
             patch("builtins.open", side_effect=mock_open), \
             patch("pickle.load", return_value=load_return), \
             patch("pickle.dump"):
            env._build_or_load_faiss_index()

        return saved_paths, mock_st_cls

    def test_cache_filename_contains_fingerprint(self):
        env, fp = self._make_env_with_mocked_init(fingerprint="deadbeef")
        saved_paths, _ = self._run_build_with_mocks(env, fingerprint="deadbeef", cache_exists=False)
        assert any("deadbeef" in p for p in saved_paths), \
            f"No path with fingerprint 'deadbeef' found in: {saved_paths}"

    def test_cache_filename_contains_model_name(self):
        env, _ = self._make_env_with_mocked_init()
        saved_paths, _ = self._run_build_with_mocks(env, fingerprint="abcd1234", cache_exists=False)
        assert any("all-mpnet-base-v2" in p or "all_mpnet_base_v2" in p for p in saved_paths), \
            f"Model name not found in any path: {saved_paths}"


# ---------------------------------------------------------------------------
# Cache validation: mismatch triggers rebuild
# ---------------------------------------------------------------------------


class TestCacheValidation:

    def _make_env(self, doc_count=3):
        from src.graph_tot.graph_env import GraphEnvironment
        from pathlib import Path
        env = object.__new__(GraphEnvironment)
        env.graph_path = Path("/fake/graph.json")
        env.faiss_cache_dir = Path("/fake/cache")
        env.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        env.doc_lookup = [f"node{i}" for i in range(doc_count)]
        env.doc_type = ["Disease"] * doc_count
        env.graph_index = {
            f"node{i}": {"features": {"name": f"Node{i}"}, "neighbors": {}}
            for i in range(doc_count)
        }
        env.faiss_index = None
        env.embed_model = None
        return env

    def _run_with_mocks(self, env, fingerprint, cached_embeddings, caplog=None):
        """Run _build_or_load_faiss_index with sys.modules-level mocks."""
        import sys
        import numpy as np
        import logging

        fresh_embeddings = np.zeros((len(env.doc_lookup), 4), dtype=np.float32)
        fake_index = MagicMock()
        fake_index.ntotal = len(env.doc_lookup)
        mock_st_model = MagicMock()
        mock_st_model.encode.return_value = fresh_embeddings
        mock_st_cls = MagicMock(return_value=mock_st_model)
        mock_st_module = MagicMock()
        mock_st_module.SentenceTransformer = mock_st_cls
        mock_faiss = MagicMock()
        mock_faiss.IndexFlatIP.return_value = fake_index

        def mock_open(path, mode="r", **kwargs):
            m = MagicMock()
            m.__enter__ = MagicMock(return_value=m)
            m.__exit__ = MagicMock(return_value=False)
            return m

        ctx = patch.dict(sys.modules, {"faiss": mock_faiss, "sentence_transformers": mock_st_module})
        with ctx, \
             patch("src.graph_tot.graph_env._graph_fingerprint", return_value=fingerprint), \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", side_effect=mock_open), \
             patch("pickle.load", return_value=cached_embeddings), \
             patch("pickle.dump") as mock_dump:
            if caplog is not None:
                with caplog.at_level(logging.WARNING):
                    env._build_or_load_faiss_index()
            else:
                env._build_or_load_faiss_index()

        return mock_st_cls, mock_dump

    def test_mismatch_triggers_rebuild(self, caplog):
        """Cache with wrong embedding count is discarded and rebuilt."""
        import numpy as np
        env = self._make_env(doc_count=3)
        # Cache has 5 embeddings, graph has 3 nodes → mismatch
        stale_embeddings = np.zeros((5, 4), dtype=np.float32)
        mock_st_cls, _ = self._run_with_mocks(env, "aaaa1111", stale_embeddings, caplog=caplog)
        assert any("mismatch" in r.message.lower() for r in caplog.records)
        mock_st_cls.assert_called_once()

    def test_matching_cache_is_reused(self):
        """Cache with correct embedding count is used: no new pickle is written."""
        import numpy as np
        env = self._make_env(doc_count=3)
        good_embeddings = np.zeros((3, 4), dtype=np.float32)
        _, mock_dump = self._run_with_mocks(env, "bbbb2222", good_embeddings)
        # No re-encoding → no new pickle written
        mock_dump.assert_not_called()
