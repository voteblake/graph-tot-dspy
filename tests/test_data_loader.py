"""
Tests for data_loader pure functions that don't require network or graph files.

find_graph_file, check_graph_available, train_val_split, and make_dspy_examples
are all exercisable with tmp_path and in-memory data.
"""

import dspy
import pytest

from src.graph_tot.data_loader import (
    GRAPH_DRIVE_URL,
    GRAPH_FILE_CANDIDATES,
    QAPair,
    check_graph_available,
    find_graph_file,
    make_dspy_examples,
    train_val_split,
)


# ---------------------------------------------------------------------------
# find_graph_file
# ---------------------------------------------------------------------------


class TestFindGraphFile:

    def test_returns_path_when_candidate_file_exists(self, tmp_path):
        graph = tmp_path / "graph.json"
        graph.write_text("{}")
        result = find_graph_file(tmp_path)
        assert result == graph

    def test_accepts_direct_file_path(self, tmp_path):
        graph = tmp_path / "myfile.json"
        graph.write_text("{}")
        result = find_graph_file(graph)
        assert result == graph

    def test_raises_file_not_found_when_directory_missing(self, tmp_path):
        missing = tmp_path / "no_such_dir"
        with pytest.raises(FileNotFoundError):
            find_graph_file(missing)

    def test_raises_when_no_candidates_present(self, tmp_path):
        (tmp_path / "unrelated.txt").write_text("hi")
        with pytest.raises(FileNotFoundError):
            find_graph_file(tmp_path)

    def test_error_message_contains_drive_url(self, tmp_path):
        missing = tmp_path / "absent"
        with pytest.raises(FileNotFoundError, match="drive.google.com"):
            find_graph_file(missing)

    def test_finds_pkl_when_json_absent(self, tmp_path):
        pkl = tmp_path / "graph.pkl"
        pkl.write_bytes(b"data")
        result = find_graph_file(tmp_path)
        assert result == pkl

    def test_prefers_first_candidate_in_list(self, tmp_path):
        # Create two candidates; the first in GRAPH_FILE_CANDIDATES order should win
        first = GRAPH_FILE_CANDIDATES[0]
        second = GRAPH_FILE_CANDIDATES[1]
        (tmp_path / second).write_bytes(b"second")
        (tmp_path / first).write_text("{}")
        result = find_graph_file(tmp_path)
        assert result.name == first


# ---------------------------------------------------------------------------
# check_graph_available
# ---------------------------------------------------------------------------


class TestCheckGraphAvailable:

    def test_returns_true_when_graph_found(self, tmp_path):
        (tmp_path / "graph.json").write_text("{}")
        assert check_graph_available(tmp_path) is True

    def test_returns_false_when_directory_absent(self, tmp_path):
        assert check_graph_available(tmp_path / "no_such") is False

    def test_returns_false_when_no_candidate_files(self, tmp_path):
        (tmp_path / "unrelated.csv").write_text("data")
        assert check_graph_available(tmp_path) is False

    def test_never_raises(self, tmp_path):
        result = check_graph_available(tmp_path / "totally_missing")
        assert isinstance(result, bool)



# ---------------------------------------------------------------------------
# train_val_split
# ---------------------------------------------------------------------------


def _make_pairs(n: int) -> list[QAPair]:
    return [QAPair(qid=str(i), question=f"Q{i}?", answer=f"A{i}", level="easy") for i in range(n)]


class TestTrainValSplit:

    def test_sizes_sum_to_total(self):
        pairs = _make_pairs(10)
        train, val = train_val_split(pairs, train_frac=0.2)
        assert len(train) + len(val) == 10

    def test_train_fraction_respected(self):
        pairs = _make_pairs(10)
        train, val = train_val_split(pairs, train_frac=0.3)
        assert len(train) == 3
        assert len(val) == 7

    def test_minimum_one_train_example(self):
        pairs = _make_pairs(5)
        train, val = train_val_split(pairs, train_frac=0.0)
        assert len(train) >= 1

    def test_reproducible_with_same_seed(self):
        pairs = _make_pairs(20)
        train1, _ = train_val_split(pairs, seed=42)
        train2, _ = train_val_split(pairs, seed=42)
        assert [e.question for e in train1] == [e.question for e in train2]

    def test_different_seeds_produce_different_splits(self):
        pairs = _make_pairs(20)
        train1, _ = train_val_split(pairs, seed=42)
        train2, _ = train_val_split(pairs, seed=99)
        assert [e.question for e in train1] != [e.question for e in train2]

    def test_returns_dspy_examples(self):
        pairs = _make_pairs(5)
        train, val = train_val_split(pairs)
        assert all(isinstance(e, dspy.Example) for e in train)
        assert all(isinstance(e, dspy.Example) for e in val)



# ---------------------------------------------------------------------------
# make_dspy_examples
# ---------------------------------------------------------------------------


class TestMakeDspyExamples:

    def test_fields_mapped_from_qa_pair(self):
        pairs = [QAPair("1", "What is diabetes?", "A metabolic condition", "easy")]
        ex = make_dspy_examples(pairs)[0]
        assert ex.question == "What is diabetes?"
        assert ex.answer == "A metabolic condition"
