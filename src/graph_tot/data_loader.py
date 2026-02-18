"""
Data loading utilities for GRBench benchmark.

  load_grbench_qa()     — load QA pairs from HuggingFace (PeterJinGo/GRBench)
  find_graph_file()     — locate graph data file in a directory
  check_graph_available() — silent boolean check (no exception)

Graph data (nodes + edges) must be downloaded manually from Google Drive:
  https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GRBENCH_HF_REPO = "PeterJinGo/GRBench"
GRAPH_DRIVE_URL = (
    "https://drive.google.com/drive/folders/1DJIgRZ3G-TOf7h0-Xub5_sE4slBUEqy9"
)

# Candidate filenames probed by find_graph_file()
GRAPH_FILE_CANDIDATES = [
    "graph.json",
    "graph.pkl",
    "graph.pickle",
    "biomedical.json",
    "biomedical.pkl",
    "healthcare.json",
    "healthcare.pkl",
    "hetionet.json",
    "hetionet.pkl",
    "data.json",
    "data.pkl",
]

DOWNLOAD_INSTRUCTIONS = """\
Graph data files for the GRBench Healthcare domain are not present.

To download:
  1. Visit the Google Drive folder:
       {drive_url}
  2. Download the 'healthcare' (or equivalent) folder.
  3. Place the graph file at:
       {graph_dir}

The expected file is a JSON or pickle with this structure:
  {{
    "Disease": {{
      "<node_id>": {{
        "features": {{"name": "...", ...}},
        "neighbors": {{"Compound-treats-Disease": ["<id>", ...], ...}}
      }},
      ...
    }},
    "Compound": {{ ... }},
    ...
  }}

After downloading, re-run this script.
"""


@dataclass
class QAPair:
    """A single question-answer pair from GRBench."""
    qid: str
    question: str
    answer: str
    level: str  # 'easy', 'medium', or 'hard'


def load_grbench_qa(
    domain: str = "healthcare",
    split: str = "test",
    max_samples: Optional[int] = None,
    level_filter: Optional[str] = None,
) -> list[QAPair]:
    """
    Load QA pairs from the GRBench HuggingFace dataset.

    Args:
        domain:       Dataset config name (e.g., 'healthcare', 'biology').
        split:        Dataset split — 'test' is the only available split.
        max_samples:  Cap the number of returned pairs; None = all.
        level_filter: If set, only return questions of this difficulty
                      ('easy', 'medium', or 'hard').

    Returns:
        List of QAPair objects.
    """
    from datasets import load_dataset

    logger.info(
        "Loading GRBench '%s' (split=%s) from HuggingFace...", domain, split
    )
    dataset = load_dataset(GRBENCH_HF_REPO, name=domain, split=split)

    pairs: list[QAPair] = []
    for row in dataset:
        if level_filter and row["level"] != level_filter:
            continue
        pairs.append(QAPair(
            qid=str(row["qid"]),
            question=str(row["question"]),
            answer=str(row["answer"]),
            level=str(row["level"]),
        ))

    if max_samples is not None and max_samples > 0:
        pairs = pairs[:max_samples]

    logger.info(
        "Loaded %d QA pairs (domain=%s, level=%s)",
        len(pairs),
        domain,
        level_filter or "all",
    )
    return pairs


def find_graph_file(graph_dir: str | Path) -> Path:
    """
    Locate the graph data file within graph_dir.

    If graph_dir is itself a file path, return it directly.
    Otherwise, probe a list of candidate filenames within the directory.

    Raises:
        FileNotFoundError with download instructions if no file is found.
    """
    graph_dir = Path(graph_dir)

    if graph_dir.is_file():
        return graph_dir

    if not graph_dir.exists():
        raise FileNotFoundError(
            DOWNLOAD_INSTRUCTIONS.format(
                drive_url=GRAPH_DRIVE_URL,
                graph_dir=graph_dir / "graph.json",
            )
        )

    for name in GRAPH_FILE_CANDIDATES:
        candidate = graph_dir / name
        if candidate.exists():
            logger.info("Found graph file: %s", candidate)
            return candidate

    existing = sorted(p.name for p in graph_dir.iterdir() if p.is_file())
    raise FileNotFoundError(
        f"No recognised graph file found in: {graph_dir}\n"
        f"Files present: {existing}\n\n"
        + DOWNLOAD_INSTRUCTIONS.format(
            drive_url=GRAPH_DRIVE_URL,
            graph_dir=graph_dir / "graph.json",
        )
    )


def check_graph_available(graph_dir: str | Path) -> bool:
    """
    Return True if a graph data file is found in graph_dir, False otherwise.

    Never raises; intended for pre-flight checks before expensive operations.
    """
    try:
        find_graph_file(graph_dir)
        return True
    except FileNotFoundError:
        return False
