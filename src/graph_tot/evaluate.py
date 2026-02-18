"""
Evaluation utilities for Graph ToT QA results.

Implements ROUGE-L F1 scoring as used in arXiv:2502.13247, with per-difficulty
breakdown (easy / medium / hard) and JSON result export.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    qid: str
    question: str
    gold_answer: str
    predicted_answer: str
    level: str
    rouge_l: float


@dataclass
class EvalReport:
    """Aggregated evaluation results across all questions."""
    results: list[EvalResult] = field(default_factory=list)

    @property
    def overall_rouge_l(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.rouge_l for r in self.results) / len(self.results)

    def rouge_l_by_level(self) -> dict[str, float]:
        by_level: dict[str, list[float]] = {}
        for r in self.results:
            by_level.setdefault(r.level, []).append(r.rouge_l)
        return {
            level: sum(scores) / len(scores)
            for level, scores in by_level.items()
        }

    def count_by_level(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.results:
            counts[r.level] = counts.get(r.level, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            f"Overall ROUGE-L : {self.overall_rouge_l:.4f}",
            f"Total questions : {len(self.results)}",
            "",
            "Per-difficulty breakdown:",
        ]
        counts = self.count_by_level()
        for level, score in sorted(self.rouge_l_by_level().items()):
            n = counts.get(level, 0)
            lines.append(f"  {level:8s} ({n:3d} questions): ROUGE-L = {score:.4f}")
        return "\n".join(lines)

    def save_json(self, output_path: str | Path) -> None:
        """Save full results and aggregates to a JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "overall_rouge_l": self.overall_rouge_l,
            "by_level": self.rouge_l_by_level(),
            "count_by_level": self.count_by_level(),
            "results": [asdict(r) for r in self.results],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation results saved to %s", output_path)


class RougeEvaluator:
    """
    Computes ROUGE-L F1 scores using Google's rouge-score library.

    ROUGE-L measures the longest common subsequence between the predicted
    answer and the gold answer, as used in arXiv:2502.13247.
    """

    def __init__(self, use_stemmer: bool = True) -> None:
        from rouge_score import rouge_scorer
        self._scorer = rouge_scorer.RougeScorer(
            ["rougeL"],
            use_stemmer=use_stemmer,
        )

    def score_single(self, prediction: str, reference: str) -> float:
        """Return ROUGE-L F1 for a single prediction/reference pair."""
        scores = self._scorer.score(
            target=reference.strip(),
            prediction=prediction.strip(),
        )
        return scores["rougeL"].fmeasure

    def evaluate(
        self,
        qa_pairs,           # list of QAPair
        predictions: list[str],
    ) -> EvalReport:
        """
        Evaluate predicted answers against gold answers.

        Args:
            qa_pairs:    List of QAPair objects (from data_loader).
            predictions: List of predicted answer strings, same order as qa_pairs.

        Returns:
            EvalReport with per-question results and aggregated metrics.
        """
        if len(qa_pairs) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(qa_pairs)} QA pairs "
                f"but {len(predictions)} predictions."
            )

        report = EvalReport()
        for qa, pred in zip(qa_pairs, predictions):
            rouge_l = self.score_single(prediction=pred, reference=qa.answer)
            report.results.append(EvalResult(
                qid=qa.qid,
                question=qa.question,
                gold_answer=qa.answer,
                predicted_answer=pred,
                level=qa.level,
                rouge_l=rouge_l,
            ))

        logger.info(
            "Evaluation complete: %d questions, overall ROUGE-L = %.4f",
            len(report.results),
            report.overall_rouge_l,
        )
        return report
