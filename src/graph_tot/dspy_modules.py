"""
DSPy modules implementing Graph Tree-of-Thought with Agent-based retrieval.

Based on arXiv:2502.13247 "Grounding LLM Reasoning with Knowledge Graphs".

Architecture:
  GraphToTAgent         — single ReAct agent trace over the knowledge graph
  TreeOfThoughtEvaluator — scores/ranks a set of candidate branches
  GraphToTSolver        — beam search orchestrator: generates k branches,
                          prunes to top-b, optionally repeats for R rounds
"""

import logging
from dataclasses import dataclass, field

import dspy

logger = logging.getLogger(__name__)


# ===========================================================================
# DSPy Signatures
# ===========================================================================


class GraphQASignature(dspy.Signature):
    """You are a medical knowledge graph expert. Use the available graph tools
    to answer questions about diseases, compounds, genes, symptoms, and their
    relationships. Always start by using retrieve_node to locate relevant
    entities in the graph, then explore their features and neighbors to build
    a complete, factual answer. Be precise and cite specific node IDs when
    possible."""

    question: str = dspy.InputField(
        desc="The medical question to answer using the knowledge graph."
    )
    context: str = dspy.InputField(
        desc="Optional prior reasoning or partial answer from an earlier branch. "
             "Use this as a starting point and try to improve or verify it.",
        default="",
    )
    answer: str = dspy.OutputField(
        desc="A concise, factual answer derived from knowledge graph exploration."
    )


class ScoreVoteSignature(dspy.Signature):
    """You are evaluating the quality of a knowledge graph reasoning trace.
    Assess it on: (1) logical soundness of graph traversal steps,
    (2) completeness and specificity of the answer, (3) direct relevance to
    the question. Return a single float score."""

    question: str = dspy.InputField(desc="The original medical question.")
    reasoning_trace: str = dspy.InputField(
        desc="The full Thought/Action/Observation trace from graph exploration."
    )
    candidate_answer: str = dspy.InputField(
        desc="The candidate answer produced by this reasoning trace."
    )
    score: float = dspy.OutputField(
        desc="Quality score from 0.0 (worst) to 1.0 (best)."
    )


class SelectionVoteSignature(dspy.Signature):
    """You are selecting the best answer from multiple candidates that each
    explored a medical knowledge graph. Choose the one that is most correct,
    complete, and well-supported by the reasoning trace."""

    question: str = dspy.InputField(desc="The original medical question.")
    candidates: str = dspy.InputField(
        desc="Numbered list (0-indexed) of candidate answers with reasoning summaries."
    )
    best_index: int = dspy.OutputField(
        desc="0-based integer index of the best candidate answer."
    )


# ===========================================================================
# Branch dataclass
# ===========================================================================


@dataclass
class Branch:
    """Represents one complete reasoning trace from a GraphToTAgent run."""

    answer: str
    trace: str
    prediction: dspy.Prediction
    score: float = 0.0
    parent_context: str = ""

    def as_dict(self) -> dict:
        return {
            "answer": self.answer,
            "trace": self.trace,
            "score": self.score,
            "parent_context": self.parent_context,
        }


# ===========================================================================
# Module 1: GraphToTAgent — single ReAct trace
# ===========================================================================


class GraphToTAgent(dspy.Module):
    """
    A single Graph Tree-of-Thought reasoning branch.

    Wraps dspy.ReAct with the four graph tools (retrieve_node, node_feature,
    neighbour_check, node_degree). Can be called multiple times independently
    to produce diverse candidate reasoning traces for beam search.
    """

    def __init__(self, graph_env, max_iters: int = 10) -> None:
        super().__init__()
        self.graph_env = graph_env
        self.react = dspy.ReAct(
            signature=GraphQASignature,
            tools=graph_env.get_tools(),
            max_iters=max_iters,
        )

    def forward(self, question: str, context: str = "") -> dspy.Prediction:
        """Run the ReAct agent and return a prediction with trajectory."""
        try:
            return self.react(question=question, context=context)
        except Exception as exc:
            logger.warning("GraphToTAgent.forward failed: %s", exc)
            return dspy.Prediction(answer=f"Agent error: {exc}", trajectory={})

    def get_trajectory_text(self, prediction: dspy.Prediction) -> str:
        """Format the trajectory dict into a human-readable Thought/Action/Observation log."""
        traj = getattr(prediction, "trajectory", {}) or {}
        if not traj:
            return ""
        lines: list[str] = []
        i = 0
        while f"thought_{i}" in traj:
            thought = traj.get(f"thought_{i}", "")
            tool = traj.get(f"tool_name_{i}", "")
            args = traj.get(f"tool_args_{i}", "")
            obs = traj.get(f"observation_{i}", "")
            lines.append(f"Thought {i + 1}: {thought}")
            lines.append(f"Action  {i + 1}: {tool}({args})")
            lines.append(f"Obs     {i + 1}: {obs}")
            i += 1
        return "\n".join(lines)


# ===========================================================================
# Module 2: TreeOfThoughtEvaluator — score and rank branches
# ===========================================================================


class TreeOfThoughtEvaluator(dspy.Module):
    """
    Scores and ranks candidate branches produced by GraphToTAgent.

    Two evaluation modes (matching the paper):
      score_vote     — each branch is scored independently (0-1 float)
      selection_vote — one LLM call selects the best branch by index
    """

    def __init__(self, mode: str = "score_vote") -> None:
        super().__init__()
        if mode not in ("score_vote", "selection_vote"):
            raise ValueError("mode must be 'score_vote' or 'selection_vote'")
        self.mode = mode
        self.score_voter = dspy.Predict(ScoreVoteSignature)
        self.selection_voter = dspy.Predict(SelectionVoteSignature)

    def forward(self, question: str, branches: list[dict]) -> list[dict]:
        """
        Score branches and return them sorted best-first with a 'score' key added.

        Args:
            question: The original question.
            branches: List of dicts with keys 'answer', 'trace', 'prediction'.

        Returns:
            The same list sorted descending by score, with 'score' added.
        """
        if not branches:
            return []
        if self.mode == "score_vote":
            return self._score_vote(question, branches)
        return self._selection_vote(question, branches)

    def _score_vote(self, question: str, branches: list[dict]) -> list[dict]:
        scored: list[dict] = []
        for branch in branches:
            try:
                result = self.score_voter(
                    question=question,
                    reasoning_trace=branch.get("trace", ""),
                    candidate_answer=branch.get("answer", ""),
                )
                score = float(result.score)
                score = max(0.0, min(1.0, score))
            except Exception as exc:
                logger.warning("Score vote failed: %s; defaulting to 0.0", exc)
                score = 0.0
            scored.append({**branch, "score": score})
        return sorted(scored, key=lambda x: x["score"], reverse=True)

    def _selection_vote(self, question: str, branches: list[dict]) -> list[dict]:
        candidates_text = "\n\n".join(
            f"[{i}] Answer: {b.get('answer', '')}\n"
            f"    Reasoning: {b.get('trace', '')[:300]}..."
            for i, b in enumerate(branches)
        )
        try:
            result = self.selection_voter(
                question=question,
                candidates=candidates_text,
            )
            best_idx = int(result.best_index)
            best_idx = max(0, min(best_idx, len(branches) - 1))
        except Exception as exc:
            logger.warning("Selection vote failed: %s; defaulting to index 0", exc)
            best_idx = 0

        scored = [
            {**b, "score": 1.0 if i == best_idx else 0.0}
            for i, b in enumerate(branches)
        ]
        return sorted(scored, key=lambda x: x["score"], reverse=True)


# ===========================================================================
# Module 3: GraphToTSolver — beam search orchestrator
# ===========================================================================


class GraphToTSolver(dspy.Module):
    """
    Orchestrates Tree-of-Thought beam search over GraphToTAgent traces.

    Algorithm (matching arXiv:2502.13247):
      1. Generate k candidate branches by running k independent ReAct agent calls.
      2. Score all branches via TreeOfThoughtEvaluator.
      3. Keep the top-b branches (beam pruning).
      4. Optionally repeat for max_rounds rounds (beam expansion), passing each
         survivor's answer as 'context' to the next round's agents.
      5. Return the best branch's answer directly.

    Parameters:
      graph_env  : GraphEnvironment instance with the 4 graph tools.
      k          : Branching factor — number of independent agent calls per round.
      b          : Beam width — number of branches kept after scoring.
      max_rounds : Number of beam search rounds (1 = single-pass ToT).
      max_iters  : Maximum ReAct iterations per agent branch.
      eval_mode  : 'score_vote' or 'selection_vote'.

    Note on diversity: set temperature >= 0.7 on dspy.LM for meaningfully
    different branches. At temperature=0.0, all k branches will be identical.
    """

    def __init__(
        self,
        graph_env,
        k: int = 3,
        b: int = 1,
        max_rounds: int = 1,
        max_iters: int = 10,
        eval_mode: str = "score_vote",
    ) -> None:
        super().__init__()
        self.k = k
        self.b = b
        self.max_rounds = max_rounds

        self.agent = GraphToTAgent(graph_env=graph_env, max_iters=max_iters)
        self.evaluator = TreeOfThoughtEvaluator(mode=eval_mode)

    def forward(self, question: str) -> dspy.Prediction:
        """
        Run ToT beam search and return the final answer.

        Returns a dspy.Prediction with fields:
          answer      — answer string from best branch
          best_trace  — reasoning trace from the best-scoring branch
          best_score  — float score of the best branch
          all_branches — list of branch dicts (for logging/inspection)
        """
        # Round 0: seed beam with k independent branches
        logger.info("ToT Round 0: generating %d branches", self.k)
        current_beam = self._generate_branches(question, contexts=[""] * self.k)

        for round_idx in range(1, self.max_rounds + 1):
            # Score and prune
            scored_dicts = self.evaluator(
                question=question,
                branches=[b.as_dict() for b in current_beam],
            )
            current_beam = self._dicts_to_branches(scored_dicts, current_beam)
            survivors = current_beam[: self.b]

            logger.info(
                "ToT Round %d: top-%d scores = %s",
                round_idx,
                self.b,
                [f"{s.score:.3f}" for s in survivors],
            )

            if round_idx == self.max_rounds:
                current_beam = survivors
                break

            # Expand survivors: each generates k new branches, guided by
            # the survivor's answer as context
            new_beam: list[Branch] = []
            for survivor in survivors:
                # Truncate context to avoid hitting context-window limits
                context = survivor.answer[:800]
                new_beam.extend(
                    self._generate_branches(question, contexts=[context] * self.k)
                )
            current_beam = new_beam

        # Return best branch's answer directly (no synthesis step, per the paper)
        best = current_beam[0] if current_beam else Branch(
            answer="Unable to determine an answer.",
            trace="",
            prediction=dspy.Prediction(),
        )

        return dspy.Prediction(
            answer=best.answer,
            best_trace=best.trace,
            best_score=best.score,
            all_branches=[b.as_dict() for b in current_beam],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_branches(self, question: str, contexts: list[str]) -> list[Branch]:
        """Run len(contexts) independent agent calls and return Branch objects."""
        branches: list[Branch] = []
        for context in contexts:
            pred = self.agent(question=question, context=context)
            trace = self.agent.get_trajectory_text(pred)
            answer = getattr(pred, "answer", "") or "No answer produced."
            branches.append(Branch(
                answer=answer,
                trace=trace,
                prediction=pred,
                parent_context=context,
            ))
        return branches

    def _dicts_to_branches(
        self,
        scored_dicts: list[dict],
        original_branches: list[Branch],
    ) -> list[Branch]:
        """Merge scores from evaluator output back into Branch objects."""
        result: list[Branch] = []
        # evaluator returns scored_dicts in sorted order; match by answer text
        orig_by_answer = {b.answer: b for b in original_branches}
        for d in scored_dicts:
            orig = orig_by_answer.get(d.get("answer", ""))
            if orig is None:
                # Fallback: reconstruct from dict
                orig = Branch(
                    answer=d.get("answer", ""),
                    trace=d.get("trace", ""),
                    prediction=dspy.Prediction(),
                )
            result.append(Branch(
                answer=orig.answer,
                trace=orig.trace,
                prediction=orig.prediction,
                score=d.get("score", 0.0),
                parent_context=orig.parent_context,
            ))
        return result
