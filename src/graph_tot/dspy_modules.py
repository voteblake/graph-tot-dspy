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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import dspy
import litellm

logger = logging.getLogger(__name__)


# ===========================================================================
# DSPy Signatures
# ===========================================================================


class GraphQASignature(dspy.Signature):
    """Answer a question by exploring a domain-specific knowledge graph."""

    question: str = dspy.InputField(desc="Question to answer using the knowledge graph.")
    context: str = dspy.InputField(
        desc="Prior reasoning from an earlier branch, if available.",
        default="",
    )
    answer: str = dspy.OutputField(
        desc="Concise, factual answer derived from the knowledge graph."
    )


class ScoreVoteSignature(dspy.Signature):
    """Estimate the probability that a reasoning trace produces the correct answer."""

    question: str = dspy.InputField(desc="The original question.")
    reasoning_trace: str = dspy.InputField(
        desc="Thought/Action/Observation trace from graph exploration."
    )
    candidate_answer: str = dspy.InputField(
        desc="The candidate answer to evaluate."
    )
    score: float = dspy.OutputField(
        desc="P(correct) from 0.0 to 1.0."
    )


class SelectionVoteSignature(dspy.Signature):
    """Select the best answer from a set of knowledge-graph-grounded candidates."""

    question: str = dspy.InputField(desc="The original question.")
    candidates: str = dspy.InputField(
        desc="Numbered candidate answers with reasoning summaries."
    )
    best_index: int = dspy.OutputField(
        desc="0-based index of the best candidate."
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
        return self.react(question=question, context=context)

    @staticmethod
    def get_trajectory_text(prediction: dspy.Prediction) -> str:
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

    def forward(self, question: str, branches: list[Branch]) -> list[Branch]:
        """Score branches and return them sorted best-first.

        Args:
            question: The original question.
            branches: Branch objects to evaluate.

        Returns:
            Branches sorted descending by score.
        """
        if not branches:
            return []
        if self.mode == "score_vote":
            return self._score_vote(question, branches)
        return self._selection_vote(question, branches)

    def _score_vote(self, question: str, branches: list[Branch]) -> list[Branch]:
        scored: list[Branch] = []
        for branch in branches:
            result = self.score_voter(
                question=question,
                reasoning_trace=branch.trace,
                candidate_answer=branch.answer,
            )
            # DSPy's typed field guarantees result.score is a float;
            # clamp to valid probability range.
            score = max(0.0, min(1.0, result.score))
            scored.append(Branch(
                answer=branch.answer,
                trace=branch.trace,
                prediction=branch.prediction,
                score=score,
                parent_context=branch.parent_context,
            ))
        return sorted(scored, key=lambda b: b.score, reverse=True)

    def _selection_vote(self, question: str, branches: list[Branch]) -> list[Branch]:
        candidates_text = "\n\n".join(
            f"[{i}] Answer: {b.answer}\n"
            f"    Reasoning: {b.trace[:2000]}"
            for i, b in enumerate(branches)
        )
        result = self.selection_voter(
            question=question,
            candidates=candidates_text,
        )
        # DSPy's typed field guarantees result.best_index is an int;
        # clamp to valid index range.
        best_idx = max(0, min(result.best_index, len(branches) - 1))

        scored = [
            Branch(
                answer=b.answer,
                trace=b.trace,
                prediction=b.prediction,
                score=1.0 if i == best_idx else 0.0,
                parent_context=b.parent_context,
            )
            for i, b in enumerate(branches)
        ]
        return sorted(scored, key=lambda b: b.score, reverse=True)


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
        parallel: bool = True,
    ) -> None:
        super().__init__()
        self.k = k
        self.b = b
        self.max_rounds = max_rounds
        self.parallel = parallel

        # Store for creating per-thread agent instances in parallel mode
        self.graph_env = graph_env
        self.max_iters = max_iters

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

        all_branch_dicts: list[dict] = []  # accumulate ALL scored branches across all rounds
        round_log: list[dict] = []  # per-round summaries for observability

        for round_idx in range(1, self.max_rounds + 1):
            # Score and prune — evaluator accepts and returns Branch objects directly
            current_beam = self.evaluator(question=question, branches=current_beam)
            survivors = current_beam[: self.b]

            # Accumulate scored branches with round-level metadata for inspection
            for rank, branch in enumerate(current_beam):
                d = branch.as_dict()
                d["round_idx"] = round_idx
                d["rank"] = rank
                d["is_survivor"] = rank < self.b
                all_branch_dicts.append(d)

            round_log.append({
                "round": round_idx,
                "num_branches": len(current_beam),
                "scores": [b.score for b in current_beam],
                "survivor_answers": [s.answer[:120] for s in survivors],
                "context_from_previous": current_beam[0].parent_context[:120] if current_beam and current_beam[0].parent_context else None,
            })

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
            all_branches=all_branch_dicts,
            round_log=round_log,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _generate_branches(self, question: str, contexts: list[str]) -> list[Branch]:
        """Run len(contexts) independent agent calls and return Branch objects.

        When ``self.parallel`` is True and there are multiple contexts, branches
        are generated concurrently using a :class:`~concurrent.futures.ThreadPoolExecutor`.
        Each thread gets its own ``GraphToTAgent`` instance to avoid any
        thread-safety issues with DSPy's internal ReAct state.  The shared
        ``GraphEnvironment`` is safe to access concurrently since graph lookups
        and FAISS searches are read-only.
        """
        if self.parallel and len(contexts) > 1:
            return self._generate_branches_parallel(question, contexts)
        return self._generate_branches_sequential(question, contexts)

    def _generate_branches_sequential(
        self, question: str, contexts: list[str],
    ) -> list[Branch]:
        """Generate branches one at a time using the shared agent instance."""
        branches: list[Branch] = []
        for context in contexts:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    pred = self.agent(question=question, context=context)
                    trace = self.agent.get_trajectory_text(pred)
                    answer = getattr(pred, "answer", "") or "No answer produced."
                    branches.append(Branch(
                        answer=answer,
                        trace=trace,
                        prediction=pred,
                        parent_context=context,
                    ))
                    break
                except litellm.RateLimitError:
                    if attempt < max_retries - 1:
                        wait = 60 * (attempt + 1)
                        logger.warning(
                            "Rate limited (attempt %d/%d), waiting %ds before retry",
                            attempt + 1, max_retries, wait,
                        )
                        time.sleep(wait)
                    else:
                        logger.error("Rate limit retries exhausted")
                        branches.append(Branch(
                            answer="Rate limit exceeded — try again later or reduce --k.",
                            trace="",
                            prediction=dspy.Prediction(answer="Rate limit exceeded", trajectory={}),
                            parent_context=context,
                        ))
        return branches

    def _generate_single_branch(
        self, question: str, context: str,
    ) -> Branch:
        """Generate one branch with its own agent instance (thread-safe).

        Creating a dedicated ``GraphToTAgent`` per call avoids sharing any
        mutable DSPy ReAct state across threads while the underlying
        ``GraphEnvironment`` (read-only graph + FAISS index) is safely shared.

        If litellm's built-in retries are exhausted on a rate-limit error,
        this method retries with longer backoff (60 s, 120 s) appropriate for
        per-minute token limits before falling back to an error prediction.
        """
        agent = GraphToTAgent(
            graph_env=self.graph_env, max_iters=self.max_iters,
        )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pred = agent(question=question, context=context)
                trace = agent.get_trajectory_text(pred)
                answer = getattr(pred, "answer", "") or "No answer produced."
                return Branch(
                    answer=answer,
                    trace=trace,
                    prediction=pred,
                    parent_context=context,
                )
            except litellm.RateLimitError:
                if attempt < max_retries - 1:
                    wait = 60 * (attempt + 1)  # 60s, 120s
                    logger.warning(
                        "Rate limited (attempt %d/%d), waiting %ds before retry",
                        attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Rate limit retries exhausted after %d attempts",
                        max_retries,
                    )
                    return Branch(
                        answer="Rate limit exceeded — try again later or reduce --k.",
                        trace="",
                        prediction=dspy.Prediction(answer="Rate limit exceeded", trajectory={}),
                        parent_context=context,
                    )

    def _generate_branches_parallel(
        self, question: str, contexts: list[str],
    ) -> list[Branch]:
        """Generate branches concurrently via ThreadPoolExecutor.

        LLM API calls are I/O-bound, so threads provide near-ideal speed-up
        (≈ k× wall-clock reduction).  Each thread creates its own
        ``GraphToTAgent`` to guarantee thread safety.  Results are collected
        in submission order so branch indices remain deterministic.
        """
        logger.info(
            "Generating %d branches in parallel", len(contexts),
        )
        with ThreadPoolExecutor(max_workers=len(contexts)) as pool:
            futures = [
                pool.submit(self._generate_single_branch, question, ctx)
                for ctx in contexts
            ]
            # Collect in submission order to keep branch indices stable
            return [f.result() for f in futures]

