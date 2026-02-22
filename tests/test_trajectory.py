"""
Tests for GraphToTAgent.get_trajectory_text.

Pure static method — no mocking needed. Reads a trajectory dict from a
dspy.Prediction and formats it into the Thought/Action/Obs log that feeds
ScoreVoteSignature's reasoning_trace field.
"""

import dspy

from src.graph_tot.dspy_modules import GraphToTAgent


def _pred(trajectory: dict) -> dspy.Prediction:
    return dspy.Prediction(answer="x", trajectory=trajectory)


class TestGetTrajectoryText:

    def test_empty_trajectory_returns_empty_string(self):
        pred = _pred({})
        assert GraphToTAgent.get_trajectory_text(pred) == ""

    def test_no_trajectory_attribute_returns_empty_string(self):
        pred = dspy.Prediction(answer="x")
        assert GraphToTAgent.get_trajectory_text(pred) == ""

    def test_single_step_contains_thought(self):
        pred = _pred({
            "thought_0": "I need to look up diabetes",
            "tool_name_0": "retrieve_node",
            "tool_args_0": {"keyword": "diabetes"},
            "observation_0": "Node ID: Disease::D001",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        assert "I need to look up diabetes" in text

    def test_single_step_contains_action(self):
        pred = _pred({
            "thought_0": "step",
            "tool_name_0": "retrieve_node",
            "tool_args_0": {},
            "observation_0": "result",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        assert "retrieve_node" in text

    def test_single_step_contains_observation(self):
        pred = _pred({
            "thought_0": "step",
            "tool_name_0": "t",
            "tool_args_0": {},
            "observation_0": "Node ID: Disease::D001",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        assert "Node ID: Disease::D001" in text

    def test_step_numbers_are_1_indexed(self):
        pred = _pred({
            "thought_0": "first step",
            "tool_name_0": "t",
            "tool_args_0": {},
            "observation_0": "obs",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        assert "Thought 1:" in text
        assert "Thought 0:" not in text

    def test_two_steps_produce_six_lines(self):
        pred = _pred({
            "thought_0": "A",
            "tool_name_0": "t1",
            "tool_args_0": {},
            "observation_0": "B",
            "thought_1": "C",
            "tool_name_1": "t2",
            "tool_args_1": {},
            "observation_1": "D",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        non_empty_lines = [line for line in text.split("\n") if line.strip()]
        assert len(non_empty_lines) == 6

    def test_terminates_at_gap_in_thought_keys(self):
        """Loop stops when thought_N is absent; orphan steps beyond the gap are skipped."""
        pred = _pred({
            "thought_0": "step zero",
            "tool_name_0": "t0",
            "tool_args_0": {},
            "observation_0": "obs0",
            # thought_1 intentionally absent
            "thought_2": "orphan step",
            "tool_name_2": "t2",
            "tool_args_2": {},
            "observation_2": "obs2",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        assert "step zero" in text
        assert "orphan step" not in text

    def test_second_step_labeled_thought_2(self):
        pred = _pred({
            "thought_0": "A",
            "tool_name_0": "t",
            "tool_args_0": {},
            "observation_0": "B",
            "thought_1": "C",
            "tool_name_1": "t",
            "tool_args_1": {},
            "observation_1": "D",
        })
        text = GraphToTAgent.get_trajectory_text(pred)
        assert "Thought 2:" in text
        assert "Action  2:" in text
        assert "Obs     2:" in text
