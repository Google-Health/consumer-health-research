"""Regression tests for expert answers and final response evidence."""

from unittest.mock import Mock

import pytest

from pha.agents.orchestrator import MultiAgentOrchestrator
from pha.prompts.orchestrator_prompts import (
    AGENT_NAME_DATA_SCIENCE,
    AGENT_NAME_DOMAIN_EXPERT,
    FINISH_DESC_PROMPT,
    GOAL_PROMPT,
    ORCHESTRATOR_FINAL_RESPONSE_IMPROVEMENT_PROMPT,
    TOPIC_PROMPT,
)


@pytest.fixture
def expert():
    pytest.importorskip("onetwo.agents.react")
    from pha.agents.domain_expert_agent import DomainExpertAgent, is_react_available

    if not is_react_available():
        pytest.skip("Domain expert ReAct dependencies are unavailable")
    agent = DomainExpertAgent(search_backend="duckduckgo")
    agent.setup_agent()
    return agent


@pytest.mark.parametrize("finish_marker", ["[Finish]:", "[Finish]"])
def test_expert_retains_multiline_final_answer(expert, finish_marker):
    answer = (
        "Sleep summary\n\n"
        "There are 30 valid nights from 2024-11-16 to 2024-12-15. "
        "Mean sleep duration is 412.63 minutes.\n\n"
        "Sixteen nights were below seven hours. "
        "Duration alone does not establish overall sleep quality."
    )
    step = expert.domain_expert_agent.parse(f"{finish_marker} {answer}")
    state = Mock(updates=[step])

    assert step.is_finished
    assert expert._extract_clean_answer(step.observation, state) == answer


def test_expert_still_parses_search_actions(expert):
    step = expert.domain_expert_agent.parse(
        "[Act]: `search('adult sleep duration')`"
    )

    assert not step.is_finished
    assert step.action.function_name == "search"
    assert tuple(step.action.args) == ("adult sleep duration",)


@pytest.fixture
def final_response_turn(monkeypatch):
    """Exercise take_turn while isolating provider and supporting-agent calls."""
    def run(agent_names, supporting_facts=(), reflect=False):
        orchestrator = MultiAgentOrchestrator()
        orchestrator.agent_name_list = agent_names
        orchestrator.question_last = "How was my sleep over the past month?"
        orchestrator.main_agent = AGENT_NAME_DOMAIN_EXPERT
        orchestrator.supporting_agents = (
            AGENT_NAME_DATA_SCIENCE if supporting_facts else ""
        )
        orchestrator.domain_expert_agent = Mock()
        orchestrator.domain_expert_agent.call_agent.return_value = (
            "Clinical reference standards are unavailable."
        )
        orchestrator.data_science_agent = Mock()
        orchestrator.data_science_agent.query.return_value = (
            "Mean sleep duration is 412.63 minutes across 30 valid nights."
        )
        final_prompts = []
        facts = iter(supporting_facts)

        def call_supporting_agents(**kwargs):
            if supporting_facts:
                orchestrator.memory["data_science_agent_insights"] = next(facts)

        def call_llm(prompt, **kwargs):
            if prompt.startswith(TOPIC_PROMPT):
                return "Sleep"
            if prompt.startswith(GOAL_PROMPT):
                return "Describe the recorded sleep data"
            if prompt.startswith(FINISH_DESC_PROMPT):
                return "CONTINUE"
            if prompt.startswith(ORCHESTRATOR_FINAL_RESPONSE_IMPROVEMENT_PROMPT):
                final_prompts.append(prompt)
                return "Final response"
            return "No supporting agents are configured."

        monkeypatch.setattr(orchestrator, "_call_llm", call_llm)
        monkeypatch.setattr(orchestrator, "_update_agent_team_structure", Mock())
        monkeypatch.setattr(
            orchestrator, "_call_supporting_agents", call_supporting_agents
        )
        monkeypatch.setattr(orchestrator, "_merge_and_store_agent_insights", Mock())
        monkeypatch.setattr(
            orchestrator,
            "_reflect_before_final_response",
            Mock(return_value={
                "decision": "YES" if reflect else "NO",
                "reflection_questions": {
                    AGENT_NAME_DATA_SCIENCE: ["Check the valid-night count"]
                } if reflect else {},
            }),
        )

        result = orchestrator.take_turn()
        assert result == "Final response"
        assert len(final_prompts) == 1
        return final_prompts[0]

    return run


def test_final_rewrite_receives_supporting_facts(final_response_turn):
    facts = (
        "window=2024-11-16..2024-12-15; valid_nights=30; "
        "mean_sleep_minutes=412.633333; below_7h=16"
    )
    prompt = final_response_turn(
        [AGENT_NAME_DOMAIN_EXPERT, AGENT_NAME_DATA_SCIENCE],
        supporting_facts=(facts,),
    )

    assert "Clinical reference standards are unavailable." in prompt
    assert "\n[SUPPORTING_AGENT_INSIGHTS]\n" in prompt
    assert facts in prompt


def test_final_rewrite_receives_refined_facts(final_response_turn):
    initial = "mean_sleep_minutes=399.125; valid_nights=29"
    refined = "mean_sleep_minutes=412.633333; valid_nights=30"
    prompt = final_response_turn(
        [AGENT_NAME_DOMAIN_EXPERT, AGENT_NAME_DATA_SCIENCE],
        supporting_facts=(initial, refined),
        reflect=True,
    )

    assert refined in prompt
    assert initial not in prompt


@pytest.mark.parametrize("agent_names", [[AGENT_NAME_DATA_SCIENCE], []])
def test_final_rewrite_without_supporting_agents(final_response_turn, agent_names):
    prompt = final_response_turn(agent_names)

    assert "\n[SUPPORTING_AGENT_INSIGHTS]\n" not in prompt
    if agent_names:
        assert "412.63 minutes across 30 valid nights" in prompt
    else:
        assert "No supporting agents are configured." in prompt
