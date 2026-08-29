"""Tests for prompt generation utilities."""

import pytest

from pha.prompts.data_science_prompts import (
    generate_approach_prompt,
    generate_codegen_prompt,
    generate_debug_prompt_v2,
    generate_result_communication_prompt,
)
from pha.prompts.orchestrator_prompts import (
    TEAM_STRUCTURE_PROMPT,
    REPHRASE_PROMPT_SUPPORTING_AGENTS,
    REFLECT_PROMPT,
    AGENT_NAME_DATA_SCIENCE,
    AGENT_NAME_HEALTH_COACH,
    AGENT_NAME_DOMAIN_EXPERT,
)
from pha.prompts.health_coach_prompts import (
    GOAL_PROMPT,
    HEALTH_COACH_SIMPLE_PROMPT,
    FINISH_DESC_PROMPT,
)


class TestDataSciencePrompts:
    """Tests for Data Science Agent prompts."""
    
    def test_approach_prompt_includes_query(self):
        """Approach prompt includes the user's question."""
        prompt = generate_approach_prompt(
            query="What is my average step count?",
            dfs_str="summary_df: daily metrics",
        )
        
        assert "average step count" in prompt
    
    def test_approach_prompt_includes_dataframes(self):
        """Approach prompt includes DataFrame descriptions."""
        prompt = generate_approach_prompt(
            query="Analyze my sleep",
            dfs_str="summary_df: Contains daily sleep_minutes, deep_sleep, rem_sleep",
        )
        
        assert "summary_df" in prompt
        assert "sleep" in prompt.lower()
    
    def test_codegen_prompt_includes_approach(self):
        """Codegen prompt includes the generated approach."""
        approach = "1. Load summary_df\n2. Calculate mean of steps column"
        prompt = generate_codegen_prompt(
            query="Average steps?",
            approach_str=approach,
            dfs_str="summary_df: steps data",
            dfs_info_str="summary_df: daily health metrics",
        )
        
        assert "Calculate mean" in prompt
        assert "summary_df" in prompt
    
    def test_codegen_prompt_includes_df_info(self):
        """Codegen prompt includes DataFrame schema info."""
        prompt = generate_codegen_prompt(
            query="Analyze trends",
            approach_str="Look at trends",
            dfs_str="summary_df with columns",
            dfs_info_str="**summary_df**\n  description: Daily health metrics\n  variable_name: summary_df",
        )
        
        assert "summary_df" in prompt
    
    def test_debug_prompt_includes_error(self):
        """Debug prompt includes the error information."""
        prompt = generate_debug_prompt_v2(
            approach_str="Print the variable",
            code="print(undefined_var)",
            code_result="NameError: name 'undefined_var' is not defined",
        )
        
        assert "NameError" in prompt or "undefined" in prompt or "error" in prompt.lower()
    
    def test_result_communication_prompt_includes_results(self):
        """Result communication prompt includes execution output."""
        prompt = generate_result_communication_prompt(
            query="What is my average step count?",
            approach_str="Calculate mean of steps",
            code="print(summary_df['steps'].mean())",
            execution_results="8500.0",
        )
        
        assert "8500" in prompt
        assert "step" in prompt.lower()


class TestOrchestratorPrompts:
    """Tests for Orchestrator prompts."""
    
    def test_team_structure_prompt_includes_agents(self):
        """Team structure prompt lists available agents."""
        # TEAM_STRUCTURE_PROMPT is a string template, not a function
        prompt = TEAM_STRUCTURE_PROMPT
        
        # Should contain placeholders or agent references
        assert "agent" in prompt.lower()
    
    def test_team_structure_prompt_requests_json(self):
        """Team structure prompt asks for JSON output."""
        prompt = TEAM_STRUCTURE_PROMPT
        
        # Should mention JSON format
        assert "json" in prompt.lower() or "JSON" in prompt
    
    def test_rephrase_prompt_is_callable(self):
        """REPHRASE_PROMPT_SUPPORTING_AGENTS is callable."""
        prompt = REPHRASE_PROMPT_SUPPORTING_AGENTS(
            original_prompt="How is my sleep?",
            main_agent=AGENT_NAME_HEALTH_COACH,
            supporting_agents=AGENT_NAME_DATA_SCIENCE,
            collaboration_workflow="sequential",
        )
        
        assert "sleep" in prompt.lower()
    
    def test_reflect_prompt_includes_response(self):
        """Reflect prompt includes main agent response."""
        prompt = REFLECT_PROMPT(
            original_prompt="How is my sleep?",
            main_agent=AGENT_NAME_HEALTH_COACH,
            supporting_agents=AGENT_NAME_DATA_SCIENCE,
            collaboration_workflow="sequential",
            main_agent_response="Your sleep has been irregular. What time do you usually go to bed?",
            supporting_agent_insights="Average sleep: 6.5 hours",
        )
        
        assert "go to bed" in prompt
        assert "6.5 hours" in prompt
    
    def test_reflect_prompt_asks_for_json(self):
        """Reflect prompt requests JSON output."""
        prompt = REFLECT_PROMPT(
            original_prompt="Test",
            main_agent="agent",
            supporting_agents="agent2",
            collaboration_workflow="test",
            main_agent_response="response",
            supporting_agent_insights="insights",
        )
        
        assert "json" in prompt.lower() or "JSON" in prompt
        assert "YES" in prompt or "NO" in prompt


class TestHealthCoachPrompts:
    """Tests for Health Coach prompts."""
    
    def test_goal_prompt_exists(self):
        """GOAL_PROMPT is defined and non-empty."""
        assert GOAL_PROMPT is not None
        assert len(GOAL_PROMPT) > 0
    
    def test_simple_prompt_exists(self):
        """HEALTH_COACH_SIMPLE_PROMPT is defined and non-empty."""
        assert HEALTH_COACH_SIMPLE_PROMPT is not None
        assert len(HEALTH_COACH_SIMPLE_PROMPT) > 0
    
    def test_finish_desc_prompt_exists(self):
        """FINISH_DESC_PROMPT is defined and non-empty."""
        assert FINISH_DESC_PROMPT is not None
        assert len(FINISH_DESC_PROMPT) > 0
    
    def test_prompts_mention_conversation(self):
        """Prompts reference conversation context."""
        # At least one prompt should mention conversation
        prompts_text = GOAL_PROMPT + HEALTH_COACH_SIMPLE_PROMPT + FINISH_DESC_PROMPT
        assert "conversation" in prompts_text.lower() or "CONVERSATION" in prompts_text


class TestAgentNames:
    """Tests for agent name constants."""
    
    def test_agent_names_are_strings(self):
        """Agent names are non-empty strings."""
        assert isinstance(AGENT_NAME_DATA_SCIENCE, str)
        assert isinstance(AGENT_NAME_HEALTH_COACH, str)
        assert isinstance(AGENT_NAME_DOMAIN_EXPERT, str)
        
        assert len(AGENT_NAME_DATA_SCIENCE) > 0
        assert len(AGENT_NAME_HEALTH_COACH) > 0
        assert len(AGENT_NAME_DOMAIN_EXPERT) > 0
    
    def test_agent_names_are_distinct(self):
        """Agent names are unique."""
        names = [AGENT_NAME_DATA_SCIENCE, AGENT_NAME_HEALTH_COACH, AGENT_NAME_DOMAIN_EXPERT]
        assert len(names) == len(set(names))
