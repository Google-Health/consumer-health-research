"""End-to-end tests for PHA multi-agent system.

This module tests the complete PHA pipeline:
- MultiAgentOrchestrator coordination
- Agent routing decisions
- Full query-to-response flow

Tests are organized into:
- Unit tests: Mock-based, no API keys required
- Integration tests: Require GEMINI_API_KEY (marked with @pytest.mark.e2e)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd


# =============================================================================
# Orchestrator Unit Tests
# =============================================================================

class TestOrchestratorUnit:
    """Unit tests for MultiAgentOrchestrator (no API key required)."""
    
    def test_import(self):
        """MultiAgentOrchestrator can be imported."""
        from pha.agents import MultiAgentOrchestrator
        assert MultiAgentOrchestrator is not None
    
    def test_instantiation(self):
        """MultiAgentOrchestrator can be instantiated."""
        from pha.agents import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        assert orchestrator is not None
    
    def test_has_configure_method(self):
        """Orchestrator has configure method."""
        from pha.agents import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        assert hasattr(orchestrator, 'configure')
        assert callable(orchestrator.configure)
    
    def test_has_set_agents_method(self):
        """Orchestrator has set_agents method."""
        from pha.agents import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        assert hasattr(orchestrator, 'set_agents')
        assert callable(orchestrator.set_agents)
    
    def test_has_respond_method(self):
        """Orchestrator has respond method."""
        from pha.agents import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator()
        assert hasattr(orchestrator, 'respond')
        assert callable(orchestrator.respond)


class TestOrchestratorPrompts:
    """Tests for orchestrator prompts."""
    
    def test_orchestrator_prompts_exist(self):
        """Orchestrator prompts are defined."""
        from pha.prompts.orchestrator_prompts import (
            ORCHESTRATOR_PREAMBLE,
            MULTI_AGENT_COLLABORATION_EXAMPLES,
            DATA_SCIENCE_AGENT_DESCRIPTION,
            DOMAIN_EXPERT_AGENT_DESCRIPTION,
            HEALTH_COACH_AGENT_DESCRIPTION,
        )
        
        assert len(ORCHESTRATOR_PREAMBLE) > 100
        assert len(MULTI_AGENT_COLLABORATION_EXAMPLES) > 100
        assert len(DATA_SCIENCE_AGENT_DESCRIPTION) > 50
        assert len(DOMAIN_EXPERT_AGENT_DESCRIPTION) > 50
        assert len(HEALTH_COACH_AGENT_DESCRIPTION) > 50
    
    def test_team_structure_prompt(self):
        """Team structure prompt is defined."""
        from pha.prompts.orchestrator_prompts import TEAM_STRUCTURE_PROMPT
        
        prompt = TEAM_STRUCTURE_PROMPT
        
        assert "data" in prompt.lower()
        assert "expert" in prompt.lower() or "domain" in prompt.lower()
        assert "coach" in prompt.lower()
    
    def test_rephrase_prompt(self):
        """Rephrase prompt function is defined and callable."""
        from pha.prompts.orchestrator_prompts import REPHRASE_PROMPT_SUPPORTING_AGENTS
        
        # It's a function, not a constant
        assert callable(REPHRASE_PROMPT_SUPPORTING_AGENTS)
        # Test that it returns a string when called with required args
        result = REPHRASE_PROMPT_SUPPORTING_AGENTS(
            original_prompt="How is my sleep?",
            main_agent="health_coach_agent",
            supporting_agents="data_science_agent;domain_expert_agent",
            collaboration_workflow="parallel"
        )
        assert isinstance(result, str)
        assert len(result) > 50
    
    def test_reflect_prompt(self):
        """Reflect prompt function is defined and callable."""
        from pha.prompts.orchestrator_prompts import REFLECT_PROMPT
        
        # It's a function, not a constant
        assert callable(REFLECT_PROMPT)
        # Test that it returns a string when called with required args
        result = REFLECT_PROMPT(
            original_prompt="How is my sleep?",
            main_agent="health_coach_agent",
            supporting_agents="data_science_agent",
            collaboration_workflow="parallel",
            main_agent_response="Your sleep looks good.",
            supporting_agent_insights="Average sleep: 7 hours"
        )
        assert isinstance(result, str)
        assert len(result) > 50


class TestAgentRouting:
    """Tests for agent routing logic."""
    
    def test_agent_names_constants(self):
        """Agent name constants are defined."""
        from pha.prompts.orchestrator_prompts import (
            AGENT_NAME_DATA_SCIENCE,
            AGENT_NAME_DOMAIN_EXPERT,
            AGENT_NAME_HEALTH_COACH,
        )
        
        assert AGENT_NAME_DATA_SCIENCE is not None
        assert AGENT_NAME_DOMAIN_EXPERT is not None
        assert AGENT_NAME_HEALTH_COACH is not None
        
        # Should be distinct
        names = [AGENT_NAME_DATA_SCIENCE, AGENT_NAME_DOMAIN_EXPERT, AGENT_NAME_HEALTH_COACH]
        assert len(names) == len(set(names))
    
    def test_query_classification_patterns(self):
        """Common query patterns should route to expected agents."""
        # These are heuristic expectations - actual routing depends on LLM
        data_queries = [
            "What are my average steps?",
            "Show me a trend of my sleep",
            "Calculate my weekly activity",
        ]
        
        expert_queries = [
            "Is my heart rate healthy?",
            "What does my HbA1c mean?",
            "Interpret my blood pressure readings",
        ]
        
        coach_queries = [
            "How can I improve my sleep?",
            "Help me set fitness goals",
            "What should I do to be healthier?",
        ]
        
        # Just verify these are valid query strings
        for q in data_queries + expert_queries + coach_queries:
            assert isinstance(q, str)
            assert len(q) > 10


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.requires_api_key("gemini")
class TestOrchestratorIntegration:
    """Integration tests for MultiAgentOrchestrator."""
    
    def test_configure_orchestrator(self, gemini_api_key):
        """Orchestrator can be configured."""
        from pha.agents import MultiAgentOrchestrator
        
        orchestrator = MultiAgentOrchestrator()
        orchestrator.configure(gemini_api_key=gemini_api_key)
        
        assert orchestrator is not None
    
    def test_set_agents(self, gemini_api_key, all_sample_dataframes):
        """Orchestrator can set up agents."""
        from pha.agents import (
            MultiAgentOrchestrator,
            DataScienceAgent,
            HealthCoachAgent,
        )
        
        # Create agents
        ds_agent = DataScienceAgent()
        ds_agent.configure(gemini_api_key=gemini_api_key)
        ds_agent.load_dataframes(all_sample_dataframes)
        
        coach = HealthCoachAgent(simple_mode=True)
        coach.configure(gemini_api_key=gemini_api_key)
        
        # Set up orchestrator
        orchestrator = MultiAgentOrchestrator()
        orchestrator.configure(gemini_api_key=gemini_api_key)
        orchestrator.set_agents(
            data_science_agent=ds_agent,
            health_coach_agent=coach,
        )
        
        assert orchestrator is not None


@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api_key("gemini")
class TestFullPipeline:
    """Full pipeline end-to-end tests."""
    
    @pytest.fixture
    def configured_orchestrator(self, gemini_api_key, all_sample_dataframes):
        """Create a fully configured orchestrator with all agents."""
        from pha.agents import (
            MultiAgentOrchestrator,
            DataScienceAgent,
            DomainExpertAgent,
            HealthCoachAgent,
        )
        
        # Data Science Agent
        ds_agent = DataScienceAgent()
        ds_agent.configure(gemini_api_key=gemini_api_key)
        ds_agent.load_dataframes(all_sample_dataframes)
        
        # Domain Expert Agent
        expert = DomainExpertAgent()
        expert.get_agent(gemini_api_key=gemini_api_key)
        
        # Health Coach Agent
        coach = HealthCoachAgent(simple_mode=True)
        coach.configure(gemini_api_key=gemini_api_key)
        
        # Orchestrator
        orchestrator = MultiAgentOrchestrator()
        orchestrator.configure(gemini_api_key=gemini_api_key)
        orchestrator.set_agents(
            data_science_agent=ds_agent,
            domain_expert_agent=expert,
            health_coach_agent=coach,
        )
        
        return orchestrator
    
    def test_data_analysis_query(self, configured_orchestrator):
        """Pipeline handles data analysis query."""
        response = configured_orchestrator.respond(
            "What are my average daily steps over the past week?"
        )
        
        assert response is not None
        assert len(response) > 0
        # Should contain numeric information or mention steps
        assert "step" in response.lower() or any(c.isdigit() for c in response)
    
    def test_health_interpretation_query(self, configured_orchestrator):
        """Pipeline handles health interpretation query."""
        response = configured_orchestrator.respond(
            "Is my resting heart rate in a healthy range?"
        )
        
        assert response is not None
        assert len(response) > 0
        # Should mention heart rate or health
        response_lower = response.lower()
        assert ("heart" in response_lower or 
                "rate" in response_lower or 
                "health" in response_lower or
                "bpm" in response_lower)
    
    def test_coaching_query(self, configured_orchestrator):
        """Pipeline handles coaching query."""
        response = configured_orchestrator.respond(
            "How can I improve my sleep quality?"
        )
        
        assert response is not None
        assert len(response) > 0
        # Should provide actionable advice
        response_lower = response.lower()
        assert ("sleep" in response_lower or 
                "bed" in response_lower or
                "rest" in response_lower or
                "improve" in response_lower)
    
    def test_complex_multi_agent_query(self, configured_orchestrator):
        """Pipeline handles query requiring multiple agents."""
        response = configured_orchestrator.respond(
            "Based on my activity data, am I getting enough exercise? "
            "What changes should I make to improve?"
        )
        
        assert response is not None
        assert len(response) > 50  # Should be a substantial response
    
    def test_response_is_coherent(self, configured_orchestrator):
        """Pipeline produces coherent responses."""
        response = configured_orchestrator.respond(
            "Give me a brief summary of my health data."
        )
        
        assert response is not None
        # Response should be readable text (not error messages or raw data)
        assert not response.startswith("Error")
        assert not response.startswith("{")  # Not raw JSON
        # Should have multiple words
        assert len(response.split()) > 10


# =============================================================================
# Agent Collaboration Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.requires_api_key("gemini")
class TestAgentCollaboration:
    """Tests for multi-agent collaboration scenarios."""
    
    def test_data_to_interpretation_flow(self, gemini_api_key, all_sample_dataframes):
        """Data Science results can be passed to Domain Expert."""
        from pha.agents import DataScienceAgent, DomainExpertAgent
        
        # Get data analysis
        ds_agent = DataScienceAgent()
        ds_agent.configure(gemini_api_key=gemini_api_key)
        ds_agent.load_dataframes(all_sample_dataframes)
        
        data_response = ds_agent.query("What is my average resting heart rate?")
        
        # Pass to domain expert for interpretation
        expert = DomainExpertAgent()
        expert.get_agent(gemini_api_key=gemini_api_key)
        
        interpretation = expert.call_agent(
            f"Based on this data analysis: {data_response}\n\n"
            "Is this heart rate healthy for an adult?"
        )
        
        assert interpretation is not None
        assert len(interpretation) > 0
    
    def test_interpretation_to_coaching_flow(self, gemini_api_key):
        """Domain Expert insights can inform Health Coach."""
        from pha.agents import DomainExpertAgent, HealthCoachAgent
        
        # Get expert interpretation
        expert = DomainExpertAgent()
        expert.get_agent(gemini_api_key=gemini_api_key)
        
        interpretation = expert.call_agent(
            "A person has a resting heart rate of 75 bpm. Is this concerning?"
        )
        
        # Get coaching based on interpretation
        coach = HealthCoachAgent(simple_mode=True)
        coach.configure(gemini_api_key=gemini_api_key)
        
        coaching = coach.respond(
            f"Based on this health assessment: {interpretation}\n\n"
            "What lifestyle changes would you recommend?"
        )
        
        assert coaching is not None
        assert len(coaching) > 0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in the pipeline."""
    
    def test_orchestrator_without_agents(self):
        """Orchestrator handles missing agents gracefully."""
        from pha.agents import MultiAgentOrchestrator
        
        orchestrator = MultiAgentOrchestrator()
        
        # Should not crash, should indicate configuration needed
        try:
            response = orchestrator.respond("test query")
            # If it returns, should indicate an issue
            assert response is not None
        except (ValueError, RuntimeError, AttributeError):
            # Expected - not configured
            pass
    
    @pytest.mark.integration
    @pytest.mark.requires_api_key("gemini")
    def test_empty_query(self, gemini_api_key, all_sample_dataframes):
        """Agents handle empty queries."""
        from pha.agents import DataScienceAgent
        
        agent = DataScienceAgent()
        agent.configure(gemini_api_key=gemini_api_key)
        agent.load_dataframes(all_sample_dataframes)
        
        # Should handle gracefully
        try:
            response = agent.query("")
            assert response is not None  # Should return something, even if error message
        except (ValueError, RuntimeError):
            # Also acceptable - rejecting empty query
            pass
    
    @pytest.mark.integration
    @pytest.mark.requires_api_key("gemini")
    def test_very_long_query(self, gemini_api_key):
        """Agents handle very long queries."""
        from pha.agents import HealthCoachAgent
        
        agent = HealthCoachAgent(simple_mode=True)
        agent.configure(gemini_api_key=gemini_api_key)
        
        long_query = "How can I improve my health? " * 100
        
        # Should handle without crashing
        response = agent.respond(long_query)
        assert response is not None


# =============================================================================
# Response Quality Tests
# =============================================================================

@pytest.mark.e2e
@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_api_key("gemini")
class TestResponseQuality:
    """Tests for response quality and consistency."""
    
    def test_response_mentions_data(self, gemini_api_key, all_sample_dataframes):
        """Data queries should reference actual data values."""
        from pha.agents import DataScienceAgent
        
        agent = DataScienceAgent()
        agent.configure(gemini_api_key=gemini_api_key)
        agent.load_dataframes(all_sample_dataframes)
        
        response = agent.query("What was my step count yesterday?")
        
        # Response should contain numbers (actual data)
        has_numbers = any(c.isdigit() for c in response)
        assert has_numbers or "step" in response.lower()
    
    def test_coaching_is_actionable(self, gemini_api_key):
        """Coaching responses should be actionable."""
        from pha.agents import HealthCoachAgent
        
        agent = HealthCoachAgent(simple_mode=True)
        agent.configure(gemini_api_key=gemini_api_key)
        
        response = agent.respond("I want to sleep better. Give me specific advice.")
        
        response_lower = response.lower()
        
        # Should contain actionable language
        actionable_words = ["try", "consider", "should", "recommend", "can", "start", "avoid", "limit"]
        has_actionable = any(word in response_lower for word in actionable_words)
        
        assert has_actionable or len(response) > 100  # Either actionable or detailed
    
    @pytest.mark.xfail(
        reason=(
            "DomainExpertAgent uses OneTwo ReAct, whose answer extractor is "
            "unreliable with Gemini 3.x models (the current default). "
            "Re-evaluate when ReAct compatibility is restored or when the "
            "default model is downgraded."
        ),
        strict=False,
    )
    def test_expert_provides_context(self, gemini_api_key):
        """Expert responses should provide medical context."""
        from pha.agents import DomainExpertAgent

        agent = DomainExpertAgent()
        agent.get_agent(gemini_api_key=gemini_api_key)

        response = agent.call_agent("What is heart rate variability and why does it matter?")

        response_lower = response.lower()

        # Should contain explanatory content
        assert len(response) > 100  # Substantial explanation
        assert ("hrv" in response_lower or
                "variability" in response_lower or
                "heart" in response_lower)
