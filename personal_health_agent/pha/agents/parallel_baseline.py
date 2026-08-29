"""Parallel Multi-Agent Baseline for PHA.

This baseline runs all three agents (Data Science, Domain Expert, Health Coach)
in parallel on the same query, then uses an LLM to synthesize their responses.

This serves as a comparison baseline to show the value of intelligent routing
in the main PHA orchestrator.

Usage:
    from pha.agents import ParallelMultiAgentBaseline
    
    baseline = ParallelMultiAgentBaseline()
    baseline.configure(gemini_api_key="your-key")
    
    response = baseline.respond("How is my sleep quality?")
"""

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, Tuple


def clean_agent_response(text: str) -> str:
    """Clean internal agent labels from response text.
    
    Removes internal markers that shouldn't be shown to users:
    - [SUMMARY] tags from data science agent
    - [CODE UPDATE] tags from data science agent  
    - Coach: prefixes from health coach agent
    - Expert: prefixes from domain expert agent
    - [Finish]: prefix from ReAct agents
    - JSON-escaped newlines
    - Quoted string wrappers
    
    Args:
        text: Raw response text from agents.
        
    Returns:
        Cleaned response suitable for user display.
    """
    if not text:
        return text
    
    # Handle JSON-escaped newlines (convert \n literal to actual newlines)
    if '\\n' in text:
        text = text.replace('\\n', '\n')
    
    # Remove surrounding quotes if the entire text is quoted
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    # Remove [SUMMARY] and [CODE UPDATE] tags (keep content after)
    text = re.sub(r'^\s*\[SUMMARY\]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\[CODE UPDATE\]\s*', '', text, flags=re.MULTILINE)
    
    # Remove Coach: prefix at start of response
    text = re.sub(r'^Coach:\s*', '', text, flags=re.MULTILINE)
    
    # Remove Expert: prefix at start of response  
    text = re.sub(r'^Expert:\s*', '', text, flags=re.MULTILINE)
    
    # Remove [Finish]: prefix (ReAct final answer marker)
    # Handle various formats: [Finish]:, [Finish]: , [Finish]:\n, etc.
    text = re.sub(r'^\s*\[Finish\]:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[Finish\]:\s*\n*', '', text)  # Also catch mid-text occurrences
    
    # Detect and reject system prompt leakage
    # These patterns indicate the model is outputting its instructions instead of a response
    # Normalize text for detection (handle ellipsis, extra spaces, etc.)
    text_normalized = text.lower().replace('…', '...').replace('  ', ' ')
    
    prompt_leak_indicators = [
        # From orchestrator/domain expert prompts
        'you must do so safely and transparently',
        'quantify uncertainty when appropriate',
        'avoid diagnosing or prescribing',
        'always recommend appropriate follow-up',
        'you should use the provided tools',
        'if a tool fails or required user context',
        'explicitly state the limitation',
        # From domain expert agent
        'clinically accurate, evidence-based',
        'safety-conscious guidance',
        'explicitly separate objective findings',
        'state uncertainty and limitations when data',
        'avoid diagnosing when criteria are not met',
        'recommend appropriate follow-up',
        'cite authoritative sources with links',
        'request only the minimum additional data',
        'wearables, labs, imaging, notes',
        'urgent/emergent evaluation when red flags',
        # Generic instruction patterns
        'you will analyze and synthesize',
        'you will avoid',
        'you will cite',
        'you will request',
        'with the goal of delivering',
        'delivering clinically accurate',
        # Tool instruction patterns (shouldn't be in final response)
        'e.g., `tool_code` for math',
        'for math, `compare_blood_test',
        '`datacommons_natural_language_query`',
        'via the `search` tool',
        'use the provided tools',
        # Instruction-style patterns
        'you are tasked with',
        'as a domain expert',
        'you are a domain expert',
        'you have access to a number of tools',
        'it is imperative that you',
    ]
    for indicator in prompt_leak_indicators:
        if indicator.lower() in text_normalized:
            print(f"[clean_agent_response:parallel] PROMPT LEAK detected: matched '{indicator}'")
            print(f"[clean_agent_response:parallel] Raw text (first 300 chars): {text[:300]}")
            return (
                "⚠️ **Response contained system prompt leakage** — the model echoed its instructions "
                "instead of answering your question.\n\n"
                f"**Matched pattern:** `{indicator}`\n\n"
                "This is a known issue with some models. Please try again, rephrase your question, "
                "or switch to a different model."
            )
    
    # Clean up any extra whitespace
    text = text.strip()
    
    return text


# Try to import LLM backends
_GENAI_AVAILABLE = False
try:
    import google.genai as genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    pass

from .data_science_agent import DataScienceAgent
from .domain_expert_agent import DomainExpertAgent
from .health_coach_agent import HealthCoachAgent


# =============================================================================
# Synthesis Prompt
# =============================================================================

PARALLEL_SYNTHESIS_PROMPT = """
[USER PROFILE]
{user_context}

[USER QUERY]
{query}

[DATA SCIENCE AGENT RESPONSE]
{ds_response}

[DOMAIN EXPERT AGENT RESPONSE]
{de_response}

[HEALTH COACH AGENT RESPONSE]
{coach_response}

Given these insights and the conversation history,
now synthesize these insights and answer the user's request.
Do not directly include any of the insights in the response. Do not refer to any specific agent.
Just act as an orchestrator and synthesize the response based on the insights.

Synthesized Response:"""


# =============================================================================
# Parallel Baseline Class
# =============================================================================

class ParallelMultiAgentBaseline:
    """Parallel Multi-Agent Baseline.
    
    Runs all three agents in parallel on every query, then synthesizes
    their responses. This serves as a baseline to compare against the
    orchestrated PHA system.
    
    Key differences from PHA:
    - No intelligent routing - all agents always run
    - No main/supporting agent hierarchy - all agents are peers
    - Simple synthesis instead of orchestrated collaboration
    """
    
    def __init__(
        self,
        debug_verbose: bool = False,
    ):
        """Initialize the parallel baseline.
        
        Args:
            debug_verbose: If True, print debug information.
        """
        self.debug_verbose = debug_verbose
        
        # Agent instances
        self.data_science_agent: Optional[DataScienceAgent] = None
        self.domain_expert_agent: Optional[DomainExpertAgent] = None
        self.health_coach_agent: Optional[HealthCoachAgent] = None
        
        # LLM for synthesis
        self._llm_client = None
        self._model_name = None
        
        # User context for synthesis (profile info)
        self.user_context: str = ""
        
        # Conversation state
        self.conversation_history: list = []
        
        # Last response details (for debugging/analysis)
        self.last_ds_response: str = ""
        self.last_de_response: str = ""
        self.last_coach_response: str = ""
        self.last_synthesis: str = ""
    
    def configure(
        self,
        api_key: Optional[str] = None,
        provider: str = "gemini",
        model_name: Optional[str] = None,
        # Legacy parameter for backwards compatibility
        gemini_api_key: Optional[str] = None,
    ) -> "ParallelMultiAgentBaseline":
        """Configure the baseline with API credentials.
        
        Args:
            api_key: API key for the selected provider.
            provider: "gemini", "openai", or "anthropic".
            model_name: Model to use for synthesis.
            gemini_api_key: DEPRECATED - use api_key instead.
            
        Returns:
            Self for method chaining.
        """
        import os
        
        # Handle legacy parameter
        if gemini_api_key and not api_key:
            api_key = gemini_api_key
            provider = "gemini"
        
        # Get API key from environment if not provided
        env_vars = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        provider = provider.lower()
        if not api_key:
            api_key = os.environ.get(env_vars.get(provider, ""))
        
        # Set default model names per provider
        default_models = {
            "gemini": "models/gemini-3.1-flash-lite-preview",
            "openai": "gpt-4o",
            "anthropic": "claude-sonnet-4-5-20250929",
        }
        model_name = model_name or default_models.get(provider)
        
        self._provider = provider
        self._model_name = model_name
        self._backend = None
        self._llm_client = None
        
        # Use google-genai directly for Gemini
        if provider == "gemini":
            if not _GENAI_AVAILABLE:
                raise ImportError(
                    "google-genai is required for Gemini. Install with: pip install google-genai"
                )
            self._llm_client = genai.Client(api_key=api_key)
        else:
            # Use the multi-provider backend for OpenAI/Anthropic
            from ..llm.backend_multimodel import get_llm_backend
            self._backend = get_llm_backend(
                backend_type=provider,
                api_key=api_key,
                model_name=model_name,
            )
        
        return self
    
    def set_agents(
        self,
        data_science_agent: Optional[DataScienceAgent] = None,
        domain_expert_agent: Optional[DomainExpertAgent] = None,
        health_coach_agent: Optional[HealthCoachAgent] = None,
    ) -> "ParallelMultiAgentBaseline":
        """Set the agent instances to use.
        
        Args:
            data_science_agent: Configured DataScienceAgent instance.
            domain_expert_agent: Configured DomainExpertAgent instance.
            health_coach_agent: Configured HealthCoachAgent instance.
            
        Returns:
            Self for method chaining.
        """
        if data_science_agent:
            self.data_science_agent = data_science_agent
        if domain_expert_agent:
            self.domain_expert_agent = domain_expert_agent
        if health_coach_agent:
            self.health_coach_agent = health_coach_agent
        
        return self
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM for synthesis.
        
        Args:
            prompt: The synthesis prompt.
            
        Returns:
            The LLM response text.
        """
        if self._backend:
            # Use multi-provider backend (OpenAI/Anthropic)
            return self._backend.generate(prompt)
        elif self._llm_client:
            # Use google-genai for Gemini
            response = self._llm_client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=4096,
                ),
            )
            return response.text
        else:
            raise RuntimeError("LLM not configured. Call configure() first.")
    
    def _call_data_science_agent(self, query: str) -> Tuple[str, str]:
        """Call the Data Science Agent.
        
        Returns:
            Tuple of (agent_name, response).
        """
        if not self.data_science_agent:
            return ("data_science", "[Data Science Agent not configured]")
        
        try:
            response = self.data_science_agent.query(query)
            return ("data_science", str(response))
        except Exception as e:
            if self.debug_verbose:
                print(f"Data Science Agent error: {e}")
            return ("data_science", f"[Error: {str(e)}]")
    
    def _call_domain_expert_agent(self, query: str) -> Tuple[str, str]:
        """Call the Domain Expert Agent.
        
        Returns:
            Tuple of (agent_name, response).
        """
        if not self.domain_expert_agent:
            return ("domain_expert", "[Domain Expert Agent not configured]")
        
        try:
            response = self.domain_expert_agent.call_agent(query)
            return ("domain_expert", str(response))
        except Exception as e:
            if self.debug_verbose:
                print(f"Domain Expert Agent error: {e}")
            return ("domain_expert", f"[Error: {str(e)}]")
    
    def _call_health_coach_agent(self, query: str) -> Tuple[str, str]:
        """Call the Health Coach Agent.
        
        Returns:
            Tuple of (agent_name, response).
        """
        if not self.health_coach_agent:
            return ("health_coach", "[Health Coach Agent not configured]")
        
        try:
            response = self.health_coach_agent.respond(query)
            return ("health_coach", str(response))
        except Exception as e:
            if self.debug_verbose:
                print(f"Health Coach Agent error: {e}")
            return ("health_coach", f"[Error: {str(e)}]")
    
    def _call_agents_parallel(self, query: str) -> Dict[str, str]:
        """Call all agents in parallel.
        
        Args:
            query: The user query.
            
        Returns:
            Dictionary mapping agent names to their responses.
        """
        from datetime import datetime
        from pha.streaming import emit_step
        ts = datetime.now().strftime("%H:%M:%S")
        emit_step("Parallel", "Calling Data Science, Domain Expert, and Health Coach agents in parallel...")
        print(f"\n[Parallel:{ts}] Calling all 3 agents in parallel...")
        
        responses = {}
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(self._call_data_science_agent, query): "data_science",
                executor.submit(self._call_domain_expert_agent, query): "domain_expert",
                executor.submit(self._call_health_coach_agent, query): "health_coach",
            }
            
            for future in as_completed(futures):
                agent_name, response = future.result()
                responses[agent_name] = response
                ts2 = datetime.now().strftime("%H:%M:%S")
                is_err = response.startswith("[Error") or response.startswith("[Data Science Agent not") or response.startswith("[Domain Expert Agent not") or response.startswith("[Health Coach Agent not")
                status = "error" if is_err else "completed"
                emit_step("Parallel", f"{agent_name} {status}", detail=response[:100])
                print(f"[Parallel:{ts2}] {agent_name} completed: {'ERROR' if is_err else 'OK'} ({len(response)} chars): {response[:150]}...")
        
        return responses
    
    def _synthesize_responses(
        self,
        query: str,
        ds_response: str,
        de_response: str,
        coach_response: str,
    ) -> str:
        """Synthesize the three agent responses into one.
        
        Args:
            query: The original user query.
            ds_response: Data Science Agent response.
            de_response: Domain Expert Agent response.
            coach_response: Health Coach Agent response.
            
        Returns:
            Synthesized response.
        """
        prompt = PARALLEL_SYNTHESIS_PROMPT.format(
            user_context=self.user_context or "No profile information available.",
            query=query,
            ds_response=ds_response,
            de_response=de_response,
            coach_response=coach_response,
        )
        
        return self._call_llm(prompt)
    
    def respond(self, user_message: str) -> str:
        """Process a user message and return a synthesized response.
        
        This is the main entry point. It:
        1. Calls all three agents in parallel
        2. Synthesizes their responses
        3. Returns the final response (cleaned of internal labels)
        
        Args:
            user_message: The user's query.
            
        Returns:
            The synthesized response from all agents.
        """
        from datetime import datetime
        from pha.streaming import emit_step
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"[Parallel:{ts}] ═══ ParallelBaseline.respond ═══")
        print(f"[Parallel:{ts}] Query: {user_message[:100]}...")
        print(f"{'═' * 60}")
        
        # Call all agents in parallel
        responses = self._call_agents_parallel(user_message)
        
        # Store individual responses for debugging
        self.last_ds_response = responses.get("data_science", "")
        self.last_de_response = responses.get("domain_expert", "")
        self.last_coach_response = responses.get("health_coach", "")
        
        ts2 = datetime.now().strftime("%H:%M:%S")
        emit_step("Parallel", "All agents completed — synthesizing responses...")
        print(f"\n[Parallel:{ts2}] All agents completed, synthesizing...")
        print(f"[Parallel:{ts2}] DS: {len(self.last_ds_response)} chars")
        print(f"[Parallel:{ts2}] DE: {len(self.last_de_response)} chars")
        print(f"[Parallel:{ts2}] Coach: {len(self.last_coach_response)} chars")
        
        # Synthesize responses
        synthesis = self._synthesize_responses(
            query=user_message,
            ds_response=self.last_ds_response,
            de_response=self.last_de_response,
            coach_response=self.last_coach_response,
        )
        
        self.last_synthesis = synthesis
        
        ts3 = datetime.now().strftime("%H:%M:%S")
        print(f"[Parallel:{ts3}] Synthesis ({len(synthesis)} chars): {synthesis[:200]}...")
        
        # Update conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": synthesis,
        })
        
        # Clean internal agent labels before returning to user
        cleaned = clean_agent_response(synthesis)
        print(f"[Parallel:{ts3}] ✓ Final cleaned ({len(cleaned)} chars): {cleaned[:200]}...")
        return cleaned
    
    def get_individual_responses(self) -> Dict[str, str]:
        """Get the individual agent responses from the last query.
        
        Returns:
            Dictionary with keys 'data_science', 'domain_expert', 'health_coach'.
        """
        return {
            "data_science": self.last_ds_response,
            "domain_expert": self.last_de_response,
            "health_coach": self.last_coach_response,
        }
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.conversation_history = []
        self.last_ds_response = ""
        self.last_de_response = ""
        self.last_coach_response = ""
        self.last_synthesis = ""
        
        # Reset health coach conversation too (it maintains state)
        if self.health_coach_agent:
            self.health_coach_agent.reset_conversation()


# =============================================================================
# Factory Function
# =============================================================================

def create_parallel_baseline(
    api_key: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    settings: Optional[Any] = None,
    debug_verbose: bool = False,
    # Legacy parameter for backwards compatibility
    gemini_api_key: Optional[str] = None,
) -> ParallelMultiAgentBaseline:
    """Create a fully configured ParallelMultiAgentBaseline.
    
    This is a convenience function that creates and configures all agents
    and the baseline system.
    
    Args:
        api_key: API key for the selected provider.
        provider: "gemini", "openai", or "anthropic".
        model_name: Model to use (uses provider defaults if not specified).
        tavily_api_key: Optional Tavily API key for web search.
        settings: Optional Settings object for data paths.
        debug_verbose: If True, print debug information.
        gemini_api_key: DEPRECATED - use api_key instead.
        
    Returns:
        Configured ParallelMultiAgentBaseline instance.
        
    Example:
        ```python
        baseline = create_parallel_baseline(
            api_key="your-key",
            provider="anthropic",
            tavily_api_key="your-tavily-key",
        )
        response = baseline.respond("How is my sleep?")
        ```
    """
    # Handle legacy parameter
    if gemini_api_key and not api_key:
        api_key = gemini_api_key
        provider = "gemini"
    
    from config.settings import Settings
    from ..utils import load_persona
    
    # Use provided settings or create default
    if settings is None:
        settings = Settings()
    
    # Load persona data for context
    summary_df, activities_df, profile_df, population_df = load_persona(settings=settings)
    
    # Create user context string from profile
    user_context = _create_user_context(profile_df, summary_df)
    
    # Create Data Science Agent with multi-provider support
    ds_agent = DataScienceAgent(settings=settings)
    ds_agent.configure(api_key=api_key, provider=provider, model_name=model_name)
    
    # Domain Expert Agent uses OneTwo which supports Gemini and OpenAI
    de_agent = None
    if provider in ("gemini", "openai"):
        de_agent = DomainExpertAgent(tavily_api_key=tavily_api_key)
        de_agent.get_agent(api_key=api_key, provider=provider, model_name=model_name)
        # Pass user health data to domain expert
        health_data_md = f"## User Profile\n{profile_df.to_string()}\n\n## Recent Health Summary\n{summary_df.head(30).to_string()}"
        de_agent.set_user_health_data(health_data_md)
    else:
        print(f"Note: DomainExpertAgent uses OneTwo which supports Gemini and OpenAI. "
              f"Using {provider} without domain expert functionality (no Anthropic backend in OneTwo).")
    
    # Create Health Coach Agent with multi-provider support
    coach_agent = HealthCoachAgent(simple_mode=True)
    coach_agent.configure(api_key=api_key, provider=provider, model_name=model_name)
    coach_agent.initialize_context(user_context)
    
    # Create and configure baseline
    baseline = ParallelMultiAgentBaseline(debug_verbose=debug_verbose)
    baseline.configure(api_key=api_key, provider=provider, model_name=model_name)
    baseline.user_context = user_context  # Set context for synthesis prompt
    baseline.set_agents(
        data_science_agent=ds_agent,
        domain_expert_agent=de_agent,
        health_coach_agent=coach_agent,
    )
    
    return baseline


def _create_user_context(profile_df, summary_df) -> str:
    """Create user context string from profile and summary data."""
    context_parts = ["[USER PROFILE]"]
    
    if profile_df is not None and not profile_df.empty:
        row = profile_df.iloc[0]
        if 'age' in profile_df.columns:
            context_parts.append(f"Age: {row['age']} years old")
        if 'gender' in profile_df.columns:
            context_parts.append(f"Sex: {row['gender']}")
        if 'height_cm' in profile_df.columns:
            context_parts.append(f"Height: {row['height_cm']} cm")
        if 'weight_kg' in profile_df.columns:
            context_parts.append(f"Weight: {row['weight_kg']} kg")
        if 'averageDailySteps' in profile_df.columns:
            context_parts.append(f"Average daily steps: {row['averageDailySteps']}")
        if 'cluster' in profile_df.columns:
            context_parts.append(f"Activity profile: {row['cluster']}")
    
    if summary_df is not None and not summary_df.empty:
        context_parts.append(f"\n[AVAILABLE DATA]")
        context_parts.append(f"Health data: {len(summary_df)} days of records")
    
    return "\n".join(context_parts)
