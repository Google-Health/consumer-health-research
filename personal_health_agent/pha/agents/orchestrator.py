"""Multi-Agent Orchestrator for PHA.

This orchestrator coordinates between Data Science, Domain Expert, and Health Coach
agents to provide comprehensive health insights.

Usage:
    from pha.agents import MultiAgentOrchestrator
    
    orchestrator = MultiAgentOrchestrator()
    orchestrator.configure(gemini_api_key="your-key")
    orchestrator.set_agents(
        data_science_agent=ds_agent,
        domain_expert_agent=de_agent,
        health_coach_agent=hc_agent,
    )
    
    response = orchestrator.respond("How is my sleep quality?")
"""

# Version for debugging
__version__ = "1.9.0"  # Added end-of-request summary

import copy
import json
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

# Import shared request stats
from ..utils.request_stats import request_stats


# =============================================================================
# Agent Call Logging
# =============================================================================

def _log_agent_call(agent_name: str, query: str = None, response: str = None, error: str = None, flag_full_content: bool = False):
    """Log an agent call to the terminal for debugging.
    
    Args:
        agent_name: Name of the agent being called.
        query: The query being sent to the agent.
        response: The agent's response (will be truncated).
        error: Error message if the call failed.
        flag_full_content: If True, log the full content.
    """
    # Record stats
    request_stats.record_agent(agent_name)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    print(f"\n{'═' * 60}")
    print(f"🤖 [{timestamp}] AGENT CALL: {agent_name}")
    
    if query:
        if flag_full_content:
            print(f"   Query: {query}")
        else:
            query_preview = query[:150] + "..." if len(query) > 150 else query
            print(f"   Query: {query_preview}")
    
    if error:
        if flag_full_content:
            print(f"   Error: {error}")
        else:
            error_preview = error[:200] + "..." if len(error) > 200 else error
            print(f"   Error: {error_preview}")
    elif response is not None:
        if flag_full_content:
            print(f"   ✓ Response: {response}")
        else:
            response_preview = response[:200] + "..." if len(response) > 200 else response
            print(f"   ✓ Response: {response_preview}")
    
    print(f"{'═' * 60}\n")


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
    - Agent name references (e.g., "The Data Science agent found...")
    - Internal insight markers (e.g., [DATA_SCIENCE_AGENT_INSIGHTS])
    
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
    
    # Remove internal agent name references (users shouldn't see implementation details)
    # Patterns: "The Data Science agent...", "According to the domain_expert_agent...", etc.
    agent_ref_patterns = [
        # Bracketed agent insights markers (must be exact match, case insensitive)
        r'\[DATA_SCIENCE_AGENT_INSIGHTS\]\s*\n?',
        r'\[DOMAIN_EXPERT_AGENT_INSIGHTS\]\s*\n?',
        r'\[HEALTH_COACH_AGENT_INSIGHTS\]\s*\n?',
        r'\[SUPPORTING_AGENT_INSIGHTS\]\s*\n?',
        # Full phrases with various structures - greedy match to capture continuation
        r'(?:The |According to (?:the )?|Based on (?:the )?)?Data Science [Aa]gent(?:\'s)?(?:\s+(?:analysis|findings|calculations?|results?|data|insights?))?(?:\s+(?:found|reported|calculated|determined|analyzed|shows?|indicates?|suggests?|reveals?))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?Domain Expert [Aa]gent(?:\'s)?(?:\s+(?:analysis|findings|assessment|interpretation|insights?))?(?:\s+(?:found|reported|determined|analyzed|shows?|indicates?|suggests?|reveals?))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?Health Coach [Aa]gent(?:\'s)?(?:\s+(?:recommendations?|suggestions?|advice|guidance|insights?))?(?:\s+(?:found|reported|suggests?|recommends?|advises?))?(?:\s+that)?[,:]?\s*',
        # Snake_case versions
        r'(?:The |According to (?:the )?|Based on (?:the )?)?data_science_agent(?:\'s)?(?:\s+(?:analysis|findings))?(?:\s+(?:found|reported|calculated))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?domain_expert_agent(?:\'s)?(?:\s+(?:analysis|findings))?(?:\s+(?:found|reported))?(?:\s+that)?[,:]?\s*',
        r'(?:The |According to (?:the )?|Based on (?:the )?)?health_coach_agent(?:\'s)?(?:\s+(?:recommendations?|suggestions?))?(?:\s+(?:found|reported))?(?:\s+that)?[,:]?\s*',
    ]
    for pattern in agent_ref_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up any leftover punctuation/whitespace artifacts
    text = re.sub(r'^\s*[,;:]\s*', '', text)  # Remove leading punctuation
    # Normalize only horizontal whitespace (spaces/tabs), preserve newlines for markdown
    text = re.sub(r'[^\S\n]+', ' ', text)  # Collapse spaces/tabs but keep \n
    text = re.sub(r' *\n *', '\n', text)    # Clean spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Cap consecutive blank lines at 2
    
    # Capitalize first letter if it's now lowercase
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    
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
            # Log the leak for terminal debugging, but surface it in the response too
            print(f"[clean_agent_response] PROMPT LEAK detected: matched '{indicator}'")
            print(f"[clean_agent_response] Raw text (first 300 chars): {text[:300]}")
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

from ..prompts.orchestrator_prompts import (
    AGENT_NAME_DATA_SCIENCE,
    AGENT_NAME_DOMAIN_EXPERT,
    AGENT_NAME_HEALTH_COACH,
    ORCHESTRATOR_PREAMBLE,
    ORCHESTRATOR_FINAL_RESPONSE_IMPROVEMENT_PROMPT,
    TEAM_STRUCTURE_PROMPT,
    TOPIC_PROMPT,
    GOAL_PROMPT,
    FINISH_DESC_PROMPT,
    REPHRASE_PROMPT_SUPPORTING_AGENTS,
    SUPPORTING_AGENT_ADDITIONAL_PROMPT,
    UPDATE_TEAM_AGENT_INSIGHTS_PROMPT_TEMPLATE,
    REFLECT_PROMPT,
    FINAL_RESPONSE_PROMPT,
)


class MultiAgentOrchestrator:
    """Multi-Agent Orchestrator for PHA.
    
    Coordinates between Data Science, Domain Expert, and Health Coach agents
    to provide comprehensive, personalized health insights.
    
    The orchestrator:
    1. Determines which agents should handle a query
    2. Calls supporting agents (potentially in parallel)
    3. Synthesizes insights through the main agent
    4. Optionally reflects and refines the response
    """
    
    def __init__(
        self,
        name: str = "Personal Health Insights Agent Team",
        system_prompt: Optional[str] = None,
        debug_verbose: bool = False,
    ):
        """Initialize the orchestrator.
        
        Args:
            name: Name for the orchestrator.
            system_prompt: Custom system prompt (uses default if None).
            debug_verbose: If True, print debug information.
        """
        self.name = name
        self.system_prompt = system_prompt or ORCHESTRATOR_PREAMBLE
        self.debug_verbose = debug_verbose
        
        # Agent references
        self.data_science_agent = None
        self.domain_expert_agent = None
        self.health_coach_agent = None
        self.multi_agent_team: Dict[str, Any] = {}
        self.agent_name_list: List[str] = []
        
        # Team structure state
        self.main_agent = ""
        self.supporting_agents = ""
        self.supporting_agents_complete = ""
        self.collaboration_workflow = ""
        
        # Conversation state
        self.conv_context = ""
        self.conv_utterances = ""
        self.question_last = ""
        
        # Memory states
        self.memory = {
            "topic": "",
            "goal": "",
            "finish_desc": "",
            "data_science_agent_insights": "",
            "domain_expert_agent_insights": "",
            "health_coach_agent_insights": "",
        }
        
        # Reflection state
        self.flag_reflection_mode = False
        self.reflection_round_counter = 0
        self.reflection_round_upper_bound = 1
        self.rephrase_prompt_record: Dict[str, List[str]] = {}
        
        # Stats
        self.log_number_of_turns = 0
        
        # LLM configuration
        self._client = None
        self._model_name = "models/gemini-3.1-flash-lite-preview"
        self._temperature = 0.7
    
    def configure(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        # Legacy parameter for backwards compatibility
        gemini_api_key: Optional[str] = None,
    ) -> "MultiAgentOrchestrator":
        """Configure the LLM backend.
        
        Args:
            api_key: API key for the selected provider.
            provider: "gemini", "openai", or "anthropic".
            model_name: Model to use (uses provider defaults if not specified).
            temperature: Sampling temperature.
            gemini_api_key: DEPRECATED - use api_key instead.
            
        Returns:
            Self for chaining.
        """
        # Handle legacy parameter
        if gemini_api_key and not api_key:
            api_key = gemini_api_key
            if not provider:
                provider = "gemini"
        
        # If provider not specified, get from global config
        from ..llm.config import get_config
        config = get_config()
        if not provider:
            provider = config.provider
            
        # Update global configuration
        from ..llm.config import configure_global
        configure_global(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        
        # Initialize backend
        from ..llm.backend_multimodel import get_llm_backend
        self._backend = get_llm_backend(
            backend_type=provider,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
        )
        
        return self
    
    def set_agents(
        self,
        data_science_agent=None,
        domain_expert_agent=None,
        health_coach_agent=None,
    ) -> "MultiAgentOrchestrator":
        """Set the agent team.
        
        Args:
            data_science_agent: DataScienceAgent instance.
            domain_expert_agent: DomainExpertAgent instance.
            health_coach_agent: HealthCoachAgent instance.
            
        Returns:
            Self for chaining.
        """
        self.multi_agent_team = {}
        
        if data_science_agent is not None:
            self.data_science_agent = data_science_agent
            self.multi_agent_team[AGENT_NAME_DATA_SCIENCE] = data_science_agent
        
        if domain_expert_agent is not None:
            self.domain_expert_agent = domain_expert_agent
            self.multi_agent_team[AGENT_NAME_DOMAIN_EXPERT] = domain_expert_agent
        
        if health_coach_agent is not None:
            self.health_coach_agent = health_coach_agent
            self.multi_agent_team[AGENT_NAME_HEALTH_COACH] = health_coach_agent
            # Save initial system prompt for coach
            self._health_coach_initial_prompt = copy.deepcopy(
                health_coach_agent.system_prompt
            )
        
        self.agent_name_list = list(self.multi_agent_team.keys())
        
        # Always log which agents are registered
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"🤖 [{timestamp}] ORCHESTRATOR AGENTS REGISTERED")
        print(f"   Available agents: {self.agent_name_list}")
        print(f"   data_science_agent: {'✓' if data_science_agent else '✗'}")
        print(f"   domain_expert_agent: {'✓' if domain_expert_agent else '✗'}")
        print(f"   health_coach_agent: {'✓' if health_coach_agent else '✗'}")
        print(f"{'═' * 60}\n")
        
        return self
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the orchestrator's LLM.
        
        Args:
            prompt: The prompt to send.
            max_retries: Maximum number of retries.
            
        Returns:
            Model response text.
        """
        for attempt in range(max_retries):
            try:
                if self._backend:
                    return self._backend.generate(prompt)
                else:
                    return "Error: LLM not configured. Call configure() first."
            except Exception as e:
                if self.debug_verbose:
                    print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    traceback.print_exc()
                    return f"Error: {e}"
        
        return "Error: All retry attempts failed."
    
    def _clear_json_format(self, text: str) -> str:
        """Clean JSON response from LLM.
        
        Args:
            text: Raw LLM response.
            
        Returns:
            Cleaned JSON string.
        """
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return text.strip()
    
    def get_conversation_history(self) -> str:
        """Get the full conversation history."""
        return self.conv_context + "\n" + self.conv_utterances
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation."""
        self.conv_utterances += f"\nUser: {message}"
        self.question_last = message
    
    def add_response(self, response: str) -> None:
        """Add an orchestrator response to the conversation."""
        self.conv_utterances += f"\nAssistant: {response}"
    
    def _update_memory(self) -> None:
        """Update memory states based on conversation.
        
        Runs topic, goal, and finish_desc extraction in parallel since
        they are independent of each other.
        """
        conv_history = self.get_conversation_history()
        
        # All 3 extractions are independent — run in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            topic_future = executor.submit(self._call_llm, TOPIC_PROMPT + conv_history)
            goal_future = executor.submit(self._call_llm, GOAL_PROMPT + conv_history)
            finish_future = executor.submit(self._call_llm, FINISH_DESC_PROMPT + conv_history)
            
            self.memory["topic"] = topic_future.result()
            self.memory["goal"] = goal_future.result()
            self.memory["finish_desc"] = finish_future.result()
    
    def _update_agent_team_structure(self) -> None:
        """Determine which agents should handle the current query."""
        prompt = (
            self.system_prompt + "\n\n" +
            TEAM_STRUCTURE_PROMPT +
            "\n[CONVERSATION]\n" + self.get_conversation_history() +
            "\n[LAST_QUESTION]\n" + self.question_last +
            "\n[TOPIC]\n" + self.memory.get("topic", "")
        )
        
        for attempt in range(5):
            response = self._call_llm(prompt)
            try:
                # Log the raw response for debugging
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{timestamp}] Team structure LLM response (attempt {attempt+1}):")
                print(f"   {response[:300]}..." if len(response) > 300 else f"   {response}")
                
                response = self._clear_json_format(response)
                result = json.loads(response)
                
                # Note: dict.get(key, default) only returns `default` when the
                # key is missing — not when its value is "". Some models (notably
                # Claude Sonnet 4.5) occasionally return empty agent strings,
                # which previously short-circuited the entire team and made
                # the orchestrator fall back to a context-less LLM call.
                raw_main = (result.get("main_agent") or "").strip()
                if raw_main and raw_main in self.agent_name_list:
                    self.main_agent = raw_main
                else:
                    # No valid main agent chosen — use health coach if available
                    # (it's typically the user-facing voice), otherwise the first
                    # available agent.
                    self.main_agent = (
                        AGENT_NAME_HEALTH_COACH
                        if AGENT_NAME_HEALTH_COACH in self.agent_name_list
                        else (self.agent_name_list[0] if self.agent_name_list else "")
                    )

                raw_supporting = (result.get("supporting_agents") or "").strip()
                if raw_supporting:
                    self.supporting_agents = ";".join([
                        a for a in raw_supporting.split(";")
                        if a and a != self.main_agent and a in self.agent_name_list
                    ])
                else:
                    # Empty supporting list paired with a successfully chosen
                    # main agent is fine. But if the model returned empty for
                    # both, route every other available agent as supporting so
                    # we don't silently lose data-access (e.g. data_science).
                    if not raw_main:
                        self.supporting_agents = ";".join([
                            a for a in self.agent_name_list if a != self.main_agent
                        ])
                    else:
                        self.supporting_agents = ""

                self.collaboration_workflow = result.get("collaboration_workflow", "")

                # Always log team structure for debugging
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n{'═' * 60}")
                print(f"📋 [{timestamp}] TEAM STRUCTURE DECISION")
                print(f"   Main Agent: {self.main_agent}")
                print(f"   Supporting Agents: {self.supporting_agents or '(none)'}")
                print(f"   Workflow: {self.collaboration_workflow[:100]}..." if len(self.collaboration_workflow) > 100 else f"   Workflow: {self.collaboration_workflow}")
                print(f"{'═' * 60}\n")
                
                break
            except Exception as e:
                if attempt == 4:
                    # Fallback: health coach as main, others as supporting
                    self.main_agent = AGENT_NAME_HEALTH_COACH if AGENT_NAME_HEALTH_COACH in self.agent_name_list else self.agent_name_list[0]
                    self.supporting_agents = ";".join([a for a in self.agent_name_list if a != self.main_agent])
                    self.collaboration_workflow = "Supporting agents provide insights, main agent responds."
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"\n{'═' * 60}")
                    print(f"📋 [{timestamp}] TEAM STRUCTURE (FALLBACK)")
                    print(f"   Main Agent: {self.main_agent}")
                    print(f"   Supporting Agents: {self.supporting_agents}")
                    print(f"   Error: {str(e)[:100]}")
                    print(f"{'═' * 60}\n")
    
    def _log_rephrase_prompt_record(self, phase: str) -> None:
        """Log the current rephrase prompt record for debugging."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"📝 [{timestamp}] REPHRASE PROMPT RECORD ({phase})")
        if not self.rephrase_prompt_record:
            print("   (empty)")
        else:
            for agent_name, prompts in self.rephrase_prompt_record.items():
                prompt_preview = prompts[-1] if prompts else ""
                prompt_preview = (
                    prompt_preview[:200] + "..."
                    if len(prompt_preview) > 200
                    else prompt_preview
                )
                print(f"   {agent_name}: {prompt_preview}")
        print(f"{'═' * 60}\n")

    def _build_supporting_agent_rephrase_prompts(
        self,
        supporting_agent_list: List[str],
    ) -> None:
        """Build rephrased prompts for supporting agents using orchestrator LLM."""
        self.rephrase_prompt_record = {}

        rephrase_prompt = REPHRASE_PROMPT_SUPPORTING_AGENTS(
            original_prompt=self.question_last,
            main_agent=self.main_agent,
            supporting_agents=self.supporting_agents,
            collaboration_workflow=self.collaboration_workflow,
        )

        rephrase_dict: Dict[str, Any] = {}
        for attempt in range(3):
            response = self._call_llm(rephrase_prompt)
            try:
                parsed = json.loads(self._clear_json_format(response))
                if isinstance(parsed, dict):
                    rephrase_dict = parsed
                    break
            except Exception:
                if self.debug_verbose:
                    print(f"Rephrase parsing failed (attempt {attempt + 1}/3)")
                continue

        for agent_name in supporting_agent_list:
            # Keep data science on raw question for analysis fidelity.
            if agent_name == AGENT_NAME_DATA_SCIENCE:
                self.rephrase_prompt_record[agent_name] = [self.question_last]
                continue

            rephrase_question = str(rephrase_dict.get(agent_name, "") or "").strip()
            if rephrase_question:
                prompt = (
                    f"Answer the user's raw question: {self.question_last}\n\n"
                    f"In addition, focus specifically on: {rephrase_question}"
                )
            else:
                prompt = self.question_last
            self.rephrase_prompt_record[agent_name] = [prompt]

        self._log_rephrase_prompt_record("initial")

    @staticmethod
    def _is_error_like_payload(text: str) -> bool:
        """Return True if text looks like technical error output."""
        if not text:
            return True
        normalized = text.strip().lower()
        if not normalized:
            return True
        error_signals = ("error:", "traceback", "exception:", "failed:", "technical issue", "⚠️", "stack trace")
        for err in error_signals:
            if normalized.startswith(err):
                return True
        return False

    def _merge_and_store_agent_insights(self, agent_name: str, new_insights: str) -> None:
        """Merge new insight text into persisted per-agent insights."""
        memory_key = f"{agent_name}_insights"
        old_insights = str(self.memory.get(memory_key, "") or "").strip()
        cleaned_new = clean_agent_response(str(new_insights or "")).strip()

        # Ignore empty or technical error payloads from supporting agents.
        if self._is_error_like_payload(cleaned_new):
            return

        agent_token = agent_name.upper()
        update_prompt = (
            UPDATE_TEAM_AGENT_INSIGHTS_PROMPT_TEMPLATE(agent_name)
            + f"\n[{agent_token}_INSIGHTS]\n{old_insights}"
            + f"\n[NEW_{agent_token}_INSIGHTS]\n{cleaned_new}"
        )

        merged_response = clean_agent_response(self._call_llm(update_prompt)).strip()
        if not merged_response:
            # If merge call is malformed, preserve old insights.
            return

        if re.search(r"\bEMPTY\b", merged_response.upper()):
            self.memory[memory_key] = ""
            return

        if self._is_error_like_payload(merged_response):
            # Guard against merge failures contaminating memory.
            if not old_insights:
                self.memory[memory_key] = cleaned_new
            return

        self.memory[memory_key] = merged_response

    def _build_supporting_agent_input_prompt(self, agent_name: str, question_prompt: str) -> str:
        """Build a supporting-agent prompt with existing insights context."""
        prior_insights = str(self.memory.get(f"{agent_name}_insights", "") or "").strip()
        additional_context = SUPPORTING_AGENT_ADDITIONAL_PROMPT(agent_name).strip()
        agent_token = agent_name.upper()

        prompt_parts = []
        if additional_context:
            prompt_parts.append(additional_context)

        if prior_insights:
            prompt_parts.append(
                "You have previously produced insights for this conversation. "
                "Reuse and extend them instead of repeating the same analysis."
            )
            prompt_parts.append(f"[{agent_token}_INSIGHTS]\n{prior_insights}")

        prompt_parts.append(f"[LAST_QUESTION]\n{question_prompt}")
        return "\n\n".join(prompt_parts)

    def _call_supporting_agents(
        self,
        parallel: bool = True,
        flag_rephrase_prompt: bool = True,
    ) -> None:
        """Call supporting agents to gather insights.
        
        Args:
            parallel: If True, call agents in parallel.
            flag_rephrase_prompt: If True, regenerate prompts for supporting agents.
        """
        if not self.supporting_agents:
            return
        
        supporting_agent_list = [a for a in self.supporting_agents.split(";") if a]
        if flag_rephrase_prompt:
            self._build_supporting_agent_rephrase_prompts(supporting_agent_list)
        
        if self.debug_verbose:
            print(f"Calling supporting agents: {supporting_agent_list}")
        
        def call_agent(agent_name: str) -> Tuple[str, str]:
            """Call a single supporting agent."""
            try:
                question_prompt = self.rephrase_prompt_record.get(agent_name, [self.question_last])[-1]
                if isinstance(question_prompt, str) and question_prompt.startswith("reflection_questions::"):
                    follow_up = question_prompt.split("reflection_questions::", 1)[-1].strip()
                    question_prompt = (
                        f"Answer the user's raw question: {self.question_last}\n\n"
                        f"Additional follow-up requested by orchestrator: {follow_up}"
                    )
                prompt = self._build_supporting_agent_input_prompt(agent_name, question_prompt)
                _log_agent_call(agent_name, query=prompt, flag_full_content=True)
                
                if agent_name == AGENT_NAME_DATA_SCIENCE and self.data_science_agent:
                    result = self.data_science_agent.query(prompt)
                    # _log_agent_call(agent_name, response=str(result)[:500])
                    # log the full response for debugging purposes
                    _log_agent_call(agent_name, response=str(result), flag_full_content=True)
                    return agent_name, str(result)
                
                elif agent_name == AGENT_NAME_DOMAIN_EXPERT and self.domain_expert_agent:
                    result = self.domain_expert_agent.call_agent(prompt)
                    _log_agent_call(agent_name, response=str(result)[:500])
                    return agent_name, str(result)
                
                elif agent_name == AGENT_NAME_HEALTH_COACH and self.health_coach_agent:
                    # For health coach as supporting, just get a response
                    result = self.health_coach_agent.respond(prompt)
                    _log_agent_call(agent_name, response=str(result)[:500])
                    return agent_name, str(result)
                
                return agent_name, ""
            
            except Exception as e:
                _log_agent_call(agent_name, error=str(e))
                if self.debug_verbose:
                    print(f"Error calling {agent_name}: {e}")
                return agent_name, f"Error: {e}"
        
        if parallel and len(supporting_agent_list) > 1:
            # Call agents in parallel
            with ThreadPoolExecutor(max_workers=len(supporting_agent_list)) as executor:
                futures = {executor.submit(call_agent, name): name for name in supporting_agent_list}
                for future in as_completed(futures):
                    agent_name, result = future.result()
                    self._merge_and_store_agent_insights(agent_name, result)
        else:
            # Call agents sequentially
            for agent_name in supporting_agent_list:
                _, result = call_agent(agent_name)
                self._merge_and_store_agent_insights(agent_name, result)
    
    def _accumulate_supporting_agents_complete(self) -> None:
        """Update the running union of all supporting agents consulted so far."""
        if self.supporting_agents:
            if self.supporting_agents_complete:
                existing = set(self.supporting_agents_complete.split(";"))
                new = set(self.supporting_agents.split(";"))
                self.supporting_agents_complete = ";".join(existing | new)
            else:
                self.supporting_agents_complete = self.supporting_agents
    
    def _organize_supporting_agent_insights(self) -> str:
        """Organize all supporting agent insights into a formatted string."""
        insights = []

        allowed_agents = [
            a.strip()
            for a in str(self.supporting_agents_complete or "").split(";")
            if a.strip()
        ]

        if AGENT_NAME_DATA_SCIENCE in allowed_agents and self.memory.get("data_science_agent_insights"):
            insights.append(
                f"[DATA_SCIENCE_AGENT_INSIGHTS]\n{self.memory['data_science_agent_insights']}"
            )

        if AGENT_NAME_DOMAIN_EXPERT in allowed_agents and self.memory.get("domain_expert_agent_insights"):
            insights.append(
                f"[DOMAIN_EXPERT_AGENT_INSIGHTS]\n{self.memory['domain_expert_agent_insights']}"
            )

        if AGENT_NAME_HEALTH_COACH in allowed_agents and self.memory.get("health_coach_agent_insights"):
            insights.append(
                f"[HEALTH_COACH_AGENT_INSIGHTS]\n{self.memory['health_coach_agent_insights']}"
            )

        return "\n\n".join(insights)
    
    def _call_main_agent(self) -> Tuple[str, str]:
        """Call the main agent with supporting insights.
        
        Returns:
            Tuple of (main agent response, supporting insights).
        """
        supporting_insights = self._organize_supporting_agent_insights()
        
        if self.main_agent == AGENT_NAME_HEALTH_COACH and self.health_coach_agent:
            # Health coach as main agent with analysis from others
            _log_agent_call(f"{self.main_agent} (MAIN)", query=self.question_last)
            if supporting_insights:
                response = self.health_coach_agent.respond_with_analysis(
                    self.question_last,
                    supporting_insights
                )
            else:
                response = self.health_coach_agent.respond(self.question_last)
            _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response)[:500])
            return response, supporting_insights
        
        elif self.main_agent == AGENT_NAME_DOMAIN_EXPERT and self.domain_expert_agent:
            # Domain expert as main with synthesis
            prompt = self.question_last
            if supporting_insights:
                prompt += f"\n\n[SUPPORTING_AGENT_INSIGHTS]\n{supporting_insights}"
            _log_agent_call(f"{self.main_agent} (MAIN)", query=prompt[:300])
            response = self.domain_expert_agent.call_agent(prompt)
            _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response)[:500])
            return str(response), supporting_insights
        
        elif self.main_agent == AGENT_NAME_DATA_SCIENCE and self.data_science_agent:
            # Data science as main
            _log_agent_call(f"{self.main_agent} (MAIN)", query=self.question_last)
            response = self.data_science_agent.query(self.question_last)
            # _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response)[:500])
            # log the full response for debugging purposes
            _log_agent_call(f"{self.main_agent} (MAIN)", response=str(response), flag_full_content=True)
            return str(response), supporting_insights
        
        else:
            # Fallback: use orchestrator LLM
            prompt = FINAL_RESPONSE_PROMPT(
                self.main_agent,
                self.supporting_agents,
                self.supporting_agents_complete,
                self.collaboration_workflow
            )
            prompt += f"\n[CONVERSATION]\n{self.get_conversation_history()}"
            prompt += f"\n[SUPPORTING_AGENT_INSIGHTS]\n{supporting_insights}"
            prompt += f"\n[MAIN_AGENT_RESPONSE]\n(Generate response)"
            
            response = self._call_llm(prompt)
            return response, supporting_insights
    
    def take_turn(self) -> str:
        """Take a single conversation turn.
        
        This is the main orchestration method that:
        1. Updates memory states
        2. Determines agent team structure
        3. Calls supporting agents
        4. Calls main agent
        5. Reflects and optionally refines (catches questions answerable by data)
        
        Returns:
            The orchestrated response.
        """
        from pha.streaming import emit_step
        
        self.log_number_of_turns += 1
        self.reflection_round_counter = 0
        self.flag_reflection_mode = False
        self.rephrase_prompt_record = {}
        
        # === Parallel memory + overlapped team structure ===
        # Topic, goal, finish_desc are independent LLM calls.
        # Team structure only needs topic, so we start it as soon as
        # topic resolves — overlapping with the still-running goal/finish calls.
        emit_step("Orch", "Analyzing query and updating memory...")
        conv_history = self.get_conversation_history()
        
        with ThreadPoolExecutor(max_workers=3) as mem_executor:
            topic_future = mem_executor.submit(
                self._call_llm, TOPIC_PROMPT + conv_history
            )
            goal_future = mem_executor.submit(
                self._call_llm, GOAL_PROMPT + conv_history
            )
            finish_future = mem_executor.submit(
                self._call_llm, FINISH_DESC_PROMPT + conv_history
            )
            
            # Block on topic (required by team structure)
            self.memory["topic"] = topic_future.result()
            
            if len(self.agent_name_list) > 1:
                # Overlap: determine team structure while goal/finish still computing
                emit_step("Orch", "Determining which agents to use...")
                self._update_agent_team_structure()
            
            # Collect remaining memory results (likely already done by now)
            self.memory["goal"] = goal_future.result()
            self.memory["finish_desc"] = finish_future.result()
        
        if len(self.agent_name_list) > 1:
            emit_step("Orch", f"Team: main={self.main_agent}, supporting={self.supporting_agents or 'none'}")
            
            # Call supporting agents
            if self.supporting_agents:
                emit_step("Orch", f"Calling supporting agents: {self.supporting_agents}")
            self._call_supporting_agents(flag_rephrase_prompt=True)
            self._accumulate_supporting_agents_complete()
            
            # Call main agent
            emit_step("Orch", f"Calling main agent: {self.main_agent}")
            main_response, supporting_insights = self._call_main_agent()

            # Persist main-agent output as that agent's evolving insight memory.
            # This preserves continuity when the same agent is a supporting agent later.
            if self.main_agent in self.agent_name_list:
                self._merge_and_store_agent_insights(self.main_agent, main_response)
            
            # Reflect before final response (catch questions answerable by data)
            emit_step("Orch", "Reflecting on response quality...")
            reflection_result = self._reflect_before_final_response(
                main_response, supporting_insights
            )
            
            # If reflection found improvements needed, call supporting agents again
            if reflection_result.get("decision") == "YES":
                self.flag_reflection_mode = True
                self.reflection_round_counter += 1
                
                if self.reflection_round_counter <= self.reflection_round_upper_bound:
                    emit_step("Orch", "Reflection found improvements needed — re-querying agents...")
                    # Update rephrase prompts with reflection questions
                    for agent_name, questions in reflection_result.get("reflection_questions", {}).items():
                        if isinstance(questions, list):
                            questions = ". ".join(questions)
                        if agent_name not in self.rephrase_prompt_record:
                            self.rephrase_prompt_record[agent_name] = [self.question_last]
                        self.rephrase_prompt_record[agent_name].append(
                            f"reflection_questions::{questions}"
                        )
                    self._log_rephrase_prompt_record("reflection update")
                    
                    # Call supporting agents again with new questions
                    self._call_supporting_agents(flag_rephrase_prompt=False)
                    self._accumulate_supporting_agents_complete()
                    
                    # Call main agent again with new insights
                    main_response, supporting_insights = self._call_main_agent()
                    if self.main_agent in self.agent_name_list:
                        self._merge_and_store_agent_insights(self.main_agent, main_response)
            
            emit_step("Orch", "Composing final response...")
        
        elif len(self.agent_name_list) == 1:
            # Single agent mode
            self.main_agent = self.agent_name_list[0]
            emit_step("Orch", f"Single-agent mode: calling {self.main_agent}")
            main_response, _ = self._call_main_agent()
            if self.main_agent in self.agent_name_list:
                self._merge_and_store_agent_insights(self.main_agent, main_response)
        
        else:
            # No agents configured - use orchestrator directly
            main_response = self._call_llm(
                self.system_prompt + 
                "\n[CONVERSATION]\n" + self.get_conversation_history() +
                "\n\nRespond to the user's question: " + self.question_last
            )
            if self.main_agent in self.agent_name_list:
                self._merge_and_store_agent_insights(self.main_agent, main_response)


        main_response_new = self._call_llm(
            ORCHESTRATOR_FINAL_RESPONSE_IMPROVEMENT_PROMPT +
            "\n[CONVERSATION]\n" + self.get_conversation_history() +
            "\n\nRespond to the user's question: " + self.question_last +
            "\n[FINAL_RESPONSE]\n" + main_response
        )
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"[Orch:{ts}] updating main response from '{main_response}'")
        print(f"\n{'═' * 10}")
        print(f"[Orch:{ts}] to new main response: '{main_response_new}'")
        print(f"\n{'═' * 60}")
        main_response = main_response_new

        # Check for finish
        if "FINISH" in self.memory.get("finish_desc", ""):
            return main_response + "\n\n I look forward to speaking with you again soon!"
        
        return main_response
    
    def _reflect_before_final_response(
        self,
        main_agent_response: str,
        supporting_agent_insights: str,
    ) -> Dict[str, Any]:
        """Reflect on whether the response can be improved by additional agent queries.
        
        This is critical for catching when the health coach asks questions that
        can be answered by the data science agent from the user's data.
        
        Args:
            main_agent_response: Response from the main agent.
            supporting_agent_insights: Insights from supporting agents.
            
        Returns:
            Dictionary with 'decision' (YES/NO) and 'reflection_questions'.
        """
        from ..prompts.orchestrator_prompts import (
            REFLECT_PROMPT, 
            AGENT_NAME_DATA_SCIENCE,
        )
        
        reflection_results = {
            "decision": "NO",
            "reflection_questions": {},
        }
        
        # Skip if no supporting agents
        if not self.supporting_agents:
            return reflection_results
        
        # Skip if data science agent is not in supporting agents
        # (reflection mainly catches data-answerable questions)
        if AGENT_NAME_DATA_SCIENCE not in self.supporting_agents:
            return reflection_results
        
        reflect_prompt = REFLECT_PROMPT(
            original_prompt=self.question_last,
            main_agent=self.main_agent,
            supporting_agents=self.supporting_agents,
            collaboration_workflow=self.collaboration_workflow,
            main_agent_response=main_agent_response,
            supporting_agent_insights=supporting_agent_insights,
        )
        
        trial_count = 0
        while trial_count < 3:
            trial_count += 1
            try:
                reflect_response = self._call_llm(reflect_prompt)
                reflect_response = self._clear_json_format(reflect_response)
                reflection_results = json.loads(reflect_response)
                
                if "YES" in reflection_results.get("decision", ""):
                    # Validate that reflection questions target known agents
                    reflection_agents = list(reflection_results.get("reflection_questions", {}).keys())
                    if not set(reflection_agents).issubset(set(self.agent_name_list)):
                        # Invalid agents referenced, skip reflection
                        reflection_results = {"decision": "NO", "reflection_questions": {}}
                    else:
                        # Update supporting agents for next round
                        self.supporting_agents = ";".join(reflection_agents)
                break
            except Exception as e:
                if trial_count >= 3:
                    reflection_results = {"decision": "NO", "reflection_questions": {}}
                continue
        
        return reflection_results
    
    def respond(self, user_message: str, max_retries: int = 2) -> str:
        """Convenience method for single-turn response.
        
        Args:
            user_message: The user's message.
            max_retries: Maximum number of retries for problematic responses.
            
        Returns:
            The orchestrated response (cleaned of internal labels).
        """
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n{'═' * 60}")
        print(f"[Orch:{ts}] ═══ Orchestrator.respond ═══")
        print(f"[Orch:{ts}] Query: {user_message[:100]}...")
        print(f"[Orch:{ts}] Agents: {self.agent_name_list}")
        print(f"{'═' * 60}")
        
        # Reset stats for this request
        request_stats.reset()
        
        self.add_user_message(user_message)
        
        for attempt in range(max_retries):
            print(f"\n[Orch:{ts}] --- Attempt {attempt+1}/{max_retries} ---")
            response = self.take_turn()
            
            print(f"[Orch:{ts}] take_turn returned ({len(response) if response else 0} chars): {repr(response[:200] if response else 'None')}...")
            
            # Check for problematic response patterns before cleaning
            response_lower = response.lower() if response else ""
            
            # Detect prompt leakage (model echoing instructions)
            prompt_leak_patterns = [
                'you must do so safely',
                'you will analyze and synthesize',
                'with the goal of delivering',
                'clinically accurate, evidence-based',
                'you are a domain expert',
            ]
            is_prompt_leak = any(p in response_lower for p in prompt_leak_patterns)
            
            # Detect raw search results (not synthesized)
            is_raw_search = response_lower.count('url:') >= 2 or response_lower.count('relevance score:') >= 2
            
            if is_prompt_leak:
                print(f"[Orch:{ts}] *** PROMPT LEAK DETECTED in response")
            if is_raw_search:
                print(f"[Orch:{ts}] *** RAW SEARCH RESULTS detected in response")
            
            if (is_prompt_leak or is_raw_search) and attempt < max_retries - 1:
                print(f"[Orch:{ts}] Retrying due to problematic response...")
                
                # Add explicit synthesis request
                synthesis_prompt = (
                    "Please provide a direct, synthesized answer to the user's question. "
                    "Do not output raw data or search results. "
                    "Summarize the key findings in natural language."
                )
                
                # For raw search results, try to synthesize
                if is_raw_search and self.domain_expert_agent:
                    try:
                        synthesis_request = f"Based on these search results, answer the user's question: {user_message}\n\nSearch results:\n{response[:2000]}\n\n{synthesis_prompt}"
                        synthesized = self.domain_expert_agent.call_agent(synthesis_request)
                        if synthesized and len(synthesized) > 50:
                            cleaned = clean_agent_response(synthesized)
                            if cleaned and "⚠️" not in cleaned:
                                self.add_response(cleaned)
                                print(f"[Orch:{ts}] ✓ Synthesized response ({len(cleaned)} chars)")
                                request_stats.print_summary()
                                return cleaned
                    except Exception as e:
                        print(f"[Orch:{ts}] Synthesis failed: {e}")

                # Handle problematic cases outside the raw-search + domain-expert path.
                # Keep existing logic above unchanged, and add a fallback repair pass.
                if is_prompt_leak:
                    try:
                        fallback_request = (
                            f"User question: {user_message}\n\n"
                            f"Draft response (may contain leaked prompt or raw output):'''\n{response}'''\n\n"
                            f"{synthesis_prompt}\n"
                            "Do not include role or instruction text in the final answer."
                        )
                        repaired = self._call_llm(fallback_request)
                        if repaired and len(repaired) > 50:
                            cleaned_repaired = clean_agent_response(str(repaired))
                            if cleaned_repaired and "⚠️" not in cleaned_repaired:
                                self.add_response(cleaned_repaired)
                                print(f"[Orch:{ts}] ✓ Fallback repaired response ({len(cleaned_repaired)} chars)")
                                request_stats.print_summary()
                                return cleaned_repaired
                    except Exception as e:
                        print(f"[Orch:{ts}] Fallback repair failed: {e}")
                
                continue
            
            self.add_response(response)
            cleaned = clean_agent_response(response)
            print(f"[Orch:{ts}] ✓ Final response ({len(cleaned)} chars): {cleaned[:200]}...")
            request_stats.print_summary()
            return cleaned
        
        # If all retries failed, return the last cleaned response
        cleaned = clean_agent_response(response)
        print(f"[Orch:{ts}] ✗ All retries exhausted, returning last response ({len(cleaned)} chars)")
        request_stats.print_summary()
        return cleaned
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.conv_context = ""
        self.conv_utterances = ""
        self.question_last = ""
        self.memory = {k: "" for k in self.memory}
        self.main_agent = ""
        self.supporting_agents = ""
        self.supporting_agents_complete = ""
        self.collaboration_workflow = ""
        self.log_number_of_turns = 0
        self.rephrase_prompt_record = {}
        
        # Reset health coach conversation if present
        if self.health_coach_agent:
            self.health_coach_agent.reset_conversation()
    
    def get_team_structure(self) -> Dict[str, str]:
        """Get the current agent team structure."""
        return {
            "main_agent": self.main_agent,
            "supporting_agents": self.supporting_agents,
            "collaboration_workflow": self.collaboration_workflow,
        }
    
    def get_agent_insights(self) -> Dict[str, str]:
        """Get insights from each agent."""
        return {
            "data_science": self.memory.get("data_science_agent_insights", ""),
            "domain_expert": self.memory.get("domain_expert_agent_insights", ""),
            "health_coach": self.memory.get("health_coach_agent_insights", ""),
        }


def create_orchestrator(
    api_key: Optional[str] = None,
    provider: str = "gemini",
    model_name: Optional[str] = None,
    data_science_agent=None,
    domain_expert_agent=None,
    health_coach_agent=None,
    debug_verbose: bool = False,
    # Legacy parameter for backwards compatibility
    gemini_api_key: Optional[str] = None,
) -> MultiAgentOrchestrator:
    """Factory function to create a configured orchestrator.
    
    Args:
        api_key: API key for the selected provider.
        provider: "gemini", "openai", or "anthropic".
        model_name: Model to use (uses provider defaults if not specified).
        data_science_agent: DataScienceAgent instance.
        domain_expert_agent: DomainExpertAgent instance.
        health_coach_agent: HealthCoachAgent instance.
        debug_verbose: If True, print debug information.
        gemini_api_key: DEPRECATED - use api_key instead.
        
    Returns:
        Configured MultiAgentOrchestrator.
    """
    # Handle legacy parameter
    if gemini_api_key and not api_key:
        api_key = gemini_api_key
        provider = "gemini"
    
    orchestrator = MultiAgentOrchestrator(debug_verbose=debug_verbose)
    orchestrator.configure(
        api_key=api_key,
        provider=provider,
        model_name=model_name,
    )
    orchestrator.set_agents(
        data_science_agent=data_science_agent,
        domain_expert_agent=domain_expert_agent,
        health_coach_agent=health_coach_agent,
    )
    return orchestrator
