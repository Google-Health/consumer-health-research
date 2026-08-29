"""Health Coach Agent for conversational health guidance.

This agent provides the user-facing conversational interface that:
- Maintains conversation history and context
- Guides users through health discussions
- Provides personalized recommendations

Usage:
    from pha.agents import HealthCoachAgent
    
    coach = HealthCoachAgent()
    coach.configure(gemini_api_key="your-key")
    
    # Single turn
    response = coach.respond("I'm worried about my sleep quality")
    
    # Multi-turn conversation
    coach.add_user_message("I'm worried about my sleep")
    response = coach.take_turn()
"""

import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Literal

# Try to import LLM backends
_GENAI_AVAILABLE = False
_LANGFUN_AVAILABLE = False

try:
    import google.genai as genai
    from google.genai import types
    _GENAI_AVAILABLE = True
except ImportError:
    pass

try:
    import langfun as lf
    _LANGFUN_AVAILABLE = True
except ImportError:
    pass

from ..prompts.health_coach_prompts import (
    HEALTH_COACH_SYSTEM_PROMPT,
    HEALTH_COACH_SIMPLE_PROMPT,
    GOAL_PROMPT,
    CONSTRAINTS_PROMPT,
    CONSIDERED_PROMPT,
    PROFILE_PROMPT,
    REC_DESC_PROMPT,
    REC_PROMPT,
    FINISH_DESC_PROMPT,
    FINISH_PROMPT,
    FOLLOW_UP_DESC_PROMPT,
    FOLLOW_UP_PROMPT,
)


__version__ = "0.1.0"


class HealthCoachAgent:
    """Conversational Health Coach Agent.
    
    This agent manages multi-turn health conversations, maintaining context
    and memory states to provide personalized guidance.
    
    Attributes:
        name: Agent identifier
        system_prompt: Main system prompt for the coach
        conv_context: Conversation context/preamble
        conv_utterances: History of conversation turns
        memory: Dictionary of memory states (goal, profile, etc.)
    """
    
    def __init__(
        self,
        name: str = "Coach",
        system_prompt: Optional[str] = None,
        simple_mode: bool = False,
    ):
        """Initialize the Health Coach Agent.
        
        Args:
            name: Name for the coach agent.
            system_prompt: Custom system prompt (uses default if None).
            simple_mode: If True, use simpler prompt for faster responses.
        """
        self.name = name
        self.system_prompt = system_prompt or (
            HEALTH_COACH_SIMPLE_PROMPT if simple_mode else HEALTH_COACH_SYSTEM_PROMPT
        )
        self.simple_mode = simple_mode
        
        # Conversation state
        self.conv_context = ""
        self.conv_utterances = ""
        
        # Memory states
        self.prompts = {
            "goal": GOAL_PROMPT,
            "constraints": CONSTRAINTS_PROMPT,
            "considered": CONSIDERED_PROMPT,
            "profile": PROFILE_PROMPT,
            "rec_desc": REC_DESC_PROMPT,
            "rec": REC_PROMPT,
            "finish_desc": FINISH_DESC_PROMPT,
            "finish": FINISH_PROMPT,
            "follow_desc": FOLLOW_UP_DESC_PROMPT,
            "follow": FOLLOW_UP_PROMPT,
        }
        self.memory_tags = {
            "goal": "[GOAL]",
            "constraints": "[CONSTRAINTS]",
            "considered": "[CONSIDERED]",
            "profile": "[PROFILE]",
            "rec_desc": "[REC_DESC]",
            "rec": "[REC]",
            "finish_desc": "[FINISH_DESC]",
            "finish": "[FINISH]",
            "follow_desc": "[FOLLOW_DESC]",
            "follow": "[FOLLOW]",
        }
        self.memory: Dict[str, str] = {key: "" for key in self.prompts.keys()}
        
        # Conversation statistics
        self.num_turns = 0
        self.avg_user_words_per_turn = 0.0
        self.avg_coach_words_per_turn = 0.0
        
        # Flags
        self.finish_flag = False
        self.rec_flag = False
        
        # LLM configuration
        self._client = None
        self._model_name = "models/gemini-3.1-flash-lite-preview"
        self._temperature = 0.7
        self._use_langfun = False
        self._lm = None
    
    def configure(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        use_langfun: bool = False,
        # Legacy parameter for backwards compatibility
        gemini_api_key: Optional[str] = None,
    ) -> "HealthCoachAgent":
        """Configure the LLM backend.
        
        Args:
            api_key: API key for the selected provider.
            provider: "gemini", "openai", or "anthropic".
            model_name: Model to use (uses provider defaults if not specified).
            temperature: Sampling temperature.
            use_langfun: If True, use langfun (Gemini only).
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
        
        self._model_name = model_name
        self._temperature = temperature
        self._use_langfun = use_langfun
        self._provider = provider
        self._api_key = api_key
        self._backend = None
        self._client = None
        self._lm = None
        
        # Use langfun for Gemini if requested
        if use_langfun and _LANGFUN_AVAILABLE and provider == "gemini":
            self._lm = lf.llms.GenAI(
                model=model_name,
                api_key=api_key,
            )
        else:
            # Use unified backend for all providers (including Gemini)
            from ..llm.backend_multimodel import get_llm_backend
            self._backend = get_llm_backend(
                backend_type=provider,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
            )
        
        return self
    
    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM with retry logic.
        
        Args:
            prompt: The prompt to send.
            max_retries: Maximum number of retries.
            
        Returns:
            Model response text.
        """
        for attempt in range(max_retries):
            try:
                if self._use_langfun and self._lm:
                    response = lf.query(prompt, lm=self._lm)
                    return str(response)
                elif self._backend:
                    # Use the multi-provider backend
                    return self._backend.generate(prompt)
                else:
                    return "Error: LLM not configured. Call configure() first."
            except Exception as e:
                print(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    traceback.print_exc()
                    return f"Error: {e}"
        
        return "Error: All retry attempts failed."
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the conversation history.
        
        Args:
            message: The user's message.
        """
        self._update_conv_state("User", message)
    
    def add_coach_message(self, message: str) -> None:
        """Add a coach message to the conversation history.
        
        Args:
            message: The coach's message.
        """
        self._update_conv_state("Coach", message)
    
    def _update_conv_state(self, agent_name: str, utterance: str) -> None:
        """Update conversation state with a new utterance.
        
        Args:
            agent_name: "User" or "Coach"
            utterance: The message text.
        """
        word_count = len(utterance.split())
        
        if agent_name == "User":
            self.avg_user_words_per_turn = (
                self.avg_user_words_per_turn * self.num_turns + word_count
            ) / (self.num_turns + 1) if self.num_turns > 0 else word_count
            self.conv_utterances += f"\nUser: {utterance}"
        elif agent_name == "Coach":
            self.avg_coach_words_per_turn = (
                self.avg_coach_words_per_turn * self.num_turns + word_count
            ) / (self.num_turns + 1) if self.num_turns > 0 else word_count
            self.conv_utterances += f"\nCoach: {utterance}"
            self.num_turns += 1
    
    def get_conversation_history(self) -> str:
        """Get the full conversation history.
        
        Returns:
            Formatted conversation history.
        """
        return self.conv_context + "\n" + self.conv_utterances
    
    def _call_llm_parallel(self, prompts: List[str], num_workers: int = 3) -> List[str]:
        """Call the LLM in parallel for multiple prompts.
        
        Args:
            prompts: List of prompts to send.
            num_workers: Number of parallel workers.
            
        Returns:
            List of responses in the same order as prompts.
        """
        results = [None] * len(prompts)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks with their index
            future_to_idx = {
                executor.submit(self._call_llm, prompt): idx 
                for idx, prompt in enumerate(prompts)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = f"Error: {e}"
        
        return results
    
    def _update_memory(self, mode: str = "GOAL") -> None:
        """Update internal memory states based on conversation.
        
        Uses parallel LLM calls for faster updates.
        
        Args:
            mode: "GOAL" for full update, "SIMPLE" for quick update.
        """
        conv_history = self.get_conversation_history()
        
        if mode == "SIMPLE":
            # Quick update: goal, follow-up, recommendation check (in parallel)
            prompts = [
                self.prompts["goal"] + conv_history,
                self.prompts["follow_desc"] + conv_history,
                self.prompts["rec_desc"] + conv_history,
            ]
            results = self._call_llm_parallel(prompts, num_workers=3)
            
            self._overwrite_memory("goal", results[0])
            self._overwrite_memory("follow_desc", results[1])
            self._overwrite_memory("rec_desc", results[2])
        else:
            # Full update: goal, recommendation check, finish check (in parallel)
            prompts = [
                self.prompts["goal"] + conv_history,
                self.prompts["rec_desc"] + conv_history,
                self.prompts["finish_desc"] + conv_history,
            ]
            results = self._call_llm_parallel(prompts, num_workers=3)
            
            self._overwrite_memory("goal", results[0])
            self._overwrite_memory("rec_desc", results[1])
            self._overwrite_memory("finish_desc", results[2])
    
    def _overwrite_memory(self, key: str, value: str) -> None:
        """Overwrite a memory state if valid.
        
        Args:
            key: Memory key.
            value: New value.
        """
        if "NOT APPLICABLE" not in value:
            self.memory[key] = value
    
    def _append_memory(self, key: str, value: str) -> None:
        """Append to a memory state if valid.
        
        Args:
            key: Memory key.
            value: Value to append.
        """
        if "NOT APPLICABLE" not in value:
            self.memory[key] += " " + value
    
    def dump_memory(self) -> str:
        """Dump all memory states for debugging.
        
        Returns:
            Formatted memory dump.
        """
        dump = ""
        for key in self.memory:
            tag = self.memory_tags.get(key, f"[{key.upper()}]")
            dump += f"\n{tag}: {self.memory[key]}"
        return dump
    
    def take_turn(self) -> str:
        """Take a single conversation turn.
        
        This is the main method for generating a response based on
        the current conversation state.
        
        Returns:
            The coach's response.
        """
        if self.simple_mode:
            return self._take_turn_simple()
        
        # Update memory states
        self._update_memory(mode="GOAL")
        
        conv_history = self.get_conversation_history()
        
        # Check if conversation should end
        if "FINISH" in self.memory.get("finish_desc", ""):
            if not self.finish_flag:
                self.finish_flag = True
                finish_response = self._call_llm(
                    self.system_prompt + self.prompts["finish"] + 
                    "\n\n*[CONVERSATION]*:\n" + conv_history
                )
                return finish_response
            else:
                self.finish_flag = False
        
        # Check if it's time for a recommendation
        elif "YESREC" in self.memory.get("rec_desc", ""):
            if not self.rec_flag:
                self.rec_flag = True
                rec_response = self._call_llm(
                    self.system_prompt + self.prompts["rec"] + 
                    "\n\n*[CONVERSATION]*:\n" + conv_history
                )
                return rec_response
            else:
                self.rec_flag = False
        
        # Regular response
        input_prompt = (
            self.system_prompt + 
            "\n\n*[[CONVERSATION]]*:" + conv_history
        )
        
        response = self._call_llm(input_prompt)
        return response
    
    def _take_turn_simple(self) -> str:
        """Take a simple turn without full memory update.
        
        Returns:
            The coach's response.
        """
        self._update_memory(mode="SIMPLE")
        conv_history = self.get_conversation_history()
        
        # Check for recommendation
        if "YESREC" in self.memory.get("rec_desc", ""):
            rec_response = self._call_llm(
                self.prompts["rec"] + conv_history
            )
            return rec_response
        
        # Check for follow-up
        elif "YESFOLLOW" in self.memory.get("follow_desc", ""):
            follow_response = self._call_llm(
                self.prompts["follow"] + conv_history
            )
            return follow_response
        
        # Regular response
        input_prompt = self.system_prompt + "\n[CONVERSATION]:\n" + conv_history
        return self._call_llm(input_prompt)
    
    def respond(self, user_message: str) -> str:
        """Convenience method for single-turn response.
        
        Args:
            user_message: The user's message.
            
        Returns:
            The coach's response.
        """
        from datetime import datetime
        from pha.streaming import emit_step
        ts = datetime.now().strftime("%H:%M:%S")
        emit_step("Coach", "Health Coach generating response...")
        print(f"\n[Coach:{ts}] respond: {user_message[:80]}...")
        self.add_user_message(user_message)
        response = self.take_turn()
        self.add_coach_message(response)
        emit_step("Coach", "Health Coach response ready", detail=response[:100])
        print(f"[Coach:{ts}] Response ({len(response)} chars): {response[:200]}...")
        return response
    
    def respond_with_analysis(self, user_message: str, analysis: str) -> str:
        """Respond with analysis from other agents (called by orchestrator).
        
        Takes analysis from external sources and incorporates it into the response.
        
        Args:
            user_message: The user's message.
            analysis: Analysis/insights from other agents.
            
        Returns:
            The coach's response incorporating the analysis.
        """
        from datetime import datetime
        from pha.streaming import emit_step
        ts = datetime.now().strftime("%H:%M:%S")
        emit_step("Coach", "Health Coach synthesizing agent insights...")
        print(f"\n[Coach:{ts}] respond_with_analysis: {user_message[:80]}...")
        print(f"[Coach:{ts}] Analysis ({len(analysis)} chars): {analysis[:200]}...")
        self.add_user_message(user_message)
        
        # Build prompt with analysis
        input_prompt = (
            self.system_prompt +
            self.get_conversation_history() +
            "\n[ANALYSIS]:\n" + analysis
        )
        
        response = self._call_llm(input_prompt)
        self.add_coach_message(response)
        print(f"[Coach:{ts}] Response ({len(response)} chars): {response[:200]}...")
        return response
    
    def reset_conversation(self) -> None:
        """Reset the conversation state."""
        self.conv_context = ""
        self.conv_utterances = ""
        self.memory = {key: "" for key in self.prompts.keys()}
        self.num_turns = 0
        self.avg_user_words_per_turn = 0.0
        self.avg_coach_words_per_turn = 0.0
        self.finish_flag = False
        self.rec_flag = False
    
    def initialize_context(self, context: str = "") -> None:
        """Initialize conversation context.
        
        Args:
            context: Additional context to add (e.g., user profile, health data).
        """
        self.conv_context = context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics.
        
        Returns:
            Dictionary of conversation stats.
        """
        return {
            "num_turns": self.num_turns,
            "avg_user_words_per_turn": self.avg_user_words_per_turn,
            "avg_coach_words_per_turn": self.avg_coach_words_per_turn,
            "memory_states": {k: len(v) for k, v in self.memory.items()},
        }


def create_health_coach(
    gemini_api_key: Optional[str] = None,
    simple_mode: bool = False,
    **kwargs
) -> HealthCoachAgent:
    """Factory function to create a configured Health Coach.
    
    Args:
        gemini_api_key: API key for Gemini.
        simple_mode: If True, use simpler prompts.
        **kwargs: Additional configuration options.
        
    Returns:
        Configured HealthCoachAgent.
    """
    coach = HealthCoachAgent(simple_mode=simple_mode)
    coach.configure(gemini_api_key=gemini_api_key, **kwargs)
    return coach
