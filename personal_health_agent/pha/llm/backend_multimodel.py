"""LLM backend abstraction layer with multi-provider support.

This module provides the LLM backend interface for PHA agents,
supporting Google Gemini, OpenAI, and Anthropic models.

It uses the unified configuration from pha.llm.config.
"""

import os
from typing import Optional, Callable, Literal
from abc import ABC, abstractmethod

from pha.llm.config import get_config, DEFAULT_MODELS, ProviderType


# =============================================================================
# Backend Interface
# =============================================================================

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (gemini, openai, anthropic)."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        pass
    
    def register(self) -> None:
        """Register this backend (no-op for most backends)."""
        pass


# =============================================================================
# Gemini Backend
# =============================================================================

class GeminiBackend(LLMBackend):
    """Gemini backend using google-genai directly."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: "
                "pip install google-genai"
            )
        
        # Use config if not explicitly provided
        config = get_config()
        self.api_key = api_key or config.api_key
        
        # Only check config if provider matches, otherwise fallback to env
        if not self.api_key and config.provider == "gemini":
             self.api_key = config.api_key
        
        if not self.api_key:
             self.api_key = os.environ.get("GEMINI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name or DEFAULT_MODELS["gemini"]
        self._types = types
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    
    @property
    def provider(self) -> str:
        return "gemini"
    
    def generate(self, prompt: str) -> str:
        try:
            from pha.streaming import emit_step
            emit_step("LLM", f"Calling {self.model_name}...", detail=f"prompt={len(prompt)} chars")
            print(f"[LLM:Gemini] Calling model={self.model_name}, prompt_len={len(prompt)}, temp={self.generation_config.temperature}")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=self.generation_config,
            )
            result = response.text
            print(f"[LLM:Gemini] Response received: {len(result)} chars")
            return result
        except Exception as e:
            print(f"[LLM:Gemini] *** ERROR: {e}")
            # Wrap error for consistent handling
            raise RuntimeError(f"Gemini API error: {e}")


# =============================================================================
# OpenAI Backend
# =============================================================================

class OpenAIBackend(LLMBackend):
    """OpenAI backend using the official SDK."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: "
                "pip install openai"
            )
        
        config = get_config()
        self.api_key = api_key or config.api_key
        
        # fallback to env
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name or DEFAULT_MODELS["openai"]
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Reasoning / GPT-5 models require `max_completion_tokens` and reject
        # any non-default temperature (only 1 is supported).
        self._is_reasoning_model = (
            self.model_name.startswith("o1")
            or self.model_name.startswith("o3")
            or self.model_name.startswith("o4")
            or self.model_name.startswith("gpt-5")
        )
        self.use_max_completion_tokens = self._is_reasoning_model

    @property
    def provider(self) -> str:
        return "openai"
    
    def generate(self, prompt: str) -> str:
        try:
            from pha.streaming import emit_step
            emit_step("LLM", f"Calling {self.model_name}...", detail=f"prompt={len(prompt)} chars")
            # Reasoning models (o1/o3/o4/gpt-5) get the temperature kwarg
            # omitted entirely — show that in the log to avoid the misleading
            # impression that we're sending a value the model rejects.
            shown_temp = "default" if self._is_reasoning_model else self.temperature
            print(f"[LLM:OpenAI] Calling model={self.model_name}, prompt_len={len(prompt)}, temp={shown_temp}")
            # Basic message structure
            messages = [{"role": "user", "content": prompt}]
            
            kwargs = {
                "model": self.model_name,
                "messages": messages,
            }

            # Reasoning models reject non-default temperature; omit the kwarg
            # so the API uses its default (1).
            if not self._is_reasoning_model:
                kwargs["temperature"] = self.temperature

            # Handle token limit parameter
            if self.use_max_completion_tokens:
                kwargs["max_completion_tokens"] = self.max_tokens
            else:
                kwargs["max_tokens"] = self.max_tokens

            response = self.client.chat.completions.create(**kwargs)
            result = response.choices[0].message.content
            print(f"[LLM:OpenAI] Response received: {len(result)} chars")
            return result
        except Exception as e:
            print(f"[LLM:OpenAI] *** ERROR: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")


# =============================================================================
# Anthropic Backend
# =============================================================================

class AnthropicBackend(LLMBackend):
    """Anthropic Claude backend using the official SDK."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: "
                "pip install anthropic"
            )
        
        config = get_config()
        self.api_key = api_key or config.api_key
        
        # fallback to env
        if not self.api_key:
            self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name or DEFAULT_MODELS["anthropic"]
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @property
    def provider(self) -> str:
        return "anthropic"
    
    def generate(self, prompt: str) -> str:
        try:
            from pha.streaming import emit_step
            emit_step("LLM", f"Calling {self.model_name}...", detail=f"prompt={len(prompt)} chars")
            print(f"[LLM:Anthropic] Calling model={self.model_name}, prompt_len={len(prompt)}, temp={self.temperature}")
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            result = response.content[0].text
            print(f"[LLM:Anthropic] Response received: {len(result)} chars")
            return result
        except Exception as e:
            print(f"[LLM:Anthropic] *** ERROR: {e}")
            raise RuntimeError(f"Anthropic API error: {e}")


# =============================================================================
# Factory Function
# =============================================================================

def get_llm_backend(
    backend_type: Optional[ProviderType] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> LLMBackend:
    """Factory function to create an LLM backend.
    
    If arguments are provided, they override the global configuration.
    If backend_type is None, usage the globally configured provider.
    """
    config = get_config()
    
    # Use config if not overridden
    backend_type = backend_type or config.provider
    backend_type = backend_type.lower()
    
    # Model name resolution: Argument > Config if provider matches > Default
    curr_model_name = model_name
    if not curr_model_name and backend_type == config.provider:
        curr_model_name = config.model_name
    
    # API Key resolution is handled by the backend classes mostly, 
    # but we can pass it if explicitly provided here.
    
    if backend_type == "gemini":
        return GeminiBackend(
            api_key=api_key,
            model_name=curr_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_type == "openai":
        return OpenAIBackend(
            api_key=api_key,
            model_name=curr_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_type == "anthropic":
        return AnthropicBackend(
            api_key=api_key,
            model_name=curr_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")

