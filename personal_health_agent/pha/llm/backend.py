"""LLM backend abstraction layer.

This module provides the LLM backend interface for PHA agents,
supporting Google Gemini, OpenAI, and Anthropic models.

Usage:
    from pha.llm import setup_backend, run_llm
    
    # Set up the Gemini backend
    setup_backend(api_key="your-key")
    
    # Generate text
    response = run_llm("What is the capital of France?")

Note:
    For best results with onetwo ReAct agents, install onetwo:
    pip install git+https://github.com/google-deepmind/onetwo
"""

import os
from typing import Optional, Callable
from abc import ABC, abstractmethod


# =============================================================================
# OpenAI API Compatibility Patch
# =============================================================================
# Newer OpenAI models (o1, o3, gpt-4o, etc.) require `max_completion_tokens`
# instead of `max_tokens`. This patch ensures compatibility with libraries
# like onetwo that still use the old parameter name.

_OPENAI_PATCHED = False

def _patch_openai_for_max_tokens():
    """Patch OpenAI client to convert max_tokens to max_completion_tokens.
    
    This is needed because newer OpenAI models don't support max_tokens,
    but some libraries (like onetwo) still use it.
    """
    global _OPENAI_PATCHED
    if _OPENAI_PATCHED:
        return
    
    try:
        from openai.resources.chat import completions
        
        _original_create = completions.Completions.create
        
        def _patched_create(self, *args, **kwargs):
            # Convert max_tokens to max_completion_tokens for newer models
            if 'max_tokens' in kwargs and 'max_completion_tokens' not in kwargs:
                kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            return _original_create(self, *args, **kwargs)
        
        completions.Completions.create = _patched_create
        _OPENAI_PATCHED = True
    except Exception:
        pass  # If patching fails, continue without it


# Try to import onetwo, fall back to simple implementation if not available
# Note: onetwo requires Python 3.12+ due to use of new type parameter syntax
_ONETWO_AVAILABLE = False
_ONETWO_OPENAI_AVAILABLE = False
_ONETWO_GENAI_AVAILABLE = False  # True if new GoogleGenAIAPI backend is available
_ONETWO_IMPORT_ERROR = None
try:
    from onetwo import ot
    from onetwo.builtins import llm
    
    # Use GoogleGenAIAPI backend (new google.genai SDK, no deprecation warning,
    # safety defaults to OFF for Gemini 2.5+ models)
    try:
        from onetwo.backends import google_genai_api
        _ONETWO_GENAI_AVAILABLE = True
    except ImportError:
        pass  # Will use FallbackGeminiBackend (google.genai SDK) instead
    
    _ONETWO_AVAILABLE = True
    
    # Try to import OpenAI backend as well
    try:
        from onetwo.backends import openai_api
        _ONETWO_OPENAI_AVAILABLE = True
        # Apply patch when OpenAI backend is available
        _patch_openai_for_max_tokens()
    except ImportError:
        pass  # OpenAI backend not available, will use fallback
except ImportError as e:
    _ONETWO_IMPORT_ERROR = str(e)
except SyntaxError as e:
    # Python version too old (onetwo requires 3.12+)
    _ONETWO_IMPORT_ERROR = f"onetwo requires Python 3.12+, you have {__import__('sys').version}"
except Exception as e:
    _ONETWO_IMPORT_ERROR = str(e)


# =============================================================================
# Backend Interface
# =============================================================================

class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM.
        
        Args:
            prompt: The input prompt.
            
        Returns:
            The generated text response.
        """
        pass
    
    @abstractmethod
    def register(self) -> None:
        """Register this backend with onetwo (if applicable)."""
        pass


# =============================================================================
# OneTwo-based Backends (preferred)
# =============================================================================

if _ONETWO_AVAILABLE:
    
    class OneTwoGeminiBackend(LLMBackend):
        """Gemini backend using onetwo's GoogleGenAIAPI (new google.genai SDK)."""
        
        def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "models/gemini-3.1-flash-lite-preview",
            temperature: float = 0.0,
            max_tokens: int = 8192,
        ):
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key required. Set GEMINI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
            
            if not _ONETWO_GENAI_AVAILABLE:
                raise ImportError(
                    "onetwo's GoogleGenAIAPI backend not available. "
                    "Update onetwo: pip install -U git+https://github.com/google-deepmind/onetwo"
                )
            
            self.backend = google_genai_api.GoogleGenAIAPI(
                api_key=self.api_key,
                generate_model_name=model_name,
                # OneTwo verifies both generate AND chat model names against
                # client.models.list(); chat_model_name defaults to
                # DEFAULT_GENERATE_MODEL (`models/gemini-2.5-flash`) which is
                # not present on every key. Pin both to the user's selection.
                chat_model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                threadpool_size=1,
            )
            self._registered = False
        
        def register(self) -> None:
            """Register this backend with onetwo's global registry."""
            if not self._registered:
                self.backend.register()
                self._registered = True
        
        def generate(self, prompt: str) -> str:
            """Generate a response using Gemini via onetwo."""
            self.register()
            response = ot.run(llm.generate_text(prompt=prompt))
            return response


if _ONETWO_OPENAI_AVAILABLE:
    
    class OneTwoOpenAIBackend(LLMBackend):
        """OpenAI backend using onetwo library.
        
        This enables full OneTwo functionality (ReAct agents, caching, etc.)
        with OpenAI models like GPT-4o.
        """
        
        def __init__(
            self,
            api_key: Optional[str] = None,
            model_name: str = "gpt-4o",
            temperature: float = 0.0,
            max_tokens: int = 8192,
        ):
            """Initialize the OpenAI backend via onetwo.
            
            Args:
                api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
                model_name: Model to use (e.g., gpt-4o, gpt-4-turbo).
                temperature: Sampling temperature.
                max_tokens: Maximum tokens in response.
            """
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )
            
            # Reasoning / GPT-5 models only accept temperature=1.0 (their
            # API default). Closest analog to the 0.7 we use elsewhere.
            if model_name.startswith(("o1", "o3", "o4", "gpt-5")):
                temperature = 1.0
            self.backend = openai_api.OpenAIAPI(
                api_key=self.api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=1,
            )
            self._registered = False
        
        def register(self) -> None:
            """Register this backend with onetwo's global registry."""
            if not self._registered:
                self.backend.register()
                self._registered = True
        
        def generate(self, prompt: str) -> str:
            """Generate a response using OpenAI via onetwo."""
            self.register()
            response = ot.run(llm.generate_text(prompt=prompt))
            return response


# =============================================================================
# Fallback Backends (when onetwo is not available)
# =============================================================================

class FallbackGeminiBackend(LLMBackend):
    """Fallback Gemini backend using google-genai directly.
    
    Used when onetwo is not installed.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "models/gemini-3.1-flash-lite-preview",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the Gemini backend.
        
        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            model_name: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError(
                "google-genai package required. Install with: "
                "pip install google-genai"
            )
        
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    
    def register(self) -> None:
        """No-op for fallback backend."""
        pass
    
    def generate(self, prompt: str) -> str:
        """Generate a response using Gemini."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self.generation_config,
        )
        return response.text


class FallbackOpenAIBackend(LLMBackend):
    """Fallback OpenAI backend.
    
    Used when onetwo is not installed.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the OpenAI backend.
        
        Args:
            api_key: OpenAI API key. If not provided, reads from OPENAI_API_KEY env var.
            model_name: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package required. Install with: "
                "pip install openai"
            )
        
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Reasoning / GPT-5 models reject any non-default temperature (only 1 is
        # supported). Omit the kwarg for those so the API uses its default.
        self._is_reasoning_model = (
            self.model_name.startswith("o1")
            or self.model_name.startswith("o3")
            or self.model_name.startswith("o4")
            or self.model_name.startswith("gpt-5")
        )

    def register(self) -> None:
        """No-op for fallback backend."""
        pass

    def generate(self, prompt: str) -> str:
        """Generate a response using OpenAI."""
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.max_tokens,
        }
        if not self._is_reasoning_model:
            kwargs["temperature"] = self.temperature
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


class FallbackAnthropicBackend(LLMBackend):
    """Fallback Anthropic Claude backend.
    
    Used when onetwo is not installed.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "claude-sonnet-4-5-20250929",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ):
        """Initialize the Anthropic backend.
        
        Args:
            api_key: Anthropic API key. If not provided, reads from ANTHROPIC_API_KEY env var.
            model_name: Model to use.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in response.
        """
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: "
                "pip install anthropic"
            )
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = Anthropic(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def register(self) -> None:
        """No-op for fallback backend."""
        pass
    
    def generate(self, prompt: str) -> str:
        """Generate a response using Anthropic Claude."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


# =============================================================================
# Factory Functions
# =============================================================================

def get_llm_backend(
    backend_type: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
    prefer_onetwo: bool = True,
) -> LLMBackend:
    """Factory function to create an LLM backend.
    
    Args:
        backend_type: "gemini", "openai", or "anthropic".
        api_key: API key for the backend.
        model_name: Model name (uses sensible defaults if not provided).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        prefer_onetwo: If True, use onetwo when available (recommended for
                      Gemini and OpenAI to enable ReAct agents and caching).
    
    Returns:
        Configured LLMBackend instance.
    
    Example:
        >>> backend = get_llm_backend("gemini", api_key="your-key")
        >>> response = backend.generate("Hello!")
    """
    backend_type = backend_type.lower()
    
    # Default model names
    defaults = {
        "gemini": "models/gemini-3.1-flash-lite-preview",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-5-20250929",
    }
    
    if backend_type not in defaults:
        raise ValueError(f"Unknown backend type: {backend_type}. Choose from: {list(defaults.keys())}")
    
    model_name = model_name or defaults[backend_type]
    
    # Use onetwo if available and preferred for Gemini (requires GoogleGenAIAPI)
    if prefer_onetwo and _ONETWO_AVAILABLE and _ONETWO_GENAI_AVAILABLE and backend_type == "gemini":
        return OneTwoGeminiBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    # Use onetwo if available and preferred for OpenAI
    if prefer_onetwo and _ONETWO_OPENAI_AVAILABLE and backend_type == "openai":
        return OneTwoOpenAIBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    
    # Otherwise use fallback backends
    if backend_type == "gemini":
        return FallbackGeminiBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_type == "openai":
        return FallbackOpenAIBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif backend_type == "anthropic":
        return FallbackAnthropicBackend(
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

# Global backend instance
_global_backend: Optional[LLMBackend] = None


def setup_backend(
    backend_type: str = "gemini",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> LLMBackend:
    """Set up and register the global LLM backend.
    
    This matches the onetwo pattern of registering a backend globally.
    
    Args:
        backend_type: "gemini", "openai", or "anthropic".
        api_key: API key for the backend.
        model_name: Model name.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
    
    Returns:
        The configured backend.
    
    Example:
        >>> setup_backend("gemini", api_key="your-key")
        >>> response = run_llm("Hello!")
    """
    global _global_backend
    _global_backend = get_llm_backend(
        backend_type=backend_type,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    _global_backend.register()
    return _global_backend


def run_llm(
    prompt: str,
    parse_function: Optional[Callable[[str], str]] = None,
) -> str:
    """Run the LLM with the global backend.
    
    Args:
        prompt: The input prompt.
        parse_function: Optional function to parse the response.
    
    Returns:
        The (optionally parsed) response.
    
    Example:
        >>> setup_backend("gemini")
        >>> response = run_llm("What is 2+2?")
    """
    if _global_backend is None:
        raise RuntimeError("No backend set up. Call setup_backend() first.")
    
    response = _global_backend.generate(prompt)
    
    if parse_function is not None:
        return parse_function(response)
    return response


def is_onetwo_available() -> bool:
    """Check if onetwo is available."""
    return _ONETWO_AVAILABLE


def is_onetwo_openai_available() -> bool:
    """Check if onetwo OpenAI backend is available."""
    return _ONETWO_OPENAI_AVAILABLE


def get_onetwo_import_error() -> str:
    """Get the error message if onetwo import failed."""
    return _ONETWO_IMPORT_ERROR or "No error"
