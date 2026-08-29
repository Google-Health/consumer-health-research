"""Model discovery utilities for LLM providers.

This module provides functions to fetch available models from each provider's API,
making it easy to populate dropdowns in the portal UI.

Usage:
    from pha.llm.models import get_available_models, get_model_options
    
    # Get all Gemini models
    models = get_available_models("gemini", api_key="your-key")
    
    # Get curated options for UI dropdown
    options = get_model_options("gemini", api_key="your-key")
"""

import os
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    provider: str
    description: str = ""
    context_window: Optional[int] = None
    supports_vision: bool = False
    supports_tools: bool = True


# =============================================================================
# Curated Model Lists (fallback when API unavailable)
# =============================================================================

CURATED_GEMINI_MODELS = [
    # Gemini 3.1 series — current default (May 2026)
    ModelInfo(
        id="models/gemini-3.1-flash-lite-preview",
        name="Gemini 3.1 Flash Lite (Preview)",
        provider="gemini",
        description="Default Gemini model",
        supports_vision=True,
    ),
    # Gemini 2.5 series — shutdown October 16, 2026
    ModelInfo(
        id="models/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="gemini",
        description="Shutdown October 16, 2026",
        supports_vision=True,
    ),
    ModelInfo(
        id="models/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="gemini",
        description="Shutdown October 16, 2026",
        supports_vision=True,
    ),
]

CURATED_OPENAI_MODELS = [
    ModelInfo(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        provider="openai",
        description="",
        context_window=128000,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-4.1",
        name="GPT-4.1",
        provider="openai",
        description="",
        context_window=128000,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        description="",
        context_window=128000,
        supports_vision=True,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        description="",
        context_window=128000,
        supports_vision=True,
    ),
]

CURATED_ANTHROPIC_MODELS = [
    ModelInfo(
        id="claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        description="",
        context_window=200000,
        supports_vision=True,
    ),
]


# =============================================================================
# Dynamic Model Fetching
# =============================================================================

def fetch_gemini_models(api_key: Optional[str] = None) -> List[ModelInfo]:
    """Fetch available Gemini models from the API.
    
    Args:
        api_key: Gemini API key. Falls back to GEMINI_API_KEY env var.
        
    Returns:
        List of available ModelInfo objects.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return CURATED_GEMINI_MODELS
    
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        models = []
        for model in client.models.list():
            # Filter to generative models only
            if hasattr(model, 'supported_generation_methods'):
                if 'generateContent' in model.supported_generation_methods:
                    models.append(ModelInfo(
                        id=model.name,
                        name=model.display_name if hasattr(model, 'display_name') else model.name,
                        provider="gemini",
                        description=model.description if hasattr(model, 'description') else "",
                        supports_vision='vision' in model.name.lower() if hasattr(model, 'name') else False,
                    ))
        
        # Sort by name, preferring newer models
        models.sort(key=lambda m: m.id, reverse=True)
        return models if models else CURATED_GEMINI_MODELS
        
    except Exception as e:
        print(f"Warning: Could not fetch Gemini models: {e}")
        return CURATED_GEMINI_MODELS


def fetch_openai_models(api_key: Optional[str] = None) -> List[ModelInfo]:
    """Fetch available OpenAI models from the API.
    
    Args:
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
        
    Returns:
        List of available ModelInfo objects.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return CURATED_OPENAI_MODELS
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        models = []
        for model in client.models.list():
            # Filter to chat models only (gpt-4, gpt-3.5)
            if 'gpt-4' in model.id or 'gpt-3.5' in model.id:
                # Skip fine-tuned models
                if ':' not in model.id:
                    models.append(ModelInfo(
                        id=model.id,
                        name=model.id.replace('-', ' ').title(),
                        provider="openai",
                        supports_vision='vision' in model.id or 'gpt-4o' in model.id,
                    ))
        
        # Sort, preferring gpt-4o models
        models.sort(key=lambda m: (0 if 'gpt-4o' in m.id else 1, m.id), reverse=False)
        return models if models else CURATED_OPENAI_MODELS
        
    except Exception as e:
        print(f"Warning: Could not fetch OpenAI models: {e}")
        return CURATED_OPENAI_MODELS


def fetch_anthropic_models(api_key: Optional[str] = None) -> List[ModelInfo]:
    """Get available Anthropic models.
    
    Note: Anthropic doesn't have a models list API, so we return curated list.
    
    Args:
        api_key: Not used, included for API consistency.
        
    Returns:
        List of available ModelInfo objects.
    """
    # Anthropic doesn't have a list models endpoint
    # Return curated list
    return CURATED_ANTHROPIC_MODELS


# =============================================================================
# Public API
# =============================================================================

ProviderType = Literal["gemini", "openai", "anthropic"]


def get_available_models(
    provider: ProviderType,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> List[ModelInfo]:
    """Get available models for a provider.
    
    Args:
        provider: The LLM provider ("gemini", "openai", "anthropic").
        api_key: API key for the provider.
        use_cache: Whether to use cached/curated list (faster) or fetch from API.
        
    Returns:
        List of ModelInfo objects.
    """
    if use_cache:
        # Return curated lists (faster, no API call)
        if provider == "gemini":
            return CURATED_GEMINI_MODELS
        elif provider == "openai":
            return CURATED_OPENAI_MODELS
        elif provider == "anthropic":
            return CURATED_ANTHROPIC_MODELS
    
    # Fetch from API
    if provider == "gemini":
        return fetch_gemini_models(api_key)
    elif provider == "openai":
        return fetch_openai_models(api_key)
    elif provider == "anthropic":
        return fetch_anthropic_models(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_model_options(
    provider: ProviderType,
    api_key: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Get model options formatted for UI dropdowns.
    
    Args:
        provider: The LLM provider.
        api_key: API key for the provider.
        
    Returns:
        List of dicts with 'value' and 'label' keys for dropdown options.
    """
    models = get_available_models(provider, api_key, use_cache=True)
    return [
        {
            "value": model.id,
            "label": model.name,
            "description": model.description,
        }
        for model in models
    ]


def get_all_model_options(
    gemini_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """Get model options for all providers.
    
    Args:
        gemini_key: Gemini API key.
        openai_key: OpenAI API key.
        anthropic_key: Anthropic API key.
        
    Returns:
        Dict mapping provider names to their model options.
    """
    return {
        "gemini": get_model_options("gemini", gemini_key),
        "openai": get_model_options("openai", openai_key),
        "anthropic": get_model_options("anthropic", anthropic_key),
    }


def get_default_model(provider: ProviderType) -> str:
    """Get the default model for a provider.
    
    Args:
        provider: The LLM provider.
        
    Returns:
        Default model ID string.
    """
    defaults = {
        "gemini": "models/gemini-3.1-flash-lite-preview",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-5-20250929",
    }
    return defaults.get(provider, "")
