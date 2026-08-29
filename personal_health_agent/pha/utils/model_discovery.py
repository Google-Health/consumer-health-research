"""Model discovery utilities for fetching available models from LLM providers.

This module provides functions to dynamically fetch available models from
Gemini, OpenAI, and Anthropic APIs.

Usage:
    from pha.utils.model_discovery import get_available_models, list_all_models
    
    # Get models for a specific provider
    gemini_models = get_available_models("gemini", api_key="your-key")
    
    # List all available models across providers
    all_models = list_all_models(gemini_key="...", openai_key="...")
"""

import os
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass


@dataclass
class ModelInfo:
    """Information about an available model.

    The `available` flag is set by the dynamic discovery functions when an
    API key is provided: True if the model appears in the provider's live
    models.list(), False otherwise. Defaults to True so unkeyed contexts
    don't grey anything out.
    """
    id: str
    name: str
    provider: str
    description: Optional[str] = None
    context_window: Optional[int] = None
    available: bool = True

    def __str__(self) -> str:
        return f"{self.provider}/{self.id}"


# =============================================================================
# Known Models (fallback when API is unavailable)
# =============================================================================

# Known models when dynamic discovery isn't available
# Note: Original PHIA paper only tested with Gemini models.
# PHIA baseline works best with Gemini due to OneTwo ReAct parser compatibility.
# OpenAI support works well with PHA/Parallel baselines.
# Last updated: January 2026

KNOWN_GEMINI_MODELS = [
    # Gemini 3.1 series — current default (May 2026)
    ModelInfo(
        id="models/gemini-3.1-flash-lite-preview",
        name="Gemini 3.1 Flash Lite (Preview)",
        provider="gemini",
        description="Default Gemini model",
    ),
    # Gemini 2.5 series — shutdown scheduled October 16, 2026
    ModelInfo(
        id="models/gemini-2.5-pro",
        name="Gemini 2.5 Pro",
        provider="gemini",
        description="Shutdown October 16, 2026",
    ),
    ModelInfo(
        id="models/gemini-2.5-flash",
        name="Gemini 2.5 Flash",
        provider="gemini",
        description="Shutdown October 16, 2026",
    ),
]

# OpenAI models — curated chat-completion set
# Note: "Pro" variants require Responses API (not Chat Completions) and are excluded.
KNOWN_OPENAI_MODELS = [
    ModelInfo(
        id="gpt-5-mini",
        name="GPT-5 Mini",
        provider="openai",
        description="",
        context_window=128000,
    ),
    ModelInfo(
        id="gpt-4.1",
        name="GPT-4.1",
        provider="openai",
        description="",
        context_window=128000,
    ),
    ModelInfo(
        id="gpt-4o",
        name="GPT-4o",
        provider="openai",
        description="",
        context_window=128000,
    ),
    ModelInfo(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider="openai",
        description="",
        context_window=128000,
    ),
]

# Anthropic models — Sonnet 4.5 only for now
KNOWN_ANTHROPIC_MODELS = [
    ModelInfo(
        id="claude-sonnet-4-5-20250929",
        name="Claude Sonnet 4.5",
        provider="anthropic",
        description="",
        context_window=200000,
    ),
]


# =============================================================================
# Dynamic Model Discovery
# =============================================================================

def _annotate_availability(
    curated: List[ModelInfo],
    available_ids: set,
) -> List[ModelInfo]:
    """Return copies of curated entries with `available` set per the live IDs.

    If a curated entry is already marked `available=False`, we keep it that way
    (e.g., models known to be closed to new customers even though they still
    appear in client.models.list()).
    """
    return [
        ModelInfo(
            id=m.id,
            name=m.name,
            provider=m.provider,
            description=m.description,
            context_window=m.context_window,
            available=(m.available and m.id in available_ids),
        )
        for m in curated
    ]


def get_gemini_models(api_key: Optional[str] = None) -> List[ModelInfo]:
    """Return the curated Gemini list, annotated with live availability.

    Without a key, every entry is marked available (we can't tell). With a
    key, we query `client.models.list()` and mark entries available iff
    their ID appears in the listing.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return KNOWN_GEMINI_MODELS

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        live_ids = {m.name for m in client.models.list()}
    except Exception as e:
        print(f"Warning: Could not fetch Gemini models for availability check: {e}")
        return KNOWN_GEMINI_MODELS

    return _annotate_availability(KNOWN_GEMINI_MODELS, live_ids)


def get_openai_models(api_key: Optional[str] = None) -> List[ModelInfo]:
    """Return the curated OpenAI list, annotated with live availability."""
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return KNOWN_OPENAI_MODELS

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        live_ids = {m.id for m in client.models.list()}
    except Exception as e:
        print(f"Warning: Could not fetch OpenAI models for availability check: {e}")
        return KNOWN_OPENAI_MODELS

    return _annotate_availability(KNOWN_OPENAI_MODELS, live_ids)


def get_anthropic_models(api_key: Optional[str] = None) -> List[ModelInfo]:
    """Return the curated Anthropic list, annotated with live availability."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return KNOWN_ANTHROPIC_MODELS

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        live_ids = {m.id for m in client.models.list()}
    except Exception as e:
        print(f"Warning: Could not fetch Anthropic models for availability check: {e}")
        return KNOWN_ANTHROPIC_MODELS

    return _annotate_availability(KNOWN_ANTHROPIC_MODELS, live_ids)


def get_available_models(
    provider: Literal["gemini", "openai", "anthropic"],
    api_key: Optional[str] = None,
) -> List[ModelInfo]:
    """Get available models for a specific provider.
    
    Args:
        provider: The LLM provider ("gemini", "openai", or "anthropic").
        api_key: API key for the provider.
        
    Returns:
        List of available ModelInfo objects.
    """
    if provider == "gemini":
        return get_gemini_models(api_key)
    elif provider == "openai":
        return get_openai_models(api_key)
    elif provider == "anthropic":
        return get_anthropic_models(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def list_all_models(
    gemini_key: Optional[str] = None,
    openai_key: Optional[str] = None,
    anthropic_key: Optional[str] = None,
) -> Dict[str, List[ModelInfo]]:
    """List all available models across all providers.
    
    Args:
        gemini_key: Gemini API key.
        openai_key: OpenAI API key.
        anthropic_key: Anthropic API key.
        
    Returns:
        Dictionary mapping provider names to lists of ModelInfo objects.
    """
    return {
        "gemini": get_gemini_models(gemini_key),
        "openai": get_openai_models(openai_key),
        "anthropic": get_anthropic_models(anthropic_key),
    }


def get_default_model(provider: Literal["gemini", "openai", "anthropic"]) -> str:
    """Get the default model ID for a provider.
    
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
    return defaults.get(provider, defaults["gemini"])


def get_recommended_models() -> Dict[str, ModelInfo]:
    """Get recommended models for each provider.
    
    These are the models recommended for PHA based on testing.
    
    Returns:
        Dictionary mapping provider names to recommended ModelInfo.
    """
    return {
        "gemini": ModelInfo(
            id="models/gemini-3.1-flash-lite-preview",
            name="Gemini 3.1 Flash Lite (Preview)",
            provider="gemini",
            description="Recommended for PHA — current default (May 2026)",
        ),
        "openai": ModelInfo(
            id="gpt-4o",
            name="GPT-4o",
            provider="openai",
            description="Recommended for PHA",
        ),
        "anthropic": ModelInfo(
            id="claude-sonnet-4-5-20250929",
            name="Claude Sonnet 4.5",
            provider="anthropic",
            description="Recommended for PHA",
        ),
    }
