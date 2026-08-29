"""Global configuration for PHA LLM backends.

This module provides a singleton configuration object to manage
the active LLM provider, API keys, and model settings globally.
"""

import os
from typing import Optional, Dict, Any, Literal

# Provider types
ProviderType = Literal["gemini", "openai", "anthropic"]

# Default models
DEFAULT_MODELS = {
    "gemini": "models/gemini-3.1-flash-lite-preview",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
}


class LLMConfig:
    """Singleton configuration for LLM settings."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._provider: ProviderType = "anthropic"
        self._api_key: Optional[str] = None
        self._model_name: Optional[str] = None
        self._temperature: float = 0.0
        self._initialized = True
        
        # Load defaults from environment variables
        self._load_from_env()

    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Check for provider override
        env_provider = os.environ.get("PHA_LLM_PROVIDER")
        if env_provider and env_provider.lower() in ["gemini", "openai", "anthropic"]:
            self._provider = env_provider.lower()
            
        # Try to find API key for the current provider
        self._api_key = self._get_env_api_key(self._provider)

    def _get_env_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment for a specific provider."""
        env_vars = {
            "gemini": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        return os.environ.get(env_vars.get(provider, ""))

    def configure(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
    ):
        """Update global configuration.
        
        Args:
            provider: "gemini", "openai", or "anthropic".
            api_key: API key. If None, tries to load from env ONLY if provider differs.
            model_name: Model name. If None, uses default for provider.
            temperature: Sampling temperature.
        """
        if provider:
            provider = provider.lower()
            if provider not in ["gemini", "openai", "anthropic"]:
                raise ValueError(f"Unknown provider: {provider}")
            
            # If switching providers, we should re-fetch API key from env unless provided
            if provider != self._provider and not api_key:
                api_key = self._get_env_api_key(provider)
            
            self._provider = provider
            
        if api_key:
            self._api_key = api_key
            
        if model_name:
            self._model_name = model_name
            
        if temperature is not None:
             self._temperature = temperature

    @property
    def provider(self) -> str:
        return self._provider
    
    @property
    def api_key(self) -> Optional[str]:
        return self._api_key
    
    @property
    def model_name(self) -> str:
        return self._model_name or DEFAULT_MODELS.get(self._provider, "gpt-4o")
    
    @property
    def temperature(self) -> float:
        return self._temperature


# Global instance
config = LLMConfig()


def get_config() -> LLMConfig:
    """Get the global configuration instance."""
    return config


def configure_global(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
):
    """Convenience function to update global configuration."""
    config.configure(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature
    )
