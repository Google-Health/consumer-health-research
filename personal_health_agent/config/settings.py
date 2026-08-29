"""Centralized configuration settings for PHA.

This module provides a single place to configure all paths, API keys,
and model settings used throughout the PHA system.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


# =============================================================================
# Default paths - can be overridden via Settings class or environment variables
# =============================================================================

# Get the root directory of the project (assumes config/ is one level down from root)
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Default data directory
DEFAULT_DATA_DIR = _PROJECT_ROOT / "data" / "sample"

# Default paths for sample data files
DEFAULT_SUMMARY_PATH = DEFAULT_DATA_DIR / "summary.csv"
DEFAULT_ACTIVITIES_PATH = DEFAULT_DATA_DIR / "activities.csv"
DEFAULT_PROFILE_PATH = DEFAULT_DATA_DIR / "profile.csv"
DEFAULT_POPULATION_PATH = DEFAULT_DATA_DIR / "population_percentiles.csv"


# =============================================================================
# LLM Configuration
# =============================================================================

# Supported LLM backends
LLMBackend = Literal["gemini", "openai", "anthropic"]

# Default model names for each backend
DEFAULT_MODELS = {
    "gemini": "models/gemini-3.1-flash-lite-preview",
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5-20250929",
}


# =============================================================================
# Agent Configuration Constants
# =============================================================================

# Code execution retry settings
DEFAULT_NUM_CODE_RETRIES = 3
DEFAULT_TOTAL_CODE_RETRIES = 5


@dataclass
class Settings:
    """Configuration settings for PHA.
    
    This class centralizes all configuration for the PHA system.
    
    Example usage:
        # Use defaults
        settings = Settings()
        
        # Custom data directory
        settings = Settings(data_dir="./my_data")
        
        # Use OpenAI instead of Gemini
        settings = Settings(llm_backend="openai")
        
        # Load API key from environment
        settings = Settings()  # Reads GEMINI_API_KEY, OPENAI_API_KEY, etc.
    """
    
    # Data paths
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    summary_path: Optional[Path] = None
    activities_path: Optional[Path] = None
    profile_path: Optional[Path] = None
    population_path: Optional[Path] = None
    
    # LLM settings
    llm_backend: LLMBackend = "gemini"
    model_name: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 8192
    
    # API keys (loaded from environment if not provided)
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Search API key for domain expert agent
    tavily_api_key: Optional[str] = None
    
    # Agent settings
    num_code_retries: int = DEFAULT_NUM_CODE_RETRIES
    total_code_retries: int = DEFAULT_TOTAL_CODE_RETRIES
    
    # Whether to temporally localize data (shift dates to appear as "today")
    temporally_localize: bool = True
    
    # Debug settings
    debug_verbose: bool = False
    
    def __post_init__(self):
        """Initialize paths and load API keys from environment."""
        # Convert string paths to Path objects and resolve relative paths
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        
        # Resolve the data_dir to an absolute path
        # This handles cases where the path is relative to the current working directory
        if not self.data_dir.is_absolute():
            self.data_dir = self.data_dir.resolve()
        
        # Set default file paths based on data_dir if not explicitly provided
        if self.summary_path is None:
            self.summary_path = self.data_dir / "summary.csv"
        if self.activities_path is None:
            self.activities_path = self.data_dir / "activities.csv"
        if self.profile_path is None:
            self.profile_path = self.data_dir / "profile.csv"
        if self.population_path is None:
            self.population_path = self.data_dir / "population_percentiles.csv"
        
        # Convert to Path objects if strings and resolve relative paths
        for attr in ['summary_path', 'activities_path', 'profile_path', 'population_path']:
            val = getattr(self, attr)
            if isinstance(val, str):
                val = Path(val)
            if val is not None and not val.is_absolute():
                val = val.resolve()
            setattr(self, attr, val)
        
        # Set default model name based on backend if not provided
        if self.model_name is None:
            self.model_name = DEFAULT_MODELS.get(self.llm_backend)
        
        # Load API keys from environment if not provided
        if self.gemini_api_key is None:
            self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.openai_api_key is None:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if self.anthropic_api_key is None:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.tavily_api_key is None:
            self.tavily_api_key = os.environ.get("TAVILY_API_KEY")
    
    def get_api_key(self) -> Optional[str]:
        """Get the API key for the current LLM backend."""
        if self.llm_backend == "gemini":
            return self.gemini_api_key
        elif self.llm_backend == "openai":
            return self.openai_api_key
        elif self.llm_backend == "anthropic":
            return self.anthropic_api_key
        return None
    
    def validate(self) -> None:
        """Validate that required settings are present.
        
        Raises:
            ValueError: If required settings are missing.
        """
        # Check API key for selected backend
        api_key = self.get_api_key()
        if api_key is None:
            raise ValueError(
                f"No API key found for {self.llm_backend}. "
                f"Set {self.llm_backend.upper()}_API_KEY environment variable "
                f"or pass {self.llm_backend}_api_key to Settings."
            )
        
        # Check that data files exist (warn but don't error - they might be created later)
        for path_attr in ['summary_path', 'activities_path', 'profile_path', 'population_path']:
            path = getattr(self, path_attr)
            if path and not path.exists():
                print(f"Warning: {path_attr} not found at {path}")


# Create a default settings instance for convenience
default_settings = Settings()
