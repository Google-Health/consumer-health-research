"""Shared pytest fixtures for PHA tests.

This module provides:
- Sample data fixtures for testing
- API key management for integration tests (with interactive prompts)
- Custom pytest markers for test categorization

API Keys Required for Full Testing:
- GEMINI_API_KEY: Required for LLM-based agent tests
- TAVILY_API_KEY: Required for web search integration tests
- OPENAI_API_KEY: Optional, for OpenAI backend tests
- ANTHROPIC_API_KEY: Optional, for Anthropic backend tests

API keys can be provided via:
1. Environment variables (checked first)
2. Interactive terminal prompt (when running with pytest -s)
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path


# =============================================================================
# API Key Cache (session-scoped)
# =============================================================================

_API_KEY_CACHE = {}
_PROMPTED_KEYS = set()  # Track which keys we've already prompted for


def _is_interactive():
    """Check if we can prompt the user interactively."""
    # Check if stdin is a TTY (interactive terminal)
    return sys.stdin.isatty()


def _prompt_for_key(key_name: str, description: str) -> str:
    """Prompt user for an API key if not in environment.
    
    Args:
        key_name: Name of the environment variable (e.g., "GEMINI_API_KEY")
        description: Human-readable description of what the key is for
        
    Returns:
        The API key string, or None if not provided
    """
    # Check environment first
    env_value = os.environ.get(key_name)
    if env_value:
        return env_value
    
    # Check cache (already prompted this session)
    if key_name in _API_KEY_CACHE:
        return _API_KEY_CACHE[key_name]
    
    # Only prompt once per key per session
    if key_name in _PROMPTED_KEYS:
        return None
    
    _PROMPTED_KEYS.add(key_name)
    
    # Try to prompt interactively
    if _is_interactive():
        print(f"\n{'='*60}")
        print(f"API Key Required: {key_name}")
        print(f"{'='*60}")
        print(f"{description}")
        print("")
        print("Enter the key below, or press Enter to skip these tests.")
        print("(Tip: Set as environment variable to avoid this prompt)\n")
        
        try:
            value = input(f"{key_name}: ").strip()
            if value:
                _API_KEY_CACHE[key_name] = value
                os.environ[key_name] = value  # Set for subprocess use
                print(f"✓ {key_name} set for this session\n")
                return value
            else:
                print(f"✗ Skipping tests that require {key_name}\n")
                return None
        except (EOFError, KeyboardInterrupt):
            print(f"\n✗ Skipping tests that require {key_name}\n")
            return None
    
    return None


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (may require API keys)"
    )
    config.addinivalue_line(
        "markers", "requires_api_key: mark test as requiring specific API key(s)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "agents: mark test as agent-specific test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )


def pytest_collection_modifyitems(config, items):
    """Handle API key requirements for tests.
    
    Note: We no longer skip tests at collection time. Instead, the fixtures
    will prompt for keys interactively when needed.
    """
    pass  # Let fixtures handle API key prompting


def pytest_report_header(config):
    """Report API key status at test start."""
    lines = ["", "PHA Test Configuration:", "-" * 60]
    
    api_keys = {
        "GEMINI_API_KEY": "Gemini (required for agent tests)",
        "TAVILY_API_KEY": "Tavily (required for search tests)",
        "OPENAI_API_KEY": "OpenAI (optional)",
        "ANTHROPIC_API_KEY": "Anthropic (optional)",
    }
    
    available = []
    missing = []
    
    for key, description in api_keys.items():
        if os.environ.get(key):
            available.append(f"  ✓ {key}: {description}")
        else:
            missing.append(f"  ✗ {key}: {description}")
    
    if available:
        lines.append("Available API Keys:")
        lines.extend(available)
    
    if missing:
        lines.append("Missing API Keys:")
        lines.extend(missing)
        if _is_interactive():
            lines.append("")
            lines.append("  💡 You will be prompted for missing keys when needed")
        else:
            lines.append("")
            lines.append("  💡 Run with 'pytest -s' to enable interactive prompts")
            lines.append("     or set environment variables before running tests")
    
    lines.append("-" * 60)
    return lines


# =============================================================================
# API Key Fixtures (with interactive prompting)
# =============================================================================

@pytest.fixture
def gemini_api_key():
    """Get Gemini API key from environment or prompt.
    
    Required for LLM-based agent tests.
    Get a key at: https://makersuite.google.com/app/apikey
    """
    key = _prompt_for_key(
        "GEMINI_API_KEY",
        "Required for Gemini LLM-based agent tests.\n"
        "Get one at: https://makersuite.google.com/app/apikey"
    )
    if not key:
        pytest.skip("GEMINI_API_KEY not provided")
    return key


@pytest.fixture
def tavily_api_key():
    """Get Tavily API key from environment or prompt.
    
    Required for web search integration tests.
    Get a key at: https://tavily.com
    """
    key = _prompt_for_key(
        "TAVILY_API_KEY",
        "Required for web search integration tests.\n"
        "Get one at: https://tavily.com"
    )
    if not key:
        pytest.skip("TAVILY_API_KEY not provided")
    return key


@pytest.fixture
def openai_api_key():
    """Get OpenAI API key from environment or prompt.
    
    Optional, for OpenAI backend tests.
    Get a key at: https://platform.openai.com
    """
    key = _prompt_for_key(
        "OPENAI_API_KEY",
        "Optional, for OpenAI backend tests.\n"
        "Get one at: https://platform.openai.com"
    )
    if not key:
        pytest.skip("OPENAI_API_KEY not provided")
    return key


@pytest.fixture
def anthropic_api_key():
    """Get Anthropic API key from environment or prompt.
    
    Optional, for Anthropic backend tests.
    Get a key at: https://console.anthropic.com
    """
    key = _prompt_for_key(
        "ANTHROPIC_API_KEY",
        "Optional, for Anthropic backend tests.\n"
        "Get one at: https://console.anthropic.com"
    )
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not provided")
    return key


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_data_dir():
    """Path to sample data directory."""
    return Path(__file__).parent.parent / "data" / "sample"


@pytest.fixture
def sample_summary_df(sample_data_dir):
    """Load sample summary DataFrame."""
    return pd.read_csv(sample_data_dir / "summary.csv")


@pytest.fixture
def sample_activities_df(sample_data_dir):
    """Load sample activities DataFrame."""
    return pd.read_csv(sample_data_dir / "activities.csv")


@pytest.fixture
def sample_profile_df(sample_data_dir):
    """Load sample profile DataFrame."""
    return pd.read_csv(sample_data_dir / "profile.csv")


@pytest.fixture
def sample_population_df(sample_data_dir):
    """Load sample population DataFrame."""
    return pd.read_csv(sample_data_dir / "population_percentiles.csv")


@pytest.fixture
def all_sample_dataframes(
    sample_summary_df,
    sample_activities_df, 
    sample_profile_df,
    sample_population_df,
):
    """All sample DataFrames as a dictionary.
    
    Keys match what DataScienceAgent.load_dataframes() expects:
    'summary', 'activities', 'profile', 'population'
    """
    return {
        "summary": sample_summary_df,
        "activities": sample_activities_df,
        "profile": sample_profile_df,
        "population": sample_population_df,
    }


@pytest.fixture
def sample_health_query():
    """Sample health-related query for testing."""
    return "How has my sleep been over the past week?"


@pytest.fixture
def sample_data_query():
    """Sample data analysis query for testing."""
    return "What are my average daily steps?"


@pytest.fixture
def sample_interpretation_query():
    """Sample interpretation query for domain expert."""
    return "Is my resting heart rate healthy for my age?"
