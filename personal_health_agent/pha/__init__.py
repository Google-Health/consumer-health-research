"""PHA - Personal Health Agent.

A multi-agent system for analyzing personal health data.
"""

__version__ = "0.1.0"

from . import utils
from . import prompts
from . import tools
from . import llm
from . import agents

# Convenience imports
from .agents import DataScienceAgent, DomainExpertAgent
from .utils import load_persona

__all__ = [
    "utils",
    "prompts", 
    "tools",
    "llm",
    "agents",
    "DataScienceAgent",
    "DomainExpertAgent",
    "load_persona",
    "__version__",
]
