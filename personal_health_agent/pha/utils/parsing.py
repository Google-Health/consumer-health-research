"""Utilities for parsing LLM outputs."""

# The V2 codegen prompt ends with this function signature, which the LLM
# continues from. We need to prepend it to the extracted code.
ANALYSIS_FUNCTION_SIGNATURE = """\
from typing import Any, Dict
import pandas as pd
import numpy as np
import datetime

def analysis(summary_df: pd.DataFrame | None = None,
    activities_df: pd.DataFrame | None = None,
    profile_df: pd.DataFrame | None = None,
    population_df: pd.DataFrame | None = None,) -> Dict[str, Any]:
"""


def _normalize_body_indentation(code: str, target_base_indent: int = 4) -> str:
    """Normalize indentation of a code block to use a consistent base indent.
    
    Args:
        code: The code to normalize
        target_base_indent: The base indentation level (default 4 for function body)
        
    Returns:
        Code with normalized indentation
    """
    lines = code.split('\n')
    
    # Find the minimum indentation in non-empty lines
    min_indent = float('inf')
    for line in lines:
        if line.strip():  # Non-empty line
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    if min_indent == float('inf') or min_indent == 0:
        min_indent = 0
    
    # Calculate the indent adjustment needed
    indent_adjustment = target_base_indent - min_indent
    
    # Adjust each line's indentation
    normalized_lines = []
    for line in lines:
        if line.strip():  # Non-empty line
            current_indent = len(line) - len(line.lstrip())
            new_indent = max(0, current_indent + indent_adjustment)
            normalized_lines.append(' ' * new_indent + line.lstrip())
        else:
            normalized_lines.append('')
    
    return '\n'.join(normalized_lines)


def parse_code_output(code_output: str) -> str:
    """Parses the code output to extract the final result.
    
    The V2 codegen prompt ends with a function signature that the LLM continues.
    If the extracted code doesn't start with 'def analysis', we need to prepend
    the function signature.
    """
    code_output = code_output.removeprefix("```python")
    code_output = code_output.removeprefix("```")
    code_output = code_output.removesuffix("```")
    
    # Only strip trailing whitespace, preserve leading whitespace for indentation detection
    code_output = code_output.rstrip()
    # Remove leading newlines only (not spaces)
    while code_output.startswith('\n'):
        code_output = code_output[1:]
    
    # Check if the code already has the function definition
    # (in case LLM repeated it or we're using V1 prompt)
    if code_output.lstrip().startswith('def analysis') or code_output.lstrip().startswith('from typing'):
        return code_output.strip()
    
    # Check if code starts with import statements (LLM included full code)
    if code_output.lstrip().startswith('import '):
        return code_output.strip()
    
    # Otherwise, prepend the function signature
    # The LLM output is the function body with its own indentation
    # We need to normalize to 4-space base indent for function body
    normalized_body = _normalize_body_indentation(code_output, target_base_indent=4)
    
    return ANALYSIS_FUNCTION_SIGNATURE + normalized_body
