"""Python code execution sandbox for the Data Science Agent.

This module provides a safe(r) way to execute LLM-generated Python code
for data analysis.

WARNING: This is a simplified sandbox. For production use, consider:
- Using RestrictedPython for additional safety
- Running in a Docker container
- Using a proper sandbox like PyPy sandbox mode
"""

import dataclasses
import io
import re
import sys
import traceback
import textwrap
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

import pandas as pd

# Import shared request stats
try:
    from ..utils.request_stats import request_stats
    _STATS_AVAILABLE = True
except ImportError:
    _STATS_AVAILABLE = False


# =============================================================================
# Tool Call Logging
# =============================================================================

def _log_tool_call(tool_name: str, inputs: dict, output: Any = None, error: str = None):
    """Log a tool call to the terminal for debugging.
    
    Args:
        tool_name: Name of the tool being called.
        inputs: Dictionary of input arguments.
        output: The tool output (will be truncated for display).
        error: Error message if the tool failed.
    """
    # Record stats
    if _STATS_AVAILABLE:
        request_stats.record_tool(tool_name, had_error=error is not None)
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Format inputs
    input_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in inputs.items())
    
    print(f"\n{'─' * 60}")
    print(f"🔧 [{timestamp}] TOOL CALL: {tool_name}")
    print(f"   Input: {input_str[:200]}{'...' if len(input_str) > 200 else ''}")
    
    if error:
        print(f"   ❌ Error: {error[:200]}")
    elif output is not None:
        # Truncate output for display
        output_str = str(output)
        if len(output_str) > 300:
            output_str = output_str[:300] + "..."
        print(f"   ✓ Output: {output_str}")
    
    print(f"{'─' * 60}\n")


# =============================================================================
# Constants
# =============================================================================

ERROR_STRING = "[ERROR]"


class ExecutionStatus(Enum):
    """Status of code execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


# =============================================================================
# Data classes
# =============================================================================

@dataclasses.dataclass
class ExecutionResult:
    """Result of executing Python code."""
    output: str
    stderr: str
    status: ExecutionStatus
    
    
@dataclasses.dataclass
class PhdaPythonRequest:
    """Request class for the PHDA Python engine."""
    request: str  # The code to execute
    include_traceback_for_errors: bool = True


@dataclasses.dataclass
class PhdaPythonReply:
    """Reply class for the PHDA Python engine."""
    execution_result: ExecutionResult
    setup_code: str
    code_ran: str
    orig_code: str

    def has_error(self) -> bool:
        return bool(self.execution_result.stderr) or (
            self.execution_result.status == ExecutionStatus.ERROR
        )


# =============================================================================
# Code manipulation utilities
# =============================================================================

def _normalize_indentation(code: str) -> str:
    """Normalize code indentation and line endings.
    
    Fixes common LLM code generation issues:
    1. Mixed tabs and spaces -> convert tabs to 4 spaces
    2. Windows line endings (CRLF) -> Unix line endings (LF)
    3. Trailing whitespace on lines
    """
    # Normalize line endings first (Windows CRLF -> Unix LF)
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    # Convert tabs to spaces
    code = code.replace('\t', '    ')
    # Remove trailing whitespace from each line (but preserve blank lines)
    lines = code.split('\n')
    lines = [line.rstrip() for line in lines]
    return '\n'.join(lines)


def _add_call_analysis_line(code: str) -> str:
    """Adds a call to the analysis() function at the end of the code.
    
    The V2 codegen prompt tells the LLM to wrap code in def analysis(...).
    This function adds the call to that function to get the result.
    
    Also normalizes indentation to fix mixed tabs/spaces issues.
    """
    # First normalize tabs to spaces
    code = _normalize_indentation(code)
    
    lines = code.rstrip().split('\n')
    
    # If last line already assigns to result at top level, we're good
    last_line = lines[-1] if lines else ''
    if last_line.lstrip().startswith('result = ') and not last_line.startswith((' ', '\t')):
        return '\n'.join(lines)
    
    call_line = ('result = analysis(summary_df=summary_df, activities_df=activities_df,'
                 ' profile_df=profile_df, population_df=population_df)')
    
    # Add blank line for readability, then the call at top level
    lines.append('')
    lines.append(call_line)
    return '\n'.join(lines)


def _remove_final_print(code: str) -> str:
    """Removes the final print statement from the code."""
    lines = code.rstrip().split('\n')
    if lines[-1].startswith('print('):
        re_match = re.fullmatch(r'print\((.*)\)', lines[-1])
        if re_match:
            args = re_match.group(1)
            lines[-1] = args.split(',')[0]
    return '\n'.join(lines)


def _wrap_code_for_execution(
    code: str,
    include_traceback_for_errors: bool = True,
) -> str:
    """Wraps code in try-except with stdout redirection."""
    # Indent code by 8 spaces (inside try + inside with)
    content = textwrap.indent(code, '        ')
    wrapped_code = f'''\
import contextlib, io, traceback, sys
_stdout_capture = io.StringIO()
_stderr_capture = io.StringIO()
_result = None
try:
    with contextlib.redirect_stdout(_stdout_capture):
{content}
        _result = result
except Exception as e:
    if {include_traceback_for_errors}:
        error_str = traceback.format_exc()
    else:
        error_str = "".join(traceback.format_exception_only(type(e), e))
    _stderr_capture.write("{ERROR_STRING}: " + error_str.rstrip() + "\\n")
'''
    return wrapped_code


# =============================================================================
# Preamble generation
# =============================================================================

def df_to_csv_string(df: pd.DataFrame, **kwargs) -> str:
    """Converts a DataFrame to a CSV string."""
    with io.StringIO() as f:
        df.to_csv(f, **kwargs)
        f.seek(0)
        return f.read()


def make_preamble(
    dict_of_df: Dict[str, pd.DataFrame],
    preamble_template: str,
) -> str:
    """Takes a dict of dataframes and makes a string to load them into python."""
    df_definitions = []
    for key, df in dict_of_df.items():
        df_csv_str = df_to_csv_string(df, index=False)
        df_parse = f'{key} = df_from_csv_string({repr(df_csv_str)})'
        df_definitions.append(df_parse)

    preamble = preamble_template.replace(
        '{{df_definitions}}', '\n'.join(df_definitions)
    )
    return preamble


# =============================================================================
# Preamble templates (from python_preambles.py, with PhdaDataFrame inlined)
# =============================================================================

# This includes the PhdaDataFrame class and enforce_persona_schema inline
PREAMBLE_TEMPLATE: str = '''\
import io
import re
import datetime
import numpy as np
import pandas as pd
from typing import Any, Dict
from pandas.api import types

import warnings
warnings.filterwarnings('ignore')

# Customize DataFrame string representation for cleaner output
pd.DataFrame.__original_str__ = pd.DataFrame.__str__
pd.DataFrame.__str__ = lambda self: self.to_string(max_rows=12, max_cols=20, line_width=180)

pd.Series.__original_str__ = pd.Series.__str__
pd.Series.__str__ = lambda self: self.to_string()


class PhdaDataFrame(pd.DataFrame):
    """A DataFrame with temporal filtering support via .during() method."""
    
    def during(self, time_expression, last_days_alt=False):
        """Returns a DataFrame filtered by the given time expression."""
        if not isinstance(self.index, pd.DatetimeIndex):
            print("Error: DataFrame index must be a DatetimeIndex")
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        if isinstance(time_expression, pd.Timestamp) and pd.isna(time_expression):
            print("Error: Time expression is an empty pd.Timestamp (NaT)")
            raise ValueError("Time expression is an empty pd.Timestamp (NaT)")
        
        if isinstance(time_expression, str):
            reference_date = self.index.max().date()
            start_date, end_date = self.get_date_range(
                time_expression, reference_date, last_days_alt=last_days_alt
            )
            start_date = start_date.date() if hasattr(start_date, "date") else start_date
            end_date = end_date.date() if hasattr(end_date, "date") else end_date
        elif isinstance(time_expression, pd.Series):
            if time_expression.empty:
                return PhdaDataFrame(columns=self.columns)
            start_date = time_expression.min().normalize().date()
            end_date = time_expression.max().normalize().date()
        elif isinstance(time_expression, pd.Timestamp):
            start_date = end_date = time_expression.normalize().date()
        else:
            print("Unsupported time expression type: ", time_expression)
            print(type(time_expression))
            raise ValueError("Unsupported time expression type")
        
        mask = (self.index.date >= start_date) & (self.index.date <= end_date)
        return self.loc[mask]
    
    def get_date_range(self, time_expression, reference_date, last_days_alt=False):
        """Gets the start and end dates for a given time expression."""
        match = re.match(r"last (\\d+) days", time_expression)
        
        if time_expression == "today":
            start_date = end_date = reference_date
        elif time_expression == "yesterday":
            start_date = end_date = reference_date - datetime.timedelta(days=1)
        elif match:
            days = int(match.group(1))
            if not last_days_alt:
                start_date = reference_date - datetime.timedelta(days=days)
                end_date = reference_date
            else:
                start_date = reference_date - datetime.timedelta(days=days - 1)
                end_date = reference_date
        else:
            raise ValueError(f"Unknown time expression: {time_expression}")
        
        return start_date, end_date


def age_to_age_group(age: int) -> str:
    """Convert an integer age to the corresponding age group string.
    
    This helper is useful when comparing user profile age with population data
    which uses age group buckets.
    
    Args:
        age: Integer age in years
        
    Returns:
        Age group string like "25-34", "35-44", etc.
    """
    if age < 18:
        return "Under 18"
    elif age <= 24:
        return "18-24"
    elif age <= 34:
        return "25-34"
    elif age <= 44:
        return "35-44"
    elif age <= 54:
        return "45-54"
    elif age <= 64:
        return "55-64"
    else:
        return "65+"


def df_from_csv_string(s, datetime_cols=[], **kwargs):
    """Load a DataFrame from a CSV string."""
    with io.StringIO(s) as f:
        df = pd.read_csv(f, **kwargs)
    # Auto-convert common datetime columns
    for col in ["bed_time", "wake", "datetime", "start_time", "end_time", "wake_up_time"]:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce")
            except Exception:
                pass  # Skip if conversion fails
    return PhdaDataFrame(df)


def enforce_persona_schema(summary, activities, population):
    """Remove useless columns and enforce datatypes for FitBit Advisor personas."""
    # Schema definitions (subset of columns we care about)
    SUMMARY_COLUMNS = {
        "datetime": "datetime64[ns]",
        "steps": "float64",
        "sleep_minutes": "float64",
        "bed_time": "datetime64[ns]",
        "wake_up_time": "datetime64[ns]",
        "resting_heart_rate": "float64",
        "heart_rate_variability": "float64",
        "active_zone_minutes": "float64",
        "deep_sleep_minutes": "float64",
        "rem_sleep_minutes": "float64",
        "light_sleep_minutes": "float64",
        "awake_minutes": "float64",
        "stress_management_score": "float64",
        "fatburn_active_zone_minutes": "float64",
        "cardio_active_zone_minutes": "float64",
        "peak_active_zone_minutes": "float64",
    }
    
    ACTIVITIES_COLUMNS = {
        "start_time": "datetime64[ns]",
        "end_time": "datetime64[ns]",
        "activity_name": "str",
        "distance": "float64",
        "duration": "float64",
        "calories": "float64",
        "steps": "float64",
        "active_zone_minutes": "float64",
    }
    
    POPULATION_COLUMNS = {
        "percentile": "int",
        "age": "str",
        "gender": "str",
        "resting_heart_rate": "float64",
        "heart_rate_variability": "float64",
        "steps": "float64",
        "fatburn_active_zone_minutes": "float64",
        "cardio_active_zone_minutes": "float64",
        "peak_active_zone_minutes": "float64",
        "rem_sleep_minutes": "float64",
        "deep_sleep_minutes": "float64",
        "light_sleep_minutes": "float64",
        "stress_management_score": "float64",
    }
    
    def _enforce(df, schema, index_col=None):
        columns = []
        for col, datatype in schema.items():
            if col in df.columns:
                if types.is_datetime64_any_dtype(df[col]) and datatype == "datetime64[ns]":
                    # Pandas will throw a timezone warning if we do this naively
                    df[col] = df[col].dt.tz_localize(None)
                    df[col] = df[col].astype("datetime64[ns]")
                else:
                    try:
                        df[col] = df[col].astype(datatype)
                    except (ValueError, TypeError):
                        pass  # Keep original dtype if conversion fails
                columns.append(col)
        if index_col is not None and index_col in df.columns:
            df.index = df[index_col]
        # Return df with available columns (don't filter to only schema columns)
        return df
    
    activities = _enforce(activities, ACTIVITIES_COLUMNS, index_col="start_time")
    activities = PhdaDataFrame(activities)
    
    summary = _enforce(summary, SUMMARY_COLUMNS, index_col="datetime")
    summary = PhdaDataFrame(summary)
    
    population = _enforce(population, POPULATION_COLUMNS)
    population = PhdaDataFrame(population)
    
    return summary, activities, population


{{df_definitions}}

# Apply schema enforcement
summary_df, activities_df, population_df = enforce_persona_schema(summary_df, activities_df, population_df)
'''


# =============================================================================
# Autofix for truncated code
# =============================================================================

def _autofix_truncated_code(code: str) -> str:
    """Attempt to fix code truncated by LLM output limits.
    
    Handles two common truncation patterns:
    1. Unclosed parentheses/brackets/braces — auto-close them
    2. Empty compound blocks (if/else/try/except with no body) — insert pass
    
    Args:
        code: The potentially truncated code string.
        
    Returns:
        Fixed code if fixable, original code otherwise.
    """
    try:
        compile(code, '<string>', 'exec')
        return code  # Already valid
    except SyntaxError as e:
        error_msg = str(e)
        
        # Fix 1: Unclosed delimiters
        if "was never closed" in error_msg:
            # Count unmatched openers
            stack = []
            match_map = {'(': ')', '[': ']', '{': '}'}
            in_string = False
            string_char = None
            
            for i, ch in enumerate(code):
                if in_string:
                    if ch == string_char and (i == 0 or code[i-1] != '\\'):
                        in_string = False
                    continue
                if ch in ('"', "'"):
                    # Check for triple quotes
                    if code[i:i+3] in ('"""', "'''"):
                        string_char = code[i:i+3]
                        in_string = True
                    else:
                        string_char = ch
                        in_string = True
                elif ch in match_map:
                    stack.append(match_map[ch])
                elif ch in (')', ']', '}'):
                    if stack and stack[-1] == ch:
                        stack.pop()
            
            if stack:
                fixed = code.rstrip() + ''.join(reversed(stack))
                try:
                    compile(fixed, '<string>', 'exec')
                    return fixed
                except SyntaxError:
                    pass
        
        # Fix 2: Empty compound block (truncated at if:/else:/try:/except:)
        if "expected an indented block" in error_msg:
            lines = code.split('\n')
            if hasattr(e, 'lineno') and e.lineno is not None:
                insert_at = e.lineno  # 1-indexed, insert after this line
                if 0 < insert_at <= len(lines):
                    prev_line = lines[insert_at - 1]
                    indent = len(prev_line) - len(prev_line.lstrip()) + 4
                    lines.insert(insert_at, ' ' * indent + 'pass')
                    fixed = '\n'.join(lines)
                    try:
                        compile(fixed, '<string>', 'exec')
                        return fixed
                    except SyntaxError:
                        pass
        
        return code  # Couldn't fix


# =============================================================================
# Main sandbox class
# =============================================================================

class PythonSandbox:
    """A simple Python sandbox for executing data analysis code.
    
    Uses exec() with captured stdout/stderr for code execution.
    For production use, consider additional sandboxing measures.
    """
    
    def __init__(
        self,
        dfs: Dict[str, pd.DataFrame],
        preamble_template: str = PREAMBLE_TEMPLATE,
    ):
        """Initialize the sandbox with DataFrames.
        
        Args:
            dfs: Dictionary mapping variable names to DataFrames.
                 Expected keys: 'summary_df', 'activities_df', 'profile_df', 'population_df'
            preamble_template: Template string for the preamble code.
        """
        self.preamble = make_preamble(dfs, preamble_template)
        self._namespace: Dict[str, Any] = {}
        self._initialized = False
    
    def _initialize_namespace(self) -> None:
        """Execute the preamble to set up the namespace."""
        if self._initialized:
            return
            
        # Execute preamble in namespace
        exec(self.preamble, self._namespace)
        self._initialized = True
    
    def execute(self, request: PhdaPythonRequest) -> PhdaPythonReply:
        """Execute Python code and return the result.
        
        Args:
            request: The code execution request.
            
        Returns:
            PhdaPythonReply with execution results.
        """
        self._initialize_namespace()
        
        orig_code = request.request
        _log_tool_call("tool_code (DataScience)", {"code": orig_code[:100] + "..." if len(orig_code) > 100 else orig_code})
        
        # Autofix truncated code before processing
        orig_code = _autofix_truncated_code(orig_code)
        
        code = _remove_final_print(orig_code)
        code = _add_call_analysis_line(code)
        
        # Wrap code for execution
        wrapped_code = _wrap_code_for_execution(
            code,
            include_traceback_for_errors=request.include_traceback_for_errors,
        )
        
        # Create a fresh namespace copy with the preamble variables
        exec_namespace = dict(self._namespace)
        
        # Capture stdout/stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        try:
            # Execute the wrapped code
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(wrapped_code, exec_namespace)
            
            # Get result and captured output
            result = exec_namespace.get('_result', None)
            stdout_from_code = exec_namespace.get('_stdout_capture', io.StringIO()).getvalue()
            stderr_from_code = exec_namespace.get('_stderr_capture', io.StringIO()).getvalue()
            
            # Determine output
            if result is not None:
                output = str(result)
            else:
                output = stdout_from_code
            
            # Determine status
            if stderr_from_code:
                status = ExecutionStatus.ERROR
                _log_tool_call("tool_code (DataScience)", {"code": "..."}, error=stderr_from_code[:200])
            else:
                status = ExecutionStatus.SUCCESS
                _log_tool_call("tool_code (DataScience)", {"code": "..."}, output=output[:200] if len(output) > 200 else output)
                
            execution_result = ExecutionResult(
                output=output,
                stderr=stderr_from_code,
                status=status,
            )
            
        except Exception as e:
            # Handle execution errors
            error_msg = traceback.format_exc()
            _log_tool_call("tool_code (DataScience)", {"code": "..."}, error=str(e)[:200])
            execution_result = ExecutionResult(
                output="",
                stderr=f"{ERROR_STRING}: {error_msg}",
                status=ExecutionStatus.ERROR,
            )
        
        return PhdaPythonReply(
            execution_result=execution_result,
            setup_code=self.preamble,
            code_ran=wrapped_code,
            orig_code=orig_code,
        )
    
    def reset_state(self) -> None:
        """Reset the sandbox state."""
        self._namespace = {}
        self._initialized = False


def get_python_sandbox(
    dfs: Dict[str, pd.DataFrame],
    preamble_template: str = PREAMBLE_TEMPLATE,
) -> PythonSandbox:
    """Creates a Python sandbox.
    
    This is the main entry point for creating a sandbox.
    
    Args:
        dfs: Dictionary of DataFrames to make available in the sandbox.
        preamble_template: Template for the preamble code.
        
    Returns:
        Configured PythonSandbox instance.
    """
    return PythonSandbox(dfs=dfs, preamble_template=preamble_template)
