"""PHIA Single-Agent Baseline for PHA.

This baseline implements the original Personal Health Insights Agent (PHIA) architecture:
a single ReAct agent that handles all health queries without specialization or orchestration.

This serves as a comparison baseline to show the value of multi-agent specialization
in the PHA system.

IMPORTANT - Model Configuration:
    This baseline uses Gemini 2.5 Pro by default, matching the original PHIA implementation.
    The original PHIA was designed for Gemini 1.5 models, but those were deprecated.
    Gemini 2.5 Pro provides the best ReAct loop completion for this architecture.
    
    For fair baseline comparisons, note that:
    - PHIA baseline uses: models/gemini-2.5-pro (fixed)
    - PHA/Parallel should use the same model family when comparing architectures

Usage:
    from pha.agents import PHIABaseline, create_phia_baseline
    
    baseline = create_phia_baseline(
        gemini_api_key="your-key",
        tavily_api_key="your-tavily-key",
    )
    response = baseline.respond("How is my sleep quality?")

Research Question: "Is specialization better than one generalist?"
"""

# Version for debugging - increment on each fix
__version__ = "1.11.0"  # Fix is_expression stripping print() from compound blocks + search recovery

import io
import os
import re
import contextlib
import textwrap
from typing import Optional, Dict, Any, List, Sequence
from datetime import datetime

import pandas as pd
import numpy as np

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


def clean_agent_response(text: str) -> str:
    """Clean internal agent labels from response text.
    
    Removes internal markers that shouldn't be shown to users:
    - [SUMMARY] tags from data science agent
    - [CODE UPDATE] tags from data science agent  
    - Coach: prefixes from health coach agent
    - Expert: prefixes from domain expert agent
    - [Finish]: prefix from ReAct agents
    - JSON-escaped newlines
    - Quoted string wrappers
    
    Args:
        text: Raw response text from agents.
        
    Returns:
        Cleaned response suitable for user display.
    """
    if not text:
        return text
    
    # Handle JSON-escaped newlines (convert \n literal to actual newlines)
    if '\\n' in text:
        text = text.replace('\\n', '\n')
    
    # Remove surrounding quotes if the entire text is quoted
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    # Remove [SUMMARY] and [CODE UPDATE] tags (keep content after)
    text = re.sub(r'^\s*\[SUMMARY\]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\[CODE UPDATE\]\s*', '', text, flags=re.MULTILINE)
    
    # Remove Coach: prefix at start of response
    text = re.sub(r'^Coach:\s*', '', text, flags=re.MULTILINE)
    
    # Remove Expert: prefix at start of response  
    text = re.sub(r'^Expert:\s*', '', text, flags=re.MULTILINE)
    
    # Remove [Finish]: prefix (ReAct final answer marker)
    # Handle various formats: [Finish]:, [Finish]: , [Finish]:\n, etc.
    text = re.sub(r'^\s*\[Finish\]:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[Finish\]:\s*\n*', '', text)  # Also catch mid-text occurrences
    
    # Detect and reject system prompt leakage
    # These patterns indicate the model is outputting its instructions instead of a response
    # Normalize text for detection (handle ellipsis, extra spaces, etc.)
    text_normalized = text.lower().replace('…', '...').replace('  ', ' ')
    
    prompt_leak_indicators = [
        # From orchestrator/domain expert prompts
        'you must do so safely and transparently',
        'quantify uncertainty when appropriate',
        'avoid diagnosing or prescribing',
        'always recommend appropriate follow-up',
        'you should use the provided tools',
        'if a tool fails or required user context',
        'explicitly state the limitation',
        # From domain expert agent
        'clinically accurate, evidence-based',
        'safety-conscious guidance',
        'explicitly separate objective findings',
        'state uncertainty and limitations when data',
        'avoid diagnosing when criteria are not met',
        'recommend appropriate follow-up',
        'cite authoritative sources with links',
        'request only the minimum additional data',
        'wearables, labs, imaging, notes',
        'urgent/emergent evaluation when red flags',
        # Generic instruction patterns
        'you will analyze and synthesize',
        'you will avoid',
        'you will cite',
        'you will request',
        'with the goal of delivering',
        'delivering clinically accurate',
        # Tool instruction patterns (shouldn't be in final response)
        'e.g., `tool_code` for math',
        'for math, `compare_blood_test',
        '`datacommons_natural_language_query`',
        'via the `search` tool',
        'use the provided tools',
        # Instruction-style patterns
        'you are tasked with',
        'as a domain expert',
        'you are a domain expert',
        'you have access to a number of tools',
        'it is imperative that you',
    ]
    for indicator in prompt_leak_indicators:
        if indicator.lower() in text_normalized:
            print(f"[PHIA:clean] *** PROMPT LEAK DETECTED: '{indicator}' found in response, replacing with error message")
            print(f"[PHIA:clean] *** Original response was ({len(text)} chars): {repr(text[:300])}")
            return (
                "⚠️ **Response contained system prompt leakage** — the model echoed its instructions "
                "instead of answering your question.\n\n"
                f"**Matched pattern:** `{indicator}`\n\n"
                "This is a known issue with some models. Please try again, rephrase your question, "
                "or switch to a different model."
            )
    
    # Clean up any extra whitespace
    text = text.strip()
    
    return text


# Try to import onetwo components
_ONETWO_AVAILABLE = False
_ONETWO_OPENAI_AVAILABLE = False
_ONETWO_GENAI_AVAILABLE = False  # True if new GoogleGenAIAPI backend is available
_ONETWO_IMPORT_ERROR = None


# =============================================================================
# OpenAI API Compatibility Patch
# =============================================================================
# Newer OpenAI models require `max_completion_tokens` instead of `max_tokens`.

_OPENAI_PATCHED = False

def _patch_openai_for_compatibility():
    """Patch OpenAI client for compatibility with newer models and OneTwo.
    
    Handles:
    - max_tokens → max_completion_tokens conversion
    - Removes 'stop' parameter for GPT-5 models (not supported)
    - Transforms tool call format for OneTwo compatibility
    """
    global _OPENAI_PATCHED
    if _OPENAI_PATCHED:
        return
    
    try:
        from openai.resources.chat import completions
        import re
        
        def _extract_balanced_parens(text: str, start: int) -> tuple:
            """Extract content inside balanced parentheses starting at start position.
            
            Returns (content, end_position) or (None, -1) if not balanced.
            """
            if start >= len(text) or text[start] != '(':
                return None, -1
            
            depth = 0
            i = start
            while i < len(text):
                if text[i] == '(':
                    depth += 1
                elif text[i] == ')':
                    depth -= 1
                    if depth == 0:
                        return text[start+1:i], i
                i += 1
            return None, -1
        
        def _transform_tool_calls_for_onetwo(text: str) -> str:
            """Transform GPT-style tool calls to OneTwo-compatible format.
            
            GPT outputs: tool_code(summary_df['steps'].mean())
            OneTwo expects: tool_code('summary_df["steps"].mean()')
            
            This function wraps the code expression in quotes so OneTwo can parse it.
            """
            if not text:
                return text
            
            result = []
            i = 0
            tool_names = ['tool_code', 'search']
            
            while i < len(text):
                found_tool = False
                for tool_name in tool_names:
                    # Check for [Act]: prefix or direct tool call
                    if text[i:].startswith('[Act]:'):
                        # Find the tool name after [Act]:
                        act_match = re.match(r'\[Act\]:\s*', text[i:])
                        if act_match:
                            prefix = act_match.group(0)
                            tool_start = i + len(prefix)
                            for tn in tool_names:
                                if text[tool_start:].startswith(tn + '('):
                                    arg_content, end_pos = _extract_balanced_parens(text, tool_start + len(tn))
                                    if arg_content is not None:
                                        arg_stripped = arg_content.strip()
                                        # Check if already properly quoted
                                        is_quoted = (
                                            (arg_stripped.startswith('"') and arg_stripped.endswith('"')) or
                                            (arg_stripped.startswith("'") and arg_stripped.endswith("'")) or
                                            arg_stripped.startswith('"""') or
                                            arg_stripped.startswith("'''")
                                        )
                                        if is_quoted:
                                            result.append(text[i:end_pos+1])
                                        else:
                                            # Wrap in quotes, escape existing quotes
                                            escaped = arg_content.replace("'", '"')
                                            result.append(f"{prefix}{tn}('{escaped}')")
                                        i = end_pos + 1
                                        found_tool = True
                                        break
                            if found_tool:
                                break
                    elif text[i:].startswith(tool_name + '('):
                        arg_content, end_pos = _extract_balanced_parens(text, i + len(tool_name))
                        if arg_content is not None:
                            arg_stripped = arg_content.strip()
                            is_quoted = (
                                (arg_stripped.startswith('"') and arg_stripped.endswith('"')) or
                                (arg_stripped.startswith("'") and arg_stripped.endswith("'")) or
                                arg_stripped.startswith('"""') or
                                arg_stripped.startswith("'''")
                            )
                            if is_quoted:
                                result.append(text[i:end_pos+1])
                            else:
                                escaped = arg_content.replace("'", '"')
                                result.append(f"{tool_name}('{escaped}')")
                            i = end_pos + 1
                            found_tool = True
                            break
                
                if not found_tool:
                    result.append(text[i])
                    i += 1
            
            return ''.join(result)
        
        def _truncate_at_stop_sequences(text: str, stop_sequences: list) -> str:
            """Truncate text at the first occurrence of any stop sequence.
            
            Backup for cases where stop parameter doesn't work as expected.
            """
            if not text or not stop_sequences:
                return text
            
            earliest_pos = len(text)
            for stop_seq in stop_sequences:
                pos = text.find(stop_seq)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            
            return text[:earliest_pos]
        
        _original_create = completions.Completions.create
        
        def _patched_create(self, *args, **kwargs):
            # Convert max_tokens to max_completion_tokens for newer API
            if 'max_tokens' in kwargs and 'max_completion_tokens' not in kwargs:
                kwargs['max_completion_tokens'] = kwargs.pop('max_tokens')
            
            model = kwargs.get('model', '')
            is_gpt5 = model and 'gpt-5' in model.lower()
            
            # Note: reasoning_effort parameter only works with Responses API, not Chat Completions
            # For Chat Completions, we rely on explicit prompting instead
            
            # Save stop sequences for backup manual truncation
            saved_stop_sequences = kwargs.get('stop')

            # GPT-5 reasoning models reject `stop` with HTTP 400. Strip it
            # before sending so the request succeeds, and rely on
            # `_truncate_at_stop_sequences` below to honor the requested stop
            # semantics on the returned content.
            if is_gpt5 and 'stop' in kwargs:
                kwargs.pop('stop')

            # Call original method
            result = _original_create(self, *args, **kwargs)
            
            # Post-process the response
            try:
                if hasattr(result, 'choices') and result.choices:
                    for choice in result.choices:
                        if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                            if choice.message.content:
                                content = choice.message.content
                                
                                # For GPT-5: backup manual truncate at stop sequences
                                # (in case the API didn't honor them)
                                if is_gpt5 and saved_stop_sequences:
                                    content = _truncate_at_stop_sequences(content, saved_stop_sequences)
                                
                                # Transform tool calls for OneTwo compatibility
                                content = _transform_tool_calls_for_onetwo(content)
                                
                                choice.message.content = content
            except Exception:
                pass
            
            return result
        
        completions.Completions.create = _patched_create
        _OPENAI_PATCHED = True
    except Exception:
        pass


try:
    from onetwo import ot
    from onetwo.agents import react
    from onetwo.stdlib.tool_use import llm_tool_use
    from onetwo.stdlib.tool_use import python_tool_use
    
    # Use GoogleGenAIAPI backend (new google.genai SDK, no deprecation warning,
    # safety defaults to OFF for Gemini 2.5+ models)
    try:
        from onetwo.backends import google_genai_api
        _ONETWO_GENAI_AVAILABLE = True
    except ImportError:
        pass  # Will error clearly at backend construction time
    
    _ONETWO_AVAILABLE = True
    
    # Try to import OpenAI backend
    try:
        from onetwo.backends import openai_api
        _ONETWO_OPENAI_AVAILABLE = True
        _patch_openai_for_compatibility()
    except ImportError:
        pass
except ImportError as e:
    _ONETWO_IMPORT_ERROR = str(e)
except SyntaxError as e:
    import sys
    _ONETWO_IMPORT_ERROR = f"onetwo requires Python 3.12+, you have {sys.version}"
except Exception as e:
    _ONETWO_IMPORT_ERROR = str(e)

# For notebook-based exemplar parsing
try:
    import nbformat
    _NBFORMAT_AVAILABLE = True
except ImportError:
    _NBFORMAT_AVAILABLE = False


# =============================================================================
# PHIA Preamble - Combined expertise prompt
# =============================================================================

PHIA_PREAMBLE = textwrap.dedent("""\
{#- Preamble: Agent functionality description -#}
I am going to ask you a question about Fitbit data. Assume that you have access
to pandas through `pd` and numpy through `np`. You DO NOT have access to matplotlib or other python libraries.

Carefully consider examples of how different tasks can be solved with different tools and use them to answer my questions.
Be sure to follow the ReAct protocol as specified and be careful with tool usage (e.g., use only one tool at a time).
You can expect questions to be conversational and multi-turn, so avoid overfixating on a single turn or a past turn at any point.

#### You have access to the two dataframes below:

- `summary_df`: This is a summary of the user's activity and sleep data. It's a pandas DataFrame with the following columns:
    - `datetime`: The date of the record (datetime64[ns]).
    - `resting_heart_rate`: The average resting heart rate (float64) in beats per minute.
    - `heart_rate_variability`: The user's heart rate variability (float64).
    - `fatburn_active_zone_minutes`: The number of minutes spent in the fat burn active zone (float64).
    - `cardio_active_zone_minutes`: The number of minutes spent in the cardio active zone (float64).
    - `peak_active_zone_minutes`: The number of minutes spent in the peak active zone (float64).
    - `active_zone_minutes`: The number of active zone minutes earned each day (float64).
    - `steps`: The number of steps taken each day (float64).
    - `rem_sleep_minutes`: The amount of REM sleep (float64) in minutes.
    - `deep_sleep_minutes`: The amount of deep sleep (float64) in minutes.
    - `awake_minutes`: The amount of awake time (float64) in minutes.
    - `light_sleep_minutes`: The amount of light sleep (float64) in minutes.
    - `sleep_minutes`: The total sleep time (float64) in minutes.
    - `bed_time`: The time the user went to bed (datetime64[ns]).
    - `wake_up_time`: The time the user woke up (datetime64[ns]).
    - `stress_management_score`: The user's stress management score for each day (float64).
    - `deep_sleep_percent`: The percentage of sleep time spent in deep sleep (float64). NOTE: This column may not be available in all datasets. If absent, compute it as `deep_sleep_minutes / sleep_minutes * 100`.
    - `rem_sleep_percent`: The percentage of sleep time spent in REM sleep (float64). NOTE: This column may not be available in all datasets. If absent, compute it as `rem_sleep_minutes / sleep_minutes * 100`.
    - `awake_percent`: The percentage of sleep time spent awake (float64). NOTE: This column may not be available in all datasets. If absent, compute it as `awake_minutes / sleep_minutes * 100`.
    - `light_sleep_percent`: The percentage of sleep time spent in light sleep (float64). NOTE: This column may not be available in all datasets. If absent, compute it as `light_sleep_minutes / sleep_minutes * 100`.

- `activities_df`: This is a list of the user's activities. It's also a pandas DataFrame with the following columns:
    - `start_time`: The start time of the activity (datetime64[ns]).
    - `end_time`: The end time of the activity (datetime64[ns]).
    - `activity_name`: The name of the activity (str). (e.g., "WALKING", "RUNNING", "OUTDOOR_BIKE", "STRENGTH_TRAINING", "WORKOUT", "ELLIPTICAL", "YOGA", "TREADMILL", "HIKING", "SWIMMING").
    - `distance`: The distance covered during the activity (float64) in miles.
    - `duration`: The duration of the activity (float64) in minutes.
    - `calories`: The number of calories burned during the activity (float64).
    - `steps`: The number of steps taken during the activity (float64).
    - `active_zone_minutes`: The number of active zone minutes earned during the activity (float64).

NOTE: The available columns may vary by dataset. If you get a KeyError, use `print(activities_df.columns)` to check available columns.

#### You also have access to a profile dataframe:

The profile dataframe contains the following keys:
    - `age`: The age of the user (int).
    - `gender`: The gender of the user (str).
    - `averageDailySteps`: The average number of steps taken each day (int).
    - `elderly`: Whether the user is elderly ("Yes" or "No") (str).
    - `height_cm`: The height of the user in centimeters (int).
    - `weight_kg`: The weight of the user in kilograms (int).

IMPORTANT: The profile dataframe already contains the user's demographic information (age, gender, height, weight). 
Always use `profile['age'].values[0]`, `profile['gender'].values[0]`, etc. to get clean scalar values - NEVER ask the user for data that is already available in the dataframes.
""")

PHIA_REACT_PROMPT_TEXT = """\
{#- Preamble: Tools description -#}
{%- role name='system' -%}
You are a ReAct agent that solves problems by thinking step-by-step and using tools.

CRITICAL: Execute the ReAct loop - do not just describe what you would do.

Here is a list of available tools:
{% for tool in tools %}
Tool name: {{ tool.name }}
Tool description: {{ tool.description }}
{% if tool.example -%}
  Tool example: {{ tool.example_str }}
{%- endif -%}
{% endfor %}

ReAct Format (MUST FOLLOW EXACTLY):
1. [Thought]: Reason about what to do next (1-2 sentences max)
2. [Act]: Call ONE tool with its arguments
3. [Observe]: Wait for and read the result
4. Repeat steps 1-3 until you have the answer
5. [Finish]: Provide your final answer to the user

FORMAT RULES:
- [Act] lines must contain ONLY the function call on ONE LINE
- tool_code() calls: put ALL code as a single quoted string argument
  CORRECT: tool_code('avg = summary_df["sleep_minutes"].mean()\\nprint(f"Average: {avg:.0f} min")')
  WRONG: tool_code(
    avg = summary_df['sleep_minutes'].mean()
  )
- For multi-line code, use \\n inside the string: tool_code('line1\\nline2\\nline3')
- Keep tool_code calls CONCISE (under 15 lines). Do NOT wrap code in if/else guards
  for empty data checks — just compute directly. If data is missing, the error will
  tell you, and you can handle it in your next step.
- Do NOT use markdown code blocks (```) 
- Do NOT use backticks (`) around tool calls  
- Do NOT output planning statements without executing tools
- ALWAYS wait for [Observe] before providing answers
- In tool_code, ALWAYS use print() with labels: print(f"Average RHR: {val:.2f} bpm")
  Do NOT use bare return/tuple — output must be labeled and human-readable.
- Use [Finish]: to provide your final synthesized answer
- The [Finish] text must be COMPLETE — include ALL details, tips, and recommendations.
  Do NOT end with a colon or promise content you don't include.
  Do NOT put detailed analysis in [Thought] and then write a short [Finish].

CORRECT EXAMPLE:
[Thought]: I need to calculate the average steps from the data.
[Act]: tool_code('avg_steps = summary_df["steps"].mean()\nprint(f"Average daily steps: {avg_steps:.0f}")')
[Observe]: Average daily steps: 6543
[Thought]: Now I have the average. Let me provide the answer.
[Finish]: Your average daily steps is 6,543 steps.

INCORRECT (DO NOT DO):
- "I need to calculate X, then search Y, then compute Z" (this is planning, not executing)
- [Act]: ```tool_code(...)```  (no markdown blocks)
- Outputting raw search results without synthesizing them

{#- Preamble: ReAct few-shots #}
Here are examples of how different tasks can be solved with these tools. Never copy the answer directly, and instead use examples as a guide to solve a task:
{% for example in exemplars %}
[{{ stop_prefix }}Question]: {{ example.inputs + '\n' }}
{%- for step in example.updates -%}
{%- if step.thought -%}
  [Thought]: {{ step.thought + '\n' }}
{%- endif -%}
{%- if step.action -%}
  [Act]: {{ step.render_action() + '\n' }}
{%- endif -%}
{%- if step.observation and step.action -%}
  [{{ stop_prefix }}Observe]: {{ step.render_observation() + '\n' }}
{%- endif -%}
{%- if step.is_finished and step.observation -%}
  [Finish]: {{ step.observation + '\n' }}
{%- endif -%}
{%- endfor -%}
{%- endfor -%}

Remember: Execute the ReAct loop step by step. Think, Act, Observe, repeat until done, then Finish with a synthesized answer.

{# Start of the processing of the actual inputs. -#}

Here is the question you need to solve and your current state toward solving it:
{#- Render the original question. -#}
{%- endrole -%}
{%- role name='user' %}
[{%- role name='system' -%}{{ stop_prefix }}{%- endrole -%}Question]: {{ state.inputs + '\n' }}
{%- endrole -%}

{# Render the current state (i.e., any steps performed up till now). -#}
{%- for step in state.updates -%}
{%- if step.thought -%}
  [Thought]: {{ step.thought + '\n' }}
{%- endif -%}
{%- if step.action -%}
  [Act]: {{ step.render_action() + '\n' }}
{%- endif -%}
{%- if step.observation and step.action -%}
  [{{ stop_prefix }}Observe]: {{ step.render_observation() + '\n' }}
{%- endif -%}
{%- if step.is_finished and step.observation -%}
  [Finish]: {{ step.observation + '\n' }}
{%- endif -%}
{%- endfor -%}

{# If force-finishing, then prompt the LLM for the final answer. -#}
{%- if force_finish -%}
  [Finish]:{{ ' ' }}
{%- endif -%}

{#- Get a response from the LLM and return it. -#}
{%- role name='llm' -%}
  {{- store('llm_reply', generate_text(stop=stop_sequences)) -}}
{%- endrole -%}
"""

QUESTION_PREFIX = (
    "Use tools such as tool_code to execute Python code and search to find "
    "external relevant information as needed. Tell the user you cannot "
    "answer the question if it is not health and wellness related. "
    "Take into account that questions may have typos or grammatical "
    "mistakes and try to reinterpret them before answering. Follow all "
    "instructions and the ReAct template carefully. Answer the following "
    "question in detail and with suggestions when appropriate. Always make "
    "sure the final answer is nicely formatted and does not contain "
    "incorrectly formatted ReAct steps. "
    "IMPORTANT: When a user expresses health concerns or worries, ALWAYS "
    "use tool_code FIRST to check their actual data (sleep, heart rate, "
    "activity, etc.) before giving general advice. Ground your response "
    "in their specific numbers. "
    "IMPORTANT: When a user asks to 'see' or 'show' their data, print "
    "actual values and statistics, NOT column names. "
    "IMPORTANT: In tool_code, always use print() with descriptive labels "
    "for results (e.g., print(f'Average RHR: {avg_rhr:.2f} bpm')). "
    "Do NOT use bare return statements or tuples — the output must be "
    "human-readable with labeled values. "
    "IMPORTANT: Always provide a COMPLETE answer in [Finish]. Include ALL "
    "details, tips, and recommendations in the finish text. Never end "
    "with a colon or promise content that you don't include. "
    "Question: "
)


# =============================================================================
# Tool Names
# =============================================================================

_PYTHON_TOOL_NAME = "tool_code"
_SEARCH_TOOL_NAME = "search"
_FINISH_TOOL_NAME = "finish"


# =============================================================================
# Helper Functions
# =============================================================================

def _autofix_truncated_code(code: str) -> str:
    """Fix code truncated during extraction from tool_code('...').
    
    When models generate long code inside tool_code('...'), the string
    can get truncated (by token limits or argument parsing), leaving
    compound statements like ``else:`` with no body — a SyntaxError
    that the model then loops on 10x without fixing.
    
    Strategy: detect truncation (empty compound-statement body) and
    insert ``pass`` so the code can execute. The truncated branch is
    almost always a non-essential guard (e.g., ``else: print("No data")``),
    while the main analysis lives in the ``if`` block.
    """
    try:
        compile(code, '<string>', 'exec')
        return code  # Already valid
    except SyntaxError as e:
        error_msg = str(e)
        if 'expected an indented block' not in error_msg:
            return code  # Different error, don't touch
        error_lineno = e.lineno
    
    lines = code.split('\n')
    
    # The error message contains the actual compound statement line:
    # "expected an indented block after 'else' statement on line 3"
    # e.lineno may point elsewhere (to the next code line), so
    # parse the message for the true source line.
    msg_match = re.search(r'on line (\d+)', error_msg)
    stmt_lineno = int(msg_match.group(1)) if msg_match else error_lineno
    
    if not stmt_lineno or stmt_lineno < 1 or stmt_lineno > len(lines):
        return code
    
    # Insert `pass` as the body of the empty compound statement.
    offending_line = lines[stmt_lineno - 1]
    indent = len(offending_line) - len(offending_line.lstrip())
    lines.insert(stmt_lineno, ' ' * (indent + 4) + 'pass')
    fixed = '\n'.join(lines)
    
    try:
        compile(fixed, '<string>', 'exec')
        print(f"[PHIA:autofix] Fixed truncated block on line {stmt_lineno}: "
              f"'{offending_line.strip()}' → added pass")
        return fixed
    except SyntaxError:
        return code  # Fix didn't help, return original


def simple_python_executor(code: str, sandbox_globals: dict) -> str:
    """Executes python code and captures the output.
    
    Args:
        code: Python code to execute.
        sandbox_globals: Global variables available to the code.
        
    Returns:
        The output of the code execution.
    """
    _log_tool_call("tool_code", {"code": code[:100] + "..." if len(code) > 100 else code})
    
    # Auto-fix truncated code blocks (e.g., else: with no body from
    # tool_code string truncation). Without this, the model loops on
    # the same SyntaxError for max_steps iterations.
    code = _autofix_truncated_code(code)
    
    local_scope = {}
    string_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(string_buffer):
            code_lines = code.strip().split('\n')
            last_line = code_lines[-1].strip() if code_lines else ""
            
            # Determine if last line is an expression
            is_expression = (
                last_line and 
                not last_line.startswith(('import ', 'from ', 'def ', 'class ', 
                                          'if ', 'for ', 'while ', 'with ', 
                                          'try:', 'except', 'finally', 'elif ', 'else:',
                                          'pass', 'break', 'continue', 'return',
                                          'raise', 'del ', 'assert ',
                                          'print(', 'print (')) and
                '=' not in last_line.split('#')[0] and
                not last_line.startswith('#')
            )
            
            if is_expression and len(code_lines) > 1:
                exec('\n'.join(code_lines[:-1]), sandbox_globals, local_scope)
                result = eval(last_line, {**sandbox_globals, **local_scope}, local_scope)
                printed_output = string_buffer.getvalue()
                output = printed_output if printed_output else str(result)
            else:
                exec(code, sandbox_globals, local_scope)
                printed_output = string_buffer.getvalue()
                
                if not printed_output and is_expression:
                    try:
                        result = eval(last_line, {**sandbox_globals, **local_scope}, local_scope)
                        output = str(result)
                    except:
                        output = printed_output or "Code executed successfully."
                else:
                    output = printed_output or "Code executed successfully."
        
        _log_tool_call("tool_code", {"code": "..."}, output=output[:200] if len(output) > 200 else output)
        return output
        
    except Exception as e:
        error_str = str(e)
        
        # Inline retry: fix truncated compound statements (e.g., else: with no body).
        # This is a second line of defense — _autofix_truncated_code above should
        # catch this, but if it didn't for any reason, handle it here.
        if 'expected an indented block' in error_str:
            try:
                # Parse the statement line from the error message
                msg_match = re.search(r'on line (\d+)', error_str)
                if msg_match:
                    fix_lineno = int(msg_match.group(1))
                    fix_lines = code.split('\n')
                    if 1 <= fix_lineno <= len(fix_lines):
                        stmt = fix_lines[fix_lineno - 1]
                        indent = len(stmt) - len(stmt.lstrip())
                        fix_lines.insert(fix_lineno, ' ' * (indent + 4) + 'pass')
                        fixed_code = '\n'.join(fix_lines)
                        
                        # Try to execute the fixed code
                        fix_buffer = io.StringIO()
                        fix_scope = {}
                        with contextlib.redirect_stdout(fix_buffer):
                            exec(fixed_code, sandbox_globals, fix_scope)
                        fix_output = fix_buffer.getvalue() or "Code executed successfully."
                        print(f"[PHIA:autofix-retry] Fixed '{stmt.strip()}' on line {fix_lineno} "
                              f"and executed successfully")
                        _log_tool_call("tool_code", {"code": "..."}, 
                                       output=fix_output[:200] if len(fix_output) > 200 else fix_output)
                        return fix_output
            except Exception as retry_err:
                print(f"[PHIA:autofix-retry] Retry also failed: {retry_err}")
        
        _log_tool_call("tool_code", {"code": "..."}, error=error_str)
        return f"Error executing code: {e}"


def tavily_search_func(query: str, api_key: Optional[str] = None) -> str:
    """Performs web search using Tavily API.
    
    Args:
        query: The search query string.
        api_key: Tavily API key.
        
    Returns:
        Formatted search results as a string.
    """
    _log_tool_call("search", {"query": query})
    
    try:
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                _log_tool_call("search", {"query": query}, error="Tavily API key not found")
                return "Error: Tavily API key not found."

        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        
        search_results = client.search(
            query=query,
            max_results=5,
            include_answer=False,
            include_raw_content=False,
            search_depth="advanced"
        )

        formatted_results = []
        
        if search_results.get('answer'):
            formatted_results.append(f"Summary: {search_results['answer']}\n")
        
        result_count = 0
        if search_results.get('results'):
            result_count = len(search_results['results'])
            formatted_results.append("Search Results:")
            for i, result in enumerate(search_results['results'], 1):
                formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
                formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
                formatted_results.append(f"   Content: {result.get('content', 'No content')[:200]}...")
                if result.get('score'):
                    formatted_results.append(f"   Relevance Score: {result['score']:.2f}")
        
        result_text = "\n".join(formatted_results) if formatted_results else "No search results found."
        _log_tool_call("search", {"query": query}, output=f"Found {result_count} results")
        return result_text
        
    except Exception as e:
        _log_tool_call("search", {"query": query}, error=str(e))
        return f"Error performing search: {str(e)}"


def build_exemplars_from_notebooks(
    notebook_paths: Sequence[str],
    verbose: bool = False,
) -> List["react.ReActState"]:
    """Construct ReAct exemplars from notebook files.
    
    Args:
        notebook_paths: Paths to notebook files containing exemplars.
        verbose: If True, print processing information.
        
    Returns:
        List of ReActState objects.
    """
    if not _NBFORMAT_AVAILABLE:
        if verbose:
            print("nbformat not available, returning empty exemplars")
        return []
    
    if not _ONETWO_AVAILABLE:
        if verbose:
            print("onetwo not available, returning empty exemplars")
        return []
    
    exemplars = []
    
    for notebook_path in notebook_paths:
        if verbose:
            print(f"Processing: {notebook_path}")
            
        if not os.path.exists(notebook_path):
            if verbose:
                print(f"  File not found: {notebook_path}")
            continue
            
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                nb = nbformat.reads(f.read(), as_version=4)
        except Exception as e:
            if verbose:
                print(f"  Error reading notebook: {e}")
            continue
        
        cells = nb.get("cells", [])
        if not cells:
            continue
        
        # First cell is the problem description
        description_cell = cells.pop(0)
        problem_description = "".join(description_cell.get("source", [])).strip()
        problem_description = problem_description.lstrip("#").strip()
        
        planner_updates = []
        
        # Process cells in pairs (markdown thought + code action)
        i = 0
        while i < len(cells):
            if (i < len(cells) - 1 and 
                cells[i].get("cell_type") == "markdown" and 
                cells[i + 1].get("cell_type") == "code"):
                
                thought = "".join(cells[i].get("source", [])).strip()
                code = "".join(cells[i + 1].get("source", [])).strip()
                code = code.replace('# @test {"skip": true}\n', "")
                
                # Determine function to call
                function_name = _SEARCH_TOOL_NAME if "search" in code.lower() else _PYTHON_TOOL_NAME
                
                if function_name == _SEARCH_TOOL_NAME:
                    # Extract search query from code like: search('query')
                    try:
                        args = (code.split("'")[1],)
                    except IndexError:
                        args = (code,)
                else:
                    args = (code,)
                
                # Get observation from cell outputs
                observation = ""
                try:
                    outputs = cells[i + 1].get("outputs", [])
                    if outputs:
                        output = outputs[0]
                        if "text" in output:
                            observation = "".join(output["text"]).strip()
                        elif "text/plain" in output.get("data", {}):
                            observation = "".join(output["data"]["text/plain"]).strip()
                except (KeyError, IndexError):
                    pass
                
                fmt = (llm_tool_use.ArgumentFormat.MARKDOWN 
                       if function_name == _PYTHON_TOOL_NAME 
                       else llm_tool_use.ArgumentFormat.PYTHON)
                
                step = react.ReActStep(
                    thought=thought,
                    is_finished=False,
                    action=llm_tool_use.FunctionCall(
                        function_name=function_name,
                        args=args,
                        kwargs={},
                    ),
                    observation=observation,
                    fmt=fmt,
                )
                planner_updates.append(step)
                i += 2
            else:
                i += 1
        
        # Handle final step with finish tool
        if planner_updates and not planner_updates[-1].is_finished:
            last_step = planner_updates[-1]
            if (last_step.action and 
                last_step.action.function_name == _PYTHON_TOOL_NAME):
                
                action_code = last_step.action.args[0]
                final_answer = None
                
                if "print(" in action_code:
                    for pattern in ['print("""', "print('", 'print("']:
                        if pattern in action_code:
                            try:
                                end_pattern = '""")' if pattern == 'print("""' else ("')" if pattern == "print('" else '")')
                                final_answer = action_code.split(pattern)[1].rsplit(end_pattern, 1)[0]
                                break
                            except IndexError:
                                continue
                    
                    if final_answer is None:
                        final_answer = action_code
                
                if final_answer is not None:
                    final_step = react.ReActStep(
                        thought=last_step.thought,
                        is_finished=True,
                        action=llm_tool_use.FunctionCall(
                            function_name=_FINISH_TOOL_NAME,
                            args=(final_answer,),
                            kwargs={},
                        ),
                        observation=final_answer,
                        fmt=llm_tool_use.ArgumentFormat.PYTHON,
                    )
                    planner_updates[-1] = final_step
        
        exemplars.append(react.ReActState(
            inputs=problem_description,
            updates=planner_updates,
        ))
    
    return exemplars


def _create_default_exemplars() -> List["react.ReActState"]:
    """Create default exemplars when notebooks are not available.
    
    Returns:
        List of hardcoded ReActState exemplars.
    """
    if not _ONETWO_AVAILABLE:
        return []
    
    exemplars = []
    
    # Example 1: Sleep analysis
    exemplars.append(react.ReActState(
        inputs="How has my sleep been over the past week?",
        updates=[
            react.ReActStep(
                thought="I need to analyze the user's sleep data from the summary_df. I'll look at sleep_minutes, deep_sleep_minutes, and rem_sleep_minutes for the last 7 days.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_PYTHON_TOOL_NAME,
                    args=("""# Get last 7 days of sleep data
last_week = summary_df.tail(7)
avg_sleep = last_week['sleep_minutes'].mean()
avg_deep = last_week['deep_sleep_minutes'].mean()
avg_rem = last_week['rem_sleep_minutes'].mean()
deep_pct = avg_deep / avg_sleep * 100 if avg_sleep > 0 else 0
rem_pct = avg_rem / avg_sleep * 100 if avg_sleep > 0 else 0
print(f"Average sleep: {avg_sleep:.0f} minutes ({avg_sleep/60:.1f} hours)")
print(f"Average deep sleep: {avg_deep:.0f} min ({deep_pct:.1f}%)")
print(f"Average REM sleep: {avg_rem:.0f} min ({rem_pct:.1f}%)")""",),
                    kwargs={},
                ),
                observation="Average sleep: 420 minutes (7.0 hours)\nAverage deep sleep: 78 min (18.5%)\nAverage REM sleep: 94 min (22.3%)",
                fmt=llm_tool_use.ArgumentFormat.MARKDOWN,
            ),
            react.ReActStep(
                thought="The user is getting about 7 hours of sleep with good deep and REM percentages. Let me provide a summary with health context.",
                is_finished=True,
                action=llm_tool_use.FunctionCall(
                    function_name=_FINISH_TOOL_NAME,
                    args=("Based on your data from the past week, you're averaging 7 hours of sleep per night, which meets the recommended 7-9 hours for adults. Your deep sleep (78 min, 18.5%) and REM sleep (94 min, 22.3%) are within healthy ranges. Deep sleep is important for physical recovery, while REM sleep supports cognitive function and memory consolidation.",),
                    kwargs={},
                ),
                observation="Based on your data from the past week, you're averaging 7 hours of sleep per night, which meets the recommended 7-9 hours for adults. Your deep sleep (78 min, 18.5%) and REM sleep (94 min, 22.3%) are within healthy ranges. Deep sleep is important for physical recovery, while REM sleep supports cognitive function and memory consolidation.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    ))
    
    # Example 2: Activity analysis with search
    exemplars.append(react.ReActState(
        inputs="What's a good resting heart rate for someone my age?",
        updates=[
            react.ReActStep(
                thought="I should first check the user's age from their profile, then search for information about healthy resting heart rate ranges.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_PYTHON_TOOL_NAME,
                    args=("""# Get user's age and recent resting heart rate
age = profile['age'].values[0]
recent_rhr = summary_df['resting_heart_rate'].tail(7).mean()
print(f"User age: {age}")
print(f"Recent average RHR: {recent_rhr:.0f} bpm")""",),
                    kwargs={},
                ),
                observation="User age: 34\nRecent average RHR: 68 bpm",
                fmt=llm_tool_use.ArgumentFormat.MARKDOWN,
            ),
            react.ReActStep(
                thought="The user is 34 years old with an RHR of 68 bpm. Let me search for healthy RHR ranges for this age group.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_SEARCH_TOOL_NAME,
                    args=("healthy resting heart rate range for adults age 30-40",),
                    kwargs={},
                ),
                observation="Search Results:\n1. Normal resting heart rate for adults is 60-100 bpm. Athletes may have RHR as low as 40 bpm. A lower RHR generally indicates better cardiovascular fitness.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
            react.ReActStep(
                thought="I have all the information needed to provide a comprehensive answer about the user's resting heart rate.",
                is_finished=True,
                action=llm_tool_use.FunctionCall(
                    function_name=_FINISH_TOOL_NAME,
                    args=("Your resting heart rate of 68 bpm is within the healthy range of 60-100 bpm for adults. At 34 years old, this indicates good cardiovascular health. A lower RHR generally suggests better fitness - athletes often have RHR in the 40-60 bpm range. To potentially lower your RHR, consider regular aerobic exercise, stress management, and adequate sleep.",),
                    kwargs={},
                ),
                observation="Your resting heart rate of 68 bpm is within the healthy range of 60-100 bpm for adults. At 34 years old, this indicates good cardiovascular health. A lower RHR generally suggests better fitness - athletes often have RHR in the 40-60 bpm range. To potentially lower your RHR, consider regular aerobic exercise, stress management, and adequate sleep.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    ))
    
    # Example 3: Emotional/worried query — ALWAYS check data first
    exemplars.append(react.ReActState(
        inputs="I'm really worried about my health lately, been feeling tired and my heart races sometimes",
        updates=[
            react.ReActStep(
                thought="The user is expressing health concerns. I should FIRST check their actual data to provide personalized reassurance, then supplement with general advice.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_PYTHON_TOOL_NAME,
                    args=("""# Check sleep, heart rate, and activity data
avg_sleep = summary_df['sleep_minutes'].mean()
avg_rhr = summary_df['resting_heart_rate'].mean()
avg_hrv = summary_df['heart_rate_variability'].mean()
avg_steps = summary_df['steps'].mean()
avg_deep = summary_df['deep_sleep_minutes'].mean()
deep_pct = avg_deep / avg_sleep * 100 if avg_sleep > 0 else 0
print(f"Avg sleep: {avg_sleep:.0f} min ({avg_sleep/60:.1f} hrs)")
print(f"Avg deep sleep: {avg_deep:.0f} min ({deep_pct:.1f}%)")
print(f"Avg resting HR: {avg_rhr:.1f} bpm")
print(f"Avg HRV: {avg_hrv:.1f}")
print(f"Avg daily steps: {avg_steps:.0f}")""",),
                    kwargs={},
                ),
                observation="Avg sleep: 418 min (7.0 hrs)\nAvg deep sleep: 72 min (17.2%)\nAvg resting HR: 62.2 bpm\nAvg HRV: 45.0\nAvg daily steps: 8234",
                fmt=llm_tool_use.ArgumentFormat.MARKDOWN,
            ),
            react.ReActStep(
                thought="The data shows healthy metrics. Let me search for information about fatigue and heart racing to give informed advice.",
                is_finished=False,
                action=llm_tool_use.FunctionCall(
                    function_name=_SEARCH_TOOL_NAME,
                    args=("causes of fatigue and heart racing in adults",),
                    kwargs={},
                ),
                observation="Search Results:\n1. Common causes include stress, caffeine, poor sleep quality, dehydration, and anxiety. Resting heart rate under 100 bpm is normal.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
            react.ReActStep(
                thought="I have both the user's data and general health context. I can now provide a reassuring, data-grounded response.",
                is_finished=True,
                action=llm_tool_use.FunctionCall(
                    function_name=_FINISH_TOOL_NAME,
                    args=("I understand you're feeling worried, and I want to help put things in perspective with your actual data. Your resting heart rate averages 62.2 bpm, which is excellent — well within the healthy 60-100 bpm range. Your HRV of 45 also looks healthy. You're averaging about 7 hours of sleep with 17% deep sleep, which is reasonable but could be improved.\n\nThe tiredness and occasional racing heart could be related to several factors: stress, caffeine intake, or inconsistent sleep timing. Here are some steps that could help:\n\n1. Try to maintain a consistent sleep schedule — going to bed and waking up at the same time daily.\n2. If you drink coffee, try reducing intake or cutting it off by early afternoon.\n3. Stay hydrated throughout the day.\n4. Your step count of 8,234 shows good activity — keep it up!\n\nThat said, if the heart racing episodes are frequent or severe, please consult a doctor to rule out any underlying conditions. Your overall data looks healthy, which is reassuring.",),
                    kwargs={},
                ),
                observation="I understand you're feeling worried, and I want to help put things in perspective with your actual data. Your resting heart rate averages 62.2 bpm, which is excellent — well within the healthy 60-100 bpm range. Your HRV of 45 also looks healthy. You're averaging about 7 hours of sleep with 17% deep sleep, which is reasonable but could be improved.\n\nThe tiredness and occasional racing heart could be related to several factors: stress, caffeine intake, or inconsistent sleep timing. Here are some steps that could help:\n\n1. Try to maintain a consistent sleep schedule — going to bed and waking up at the same time daily.\n2. If you drink coffee, try reducing intake or cutting it off by early afternoon.\n3. Stay hydrated throughout the day.\n4. Your step count of 8,234 shows good activity — keep it up!\n\nThat said, if the heart racing episodes are frequent or severe, please consult a doctor to rule out any underlying conditions. Your overall data looks healthy, which is reassuring.",
                fmt=llm_tool_use.ArgumentFormat.PYTHON,
            ),
        ],
    ))
    
    return exemplars


# =============================================================================
# Check onetwo availability
# =============================================================================

def is_onetwo_available() -> bool:
    """Check if onetwo is available for PHIA baseline."""
    return _ONETWO_AVAILABLE


def get_onetwo_import_error() -> Optional[str]:
    """Get the onetwo import error message if any."""
    return _ONETWO_IMPORT_ERROR


# =============================================================================
# PHIA Baseline Class
# =============================================================================

if _ONETWO_AVAILABLE:
    
    class PHIABaseline:
        """PHIA Single-Agent Baseline.
        
        Implements the original PHIA architecture: a single ReAct agent that
        handles all health queries using Python code execution and web search.
        
        Key differences from PHA:
        - Single generalist agent instead of 3 specialists
        - No orchestration or routing logic
        - Combined prompt covering all health domains
        """
        
        def __init__(
            self,
            tavily_api_key: Optional[str] = None,
            few_shot_dir: Optional[str] = None,
            debug_verbose: bool = False,
        ):
            """Initialize the PHIA baseline.
            
            Args:
                tavily_api_key: Tavily API key for web search.
                few_shot_dir: Directory containing few-shot notebook examples.
                debug_verbose: If True, print debug information.
            """
            self.tavily_api_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
            self.few_shot_dir = few_shot_dir
            self.debug_verbose = debug_verbose
            
            if debug_verbose:
                print(f"[PHIA Baseline] Version {__version__}")
            
            # Agent state
            self._agent = None
            self._gemini_api_key = None
            self._sandbox_globals = None
            
            # Data
            self.summary_df: Optional[pd.DataFrame] = None
            self.activities_df: Optional[pd.DataFrame] = None
            self.profile_df: Optional[pd.DataFrame] = None
            
            # Conversation history
            self.conversation_history: list = []
            self.last_response: str = ""
        
        def configure(
            self,
            api_key: Optional[str] = None,
            provider: str = "gemini",
            model_name: Optional[str] = None,
            # Legacy parameter
            gemini_api_key: Optional[str] = None,
        ) -> "PHIABaseline":
            """Configure the PHIA baseline with API credentials.
            
            Args:
                api_key: API key for the selected provider.
                provider: "gemini" or "openai".
                model_name: Model to use. Defaults vary by provider:
                    - Gemini: "models/gemini-2.5-pro"
                    - OpenAI: "gpt-4o"
                gemini_api_key: DEPRECATED - use api_key instead.
                
            Returns:
                Self for method chaining.
            """
            # Handle legacy parameter
            if gemini_api_key and not api_key:
                api_key = gemini_api_key
                provider = "gemini"
            
            if not api_key:
                raise ValueError("API key is required")
            
            self._api_key = api_key
            self._provider = provider.lower()
            
            # Set default model based on provider
            if model_name is None:
                if self._provider == "gemini":
                    model_name = "models/gemini-3.1-flash-lite-preview"
                elif self._provider == "openai":
                    model_name = "gpt-4o"
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            
            self._model_name = model_name
            
            # Configure onetwo backend using register() pattern
            if self._provider == "gemini":
                # Add models/ prefix if needed
                if not model_name.startswith("models/"):
                    model_name = f"models/{model_name}"
                
                if not _ONETWO_GENAI_AVAILABLE:
                    raise ImportError(
                        "onetwo's GoogleGenAIAPI backend not available. "
                        "Update onetwo: pip install -U git+https://github.com/google-deepmind/onetwo"
                    )
                
                print(f"[PHIA:configure] Registering GoogleGenAIAPI backend: model={model_name}, temp=0.0")
                llm_engine = google_genai_api.GoogleGenAIAPI(
                    api_key=api_key,
                    generate_model_name=model_name,
                    # Pin chat model too — OneTwo verifies both; default is
                    # `models/gemini-2.5-flash` which not all keys have.
                    chat_model_name=model_name,
                    temperature=0.0,
                    max_tokens=8192,
                    threadpool_size=1,
                )
                llm_engine.register()
            elif self._provider == "openai":
                if not _ONETWO_OPENAI_AVAILABLE:
                    raise ValueError(
                        "OneTwo OpenAI backend not available. "
                        "Please use Gemini or update onetwo."
                    )
                # Reasoning / GPT-5 models only accept temperature=1.0 (their
                # API default). Closest analog to the 0.0 we use elsewhere.
                openai_temp = 1.0 if model_name.startswith(("o1", "o3", "o4", "gpt-5")) else 0.0
                print(f"[PHIA:configure] Registering OpenAI backend: model={model_name}, temp={openai_temp}")
                llm_engine = openai_api.OpenAIAPI(
                    api_key=api_key,
                    model_name=model_name,
                    temperature=openai_temp,
                    batch_size=1,
                )
                llm_engine.register()
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
            return self
        
        def load_data(
            self,
            summary_df: pd.DataFrame,
            activities_df: pd.DataFrame,
            profile_df: pd.DataFrame,
        ) -> "PHIABaseline":
            """Load user health data.
            
            Args:
                summary_df: Daily summary DataFrame.
                activities_df: Activities DataFrame.
                profile_df: User profile DataFrame.
                
            Returns:
                Self for method chaining.
            """
            self.summary_df = summary_df
            self.activities_df = activities_df
            self.profile_df = profile_df
            
            # Set up sandbox globals for code execution
            self._sandbox_globals = {
                "pd": pd,
                "np": np,
                "summary_df": summary_df,
                "activities_df": activities_df,
                "profile": profile_df,
            }
            
            return self
        
        def _build_agent(self) -> None:
            """Build the ReAct agent with tools and exemplars."""
            if self._sandbox_globals is None:
                raise RuntimeError("Data not loaded. Call load_data() first.")
            
            # Create tools
            python_tool = llm_tool_use.Tool(
                name=_PYTHON_TOOL_NAME,
                function=lambda code: simple_python_executor(code, self._sandbox_globals),
                description='Python interpreter. Executes code on pandas DataFrames (summary_df, activities_df, profile).',
            )
            
            search_tool = llm_tool_use.Tool(
                name=_SEARCH_TOOL_NAME,
                function=lambda query: tavily_search_func(query, api_key=self.tavily_api_key),
                description='Web search engine for finding health, wellness, and fitness information.',
            )
            
            finish_tool = llm_tool_use.Tool(
                name=_FINISH_TOOL_NAME,
                function=lambda x: x,
                description='Returns the final answer.',
            )
            
            agent_tools = [python_tool, search_tool, finish_tool]
            
            # Load exemplars
            exemplars = []
            if self.few_shot_dir and os.path.isdir(self.few_shot_dir):
                notebook_files = [
                    os.path.join(self.few_shot_dir, f)
                    for f in os.listdir(self.few_shot_dir)
                    if f.endswith('.ipynb')
                ]
                exemplars = build_exemplars_from_notebooks(
                    notebook_files,
                    verbose=self.debug_verbose,
                )
            
            # Fall back to default exemplars if none loaded
            if not exemplars:
                if self.debug_verbose:
                    print("Using default hardcoded exemplars")
                exemplars = _create_default_exemplars()
            
            # Create the agent
            self._agent = react.ReActAgent(
                exemplars=exemplars,
                environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
                    tools=agent_tools,
                ),
                max_steps=10,  # Limit to prevent runaway loops
                stop_prefix="",
            )
            
            # Set the prompt
            self._agent.prompt = react.ReActPromptJ2(
                text=PHIA_PREAMBLE + PHIA_REACT_PROMPT_TEXT
            )
        
        def respond(self, user_message: str) -> str:
            """Process a user message and return a response.
            
            Args:
                user_message: The user's query.
                
            Returns:
                The agent's response (cleaned of internal labels).
            """
            # Reset stats for this request
            if _STATS_AVAILABLE:
                request_stats.reset()
            
            # Always log key info for diagnostics
            from pha.streaming import emit_step
            print(f"\n{'─' * 60}")
            print(f"[PHIA] Model: {getattr(self, '_model_name', 'unknown')}")
            print(f"[PHIA] Provider: {getattr(self, '_provider', 'unknown')}")
            print(f"[PHIA] Query: {user_message[:80]}...")
            print(f"{'─' * 60}")
            
            # Rebuild agent for each query to avoid state accumulation
            # (following original PHIA pattern where agent is stateless)
            self._build_agent()
            
            if self.debug_verbose:
                print(f"[PHIA Baseline] Processing: {user_message[:50]}...")
            
            # Add question prefix for better responses
            full_query = QUESTION_PREFIX + user_message
            
            # Run the agent using ot.run() (following original PHIA pattern)
            try:
                emit_step("PHIA", "Running ReAct agent...")
                final_answer, final_state = ot.run(
                    self._agent(inputs=full_query, return_final_state=True)
                )
                
                # Always log what ot.run() returned
                print(f"\n[PHIA] ot.run() completed")
                print(f"[PHIA] final_answer type: {type(final_answer).__name__}")
                print(f"[PHIA] final_answer: {repr(str(final_answer))[:300]}")
                if hasattr(final_state, 'updates') and final_state.updates:
                    n_steps = len(final_state.updates)
                    print(f"[PHIA] final_state has {n_steps} steps")
                    emit_step("PHIA", f"ReAct completed with {n_steps} steps")
                    for i, step in enumerate(final_state.updates):
                        thought_preview = repr(str(step.thought))[:120] if step.thought else 'None'
                        action_preview = repr(str(step.action))[:120] if step.action else 'None'
                        obs_preview = repr(str(step.observation))[:120] if step.observation else 'None'
                        print(f"[PHIA]   Step {i}: finished={step.is_finished}")
                        print(f"[PHIA]     thought: {thought_preview}")
                        print(f"[PHIA]     action:  {action_preview}")
                        print(f"[PHIA]     observe: {obs_preview}")
                else:
                    print(f"[PHIA] final_state: {repr(final_state)[:200]}")
                
                # Extract the final answer
                emit_step("PHIA", "Extracting and cleaning response...")
                response = self._extract_answer(final_answer, final_state)
                
                # If extraction failed, try to recover from OneTwo parse errors
                _FALLBACK_INDICATORS = [
                    "Could not extract a valid answer",
                    "Response contained system prompt leakage",
                ]
                if any(indicator in response for indicator in _FALLBACK_INDICATORS):
                    emit_step("PHIA", "Extraction failed — attempting recovery...")
                    recovered = self._try_recover_from_parse_error(
                        final_answer, final_state, user_message
                    )
                    if not recovered:
                        # Try loop recovery (model repeated same tool call)
                        recovered = self._try_recover_from_loop(
                            final_state, user_message
                        )
                    if recovered:
                        response = recovered
                
                # Check for truncated responses (common with 2.5 Pro)
                # Model generates long thoughts that eat into token budget,
                # leaving the [Finish] text cut off mid-sentence
                if self._is_truncated_response(response):
                    emit_step("PHIA", "Response appears truncated — generating complete answer...")
                    print(f"[PHIA] Truncated response detected ({len(response)} chars): {repr(response[:100])}")
                    direct = self._generate_direct_answer(user_message)
                    if direct:
                        response = direct
                
                # Check for low-quality responses: vague preambles or raw data dumps
                # that passed syntactic checks but aren't actually helpful
                if not self._is_substantive_response(response, final_state):
                    emit_step("PHIA", "Response lacks substance — generating enriched answer...")
                    print(f"[PHIA] Non-substantive response detected ({len(response)} chars): {repr(response[:100])}")
                    # Collect ALL context from the ReAct steps: tool output, 
                    # search results, and model's own analysis from thoughts
                    full_context = self._collect_full_step_context(final_state)
                    enriched = self._generate_enriched_answer(
                        user_message, full_context
                    )
                    if enriched:
                        response = enriched
                    else:
                        direct = self._generate_direct_answer(user_message)
                        if direct:
                            response = direct
                    
            except Exception as e:
                print(f"\n[PHIA] *** EXCEPTION in ot.run(): {e}")
                import traceback
                traceback.print_exc()
                emit_step("PHIA", f"Error: {type(e).__name__}: {str(e)[:100]}")
                response = f"I encountered an error processing your question: {str(e)}"
            
            print(f"\n[PHIA] Final response ({len(response)} chars): {response[:200]}...")
            
            self.last_response = response
            
            # Update conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message,
            })
            self.conversation_history.append({
                "role": "assistant", 
                "content": response,
            })
            
            # Print request summary
            if _STATS_AVAILABLE:
                request_stats.print_summary()
            
            # Clean internal agent labels before returning to user
            return clean_agent_response(response)
        
        def _extract_answer(self, final_answer: Any, final_state: Any) -> str:
            """Extract a clean answer from the agent's response.
            
            Args:
                final_answer: The final answer returned by ot.run().
                final_state: The final state of the agent.
                
            Returns:
                A clean response string.
            """
            # Helper to check if text is raw search results
            def is_raw_search_output(text: str) -> bool:
                text_lower = text.lower()
                url_count = text_lower.count('url:')
                relevance_count = text_lower.count('relevance score:')
                has_search_header = 'search results:' in text_lower
                return url_count >= 2 or relevance_count >= 2 or (has_search_header and url_count >= 1)
            
            print(f"\n[PHIA:extract] === Starting _extract_answer ===")
            
            # ── Early detection of API errors (rate limits, auth, etc.) ──
            # These appear as #ERROR# strings in final_answer or observations
            error_sources = []
            if final_answer:
                error_sources.append(str(final_answer))
            if hasattr(final_state, 'updates') and final_state.updates:
                for step in final_state.updates:
                    if step.observation:
                        error_sources.append(str(step.observation))
            
            for src in error_sources:
                if '#ERROR#' in src:
                    # Extract the actual error message
                    api_error_patterns = {
                        '429': 'Rate limit exceeded',
                        'Resource exhausted': 'Rate limit exceeded', 
                        'quota': 'API quota exceeded',
                        '401': 'Authentication error',
                        '403': 'Permission denied',
                        '500': 'API server error',
                        '503': 'API service unavailable',
                    }
                    for pattern, label in api_error_patterns.items():
                        if pattern.lower() in src.lower():
                            print(f"[PHIA:extract] *** API ERROR detected: {label} ***")
                            return (
                                f"⚠️ **{label}** — the model API returned an error.\n\n"
                                f"```\n{src[len('#ERROR#:'):].strip()[:300]}\n```\n\n"
                                f"Please wait a moment and try again, or switch to a different model."
                            )
            
            # If we have a direct final answer, check if it's complete
            if final_answer:
                answer = str(final_answer)
                print(f"[PHIA:extract] Step 1 - Checking final_answer ({len(answer)} chars): {repr(answer[:200])}")
                cleaned = self._clean_react_format(answer)
                print(f"[PHIA:extract]   After _clean_react_format ({len(cleaned)} chars): {repr(cleaned[:200])}")
                # Skip if it's raw search output
                if is_raw_search_output(cleaned):
                    print(f"[PHIA:extract]   REJECTED: raw search output")
                elif cleaned and self._is_complete_response(cleaned):
                    print(f"[PHIA:extract]   ACCEPTED as complete response")
                    return cleaned
                else:
                    print(f"[PHIA:extract]   REJECTED: _is_complete_response returned False (empty={not cleaned}, len={len(cleaned)})")
                    if cleaned:
                        self._debug_completeness_check(cleaned)
                    # If the text is long (>500 chars), the model likely wrote
                    # a full answer but prefixed it with planning sentences.
                    # Try stripping the planning preamble and re-checking.
                    if cleaned and len(cleaned) > 500:
                        salvaged = self._strip_planning_preamble(cleaned)
                        if salvaged and self._is_complete_response(salvaged):
                            print(f"[PHIA:extract]   SALVAGED after stripping planning preamble ({len(salvaged)} chars)")
                            return salvaged
            else:
                print(f"[PHIA:extract] Step 1 - final_answer is falsy: {repr(final_answer)}")
            
            # Try to extract from final state
            if hasattr(final_state, 'updates') and final_state.updates:
                # Look for finished steps first (these have actual final answers)
                print(f"[PHIA:extract] Step 2 - Checking finished steps...")
                found_finished = False
                for step in reversed(final_state.updates):
                    if step.is_finished and step.observation:
                        found_finished = True
                        obs = str(step.observation)
                        print(f"[PHIA:extract]   Finished step obs ({len(obs)} chars): {repr(obs[:200])}")
                        if is_raw_search_output(obs):
                            print(f"[PHIA:extract]   REJECTED: raw search output")
                            continue
                        cleaned = self._clean_react_format(obs)
                        print(f"[PHIA:extract]   After clean ({len(cleaned)} chars): {repr(cleaned[:200])}")
                        if cleaned and self._is_complete_response(cleaned):
                            print(f"[PHIA:extract]   ACCEPTED finished step")
                            return cleaned
                        else:
                            print(f"[PHIA:extract]   REJECTED: _is_complete_response False")
                            if cleaned:
                                self._debug_completeness_check(cleaned)
                            # Same salvage logic as Step 1
                            if cleaned and len(cleaned) > 500:
                                salvaged = self._strip_planning_preamble(cleaned)
                                if salvaged and self._is_complete_response(salvaged):
                                    print(f"[PHIA:extract]   SALVAGED finished step after stripping preamble ({len(salvaged)} chars)")
                                    return salvaged
                if not found_finished:
                    print(f"[PHIA:extract]   No finished steps found")
                
                # If no finished step, look for the last substantial observation
                print(f"[PHIA:extract] Step 3 - Checking all observations...")
                for i, step in enumerate(reversed(final_state.updates)):
                    if step.observation:
                        obs = str(step.observation)
                        obs_stripped = obs.strip()
                        
                        print(f"[PHIA:extract]   Obs[rev {i}] ({len(obs)} chars): {repr(obs[:150])}")
                        
                        if is_raw_search_output(obs):
                            print(f"[PHIA:extract]     SKIP: raw search output")
                            continue
                        
                        has_react_markers = any(m in obs for m in ['[Act]:', '[Thought]:', '[Observe]:'])
                        obs_no_backticks = obs_stripped.replace('`', '')
                        is_raw_tool_call = obs_no_backticks.startswith('tool_code(') or obs_no_backticks.startswith('search(')
                        is_onetwo_error = '#ERROR#' in obs or 'Invalid syntax for positional arguments' in obs
                        is_raw_pandas = 'dtype:' in obs and 'Name:' in obs  # pandas Series repr
                        is_raw_code_output = obs_stripped.startswith('(np.') or obs_stripped.startswith('np.')  # raw numpy output
                        # Detect code content (model dumped tool call as text)
                        code_indicators = ['summary_df[', 'activities_df[', 'profile[',
                                           '.mean()', '.sum()', '.max()', '.min()',
                                           'print(f"', "print(f'", '\\n']
                        is_code_content = sum(1 for ci in code_indicators if ci in obs) >= 3
                        
                        if len(obs) <= 100:
                            print(f"[PHIA:extract]     SKIP: too short ({len(obs)} <= 100)")
                        elif 'Error' in obs:
                            print(f"[PHIA:extract]     SKIP: contains 'Error'")
                        elif has_react_markers:
                            print(f"[PHIA:extract]     SKIP: has ReAct markers")
                        elif is_raw_tool_call:
                            print(f"[PHIA:extract]     SKIP: raw tool call")
                        elif is_onetwo_error:
                            print(f"[PHIA:extract]     SKIP: OneTwo error")
                        elif is_raw_pandas:
                            print(f"[PHIA:extract]     SKIP: raw pandas output")
                        elif is_raw_code_output:
                            print(f"[PHIA:extract]     SKIP: raw code output")
                        elif is_code_content:
                            print(f"[PHIA:extract]     SKIP: code content (not natural language)")
                        elif self._is_complete_response(obs):
                            print(f"[PHIA:extract]     ACCEPTED as substantial observation")
                            return f"Based on my analysis of your data:\n\n{obs}"
                        else:
                            print(f"[PHIA:extract]     SKIP: _is_complete_response False")
                            self._debug_completeness_check(obs)
                
                # Try to use the last thought if it has useful content
                print(f"[PHIA:extract] Step 4 - Checking thoughts...")
                for step in reversed(final_state.updates):
                    if step.thought:
                        thought = str(step.thought)
                        thought_lower = thought.lower()
                        planning_indicators = ['i need to', 'i need the', 'then use search', 'then use tool_code', 'from summary_df']
                        if any(p in thought_lower for p in planning_indicators):
                            print(f"[PHIA:extract]   Thought SKIP (planning): {repr(thought[:100])}")
                            continue
                        if any(word in thought_lower for word in ['average', 'found', 'shows', 'indicates', 'data']):
                            cleaned = self._clean_react_format(thought)
                            if cleaned and len(cleaned) > 50 and self._is_complete_response(cleaned):
                                print(f"[PHIA:extract]   ACCEPTED thought: {repr(cleaned[:100])}")
                                return cleaned
                            else:
                                print(f"[PHIA:extract]   Thought REJECTED: {repr(thought[:100])}")
                        else:
                            print(f"[PHIA:extract]   Thought SKIP (no analytical words): {repr(thought[:100])}")
            else:
                print(f"[PHIA:extract] Step 2 - No updates in final_state")
            
            # Fallback — surface what went wrong
            print(f"[PHIA:extract] *** ALL EXTRACTION ATTEMPTS FAILED - returning diagnostic error ***")
            
            # Gather diagnostic info
            n_steps = 0
            step_details = []
            if final_state and hasattr(final_state, 'updates') and final_state.updates:
                n_steps = len(final_state.updates)
                for i, step in enumerate(final_state.updates):
                    finished = getattr(step, 'is_finished', '?')
                    obs_preview = str(getattr(step, 'observation', ''))[:100]
                    step_details.append(f"  Step {i}: finished={finished}, obs=`{obs_preview}`")
            
            answer_preview = str(final_answer)[:300] if final_answer else "None"
            steps_str = "\n".join(step_details) if step_details else "  (no steps)"
            
            return (
                f"⚠️ **Could not extract a valid answer** from the agent's output.\n\n"
                f"**ReAct steps executed:** {n_steps}\n"
                f"```\n{steps_str}\n```\n\n"
                f"**Raw final_answer preview:**\n```\n{answer_preview}\n```\n\n"
                f"This usually means the model's output format didn't match expectations. "
                f"Try rephrasing your question or switching models."
            )
        
        def _try_recover_from_parse_error(
            self, final_answer: Any, final_state: Any, user_message: str
        ) -> Optional[str]:
            """Recover when OneTwo can't parse the model's tool calls.
            
            Gemini Flash (and sometimes other models) outputs tool calls that
            OneTwo's parser can't handle. This commonly happens when:
            - tool_code() calls span multiple lines
            - search() calls have formatting OneTwo doesn't expect
            
            Recovery strategy:
            1. Detect the parse error in step observations
            2. Collect successful intermediate results from earlier steps
            3. Extract code/search from ALL sources and execute
            4. Generate enriched answer using all context (intermediate + new)
            5. Fall back to enriched answer with just intermediate results
            """
            # Step 1: Check if the failure was a tool call parse error
            has_parse_error = False
            if hasattr(final_state, 'updates'):
                for step in final_state.updates:
                    obs = str(step.observation) if step.observation else ''
                    # OneTwo emits different error patterns:
                    # - "#ERROR#: Invalid syntax for call: tool_code("
                    # - "#ERROR#: Didn't find \[Act\]:? or \[Finish\]:?"
                    if '#ERROR#' in obs and ('Invalid syntax' in obs or 
                                              'find' in obs.lower()):
                        has_parse_error = True
                        break
            
            if not has_parse_error:
                return None
            
            print(f"\n[PHIA:recover] Detected OneTwo parse error — attempting manual recovery")
            
            # Step 2: Collect successful intermediate results from earlier steps
            intermediate_results = []
            if hasattr(final_state, 'updates'):
                for i, step in enumerate(final_state.updates):
                    if step.observation:
                        obs = str(step.observation).strip()
                        if '#ERROR#' not in obs and 'Error executing code' not in obs and len(obs) > 10:
                            intermediate_results.append(obs)
                            print(f"[PHIA:recover] Collected intermediate result from step {i}: {obs[:100]}")
            
            # Step 3: Search ALL sources for code or search queries
            sources_to_search = []
            # final_answer
            if final_answer:
                sources_to_search.append(("final_answer", str(final_answer)))
            # Step actions and thoughts (where the code usually lives)
            if hasattr(final_state, 'updates'):
                for i, step in enumerate(final_state.updates):
                    if step.action:
                        sources_to_search.append((f"step[{i}].action", str(step.action)))
                    if step.thought:
                        sources_to_search.append((f"step[{i}].thought", str(step.thought)))
                    if step.observation:
                        sources_to_search.append((f"step[{i}].observation", str(step.observation)))
            
            # 3a: Try to extract and execute code
            code = None
            for source_name, source_text in sources_to_search:
                code = self._extract_code_from_raw_output(source_text)
                if code:
                    print(f"[PHIA:recover] Found code in {source_name} ({len(code)} chars)")
                    break
            
            if code:
                print(f"[PHIA:recover] Extracted code:\n{code[:300]}")
                
                if _STATS_AVAILABLE:
                    request_stats.record_tool("tool_code (recovered)")
                result = simple_python_executor(code, self._sandbox_globals)
                if 'Error executing code' not in result:
                    print(f"[PHIA:recover] Code executed successfully: {result[:300]}")
                    summary = self._generate_summary(user_message, result, code_context=code)
                    if summary:
                        return summary
                    return f"Based on my analysis of your data:\n\n{result}"
                else:
                    print(f"[PHIA:recover] Code execution failed: {result[:200]}")
            
            # 3b: Try to extract and execute a search query
            search_result = None
            search_query = self._extract_search_from_raw_output(sources_to_search)
            if search_query:
                print(f"[PHIA:recover] Found search query: {search_query}")
                try:
                    search_result = tavily_search_func(search_query, api_key=self.tavily_api_key)
                    if search_result and 'Error' not in search_result:
                        intermediate_results.append(f"Search results for '{search_query}':\n{search_result}")
                        print(f"[PHIA:recover] Search executed successfully ({len(search_result)} chars)")
                    else:
                        print(f"[PHIA:recover] Search returned error: {str(search_result)[:100]}")
                except Exception as search_err:
                    print(f"[PHIA:recover] Search execution failed: {search_err}")
            
            # Step 4: Generate enriched answer with all collected context
            if intermediate_results:
                context = "\n\n".join(intermediate_results)
                print(f"[PHIA:recover] Generating enriched answer with {len(intermediate_results)} intermediate results")
                enriched = self._generate_enriched_answer(user_message, context)
                if enriched:
                    return enriched
            
            # Step 5: Fall back to direct LLM answer
            print(f"[PHIA:recover] Falling back to direct LLM answer")
            return self._generate_direct_answer(user_message)
        
        def _try_recover_from_loop(
            self, final_state: Any, user_message: str
        ) -> Optional[str]:
            """Recover when the model gets stuck in a tool call loop.
            
            Some models (especially GPT-4o-mini) repeat the same tool_code call
            many times without progressing to [Finish]. If we detect this pattern,
            we can use the first successful observation to generate an answer.
            """
            if not hasattr(final_state, 'updates') or not final_state.updates:
                return None
            
            # Collect non-error observations and their associated code
            observations = []
            obs_to_code = {}  # Map observation → code that produced it
            for step in final_state.updates:
                if step.observation:
                    obs = str(step.observation)
                    if '#ERROR#' not in obs and 'Error executing code' not in obs:
                        observations.append(obs)
                        # Try to capture the code that produced this observation
                        if step.action and obs not in obs_to_code:
                            action_str = str(step.action)
                            if 'tool_code' in action_str:
                                import re
                                code_match = re.search(r"args=\(['\"](.+?)['\"],?\)", action_str, re.DOTALL)
                                if code_match:
                                    obs_to_code[obs] = code_match.group(1)
            
            if len(observations) < 2:
                return None
            
            # Check if the same observation repeats 3+ times (loop detected)
            from collections import Counter
            obs_counts = Counter(observations)
            most_common_obs, count = obs_counts.most_common(1)[0]
            
            if count < 3:
                return None
            
            print(f"\n[PHIA:recover] Detected tool call loop: same observation repeated {count}x")
            print(f"[PHIA:recover] Using repeated observation: {most_common_obs[:200]}")
            
            # Collect ALL code context from steps for better interpretation
            code_context, _ = self._collect_step_context(final_state)
            
            # Use this observation to generate a direct answer
            summary = self._generate_summary(
                user_message, most_common_obs, code_context=code_context
            )
            if summary:
                return summary
            
            # Fallback to direct LLM answer
            return self._generate_direct_answer(user_message)
        
        def _generate_summary(self, user_message: str, code_result: str, code_context: str = "") -> Optional[str]:
            """Generate a user-friendly summary of code execution results."""
            # Build a prompt that includes the code context so the LLM
            # knows what each number means (prevents hallucinated labels)
            code_section = ""
            if code_context:
                code_section = (
                    f'\nThe code that produced these results was:\n'
                    f'```python\n{code_context[:1500]}\n```\n\n'
                    f'Use the variable names and print labels from the code above '
                    f'to correctly identify what each number represents.\n'
                )
            
            prompt = (
                f'The user asked: "{user_message}"\n\n'
                f'I ran an analysis on their Fitbit health data and got these results:\n'
                f'{code_result}\n'
                f'{code_section}\n'
                f'Please provide a helpful, well-formatted response to the user\'s '
                f'question based on these results. Include context about what the '
                f'numbers mean for their health. Be conversational and supportive. '
                f'Do not mention tools, code, dataframes, or technical details.'
            )
            result = self._call_llm_direct(prompt)
            if result:
                print(f"[PHIA:recover] LLM summary generated ({len(result)} chars)")
            return result
        
        def _generate_direct_answer(self, user_message: str) -> Optional[str]:
            """Generate a direct answer using the LLM when code recovery fails.
            
            Uses the data summary (already available) to answer the question
            without going through ReAct code execution.
            """
            # Build a data context from what's available in sandbox globals
            data_context = ""
            if hasattr(self, 'summary_df') and self.summary_df is not None:
                data_context = self.summary_df.tail(14).to_string()
            elif self._sandbox_globals and 'summary_df' in self._sandbox_globals:
                data_context = self._sandbox_globals['summary_df'].tail(14).to_string()
            
            if not data_context:
                return None
            
            # Add profile info if available
            profile_context = ""
            if hasattr(self, 'profile_df') and self.profile_df is not None:
                profile_context = f"\n\nUser profile:\n{self.profile_df.to_string()}"
            
            prompt = (
                f'You are a health data analyst. The user asked: "{user_message}"\n\n'
                f'Here is a summary of their recent Fitbit health data:\n'
                f'{data_context[:3000]}{profile_context}\n\n'
                f'Please provide a helpful, well-formatted response based on this data. '
                f'Reference specific numbers from the data. Be conversational and supportive. '
                f'If the data doesn\'t contain relevant information, say so honestly.'
            )
            result = self._call_llm_direct(prompt)
            if result:
                print(f"[PHIA:recover] Direct LLM answer generated ({len(result)} chars)")
            return result
        
        def _call_llm_direct(self, prompt: str) -> Optional[str]:
            """Make a direct LLM call for recovery, routing to correct provider."""
            try:
                if self._provider == "openai":
                    return self._call_openai_direct(prompt)
                else:
                    return self._call_gemini_direct(prompt)
            except Exception as e:
                print(f"[PHIA:recover] Direct LLM call failed ({self._provider}): {e}")
                return None
        
        def _call_gemini_direct(self, prompt: str) -> Optional[str]:
            """Direct Gemini API call for recovery."""
            from google import genai
            from google.genai import types as genai_types
            
            client = genai.Client(api_key=self._api_key)
            model = getattr(self, '_model_name', 'models/gemini-3.1-flash-lite-preview')
            if not model.startswith('models/'):
                model = f'models/{model}'
            
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.7,
                    max_output_tokens=4096,
                ),
            )
            if response and response.text:
                return response.text.strip()
            return None
        
        def _call_openai_direct(self, prompt: str) -> Optional[str]:
            """Direct OpenAI API call for recovery."""
            try:
                from openai import OpenAI
            except ImportError:
                print("[PHIA:recover] openai package not available for recovery")
                return None
            
            client = OpenAI(api_key=self._api_key)
            model = getattr(self, '_model_name', 'gpt-4o')
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            if response.choices and response.choices[0].message.content:
                return response.choices[0].message.content.strip()
            return None
        
        def _extract_code_from_raw_output(self, raw_text: str) -> Optional[str]:
            """Extract Python code from raw LLM output that OneTwo couldn't parse.
            
            Handles the common Flash format:
                [Act]: tool_code(
                  code_line_1
                  code_line_2
                )
            
            Uses balanced parenthesis tracking to find the full code block.
            """
            import re
            
            # Find tool_code( in the text
            idx = raw_text.find('tool_code(')
            if idx == -1:
                return None
            
            # Track balanced parentheses to find the matching close
            start = idx + len('tool_code')
            depth = 0
            i = start
            while i < len(raw_text):
                ch = raw_text[i]
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth == 0:
                        code = raw_text[start + 1:i].strip()
                        # Remove markdown backtick wrappers if present
                        code = re.sub(r'^```(?:python)?\s*', '', code)
                        code = re.sub(r'\s*```$', '', code)
                        # Remove surrounding quotes if the whole thing is quoted
                        for q in ['"""', "'''", '"', "'"]:
                            if code.startswith(q) and code.endswith(q):
                                code = code[len(q):-len(q)]
                                break
                        return code if code else None
                i += 1
            
            # Balanced parens failed — try heuristic: grab everything between
            # tool_code( and the next ReAct section marker
            code = raw_text[start + 1:]
            for marker in ['[Observe]', '[Thought]', '[Finish]', '[Act]']:
                pos = code.find(marker)
                if pos != -1:
                    code = code[:pos]
                    break
            
            # Strip trailing ) if present 
            code = code.rstrip().rstrip(')')
            code = code.strip()
            
            # Remove markdown backtick wrappers
            code = re.sub(r'^```(?:python)?\s*', '', code)
            code = re.sub(r'\s*```$', '', code)
            
            return code if code else None
        
        def _extract_search_from_raw_output(
            self, sources: list
        ) -> Optional[str]:
            """Extract a search query from raw LLM output that OneTwo couldn't parse.
            
            Looks for search('query') patterns in all sources, and also infers
            search intent from model thoughts (e.g., "I need to search for X").
            
            Args:
                sources: List of (source_name, source_text) tuples.
                
            Returns:
                Extracted search query string, or None.
            """
            # Pattern 1: Explicit search() calls
            for source_name, source_text in sources:
                for pattern in [
                    r"""search\(['"](.*?)['"]\)""",
                    r"""search\(["'](.*?)["']\)""",
                ]:
                    match = re.search(pattern, source_text)
                    if match:
                        query = match.group(1).strip()
                        if query and len(query) > 3:
                            print(f"[PHIA:recover] Extracted search query from {source_name}: '{query}'")
                            return query
            
            # Pattern 2: Search intent from thoughts
            # "I need to search for average steps in San Francisco"
            # "Now I should search for typical step count San Francisco"
            for source_name, source_text in sources:
                text_lower = source_text.lower()
                for trigger in ['search for ', 'look up ', 'find out about ',
                                'search about ', 'look for ']:
                    idx = text_lower.find(trigger)
                    if idx != -1:
                        # Extract the rest of the sentence as search query
                        query_start = idx + len(trigger)
                        remainder = source_text[query_start:]
                        # Stop at sentence boundary
                        for end_marker in ['.', '!', '\n', ',', ' and ', ' to ']:
                            end_idx = remainder.find(end_marker)
                            if end_idx != -1 and end_idx > 3:
                                remainder = remainder[:end_idx]
                                break
                        query = remainder.strip().rstrip('.')
                        if query and 5 < len(query) < 100:
                            print(f"[PHIA:recover] Inferred search query from {source_name} thought: '{query}'")
                            return query
            
            return None
        
        def _debug_completeness_check(self, text: str):
            """Print why _is_complete_response rejects a text. For diagnostics only."""
            if not text:
                print(f"[PHIA:complete?]   → REJECTED: empty text")
                return
            text_normalized = text.lower().replace('`', '').replace('  ', ' ')
            incomplete_markers = [
                '[act]:', '[thought]:', '```tool_code', '```python',
                'i need to', "i'll",
                'i need the user', 'i need your',
                'then use search', 'then use tool_code', 'then use web',
                'from summary_df', 'from activities_df',
                'and compute the', 'and calculate the',
                ', then use', ', then search', ', then calculate',
            ]
            planning_let_me = [
                'let me check', 'let me analyze', 'let me look', 'let me calculate',
                'let me search', 'let me get', 'let me find', 'let me pull',
                'let me run', 'let me compute', 'let me first',
            ]
            planning_i_will = [
                'i will check', 'i will analyze', 'i will now', 'i will look',
                'i will search', 'i will calculate', 'i will use', 'i will first',
                'i will run', 'i will compute', 'i will get',
            ]
            all_markers = incomplete_markers + planning_let_me + planning_i_will
            for marker in all_markers:
                if marker in text_normalized:
                    idx = text_normalized.find(marker)
                    ctx_start = max(0, idx - 20)
                    ctx_end = min(len(text_normalized), idx + len(marker) + 20)
                    context = text_normalized[ctx_start:ctx_end]
                    print(f"[PHIA:complete?]   → REJECTED: found marker '{marker}' in: ...{context}...")
                    return
            if text_normalized.strip().startswith('i need'):
                print(f"[PHIA:complete?]   → REJECTED: starts with 'i need'")
                return
            text_norm_stripped = text_normalized.strip()
            if text_norm_stripped.startswith('tool_code(') or text_norm_stripped.startswith('search('):
                print(f"[PHIA:complete?]   → REJECTED: raw tool call")
                return
            code_indicators = ['summary_df[', 'activities_df[', 'profile[', 
                               '.mean()', '.sum()', '.max()', '.min()',
                               'print(f"', 'print(f\'', '\\n']
            code_indicator_count = sum(1 for ci in code_indicators if ci in text)
            if code_indicator_count >= 3:
                print(f"[PHIA:complete?]   → REJECTED: code content ({code_indicator_count} indicators)")
                return
            if 'dtype:' in text and 'Name:' in text:
                print(f"[PHIA:complete?]   → REJECTED: raw pandas output")
                return
            url_count = text_normalized.count('url:')
            relevance_count = text_normalized.count('relevance score:')
            if url_count >= 2 or relevance_count >= 2:
                print(f"[PHIA:complete?]   → REJECTED: raw search results (urls={url_count}, relevance={relevance_count})")
                return
            if '#ERROR#' in text or 'Invalid syntax for positional arguments' in text:
                print(f"[PHIA:complete?]   → REJECTED: OneTwo error")
                return
            if len(text) <= 30:
                print(f"[PHIA:complete?]   → REJECTED: too short ({len(text)} <= 30)")
                return
            print(f"[PHIA:complete?]   → ACCEPTED (no markers found, len={len(text)})")
        
        def _clean_react_format(self, text: str) -> str:
            """Remove ReAct formatting artifacts from text.
            
            Args:
                text: Text that may contain ReAct formatting.
                
            Returns:
                Cleaned text.
            """
            if not text:
                return ""
            
            text = str(text)
            
            # Remove common ReAct prefixes at start
            prefixes_to_remove = [
                '[Thought]:', '[Act]:', '[Observe]:', '[Finish]:',
                'Thought:', 'Act:', 'Observe:', 'Finish:',
            ]
            for prefix in prefixes_to_remove:
                if text.startswith(prefix):
                    text = text[len(prefix):].strip()
            
            # Remove code blocks anywhere in the text
            import re
            text = re.sub(r'```[\w]*\n.*?```', '', text, flags=re.DOTALL)
            
            # Remove inline [Act]: and everything after it (incomplete response)
            for marker in ['[Act]:', '[Thought]:', '[Observe]:']:
                if marker in text:
                    text = text.split(marker)[0].strip()
            
            # Remove JSON formatting artifacts
            if text.startswith('```json') or text.strip().startswith('{'):
                return ""
            
            # If only code blocks remain, return empty
            if text.startswith('```') and text.endswith('```'):
                return ""
            
            return text.strip()
        
        def _strip_planning_preamble(self, text: str) -> str:
            """Strip planning/meta sentences from the start of a long response.
            
            Some models (especially 2.5 Flash) write their [Finish] with a
            planning sentence prefix like:
              "I have gathered the user's key health metrics. I will now
               synthesize this information... [actual good content follows]"
            
            This method removes those leading sentences so the actual content
            can pass _is_complete_response.
            
            Args:
                text: Long text (>500 chars) that was rejected by _is_complete_response.
                
            Returns:
                The text with planning preamble stripped, or empty string if
                we can't salvage anything substantial.
            """
            import re
            
            # Planning markers to check in individual sentences
            planning_markers = [
                'i will now', 'i will synthesize', 'i will analyze', 'i will check',
                'i will use', 'i will search', 'i will calculate', 'i will first',
                'i will provide', 'i will combine', 'i will compile',
                'i have gathered', 'i have analyzed', 'i have retrieved',
                'i have already', 'i have now', 'i have collected',
                'i need to', "i'll now", "i'll synthesize", "i'll analyze",
                'let me synthesize', 'let me analyze', 'let me compile',
                'let me now', 'let me provide', 'let me combine',
                'now i can', 'now i will', 'now that i have',
            ]
            
            # Split into sentences (handle ". " and ".\n")
            sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=5)
            
            if len(sentences) <= 1:
                return ""
            
            # Find how many leading sentences are planning/meta
            skip_count = 0
            for sent in sentences[:3]:  # Check at most first 3 sentences
                sent_lower = sent.lower()
                if any(m in sent_lower for m in planning_markers):
                    skip_count += 1
                else:
                    break  # Stop at first non-planning sentence
            
            if skip_count == 0:
                return ""  # No planning preamble found
            
            # Reconstruct text without the planning sentences
            remainder = ' '.join(sentences[skip_count:]).strip()
            
            # Only return if we have substantial content left
            if len(remainder) < 200:
                return ""
            
            print(f"[PHIA:salvage] Stripped {skip_count} planning sentence(s), "
                  f"{len(remainder)} chars remaining")
            return remainder
        
        def _is_complete_response(self, text: str) -> bool:
            """Check if a response looks complete (not mid-ReAct-loop).
            
            Args:
                text: Response text to check.
                
            Returns:
                True if the response looks complete.
            """
            if not text:
                return False
            
            # Normalize text: remove backticks and extra whitespace for pattern matching
            text_normalized = text.lower().replace('`', '').replace('  ', ' ')
            
            # Signs of incomplete response - model thinking out loud
            incomplete_markers = [
                '[act]:', '[thought]:', '```tool_code', '```python',
                'i need to', "i'll",
                'i need the user',  # "I need the user's average..."
                'i need your',  # "I need your average daily steps..."
                'then use search', 'then use tool_code', 'then use web',  # Planning statements
                'from summary_df', 'from activities_df',  # Reference to dataframes in plans
                'and compute the', 'and calculate the',  # Planning continuation
                ', then use', ', then search', ', then calculate',  # Multi-step plans
            ]
            
            # "let me" is a planning phrase ("let me check", "let me analyze"),
            # but NOT when used conversationally ("just let me know", "please let me know")
            # "i will" is planning ("I will now check"), but NOT when final ("I will be happy to")
            # Handle these with more context:
            planning_let_me = [
                'let me check', 'let me analyze', 'let me look', 'let me calculate',
                'let me search', 'let me get', 'let me find', 'let me pull',
                'let me run', 'let me compute', 'let me first',
            ]
            planning_i_will = [
                'i will check', 'i will analyze', 'i will now', 'i will look',
                'i will search', 'i will calculate', 'i will use', 'i will first',
                'i will run', 'i will compute', 'i will get',
            ]
            
            for marker in incomplete_markers:
                if marker in text_normalized:
                    return False
            
            for marker in planning_let_me:
                if marker in text_normalized:
                    return False
            
            for marker in planning_i_will:
                if marker in text_normalized:
                    return False
            
            # Check for planning statements that start with "I need"
            if text_normalized.strip().startswith('i need'):
                return False
            
            # Detect raw tool calls (model sometimes outputs these directly,
            # potentially wrapped in backticks)
            # Use text_normalized which has backticks already stripped
            text_norm_stripped = text_normalized.strip()
            if text_norm_stripped.startswith('tool_code(') or text_norm_stripped.startswith('search('):
                return False
            
            # Detect responses that are predominantly code rather than natural language
            # This catches cases where the model dumps its intended tool call as text
            code_indicators = ['summary_df[', 'activities_df[', 'profile[', 
                               '.mean()', '.sum()', '.max()', '.min()',
                               'print(f"', 'print(f\'', '\\n']
            code_indicator_count = sum(1 for ci in code_indicators if ci in text)
            if code_indicator_count >= 3:
                return False
            
            # Detect raw pandas output (Series repr, dtype info)
            # e.g. "Age: 0    34\nName: age, dtype: int64"
            if 'dtype:' in text and 'Name:' in text:
                return False
            
            # Detect raw search results being returned instead of synthesized answer
            # GPT-5.2 sometimes outputs raw observations as final answer
            url_count = text_normalized.count('url:')
            relevance_count = text_normalized.count('relevance score:')
            if url_count >= 2 or relevance_count >= 2:
                # Multiple URLs or relevance scores = raw search output, not synthesis
                return False
            
            # Detect OneTwo parsing errors (happens with OpenAI + PHIA)
            if '#ERROR#' in text or 'Invalid syntax for positional arguments' in text:
                return False
            
            # A complete response should have some substance
            return len(text) > 30
        
        def _is_truncated_response(self, text: str) -> bool:
            """Detect responses that got cut off mid-generation.
            
            Common with verbose models (2.5 Pro) where long thoughts eat
            into the token budget, leaving the [Finish] text truncated.
            
            Examples of truncated responses:
            - "Here's a breakdown of what your numbers mean and some dietary tips to help you maintain and even improve it."
            - "Of course! Here are some dietary tips to help you maintain a healthy lifestyle:"
            """
            if not text or len(text) > 500:
                # Long responses are unlikely to be truncated
                return False
            
            text_stripped = text.strip()
            text_lower = text_stripped.lower()
            
            # Response ends with colon — clearly promises content that never came
            if text_stripped.endswith(':'):
                print(f"[PHIA:truncation] Detected: ends with ':'")
                return True
            
            # Short response that promises a list/breakdown but doesn't deliver
            promise_phrases = [
                "here's a breakdown", "here are some", "here is a breakdown",
                "the following", "consider the following", "these tips",
                "some tips", "some suggestions", "some recommendations",
            ]
            has_promise = any(p in text_lower for p in promise_phrases)
            
            # Check if the promise is actually fulfilled (has list items or substantial content)
            has_list = any(marker in text for marker in ['- ', '* ', '1.', '•'])
            has_paragraph = text.count('\n\n') >= 1 and len(text) > 300
            
            if has_promise and not has_list and not has_paragraph:
                print(f"[PHIA:truncation] Detected: promises content ({len(text)} chars) but no list/detail")
                return True
            
            return False
        
        def _is_substantive_response(self, text: str, final_state: Any) -> bool:
            """Check if a response has actual substance — not just a vague preamble
            or raw data dump.
            
            Catches two failure modes:
            A) Model does all analysis in [Thought] but writes a short, vague [Finish]
               like "Let's look at your data together..." with no actual numbers.
            B) Extraction falls back to raw tool_code output like 
               "Average Sleep Minutes: 417.80\\nAverage Deep Sleep: 62.38..."
               which is a data dump with no interpretation.
            
            Returns:
                True if the response is substantive enough to send to the user.
            """
            if not text:
                return True  # Let other checks handle empty
            
            # Skip this check for error messages (they have their own formatting)
            if text.startswith('⚠️') or 'Could not extract' in text:
                return True
            
            # Check if tool_code was actually used (if no tools were used,
            # a conversational-only response is fine)
            tool_code_used = False
            if hasattr(final_state, 'updates') and final_state.updates:
                for step in final_state.updates:
                    if step.action and 'tool_code' in str(step.action):
                        tool_code_used = True
                        break
            
            if not tool_code_used:
                return True  # No data was computed, conversational response is fine
            
            # Fix A: Detect vague preambles — short response with no data values
            # despite tool_code having been used to compute data
            if len(text) < 500:
                import re
                # Look for actual numeric health values (e.g., "62.17 bpm", "417 min", "7.0 hrs")
                health_numbers = re.findall(r'\d+\.?\d*\s*(?:bpm|min|hrs?|hours?|%|steps|ms)', text.lower())
                # Also check for standalone meaningful numbers (not just "1." list markers)
                data_numbers = re.findall(r'\b\d{2,}\b', text)  # Numbers with 2+ digits
                
                if not health_numbers and not data_numbers:
                    print(f"[PHIA:quality] REJECTED: short response ({len(text)} chars) with "
                          f"no health data values despite tool_code usage")
                    return False
            
            # Fix B: Detect raw data dumps — mostly "Label: number" lines
            # with no conversational sentences or interpretation
            if self._is_raw_data_dump(text):
                print(f"[PHIA:quality] REJECTED: raw data dump with no interpretation")
                return False
            
            return True
        
        def _is_raw_data_dump(self, text: str) -> bool:
            """Detect responses that are just raw tool output with no interpretation.
            
            Pattern: Multiple lines of "Label: number" with no sentence structure.
            Example: "Average Sleep Minutes: 417.80\\nAverage Deep Sleep: 62.38..."
            """
            lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
            if len(lines) < 3:
                return False
            
            import re
            # Count lines that are "Label: number" format
            data_lines = 0
            sentence_lines = 0
            for line in lines:
                # Skip the "Based on my analysis:" prefix we add
                if line.startswith('Based on'):
                    continue
                # "Label: number" pattern
                if re.match(r'^[\w\s]+:\s*[\d.]+', line):
                    data_lines += 1
                # Actual sentences (has period, question mark, or exclamation, and >40 chars)
                elif len(line) > 40 and any(p in line for p in ['. ', '? ', '! ']):
                    sentence_lines += 1
            
            # If mostly data lines with no interpretation sentences, it's a dump
            if data_lines >= 3 and sentence_lines == 0:
                return True
            
            return False
        
        def _collect_step_context(self, final_state: Any) -> tuple:
            """Collect code and data from ReAct steps for recovery context.
            
            Returns:
                (code_context, data_context): 
                    code_context: The code that was executed (with variable names)
                    data_context: The output/observations from tool calls
            """
            code_snippets = []
            data_snippets = []
            
            if not hasattr(final_state, 'updates') or not final_state.updates:
                return "", ""
            
            for step in final_state.updates:
                # Extract code from actions
                if step.action:
                    action_str = str(step.action)
                    if 'tool_code' in action_str:
                        # Try to extract the code argument
                        import re
                        # Match args=('code_here',) or args=("code_here",)
                        code_match = re.search(r"args=\(['\"](.+?)['\"],?\)", action_str, re.DOTALL)
                        if code_match:
                            code_snippets.append(code_match.group(1))
                
                # Extract observations (non-error tool outputs)
                if step.observation:
                    obs = str(step.observation)
                    if ('#ERROR#' not in obs and 'Search Results' not in obs 
                            and len(obs) > 10):
                        data_snippets.append(obs)
            
            # Deduplicate data snippets (loop recovery case)
            seen = set()
            unique_data = []
            for d in data_snippets:
                if d not in seen:
                    seen.add(d)
                    unique_data.append(d)
            
            code_context = '\n'.join(code_snippets[:3])  # Max 3 code blocks
            data_context = '\n'.join(unique_data[:3])  # Max 3 unique outputs
            
            return code_context, data_context
        
        def _collect_full_step_context(self, final_state: Any) -> str:
            """Collect ALL useful context from ReAct steps for rich recovery.
            
            Unlike _collect_step_context (which returns just code + data),
            this collects thoughts, search results, and tool output — everything
            the model already reasoned about — so the recovery LLM can write
            a comprehensive answer without re-doing the work.
            
            Returns:
                A formatted string with all step context.
            """
            sections = []
            
            if not hasattr(final_state, 'updates') or not final_state.updates:
                return ""
            
            for i, step in enumerate(final_state.updates):
                step_parts = []
                
                # Thought: the model's analysis (often contains the real insight)
                if step.thought:
                    thought = str(step.thought).strip()
                    # Skip pure planning thoughts
                    thought_lower = thought.lower()
                    if not any(p in thought_lower for p in [
                        'i need to', 'i should first', 'let me first'
                    ]):
                        step_parts.append(f"Analysis: {thought}")
                
                # Observation: tool output or search results
                if step.observation:
                    obs = str(step.observation).strip()
                    if '#ERROR#' not in obs and len(obs) > 10:
                        if 'Search Results' in obs:
                            step_parts.append(f"Search findings: {obs[:800]}")
                        else:
                            step_parts.append(f"Data: {obs[:500]}")
                
                if step_parts:
                    sections.append(f"Step {i+1}:\n" + "\n".join(step_parts))
            
            return "\n\n".join(sections)
        
        def _generate_enriched_answer(
            self, user_message: str, full_context: str
        ) -> Optional[str]:
            """Generate a comprehensive answer using all available step context.
            
            Unlike _generate_summary (which only gets raw numbers), this gets
            the model's own analysis, search results, and data — everything
            needed to write a rich response without losing context.
            """
            if not full_context:
                return None
            
            prompt = (
                f'You are a friendly health data analyst. The user asked: "{user_message}"\n\n'
                f'Here is what was found by analyzing their Fitbit health data:\n\n'
                f'{full_context[:4000]}\n\n'
                f'Please provide a helpful, comprehensive response to the user\'s question '
                f'based on ALL of the above findings. Include:\n'
                f'- Specific numbers from the data analysis\n'
                f'- Context about what those numbers mean for their health\n'
                f'- Relevant information from the search results\n'
                f'- Actionable suggestions\n'
                f'Be conversational, supportive, and thorough. '
                f'Do not mention tools, code, dataframes, or technical details.'
            )
            result = self._call_llm_direct(prompt)
            if result:
                print(f"[PHIA:recover] Enriched answer generated ({len(result)} chars)")
            return result
        
        def reset_conversation(self) -> None:
            """Reset the conversation state."""
            self.conversation_history = []
            self.last_response = ""
            # Rebuild agent to clear any state
            self._agent = None


else:
    # Fallback class when onetwo is not available
    class PHIABaseline:
        """PHIA Baseline placeholder when onetwo is not available."""
        
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"PHIABaseline requires onetwo which is not available. "
                f"Error: {_ONETWO_IMPORT_ERROR}"
            )


# =============================================================================
# Factory Function
# =============================================================================

def create_phia_baseline(
    api_key: Optional[str] = None,
    provider: str = "gemini",
    tavily_api_key: Optional[str] = None,
    settings: Optional[Any] = None,
    few_shot_dir: Optional[str] = None,
    debug_verbose: bool = False,
    model_name: Optional[str] = None,
    # Legacy parameter
    gemini_api_key: Optional[str] = None,
) -> PHIABaseline:
    """Create a fully configured PHIABaseline.
    
    Args:
        api_key: API key for the selected provider.
        provider: "gemini" or "openai". Default is "gemini".
        tavily_api_key: Optional Tavily API key for web search.
        settings: Optional Settings object for data paths.
        few_shot_dir: Directory containing few-shot notebook examples.
        debug_verbose: If True, print debug information.
        model_name: Optional model name. Defaults vary by provider:
            - Gemini: "models/gemini-2.5-pro" (recommended)
            - OpenAI: "gpt-4o"
        gemini_api_key: DEPRECATED - use api_key instead.
        
    Returns:
        Configured PHIABaseline instance.
        
    Example:
        ```python
        # With Gemini
        baseline = create_phia_baseline(
            api_key="your-key",
            provider="gemini",
        )
        
        # With OpenAI
        baseline = create_phia_baseline(
            api_key="your-key",
            provider="openai",
        )
        
        response = baseline.respond("How is my sleep?")
        ```
    """
    if not _ONETWO_AVAILABLE:
        raise ImportError(
            f"PHIABaseline requires onetwo. Error: {_ONETWO_IMPORT_ERROR}"
        )
    
    from config.settings import Settings
    from ..utils import load_persona
    
    # Use provided settings or create default
    if settings is None:
        settings = Settings()
    
    # Load data
    summary_df, activities_df, profile_df, _ = load_persona(settings=settings)
    
    # Create and configure baseline
    baseline = PHIABaseline(
        tavily_api_key=tavily_api_key,
        few_shot_dir=few_shot_dir,
        debug_verbose=debug_verbose,
    )
    
    # Handle both old and new API
    # Determine provider from api_key or legacy gemini_api_key
    effective_api_key = api_key if api_key else gemini_api_key
    effective_provider = provider if api_key else "gemini"
    
    # Configure with model (use default if not specified)
    baseline.configure(
        api_key=effective_api_key,
        provider=effective_provider,
        model_name=model_name,
    )
    
    baseline.load_data(
        summary_df=summary_df,
        activities_df=activities_df,
        profile_df=profile_df,
    )
    
    return baseline


def is_onetwo_openai_available() -> bool:
    """Check if OneTwo OpenAI backend is available."""
    return _ONETWO_OPENAI_AVAILABLE