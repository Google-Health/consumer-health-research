"""Core ReACT Agent Logic with Tavily Search Integration (Simplified).
"""

from typing import Sequence, Optional
import pandas as pd
import numpy as np
import io
import contextlib
import os
from tavily import TavilyClient

from onetwo import ot
from onetwo.agents import react
from onetwo.stdlib.tool_use import llm_tool_use
from onetwo.stdlib.tool_use import python_tool_use

from prompt_templates import build_exemplars, AGENT_PREAMBLE, PHIA_REACT_PROMPT_TEXT

# For now, synchronous only tavily search functionality
def tavily_search_func(query: str, api_key: Optional[str] = None) -> str:
    """
    Performs web search using Tavily API.
    
    Args:
        query: The search query string
        api_key: Tavily API key (if not provided, will look for TAVILY_API_KEY env var)
    
    Returns:
        Formatted search results as a string
    """
    try:
        if api_key is None:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                return "Error: Tavily API key not found. Please set the key when creating the agent or set the TAVILY_API_KEY environment variable."

        client = TavilyClient(api_key=api_key)
        
        # Perform search with appropriate parameters for health/wellness queries
        search_results = client.search(
            query=query,
            max_results=5,  # Higher value returns more raw search results
            include_answer=False,  # Corresponds to AI-generated answer summary based on results
            include_raw_content=False,  # Set to True if you need full page content
            search_depth="advanced"  # Can be "basic" or "advanced", controls algos used for search
        )

        formatted_results = []
        
        # Add the AI-generated answer if available
        if search_results.get('answer'):
            formatted_results.append(f"Summary: {search_results['answer']}\n")
        
        # Add individual search results
        if search_results.get('results'):
            formatted_results.append("Search Results:")
            for i, result in enumerate(search_results['results'], 1):
                formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
                formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
                formatted_results.append(f"   Content: {result.get('content', 'No content available')[:200]}...")
                if result.get('score'):
                    formatted_results.append(f"   Relevance Score: {result['score']:.2f}")
        
        return "\n".join(formatted_results) if formatted_results else "No search results found."
        
    except Exception as e:
        return f"Error performing search: {str(e)}"

# For code generation, create a simple python executor using exec()
def simple_python_executor(code: str, globals: dict) -> str:
    """Executes python code and captures the output."""
    local_scope = {}
    # Use a string buffer to capture any `print()` statements
    string_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(string_buffer):
            # Split code into lines to handle the last expression separately
            code_lines = code.strip().split('\n')
            
            # Check if the last line is likely an expression (not an assignment or statement)
            last_line = code_lines[-1].strip() if code_lines else ""
            
            # Determine if last line is an expression that should be evaluated
            is_expression = (
                last_line and 
                not last_line.startswith(('import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally', 'elif ', 'else:')) and
                '=' not in last_line.split('#')[0] and  # No assignment (ignoring comments)
                not last_line.startswith('#')  # Not a comment
            )
            
            if is_expression and len(code_lines) > 1:
                # Execute all but the last line
                exec('\n'.join(code_lines[:-1]), globals, local_scope)
                # Evaluate the last line and capture its value
                result = eval(last_line, {**globals, **local_scope}, local_scope)
                
                # Get any printed output
                printed_output = string_buffer.getvalue()
                
                # If there was printed output, use it; otherwise use the result
                if printed_output:
                    output = printed_output
                else:
                    output = str(result)
            else:
                # Execute the entire code block
                exec(code, globals, local_scope)
                printed_output = string_buffer.getvalue()
                
                # Check if the last line stored a value in local_scope
                if not printed_output and is_expression:
                    # Try to evaluate the last line as an expression
                    try:
                        result = eval(last_line, {**globals, **local_scope}, local_scope)
                        output = str(result)
                    except:
                        output = printed_output or "Code executed successfully."
                else:
                    output = printed_output or "Code executed successfully."
        
        return output
        
    except Exception as e:
        return f"Error executing code: {e}"

def get_react_agent(
    summary_df: pd.DataFrame,
    activities_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    example_files: Sequence[str],
    tavily_api_key: Optional[str] = None,  # Add optional API key parameter
    use_mock_search: bool = False,  # Option to fall back to mock search
):
    """
    Initializes and returns an open-source compatible ReAct agent.
    
    Args:
        summary_df: Summary dataframe with user activity and sleep data
        activities_df: Activities dataframe with exercise data
        profile_df: User profile dataframe
        example_files: List of example files for few-shot learning
        tavily_api_key: Optional Tavily API key (if not provided, uses env var)
        use_mock_search: If True, uses mock search instead of Tavily
    """

    # Define the globals dictionary for our code executor
    sandbox_globals = {
        "pd": pd,
        "np": np,
        "summary_df": summary_df,
        "activities_df": activities_df,
        "profile": profile_df,
    }

    # Python tool uses our custom simple_python_executor
    python_tool = llm_tool_use.Tool(
        name='tool_code',
        function=lambda code: simple_python_executor(code, globals=sandbox_globals),
        description='Python interpreter. Executes code on pandas DataFrames (summary_df, activities_df, profile).',
    )

    # Search tool - either Tavily or mock based on configuration
    if use_mock_search:
        print("Using mock search (set use_mock_search=False to use Tavily)")
        search_tool = llm_tool_use.Tool(
            name="search",
            function=lambda query: "Web search is not available in mock mode.",
            description="Mock search engine."
        )
    else:
        # Use Tavily search with the provided API key
        search_tool = llm_tool_use.Tool(
            name="search",
            function=lambda query: tavily_search_func(query, api_key=tavily_api_key),
            description="Web search engine for finding health, wellness, and fitness information. Returns relevant articles and summaries."
        )
    
    finish_tool = llm_tool_use.Tool(
        name="finish", 
        function=lambda x: x, 
        description="Returns the final answer."
    )
    
    agent_tools = [python_tool, search_tool, finish_tool]

    # Load exemplars
    examples = build_exemplars(example_files)

    # Create the agent specifically with PythonToolUseEnvironmentConfig
    agent = react.ReActAgent(
        exemplars=examples,
        environment_config=python_tool_use.PythonToolUseEnvironmentConfig(
            tools=agent_tools,
        ),
        max_steps=10,
        stop_prefix="",
    )
    
    agent.prompt = react.ReActPromptJ2(
        text=AGENT_PREAMBLE + PHIA_REACT_PROMPT_TEXT
    )
    return agent

QUESTION_PREFIX = (
    "Use tools such as tool_code to execute Python code and search to find "
    "external relevant information as needed. Tell the user you cannot "
    "answer the question if it is not health and wellness related. "
    "Take into account that questions may have typos or grammatical "
    "mistakes and try to reinterpret them before answering. Follow all "
    "instructions and the ReAct template carefully. Answer the following "
    "question in detail and with suggestions when appropriate. Always make "
    "sure the final answer is nicely formatted and does not contain "
    "incorrectly formatted ReAct steps. Remember to use user data whenever "
    "relevant to enhance your answers! Question: "
)
