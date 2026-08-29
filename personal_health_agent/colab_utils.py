"""Utilities for Colab.

There are some cases where we need to be able to visualize or annotate
agent outputs in notebooks.
"""

import html
from onetwo.agents import react
from onetwo.stdlib.tool_use import llm_tool_use
from typing import get_origin

def format_react_state_html(state: react.ReActState) -> str:
    """Formats the onetwo.agents.react.ReActState into HTML."""
    if not isinstance(state, get_origin(react.ReActState) or react.ReActState):
        return "<p>Invalid state type for formatting.</p>"

    inputs = state.inputs
    question = inputs
    
    html_output = f'<p><span style="font-weight: bold; color: #DAA520;">[Question]:</span> {html.escape(str(question))}</p>'
    html_output += "<b>--- Reasoning ---</b>"

    for i, step in enumerate(state.updates):
        html_output += f"<br><hr><b>Step {i + 1}:</b><br>"
        thought = step.thought or ""
        html_output += f'<span style="font-weight: bold; color: #1E90FF;">[Thought]:</span> {html.escape(thought)}<br>'

        if step.action:
            action = step.action
            if isinstance(action, llm_tool_use.FunctionCall):
                function_name = action.function_name
                html_output += f'<span style="font-weight: bold; color: #FF4500;">[Act]:</span> {html.escape(function_name)}<br>'

                action_args = action.args
                if action_args:
                    if function_name == "tool_code":
                        code_content = html.escape(str(action_args[0]))
                        code_style = "display: block; background-color: #F8F8F8; color: #000000; font-family: 'Courier New', Courier, monospace; border: 1px solid #E0E0E0; border-radius: 5px; padding: 10px; margin: 5px 0; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word;"
                        html_output += f'<div style="{code_style}">{code_content}</div>'
                    else:
                        args_str = ', '.join(html.escape(str(arg)) for arg in action_args)
                        html_output += f"Args: ({args_str})<br>"
                if action.kwargs:
                    kwargs_str = ', '.join(f"{k}={html.escape(str(v))}" for k, v in action.kwargs.items())
                    html_output += f"Kwargs: {{{kwargs_str}}}<br>"
            else:
                html_output += f'<span style="font-weight: bold; color: #FF4500;">[Act]:</span> {html.escape(str(action))}<br>'

        observation = step.observation or ""
        obs_html = html.escape(str(observation))

        obs_box_style = "border: 1px solid #888; border-radius: 5px; padding: 10px; margin-top: 5px;"
        
        html_output += f'<span style="font-weight: bold; color: #2E8B57;">[Observe]:</span><div style="{obs_box_style}">{obs_html}</div>'

        if step.is_finished:
            break
            
    return html_output
