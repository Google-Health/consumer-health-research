"""Tests for parsing utilities."""

import pytest

from pha.utils.parsing import parse_code_output


def clear_json_format(text: str) -> str:
    """Helper to strip JSON markdown - mirrors orchestrator's method."""
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class TestParseCodeOutput:
    """Tests for extracting Python code from LLM responses."""
    
    def test_extracts_from_generic_code_block(self):
        """Extract code from ``` ... ``` blocks without language specifier."""
        llm_output = '''```
x = 1 + 1
print(x)
```'''
        
        code = parse_code_output(llm_output)
        
        # Should contain the code (may have wrapper added)
        assert "x = 1 + 1" in code or "1 + 1" in code
    
    def test_handles_raw_code_without_markers(self):
        """Handle code that isn't wrapped in markdown."""
        llm_output = '''import pandas as pd
df = pd.read_csv("data.csv")
print(df.head())'''
        
        code = parse_code_output(llm_output)
        
        assert "pandas" in code
    
    def test_returns_string(self):
        """Output is always a string."""
        code = parse_code_output("x = 1")
        assert isinstance(code, str)
    
    def test_strips_python_marker(self):
        """Strips ```python marker."""
        llm_output = "```python\nx = 1\n```"
        code = parse_code_output(llm_output)
        assert "```python" not in code


class TestClearJsonFormat:
    """Tests for cleaning JSON from LLM responses."""
    
    def test_removes_json_markdown_markers(self):
        """Remove ```json ... ``` markers."""
        llm_output = '''```json
{"key": "value", "count": 42}
```'''
        
        result = clear_json_format(llm_output)
        
        assert result.strip() == '{"key": "value", "count": 42}'
        assert "```" not in result
    
    def test_removes_generic_code_markers(self):
        """Remove ``` ... ``` markers without json specifier."""
        llm_output = '''```
{"decision": "YES"}
```'''
        
        result = clear_json_format(llm_output)
        
        assert "decision" in result
        assert "```" not in result
    
    def test_handles_clean_json(self):
        """Handle JSON that's already clean."""
        llm_output = '{"main_agent": "health_coach"}'
        
        result = clear_json_format(llm_output)
        
        assert result == '{"main_agent": "health_coach"}'
    
    def test_result_is_valid_json(self):
        """Result can be parsed as JSON."""
        import json
        
        llm_output = '''```json
{
    "decision": "NO",
    "reflection_questions": {}
}
```'''
        
        result = clear_json_format(llm_output)
        parsed = json.loads(result)
        
        assert parsed["decision"] == "NO"
        assert parsed["reflection_questions"] == {}
    
    def test_handles_nested_json(self):
        """Handle nested JSON structures."""
        llm_output = '''```json
{
    "main_agent": "health_coach_agent",
    "supporting_agents": ["data_science_agent"],
    "workflow": "sequential"
}
```'''
        
        result = clear_json_format(llm_output)
        
        import json
        parsed = json.loads(result)
        assert parsed["main_agent"] == "health_coach_agent"
        assert "data_science_agent" in parsed["supporting_agents"]
