"""Tests for Python sandbox execution.

NOTE: The sandbox is designed to work with LLM-generated code that follows
the `def analysis()` pattern. These tests verify the basic infrastructure.
"""

import pytest
import pandas as pd

from pha.tools.python_sandbox import (
    get_python_sandbox,
    PhdaPythonRequest,
    PhdaPythonReply,
    ExecutionStatus,
    ExecutionResult,
)


@pytest.fixture
def sample_dfs():
    """Create sample DataFrames for sandbox testing."""
    summary_df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "steps": [8000, 10000, 7500],
        "sleep_minutes": [420, 480, 390],
        "resting_heart_rate": [62, 60, 65],
    })
    
    activities_df = pd.DataFrame({
        "activity_name": ["Running", "Walking", "Cycling"],
        "duration_minutes": [30, 45, 60],
        "calories": [300, 150, 400],
    })
    
    profile_df = pd.DataFrame({
        "age": [35],
        "gender": ["male"],
        "weight": [70],
        "height": [175],
    })
    
    population_df = pd.DataFrame({
        "metric": ["steps", "sleep"],
        "p25": [5000, 360],
        "p50": [7500, 420],
        "p75": [10000, 480],
    })
    
    return {
        "summary_df": summary_df,
        "activities_df": activities_df,
        "profile_df": profile_df,
        "population_df": population_df,
    }


@pytest.fixture
def sandbox(sample_dfs):
    """Create a configured sandbox."""
    return get_python_sandbox(dfs=sample_dfs)


class TestPythonSandboxBasics:
    """Basic sandbox infrastructure tests."""
    
    def test_sandbox_creation(self, sample_dfs):
        """Sandbox can be created with DataFrames."""
        sandbox = get_python_sandbox(dfs=sample_dfs)
        assert sandbox is not None
    
    def test_request_creation(self):
        """PhdaPythonRequest can be created."""
        request = PhdaPythonRequest(
            request="x = 1",
            include_traceback_for_errors=True,
        )
        assert request.request == "x = 1"
    
    def test_sandbox_returns_reply(self, sandbox):
        """Sandbox returns a PhdaPythonReply."""
        request = PhdaPythonRequest(
            request="x = 1",
            include_traceback_for_errors=True,
        )
        result = sandbox.execute(request)
        assert isinstance(result, PhdaPythonReply)
    
    def test_reply_has_execution_result(self, sandbox):
        """Reply contains execution_result."""
        request = PhdaPythonRequest(
            request="x = 1",
            include_traceback_for_errors=True,
        )
        result = sandbox.execute(request)
        assert hasattr(result, 'execution_result')
    
    def test_captures_syntax_errors(self, sandbox):
        """Syntax errors are captured."""
        request = PhdaPythonRequest(
            request="def broken(",
            include_traceback_for_errors=True,
        )
        result = sandbox.execute(request)
        assert result.has_error()
    
    def test_stores_original_code(self, sandbox):
        """Original code is stored in result."""
        code = "x = 1"
        request = PhdaPythonRequest(
            request=code,
            include_traceback_for_errors=True,
        )
        result = sandbox.execute(request)
        assert result.orig_code == code


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""
    
    def test_create_success_result(self):
        """Can create a SUCCESS result."""
        result = ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            output="hello",
            stderr="",
        )
        assert result.status == ExecutionStatus.SUCCESS
        assert result.output == "hello"
    
    def test_create_error_result(self):
        """Can create an ERROR result."""
        result = ExecutionResult(
            status=ExecutionStatus.ERROR,
            output="",
            stderr="NameError: x",
        )
        assert result.status == ExecutionStatus.ERROR


class TestPhdaPythonReply:
    """Tests for PhdaPythonReply."""
    
    def test_has_error_method_exists(self, sandbox):
        """PhdaPythonReply has has_error() method."""
        request = PhdaPythonRequest(request="x", include_traceback_for_errors=True)
        result = sandbox.execute(request)
        assert hasattr(result, 'has_error')
        assert callable(result.has_error)
    
    def test_has_error_returns_bool(self, sandbox):
        """has_error() returns a boolean."""
        request = PhdaPythonRequest(request="x", include_traceback_for_errors=True)
        result = sandbox.execute(request)
        assert isinstance(result.has_error(), bool)
