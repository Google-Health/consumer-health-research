"""Shared request statistics tracking for debugging.

This module provides a global stats tracker that can be used across
all agent and tool modules to aggregate request-level statistics.
"""

from datetime import datetime
from typing import Dict, List


class RequestStats:
    """Track statistics for a single request."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset stats for a new request."""
        self.start_time = datetime.now()
        self.agents_called: List[str] = []
        self.tools_used: Dict[str, int] = {}  # tool_name -> count
        self.errors = 0
    
    def record_agent(self, agent_name: str):
        """Record an agent call."""
        # Clean up agent name
        clean_name = agent_name.replace(" (MAIN)", "").replace(" (supporting)", "")
        if clean_name not in self.agents_called:
            self.agents_called.append(clean_name)
    
    def record_tool(self, tool_name: str, had_error: bool = False):
        """Record a tool use."""
        self.tools_used[tool_name] = self.tools_used.get(tool_name, 0) + 1
        if had_error:
            self.errors += 1
    
    def get_duration(self) -> float:
        """Get duration in seconds."""
        return (datetime.now() - self.start_time).total_seconds()
    
    def print_summary(self):
        """Print end-of-request summary."""
        duration = self.get_duration()
        agents = ", ".join(self.agents_called) or "(none)"
        
        tools_str = ", ".join(
            f"{name} ×{count}" for name, count in self.tools_used.items()
        ) or "(none)"
        
        print(f"\n{'═' * 60}")
        print(f"📊 REQUEST SUMMARY ({duration:.1f}s)")
        print(f"   Agents called: {agents}")
        print(f"   Tools used: {tools_str}")
        print(f"   Errors: {self.errors}")
        print(f"{'═' * 60}\n")


# Global stats tracker (reset at start of each request)
request_stats = RequestStats()
