# tests/test_engine.py
"""Tests for the ADK NeuroSymbolicAgent."""

import datetime
import unittest
from unittest import mock

from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.config import config as config_module
from ns_agent_adk import engine as engine_module


class TestNeuroSymbolicAgent(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.config = config_module.Config()
    self.graph = graph_module.ChronologicalHypergraph()
    self.config.gcp_project_id = "test_project"

  @mock.patch(
      "ns_agent_adk.engine.runners.Runner.run"
  )
  @mock.patch.object(config_module.Config, "get_embedder", autospec=True)
  @mock.patch.object(config_module.Config, "get_llm_model", autospec=True)
  @mock.patch.object(
      engine_module, "generate_budgeted_linearization", autospec=True
  )
  @mock.patch.object(engine_module, "_load_preamble", autospec=True)
  def test_agent_execution(
      self,
      mock_load_preamble,
      mock_linearize,
      mock_get_llm,
      mock_get_embedder,
      mock_runner_run,
  ):
    """Tests the basic agent execution flow."""
    mock_load_preamble.return_value = "System preamble"
    mock_get_embedder.return_value = mock.MagicMock()
    mock_get_llm.return_value = "test_model_name"
    mock_linearize.return_value = "Sample skeleton view"

    mock_event = mock.MagicMock()
    mock_event.author = "NeuroSymbolicAgent"
    mock_event.content.parts = [mock.MagicMock(text="Final answer", thought=False)]
    mock_runner_run.return_value = [mock_event]

    agent = engine_module.NeuroSymbolicAgent(
        config=self.config, graph=self.graph
    )
    query = "test query"
    result, tracing = agent.execute(query)

    self.assertEqual(result, "Final answer")
    self.assertEqual(len(tracing), 1)
    mock_linearize.assert_called_once()
    mock_runner_run.assert_called_once()

  def test_load_preamble_real(self):
    """Tests that _load_preamble can actually find and render the template."""
    # This verifies that the template file is correctly included in data dependencies
    # and accessible via the relative path logic.
    preamble = engine_module._load_preamble(["2023-01-01"], datetime.datetime(2024, 5, 21, tzinfo=datetime.timezone.utc), 15)
    self.assertIn("2023-01-01", preamble)
    self.assertIn("2024-05-21", preamble)
    self.assertIn("Clinical Graph Navigator", preamble)


if __name__ == "__main__":
  unittest.main()
