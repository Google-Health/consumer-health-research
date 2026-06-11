"""NS (Neuro-Symbolic) Agent strategy.

Wraps the ns_agent_adk graph-based agent so it can be used as a strategy
in the FHIR retrieval benchmark.
"""

import copy
import os
from typing import Any, Literal

from absl import logging

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base
from fhir_retrieval_bench.utils import api
from ns_agent_adk import engine as engine_module
from ns_agent_adk.config import config as ns_agent_config_module
from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.core import linearizer as linearizer_module


class NSAgentStrategy(base.Strategy):
  """Retrieves relevant resources via keyword-based scoring as context."""

  def __init__(
      self,
      creds: api.Credentials,
      answer_config: config.LLMConfig,
      embedding_config: config.EmbeddingConfig,
      cache_dir: str | None = None,
      saliency_threshold: float | None = None,
      recency_weight: float | None = None,
      max_linearization_tokens: int | None = None,
      enabled_tools: (
          list[
              Literal[
                  "inspect_node",
                  "search_graph",
                  "follow_links",
                  "filter_graph_by_time",
              ]
          ]
          | None
      ) = None,
      linearization_strategy: Literal["greedy", "chronological"] | None = None,
  ):
    self.creds = creds
    self.answer_config = answer_config
    self.embedding_config = embedding_config
    self.cache_dir = cache_dir
    self.saliency_threshold = saliency_threshold
    self.recency_weight = recency_weight
    self.max_linearization_tokens = max_linearization_tokens
    self.enabled_tools = enabled_tools
    self.linearization_strategy = linearization_strategy

    self.context_window_limit = api.get_context_limit(self.answer_config.model)

    logging.info("Initialized NSAgentStrategy")

  def process(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> tuple[str | None, dict[str, Any] | None, str | None, str, str, int]:
    logging.debug(
        "RAGStrategy.process: patient=%s | question_context=%s",
        record.patient_id,
        record.question_context,
    )
    if not fhir_utils.verify_with_pydantic(fhir_bundle):
      raise ValueError("Failed to parse FHIR bundle with pydantic validation.")
    fhir_bundle = copy.deepcopy(fhir_bundle)
    shuffled_creds = self.creds.shuffled()
    llm_use_vertex_ai = self.answer_config.backend == "vertex"
    embedding_use_vertex_ai = self.embedding_config.backend == "vertex"

    if llm_use_vertex_ai:
      llm_api_key = None
      llm_gcp_project_and_locations = ",".join([
          f"{project_id}:{location}"
          for project_id, location in shuffled_creds.gcp_project_and_locations
      ])
    else:
      if api.model_maker_for(self.answer_config.model) == "google":
        if not shuffled_creds.genai_api_keys:
          raise ValueError("No GenAI API keys.")
        llm_api_key = shuffled_creds.genai_api_keys[0]
      elif api.model_maker_for(self.answer_config.model) == "openai":
        if not shuffled_creds.openai_api_keys:
          raise ValueError("No OpenAI API keys.")
        llm_api_key = shuffled_creds.openai_api_keys[0]
      else:
        raise ValueError(
            "Unsupported model maker:"
            f" {api.model_maker_for(self.answer_config.model)}"
        )
      llm_gcp_project_and_locations = None

    if embedding_use_vertex_ai:
      embedding_gcp_project_and_locations = ",".join([
          f"{project_id}:{location}"
          for project_id, location in shuffled_creds.gcp_project_and_locations
      ])
      embedding_api_key = None
    else:
      embedding_gcp_project_and_locations = None
      embedding_api_key = ",".join(shuffled_creds.genai_api_keys)

    ns_config_kwargs = dict(
        embedding_use_vertex_ai=embedding_use_vertex_ai,
        embedding_gcp_project_and_locations=embedding_gcp_project_and_locations,
        embedding_api_key=embedding_api_key,
        embedding_model_name=self.embedding_config.model,
        llm_use_vertex_ai=llm_use_vertex_ai,
        llm_gcp_project_and_locations=llm_gcp_project_and_locations,
        llm_api_key=llm_api_key,
        llm_model_name=self.answer_config.model,
        llm_temperature=self.answer_config.temperature,
        llm_thinking_config=self.answer_config.thinking_config,
    )
    if self.saliency_threshold is not None:
      ns_config_kwargs["saliency_threshold"] = self.saliency_threshold
    if self.recency_weight is not None:
      ns_config_kwargs["recency_weight"] = self.recency_weight
    if self.max_linearization_tokens is not None:
      ns_config_kwargs["max_linearization_tokens"] = (
          self.max_linearization_tokens
      )
    if self.enabled_tools is not None:
      ns_config_kwargs["enabled_tools"] = self.enabled_tools
    if self.linearization_strategy is not None:
      ns_config_kwargs["linearization_strategy"] = self.linearization_strategy

    ns_config = ns_agent_config_module.Config(**ns_config_kwargs)

    print("⏳ Constructing the hypergraph...")
    graph = graph_module.ChronologicalHypergraph()
    graph.build_from_bundle(fhir_bundle)
    print(f"✅ Graph built with {len(graph.spine)} hypernodes.")

    print("⏳ Precomputing node embeddings...")
    embedder = ns_config.get_embedder()

    cache_path = None
    if self.cache_dir:
      cache_path = os.path.join(
          self.cache_dir,
          f"ns_agent_embeddings_{record.patient_id}.parquet",
      )

    precomputed_node_embeddings = linearizer_module.precompute_node_embeddings(
        graph,
        embedder,
        cache_path=cache_path,
    )

    if record.question_context:
      user_context = (
          (record.question_context,)
          if isinstance(record.question_context, str)
          else tuple(record.question_context)
      )
      print(
          f"Using additional context provided for the patient: {user_context}"
      )
    else:
      print("No additional context provided for the patient.")
      user_context = None

    print("⏳ Initializing ADK NeuroSymbolicAgent...")
    ns_agent = engine_module.NeuroSymbolicAgent(
        config=ns_config,
        graph=graph,
        precomputed_node_embeddings=precomputed_node_embeddings,
        anchor_date=None,
        user_context=user_context,
    )

    query = record.question_for_answering or record.question
    print(f"⏳ Asking the agent: {query}...")
    final_response, ns_agent_trace = ns_agent.execute(query)

    ns_agent_trace = engine_module.parse_events(ns_agent_trace)
    prompt_token_count = int(
        ns_agent_trace["aggregate_summary"]["context_validation"][
            "final_prompt_tokens"
        ]
    )
    if final_response:
      return (
          final_response,
          {
              "events": ns_agent_trace,
          },
          None,
          "",
          query,
          prompt_token_count,
      )
    else:
      return (
          None,
          {
              "events": ns_agent_trace,
          },
          "NS Agent did not return a response",
          "",
          query,
          prompt_token_count,
      )
