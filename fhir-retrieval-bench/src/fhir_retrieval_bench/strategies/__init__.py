"""Strategy instantiation and dispatch."""

from absl import logging

from fhir_retrieval_bench import config
from fhir_retrieval_bench.strategies import base
from fhir_retrieval_bench.strategies import embedding_csl
from fhir_retrieval_bench.strategies import fhir2text
from fhir_retrieval_bench.strategies import flowsheet
from fhir_retrieval_bench.strategies import flowsheet_agent
from fhir_retrieval_bench.strategies import ns_agent
from fhir_retrieval_bench.strategies import ontology_guided_retrieval
from fhir_retrieval_bench.strategies import prefiltered
from fhir_retrieval_bench.utils import api


def get_strategy(
    strategy_config: config.StrategyConfig,
    model_config: config.ModelConfig,
    creds: api.Credentials,
    dataset_name: str | None = None,
    output_dir: str | None = None,
    cache_dir: str | None = None,
) -> base.Strategy:
  """Instantiate a strategy by name."""
  logging.info("get_strategy: name=%s", strategy_config.name)

  if strategy_config.name == "embedding_csl":
    return embedding_csl.EmbeddingCSLStrategy(
        creds=creds,
        answer_config=model_config.answer,
        embedding_config=model_config.embedding,
        saliency_threshold=strategy_config.embedding_csl_saliency_threshold,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
  elif strategy_config.name == "ontology_guided_retrieval":
    return ontology_guided_retrieval.OntologyGuidedRetrievalStrategy(
        creds=creds,
        answer_config=model_config.answer,
        dataset_name=dataset_name,
    )
  elif strategy_config.name == "fhir2text":
    return fhir2text.Fhir2TextStrategy(
        creds=creds,
        answer_config=model_config.answer,
    )
  elif strategy_config.name == "prefiltered":
    return prefiltered.PrefilteredStrategy(
        creds=creds,
        answer_config=model_config.answer,
        dataset_name=dataset_name,
    )
  elif strategy_config.name == "ns_agent":
    return ns_agent.NSAgentStrategy(
        creds=creds,
        answer_config=model_config.answer,
        embedding_config=model_config.embedding,
        cache_dir=cache_dir,
        saliency_threshold=strategy_config.ns_agent_saliency_threshold,
        recency_weight=strategy_config.recency_weight,
        max_linearization_tokens=strategy_config.max_linearization_tokens,
        enabled_tools=strategy_config.enabled_tools,
        linearization_strategy=strategy_config.linearization_strategy,
    )
  elif strategy_config.name == "flowsheet":
    return flowsheet.FlowsheetStrategy(
        creds=creds,
        answer_config=model_config.answer,
    )
  elif strategy_config.name == "flowsheet_agent":
    return flowsheet_agent.FlowsheetAgentStrategy(
        creds=creds,
        answer_config=model_config.answer,
        downsampling_enable_llm_control=strategy_config.downsampling_enable_llm_control,
        date_selection_enabled=strategy_config.date_selection_enabled
    )
  else:
    raise ValueError(
        f"Unknown strategy: {strategy_config.name}. Available:  embedding_csl,"
        " ontology_guided_retrieval, fhir2text, prefiltered, flowsheet,"
        " flowsheet_agent, ns_agent"
    )
