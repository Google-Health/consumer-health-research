"""Configuration loading and composition for FHIR Retrieval Bench experiments.

Fragment composition works as follows:
  1. Each CLI axis (strategy, dataset, model) maps to a YAML file under
     ``config/<axis>/<name>.yaml``.
  2. Fragments are loaded in a fixed order and deep-merged so that later
     fragments can override keys set by earlier ones.
  3. The merged dict is validated into an ``ExperimentConfig`` Pydantic model.
"""

import os
from typing import Any, Literal, Mapping

from absl import logging
import immutabledict
import pydantic
import yaml



class LLMConfig(pydantic.BaseModel):
  """Parameters for a single Gemini LLM call (answerer or judge).

  Attributes:
    model: Model identifier string (e.g. ``"gemini-2.5-pro"``).
    backend: The hosting backend, either ``"vertex"`` or ``"public"``.
    max_output_tokens: Optional cap on generated tokens.
    temperature: Sampling temperature; ``None`` uses the model default.
    thinking_config: Optional thinking config. ``None`` means "do not send any
      thinking-related parameter", so you get whatever the provider's
      server-side default is. Depending on the provider, it accepts different
      value types.
  """

  model: str
  backend: Literal["vertex", "public"]
  max_output_tokens: int | None = None
  temperature: float | None = None
  thinking_config: int | str | None = None


class EmbeddingConfig(pydantic.BaseModel):
  """Parameters for a single embedding model call.

  Attributes:
    model: Model identifier string (e.g. ``"gemini-embedding-001"``).
    backend: The hosting backend, either ``"vertex"`` or ``"public"``.
  """

  model: str
  backend: Literal["vertex", "public"]


class ModelConfig(pydantic.BaseModel):
  """LLM and embedding model configurations for each role in the evaluation pipeline.

  Attributes:
    answer: Config for the answerer model that produces predictions.
    judge: Config for the judge model that scores predictions.
    embedding: Embedding model name used by RAG strategies.
  """

  answer: LLMConfig
  judge: LLMConfig
  embedding: EmbeddingConfig


class DatasetConfig(pydantic.BaseModel):
  """Configuration for an evaluation dataset.

  Attributes:
    name: Registered dataset key (``"fhiragentbench"``, ``"ehrqa"``, or
      ``"medagentbench"``).
    qa_path: Filesystem path to the QA data.
    fhir_path: Filesystem path to the FHIR data.
  """

  name: str
  qa_path: str
  fhir_path: str


class StrategyConfig(pydantic.BaseModel):
  """Configuration for a retrieval/context strategy.

  Attributes:
    name: Registered strategy key (e.g. ``"ontology_guided_retrieval"``).
    embedding_csl_saliency_threshold: Cosine-similarity cutoff for resource
      inclusion. Only used by ``"embedding_csl"``.
    ns_agent_saliency_threshold: Cutoff applied to the per-node combined score
      (semantic * masks + recency * recency_weight + boosts) during hypergraph
      linearization. Only used by ``"ns_agent"``.
    recency_weight: Weight for the recency boost in ``"ns_agent"``'s per-node
      saliency score. Only effective for queries classified as "recent". Only
      used by ``"ns_agent"``.
    max_linearization_tokens: Maximum number of tokens to use for linearizing
      the FHIR bundle. Only used by ``"ns_agent"`` strategy.
    enabled_tools: List of tools to enable for ``ns_agent`` strategy.
    linearization_strategy: Strategy for linearizing the FHIR bundle. Only used
      by ``"ns_agent"`` strategy.
    downsampling_enable_llm_control: Whether to enable LLM control for
      downsampling. Only used by ``"flowsheet_agent"``.
    date_selection_enabled: Whether to enable date selection. Only used by
      ``"flowsheet_agent"``.
  """

  name: str
  embedding_csl_saliency_threshold: float | None = None
  ns_agent_saliency_threshold: float | None = None
  recency_weight: float | None = None
  max_linearization_tokens: int | None = None
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
  ) = None
  linearization_strategy: Literal["greedy", "chronological"] | None = None
  downsampling_enable_llm_control: bool = False
  date_selection_enabled: bool = False


class ExperimentConfig(pydantic.BaseModel):
  """Top-level config produced by composing all fragment axes.

  This is the single object passed through the entire evaluation pipeline.

  Attributes:
    dataset: Which dataset to evaluate on.
    strategy: Which strategy transforms the FHIR bundle into LLM context.
    models: LLM and embedding model settings.
  """

  dataset: DatasetConfig
  strategy: StrategyConfig
  models: ModelConfig


def _deep_merge(
    base: dict[str, Any], override: Mapping[str, Any], path: str = ""
) -> None:
  """Recursively merge *override* into *base*, mutating *base* in place.

  When both sides have a dict for the same key, the merge recurses so nested
  keys are combined.  Lists and scalars are **overwritten** (not extended):
  if a later fragment supplies a list for a key that already held a list,
  the prior list is discarded.  That case is logged at WARNING so the
  silent replacement is at least visible in the run log.

  Args:
    base: The accumulator dict that receives merged values.
    override: The dict whose keys take precedence.
    path: Dotted key path used only for log messages during recursion.
  """
  for key, value in override.items():
    key_path = f"{path}.{key}" if path else key
    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
      # Recurse; nested keys are combined.
      _deep_merge(base[key], value, path=key_path)
      continue
    if key in base and (isinstance(base[key], list) or isinstance(value, list)):
      existing_len = len(base[key]) if isinstance(base[key], list) else "n/a"
      new_len = len(value) if isinstance(value, list) else "n/a"
      logging.warning(
          "Deep merge overwriting list at %s (len %s → %s) — "
          "fragments cannot extend lists, only replace them.",
          key_path,
          existing_len,
          new_len,
      )
    base[key] = value


AXIS_DIRS = immutabledict.immutabledict({
    "strategy": "strategy",
    "dataset": "dataset",
    "model": "model",
})


def load_composed_config(
    dataset: str,
    strategy: str,
    model: str,
    config_dir: str = "config",
) -> ExperimentConfig:
  """Compose an ExperimentConfig from per-axis fragment files.

  Each axis name resolves to ``config_dir/{axis}/{name}.yaml``. Fragments
  are deep-merged in the fixed order **dataset → strategy → model** —
  later fragments overwrite keys set by earlier ones. Lists are replaced
  (not extended); see :func:`_deep_merge`.

  Args:
    dataset: The name of the dataset configuration to load.
    strategy: The name of the strategy configuration to load.
    model: The name of the model configuration to load.
    config_dir: The base directory where configuration YAML files are located.

  Returns:
    An ExperimentConfig object composed from the specified fragments.
  """
  logging.info(
      "Composing config: dataset=%s strategy=%s model=%s",
      dataset,
      strategy,
      model,
  )
  merged: dict[str, Any] = {}
  axes = {"dataset": dataset, "strategy": strategy, "model": model}
  for axis, name in axes.items():
    fragment_path = os.path.join(config_dir, AXIS_DIRS[axis], f"{name}.yaml")
    logging.debug("Loading fragment: %s", fragment_path)
    with open(fragment_path) as f:
      fragment = yaml.safe_load(f)
    _deep_merge(merged, fragment)
  result = ExperimentConfig(**merged)
  logging.info(
      "Config composed | answer_model=%s judge_model=%s embedding=%s",
      result.models.answer.model,
      result.models.judge.model,
      result.models.embedding.model,
  )
  return result
