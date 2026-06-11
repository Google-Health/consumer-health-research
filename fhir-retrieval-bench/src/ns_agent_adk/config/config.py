"""Configuration for the FHIR Neuro-Symbolic ADK Agent."""

import dataclasses
import os
from google.adk.models import lite_llm
from google.adk.models import google_llm
from google.adk.models import anthropic_llm
from typing import Any, Optional, Literal
import functools
from google.genai import client as genai_client
from google.genai import types
from ns_agent_adk.utils import embeddings as embeddings_module
from fhir_retrieval_bench.utils import api

class GeminiWithCreds(google_llm.Gemini):
  """Gemini model that accepts credential details."""

  api_key: Optional[str] = None
  gcp_project_id: Optional[str] = None
  gcp_location: Optional[str] = None

  @functools.cached_property
  def api_client(self) -> genai_client.Client:
    """Returns the GenAI client with the configured API key."""
    if self.api_key:
      return genai_client.Client(
          api_key=self.api_key,
          http_options=types.HttpOptions(
              headers=self._tracking_headers(),
              retry_options=self.retry_options,
              base_url=self.base_url,
          ),
      )
    else:
      return genai_client.Client(
          vertexai=True,
          project=self.gcp_project_id,
          location=self.gcp_location,
          http_options=types.HttpOptions(
              headers=self._tracking_headers(),
              retry_options=self.retry_options,
              base_url=self.base_url,
          ),
      )


@dataclasses.dataclass
class Config:
  """Configuration settings for the library.

  This class holds configuration for various components of the agent, including
  the embedding model, the LLM, and the linearizer. It supports using
  either the public GenAI API (requires API keys) or Vertex AI (requires
  GCP project and location information).

  Attributes:
    embedding_use_vertex_ai: If True, use Vertex AI for embeddings. Otherwise,
      use the public GenAI API.
    embedding_api_key: Comma-separated list of GenAI API keys for embedding.
      Used if embedding_use_vertex_ai is False. Example: "key1,key2,key3" When
      you have multiple API keys, the library will rotate between them.
    embedding_gcp_project_and_locations: Comma-separated list of
      gcp_project_id:gcp_location pairs for Vertex AI embedding. Used if
        embedding_use_vertex_ai is True. Example: "project1:us-central1" When
        you have multiple project_id:location pairs, the library will rotate
        between them.
    embedding_model_name: The name of the embedding model to use.
    llm_use_vertex_ai: If True, use Vertex AI for the LLM. Otherwise, use the
      public GenAI API.
    llm_api_key: Comma-separated list of GenAI API keys for the LLM. Used if
      llm_use_vertex_ai is False. Example: "key1,key2,key3" Note: Only the first
      API key is used. Key rotation is not supported yet.
    llm_gcp_project_and_locations: Comma-separated list of
      gcp_project_id:gcp_location pairs for Vertex AI LLM. Used if
        llm_use_vertex_ai is True. Example: "project1:us-central1" Note: Only
        the first project_id:location pair is used. Rotation is not supported
        yet.
    llm_model_name: The name of the LLM to use.
    llm_temperature: Sampling temperature for the LLM. If None, uses model
      default.
    llm_thinking_config: Optional configuration for thinking budget (int) or
      level (str).
    temporal_parser_model_name: The name of the model to use for temporal
      parsing.
    saliency_threshold: Threshold for saliency-based linearization.
    max_linearization_tokens: Maximum number of tokens for linearized output.
    recency_weight: Weight for recency in saliency scoring.
    always_include_types: List of FHIR resource types to always include in
      linearization.
    max_agent_steps: Maximum number of steps the agent can take.
  """

  # Embedder settings
  embedding_use_vertex_ai: bool = False
  embedding_api_key: str | None = os.getenv(
      "GENAI_API_KEY"
  )  
  embedding_gcp_project_and_locations: str | None = None
  embedding_model_name: str = "gemini-embedding-001"

  # LLM Client settings
  llm_use_vertex_ai: bool = False
  llm_api_key: str | None = os.getenv(
      "GENAI_API_KEY"
  )  
  llm_gcp_project_and_locations: str | None = None
  llm_model_name: str = "gemini-3-flash-preview"
  llm_temperature: float | None = None
  llm_thinking_config: int | str | None = None
  temporal_parser_model_name: str = "gemini-2.5-flash"

  # Linearizer settings
  saliency_threshold: float = 0.5
  max_linearization_tokens: int = 4000 # Be careful with increasing this since it can lead to TPM quota issues
  recency_weight: float = 0.5
  always_include_types: list[str] = dataclasses.field(
      default_factory=lambda: ["Condition", "AllergyIntolerance", "Patient"]
  )
  linearization_strategy: Literal["greedy", "chronological"] = "greedy"

  # Agent settings
  max_agent_steps: int = 15
  enabled_tools: list[
      Literal[
          "inspect_node",
          "search_graph",
          "follow_links",
          "filter_graph_by_time",
      ]
  ] = dataclasses.field(
      default_factory=lambda: [
          "inspect_node",
          "search_graph",
          "follow_links",
          "filter_graph_by_time",
      ]
  )

  # Cached instances
  _embedder: Any = dataclasses.field(default=None, init=False)

  def get_embedder(self):
    """Gets the configured embedder instance.

    If embedding_use_vertex_ai is True, it configures the embedder to use
    Vertex AI, requiring embedding_gcp_project_and_locations to be set.
    Otherwise, it uses the public GenAI API, requiring embedding_api_key.

    Returns:
      An instance of GenAIEmbedder.
    """
    if self._embedder is None:
      if self.embedding_use_vertex_ai:
        if not self.embedding_gcp_project_and_locations:
          raise ValueError(
              "Embedding on Vertex AI requires project_id and location."
          )
        self.api_backend = api.GenAIBackend(
            use_vertex_ai=True,
            gcp_project_and_locations=[
                tuple(project_location.split(":"))
                for project_location in (
                    self.embedding_gcp_project_and_locations.split(",")
                )
            ]
            if self.embedding_gcp_project_and_locations
            else None,
        )
      else:
        if not self.embedding_api_key:
          raise ValueError("Embedding on public GenAI requires api_key.")
        self.api_backend = api.GenAIBackend(
            genai_api_keys=self.embedding_api_key.split(",")
        )
      self._embedder = embeddings_module.GenAIEmbedder(
          backend=self.api_backend,
          model_name=self.embedding_model_name,
      )
    return self._embedder

  def get_llm_model(self):
    """Gets the configured ADK LLM model instance.

    If llm_model_name starts with 'gemini', it configures a Gemini model.
    If llm_use_vertex_ai is True, it uses Vertex AI with
    llm_gcp_project_and_locations. Otherwise, it uses the public GenAI API
    with llm_api_key.

    If llm_model_name starts with 'claude', it configures an Anthropic Claude
    model via Vertex AI, requiring llm_gcp_project_and_locations.

    If llm_model_name starts with 'openai', it configures an OpenAI model
    via LiteLLM, requiring llm_api_key.

    Returns:
      An instance of an ADK LLM model (e.g., GeminiWithCreds, Claude,
      LiteLlm).
    """
    # if self.llm_api_key:
    #   os.environ["GEMINI_API_KEY"] = self.llm_api_key
    if self.llm_model_name.startswith("gemini"):
      if self.llm_use_vertex_ai:
        if not self.llm_gcp_project_and_locations:
          raise ValueError(
              "LLM on Vertex AI requires project_id and location."
          )
        return GeminiWithCreds(
            model=self.llm_model_name,
            gcp_project_id=self.llm_gcp_project_and_locations.split(",")[0].split(":")[0],
            gcp_location=self.llm_gcp_project_and_locations.split(",")[0].split(":")[1],
        )
      else:
        if not self.llm_api_key:
          raise ValueError("LLM on public GenAI requires api_key.")
        return GeminiWithCreds(
            model=self.llm_model_name, api_key=self.llm_api_key.split(",")[0]
        )
    elif self.llm_model_name.startswith("claude"):
      # Uses ADK's native Claude class which wraps AsyncAnthropicVertex.
      # Requires GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars.
      if not self.llm_gcp_project_and_locations:
        raise ValueError(
            "LLM on Vertex AI requires project_id and location."
        )
      gcp_project_and_locations = [
          tuple(project_location.split(":"))
          for project_location in self.llm_gcp_project_and_locations.split(",")
      ] if self.llm_gcp_project_and_locations else None
      os.environ["GOOGLE_CLOUD_PROJECT"] = gcp_project_and_locations[0][0]
      os.environ["GOOGLE_CLOUD_LOCATION"] = gcp_project_and_locations[0][1]
      return anthropic_llm.Claude(model=self.llm_model_name)
    elif self.llm_model_name.startswith("openai"):
      return lite_llm.LiteLlm(
          model=self.llm_model_name, api_key=self.llm_api_key
      )
    else:
      raise ValueError(f"Unsupported LLM model: {self.llm_model_name}")


# Default configuration
CONFIG = Config()
