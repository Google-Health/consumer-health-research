"""Unified API gateway for Google GenAI, AnthropicVertex, and LiteLLM backends.

Callers pass a Credentials bundle to get_api_client and get back the single
backend they need. Retries and rotation live in _retry.
Fun names like `ThrottleSmith`, `LLMux`,  `LimitlessAI`
"""

from abc import ABC
from abc import abstractmethod
import dataclasses
import os
import random
import time
from typing import Any, Callable, Literal

from absl import logging
from anthropic.lib.vertex import AnthropicVertex
from google import genai
from google.genai import types
from immutabledict import immutabledict
import litellm
import numpy as np
import tqdm

from fhir_retrieval_bench import config

# ---------------------------------------------------------------------------
# model info
# ---------------------------------------------------------------------------

# Per-model context-window sizes (input-token limits). Register every model here.
_MODEL_CONTEXT_LIMITS: immutabledict[str, int] = immutabledict({
    # Gemini 3 series (Google)
    "gemini-2.5-flash": 1_048_576,
    "gemini-3-flash-preview": 1_048_576,
    "gemini-3.1-pro-preview": 1_048_576,
    "gemini-3.1-flash-lite-preview": 1_048_576,
    "gemini-3.1-flash": 1_048_576,
    "gemini-3-pro": 1_048_576,
    # Anthropic (Claude 4.6 Series)
    "claude-sonnet-4-6": 1_000_000,
    "claude-opus-4-6": 1_048_576,
    # OpenAI (GPT-5 family)
    "gpt-5": 1_048_576,
    "gpt-5-mini": 272_000,
    "gpt-5.4-mini": 272_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
})


def get_context_limit(model_name: str) -> int:
  """Return the input context-window size for *model_name*.

  Raises :class:`ValueError` if the model is not registered in
  :data:`_MODEL_CONTEXT_LIMITS`; fail fast rather than guess by model_maker.
  """
  model_name = model_name.replace("openai/", "")

  if model_name not in _MODEL_CONTEXT_LIMITS:
    raise ValueError(
        f"Unknown context limit for model {model_name!r}; "
        "register it in _MODEL_CONTEXT_LIMITS in utils/api.py."
    )
  return _MODEL_CONTEXT_LIMITS[model_name]


def model_maker_for(model_name: str) -> str:
  """Classify a model string into a supported model_maker route.

  Args:
      model_name: The name of the model to classify.

  Returns:
      A string representing the model_maker route (e.g., 'anthropic', 'openai',
      'google').
  """
  if model_name.startswith("claude"):
    return "anthropic"
  elif model_name.startswith("openai") or model_name.startswith("gpt-"):
    return "openai"
  elif model_name.startswith("google") or model_name.startswith("gemini"):
    return "google"
  else:
    raise ValueError(
        f"Unrecognized model name {model_name!r}; can't determine model_maker."
    )


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def _env_keys(env_var: str) -> list[str]:
  raw = os.getenv(env_var, "")
  keys = [k.strip() for k in raw.split(",") if k.strip()]
  logging.info("Loaded %d API key(s) from %s", len(keys), env_var)
  return keys


def get_genai_api_keys() -> list[str]:
  """Read Gemini API keys from ``GENAI_API_KEYS``."""
  return _env_keys("GENAI_API_KEYS")


def get_openai_api_keys() -> list[str]:
  """Read OpenAI API keys from ``OPENAI_API_KEYS``."""
  return _env_keys("OPENAI_API_KEYS")


@dataclasses.dataclass(frozen=True)
class Credentials:
  """model_maker credentials bundle passed through the pipeline.

  Carries every key/config that any backend might need; :func:`get_api_client`
  picks out only the fields relevant to the chosen backend.
  """

  genai_api_keys: tuple[str, ...] = dataclasses.field(default_factory=tuple)
  openai_api_keys: tuple[str, ...] = dataclasses.field(default_factory=tuple)
  gcp_project_and_locations: tuple[tuple[str, str], ...] = dataclasses.field(
      default_factory=tuple
  )

  @classmethod
  def from_lists(
      cls,
      genai_api_keys: list[str] | None = None,
      openai_api_keys: list[str] | None = None,
      gcp_project_and_locations: list[tuple[str, str]] | None = None,
  ) -> "Credentials":
    """Build a :class:`Credentials` from plain lists."""
    return cls(
        genai_api_keys=tuple(genai_api_keys or ()),
        openai_api_keys=tuple(openai_api_keys or ()),
        gcp_project_and_locations=tuple(gcp_project_and_locations or ()),
    )

  def shuffled(self) -> "Credentials":
    shuffled_genai = list(self.genai_api_keys)
    random.shuffle(shuffled_genai)
    shuffled_openai = list(self.openai_api_keys)
    random.shuffle(shuffled_openai)
    shuffled_gcp = list(self.gcp_project_and_locations)
    random.shuffle(shuffled_gcp)
    return Credentials(
        genai_api_keys=tuple(shuffled_genai),
        openai_api_keys=tuple(shuffled_openai),
        gcp_project_and_locations=tuple(shuffled_gcp),
    )


# ---------------------------------------------------------------------------
# Key rotation + retry
# ---------------------------------------------------------------------------


class ApiCallExhausted(Exception):
  """Raised when a retry loop exhausts all attempts.

  Backends catch this at the public API seam and return ``None`` so that
  ``generate`` / ``embed`` preserve their "``None`` on failure" contract.
  """


class KeyRing:
  """Rotating pool of API keys; fires ``on_rotate`` after each rotation."""

  def __init__(
      self, keys: list[str], on_rotate: Callable[[str], None] | None = None
  ):
    if not keys:
      raise ValueError("KeyRing requires at least one key.")
    self._keys = list(keys)
    self._idx = 0
    self._on_rotate = on_rotate

  def __len__(self) -> int:
    return len(self._keys)

  def current(self) -> str:
    return self._keys[self._idx]

  def rotate(self) -> None:
    self._idx = (self._idx + 1) % len(self._keys)
    if self._on_rotate is not None:
      self._on_rotate(self.current())


def _retry(
    fn: Callable,
    *args: Any,
    keyring: KeyRing | None = None,
    max_rounds: int = 5,
    base_sleep: float = 5.0,
) -> Any:
  """Call ``fn(*args)`` with exponential backoff and optional key rotation.

  With ``keyring`` each round tries every key once, rotating after each
  failure. Without it, one attempt per round. Raises
  :class:`ApiCallExhausted` when all rounds fail.
  """
  attempts_per_round = len(keyring) if keyring is not None else 1
  fn_name = getattr(fn, "__name__", repr(fn))
  last_exc: Exception | None = None
  for round_idx in range(max_rounds):
    for attempt in range(attempts_per_round):
      try:
        start = time.time()
        result = fn(*args)
        logging.debug(
            "%s succeeded | round=%d/%d attempt=%d/%d time=%.2fs",
            fn_name,
            round_idx + 1,
            max_rounds,
            attempt + 1,
            attempts_per_round,
            time.time() - start,
        )
        return result
      except Exception as e:
        last_exc = e
        logging.warning(
            "%s failed | round=%d/%d attempt=%d/%d: %s",
            fn_name,
            round_idx + 1,
            max_rounds,
            attempt + 1,
            attempts_per_round,
            e,
        )
        if keyring is not None:
          keyring.rotate()
    if round_idx + 1 < max_rounds:
      sleep_s = min(base_sleep * (2**round_idx), 60.0)
      logging.warning(
          "Round %d/%d failed — sleeping %.1fs%s",
          round_idx + 1,
          max_rounds,
          sleep_s,
          " (capped at 60s)" if sleep_s >= 60.0 else "",
      )
      time.sleep(sleep_s)
  logging.error("%s exhausted all retries", fn_name)
  raise ApiCallExhausted(
      f"{fn_name} exhausted after {max_rounds} round(s) x {attempts_per_round}"
      " attempt(s)"
  ) from last_exc


# ---------------------------------------------------------------------------
# Per-SDK config builders (module-level, testable in isolation)
# ---------------------------------------------------------------------------


def build_generation_config(
    model_name: str,
    backend_type: str,
    temperature: float | None = None,
    max_output_tokens: int | None = None,
    thinking_config: int | str | None = None,
    extra_config: dict[str, Any] | None = None,
) -> Any:
  """Builds the backend-specific generation configuration."""
  if backend_type == "genai":
    genai_thinking_config = None
    if thinking_config is not None:
      thinking_budget_val = None
      thinking_level = None
      if isinstance(thinking_config, int):
        thinking_budget_val = thinking_config
      elif isinstance(thinking_config, str):
        thinking_level = types.ThinkingLevel(thinking_config.upper())

      genai_thinking_config = types.ThinkingConfig(
          thinking_budget=thinking_budget_val,
          thinking_level=thinking_level,
      )

    cfg_kwargs = {}
    if temperature is not None:
      cfg_kwargs["temperature"] = temperature
    if max_output_tokens is not None:
      cfg_kwargs["max_output_tokens"] = max_output_tokens
    if genai_thinking_config is not None:
      cfg_kwargs["thinking_config"] = genai_thinking_config
    if extra_config:
      cfg_kwargs.update(extra_config)

    return types.GenerateContentConfig(**cfg_kwargs)

  elif backend_type == "anthropic_vertex":
    kwargs = {}
    if temperature is not None:
      kwargs["temperature"] = temperature
    if max_output_tokens is not None:
      kwargs["max_tokens"] = max_output_tokens
    if thinking_config is not None and isinstance(thinking_config, int):
      kwargs["thinking"] = {
          "type": "enabled",
          "budget_tokens": thinking_config,
      }
    if extra_config:
      kwargs.update(extra_config)
    return kwargs

  elif backend_type == "litellm":
    kwargs = {}
    if temperature is not None:
      kwargs["temperature"] = temperature
    if max_output_tokens is not None:
      kwargs["max_tokens"] = max_output_tokens
    if thinking_config is not None and isinstance(thinking_config, str):
      kwargs["reasoning_effort"] = thinking_config.lower()
    if extra_config:
      kwargs.update(extra_config)
    return kwargs

  else:
    raise ValueError(f"Unknown backend_type: {backend_type}")


# ---------------------------------------------------------------------------
# Base backend
# ---------------------------------------------------------------------------


class BaseBackend(ABC):
  """Generate text and count tokens for a single model_maker family."""

  @abstractmethod
  def generate(
      self,
      llm_config: config.LLMConfig,
      contents: str,
      extra_config: dict[str, Any] | None = None,
  ) -> str:
    """Generate response text."""

  @abstractmethod
  def count_tokens(self, model_name: str, text: str) -> int | None:
    """Count tokens in *text*; return ``None`` on failure."""


# ---------------------------------------------------------------------------
# GenAI backend (Google GenAI API, supports vertexai flag)
# ---------------------------------------------------------------------------


class GenAIBackend(BaseBackend):
  """Google GenAI API — generate, embed, count.

  Supports key rotation if not using Vertex.
  """

  def __init__(
      self,
      genai_api_keys: list[str] | None = None,
      use_vertex_ai: bool = False,
      gcp_project_and_locations: list[tuple[str, str]] | None = None,
  ):
    self.use_vertex_ai = use_vertex_ai
    self.gcp_project_and_locations = gcp_project_and_locations

    if self.use_vertex_ai:
      if gcp_project_and_locations is None:
        raise ValueError(
            "GenAIBackend requires gcp_project_and_locations when"
            " vertexai=True."
        )
      self._keyring = KeyRing(
          [
              f"{project_id}:{location}"
              for project_id, location in gcp_project_and_locations
          ],
          on_rotate=self._reset_client,
      )
      project_id, location = self._keyring.current().split(":")
      self._client = genai.Client(
          vertexai=True, project=project_id, location=location
      )
    else:
      if not genai_api_keys:
        raise ValueError(
            "GenAIBackend requires genai_api_keys when vertexai=False."
        )
      self._keyring = KeyRing(genai_api_keys, on_rotate=self._reset_client)
      self._client = genai.Client(
          vertexai=False, api_key=self._keyring.current()
      )

  def _reset_client(self, api_key: str) -> None:
    if self.use_vertex_ai:
      project_id, location = api_key.split(":")
      self._client = genai.Client(
          vertexai=True, project=project_id, location=location
      )
    else:
      self._client = genai.Client(vertexai=False, api_key=api_key)

  def generate(
      self,
      llm_config: config.LLMConfig,
      contents: str,
      extra_config: dict[str, Any] | None = None,
  ) -> str:
    cfg = build_generation_config(
        model_name=llm_config.model,
        backend_type="genai",
        temperature=llm_config.temperature,
        max_output_tokens=llm_config.max_output_tokens,
        thinking_config=llm_config.thinking_config,
        extra_config=extra_config,
    )

    return _retry(
        self._generate, llm_config.model, contents, cfg, keyring=self._keyring
    )

  def _generate(
      self,
      model: str,
      contents: str,
      cfg: types.GenerateContentConfig | None,
  ) -> str:
    response = self._client.models.generate_content(
        model=model,
        contents=contents,
        config=cfg,
    )
    return response.text

  def count_tokens(self, model_name: str, text: str) -> int | None:
    return self._client.models.count_tokens(
        model=model_name, contents=text
    ).total_tokens

  def embed(
      self,
      model_name: str,
      text: str,
      task_type: (
          Literal[
              "RETRIEVAL_QUERY",
              "RETRIEVAL_DOCUMENT",
              "SEMANTIC_SIMILARITY",
              "CLASSIFICATION",
              "CLUSTERING",
          ]
          | None
      ) = "SEMANTIC_SIMILARITY",
  ) -> np.ndarray:

    return _retry(
        self._embed, model_name, text, task_type, keyring=self._keyring
    )[0]

  def embed_batch(
      self,
      model_name: str,
      texts: list[str],
      task_type: (
          Literal[
              "RETRIEVAL_QUERY",
              "RETRIEVAL_DOCUMENT",
              "SEMANTIC_SIMILARITY",
              "CLASSIFICATION",
              "CLUSTERING",
          ]
          | None
      ) = "SEMANTIC_SIMILARITY",
      batch_size: int = 100,
  ) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    with tqdm.tqdm(
        total=len(texts), desc="Embedding texts", unit="text"
    ) as pbar:
      for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        out.extend(
            _retry(
                self._embed, model_name, batch, task_type, keyring=self._keyring
            )
        )
        pbar.update(len(batch))
    return out

  def _embed(
      self,
      model_name: str,
      contents: str | list[str],
      task_type: (
          Literal[
              "RETRIEVAL_QUERY",
              "RETRIEVAL_DOCUMENT",
              "SEMANTIC_SIMILARITY",
              "CLASSIFICATION",
              "CLUSTERING",
          ]
          | None
      ) = "SEMANTIC_SIMILARITY",
  ) -> list[np.ndarray]:
    """Embed text using Google GenAI API.

    Args:
      model_name: The model name to use for embedding.
      contents: The text to embed.
      task_type: The task type to use for embedding.

    Returns:
      A list of numpy arrays, each representing an embedding.
    """
    result = self._client.models.embed_content(
        model=model_name,
        contents=contents,
        config={"task_type": task_type} if task_type else None,
    )
    return [np.array(e.values) for e in result.embeddings]


# ---------------------------------------------------------------------------
# Anthropic Vertex backend (Claude via AnthropicVertex)
# ---------------------------------------------------------------------------


class AnthropicVertexBackend(BaseBackend):
  """Claude via AnthropicVertex.

  Reference: Anthropic Vertex AI Model Garden.
  """

  def __init__(self, gcp_project_and_locations: list[tuple[str, str]]):
    if gcp_project_and_locations is None:
      raise ValueError(
          "AnthropicVertexBackend requires project_id and location."
      )
    self._keyring = KeyRing(
        [
            f"{project_id}:{location}"
            for project_id, location in gcp_project_and_locations
        ],
        on_rotate=self._reset_client,
    )
    project_id, location = self._keyring.current().split(":")
    self._client = AnthropicVertex(region=location, project_id=project_id)

  def _reset_client(self, api_key: str) -> None:
    project_id, location = api_key.split(":")
    self._client = AnthropicVertex(region=location, project_id=project_id)

  def generate(
      self,
      llm_config: config.LLMConfig,
      contents: str,
      extra_config: dict[str, Any] | None = None,
  ) -> str:
    kwargs = build_generation_config(
        model_name=llm_config.model,
        backend_type="anthropic_vertex",
        temperature=llm_config.temperature,
        max_output_tokens=llm_config.max_output_tokens,
        thinking_config=llm_config.thinking_config,
        extra_config=extra_config,
    )

    return _retry(self._generate, llm_config.model, contents, kwargs)

  def _generate(
      self,
      model: str,
      contents: str,
      kwargs: dict[str, Any],
  ) -> str:
    """Generate response text via Anthropic Vertex API."""
    system_prompt = kwargs.pop("system", "")
    response = self._client.messages.create(
        model=model,
        messages=[{"role": "user", "content": contents}],
        system=system_prompt,
        **kwargs,
    )
    return response.content[0].text  # pylint: disable=attribute-error

  def count_tokens(self, model_name: str, text: str) -> int | None:
    response = self._client.messages.count_tokens(
        model=model_name, messages=[{"role": "user", "content": text}]
    )
    return response.input_tokens


# ---------------------------------------------------------------------------
# LiteLLM backend (OpenAI-compatible, rotating keys)
# ---------------------------------------------------------------------------


class LiteLLMBackend(BaseBackend):
  """LiteLLM-compatible backend with rotating keys."""

  def __init__(self, api_keys: list[str]):
    self._keyring = KeyRing(api_keys)

  def generate(
      self,
      llm_config: config.LLMConfig,
      contents: str,
      extra_config: dict[str, Any] | None = None,
  ) -> str:

    return _retry(
        self._generate,
        llm_config,
        contents,
        extra_config or {},
        keyring=self._keyring,
    )

  def _generate(
      self,
      llm_config: config.LLMConfig,
      contents: str,
      extra_config: dict[str, Any],
  ) -> str:
    """Generate response text via LiteLLM API for OpenAI-compatible models."""

    kwargs = build_generation_config(
        model_name=llm_config.model,
        backend_type="litellm",
        temperature=llm_config.temperature,
        max_output_tokens=llm_config.max_output_tokens,
        thinking_config=llm_config.thinking_config,
        extra_config=extra_config,
    )
    kwargs["model"] = llm_config.model
    kwargs["messages"] = [{"role": "user", "content": contents}]
    kwargs["api_key"] = self._keyring.current()

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content or ""

  def count_tokens(self, model_name: str, text: str) -> int | None:
    return litellm.token_counter(model=model_name, text=text)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_api_client(
    model_name: str, creds: Credentials, backend: str
) -> BaseBackend:
  """Return the backend that ``model_name`` should use under ``creds``."""
  is_vertex = backend == "vertex"
  model_maker = model_maker_for(model_name)
  if model_maker == "anthropic":
    if not is_vertex:
      raise RuntimeError(
          "Claude models are only supported via Vertex AI in this build. "
          'Please specify backend: "vertex" in your config fragment.'
      )
    if not creds.gcp_project_and_locations:
      raise RuntimeError(
          "Claude via Vertex AI requires GCP_PROJECT_LOCATIONS environment"
          " variable to be set."
      )
    return AnthropicVertexBackend(list(creds.gcp_project_and_locations))
  elif model_maker == "openai":
    if not creds.openai_api_keys:
      raise RuntimeError(
          f"OpenAI model {model_name!r} requires openai_api_keys."
      )
    return LiteLLMBackend(list(creds.openai_api_keys))
  elif model_maker == "google":
    if is_vertex:
      if not creds.gcp_project_and_locations:
        raise RuntimeError(
            "Gemini on Vertex AI requires GCP_PROJECT_LOCATIONS environment"
            " variable to be set."
        )
      return GenAIBackend(
          use_vertex_ai=True,
          gcp_project_and_locations=list(creds.gcp_project_and_locations),
      )
    if not creds.genai_api_keys:
      raise RuntimeError(
          f"Gemini model {model_name!r} on public GenAI requires api_keys."
      )
    return GenAIBackend(
        genai_api_keys=list(creds.genai_api_keys),
        use_vertex_ai=False,
    )
  else:
    raise ValueError(
        f"Unsupported model_maker {model_maker!r} for model {model_name!r}"
    )
