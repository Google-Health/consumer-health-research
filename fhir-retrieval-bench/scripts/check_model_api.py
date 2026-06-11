r"""Smoke-test Gemini API connectivity with a dummy generate + embed call.

Use this before launching a full run to verify:
  - GENAI_API_KEYS is set and each key is live
  - the configured answer/judge/embedding model names are reachable

Example:
  python scripts/check_model_api.py --model gemini_pro_3_1
"""

import logging
import os
from typing import Any

from absl import app
from absl import flags
from absl import logging as absl_logging
import yaml

from fhir_retrieval_bench import config
from fhir_retrieval_bench.utils import api
from fhir_retrieval_bench.utils import logging as logging_utils


DUMMY_PROMPT = "What is the capital of France?"
DUMMY_EMBED_TEXT = "hello world"

_MODEL = flags.DEFINE_string(
    "model",
    "gemini_pro_3_1",
    "Model config fragment name under config/model/ (default: gemini_pro_3_1).",
)
_CONFIG_DIR = flags.DEFINE_string(
    "config_dir",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    ),
    "Root config directory (default: <repository_root>/config).",
)


def _load_model_config(model_name: str, config_dir: str) -> config.ModelConfig:

  """Load a model config fragment from YAML."""
  model_path = os.path.join(config_dir, "model", f"{model_name}.yaml")
  with open(model_path) as f:
    model_frag = yaml.safe_load(f)
  return config.ModelConfig(**model_frag["models"])


def _test_single_backend_generate(
    backend: api.BaseBackend,
    model_config: config.LLMConfig,
    prompt: str,
) -> bool:
  """Test a single backend against a dummy prompt."""
  resp = backend.generate(model_config, prompt)
  if resp is None:
    print(f"response is None for model {model_config.model}")
    return False
  else:
    print(f"{prompt} -> {resp}")
    return True


def _test_single_backend_embed(
    backend: api.BaseBackend,
    model_config: config.EmbeddingConfig,
    text: str,
) -> bool:
  """Test a single backend against a dummy text for embedding."""
  if not hasattr(backend, "embed"):
    print(f"Backend for {model_config.model} does not support embed().")
    return False
  embedding = backend.embed(model_config.model, text)
  if embedding is None:
    print(f"embedding is None for model {model_config.model}")
    return False
  else:
    print(f"embedding length: {len(embedding)}")
    return True


def main(argv: list[str]) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logger = logging.getLogger()
  suppressors = [
      logging_utils.LogSuppressor(filename="caching.py", message_regex=None),
      logging_utils.LogSuppressor(filename="operation.cc", message_regex=None),
  ]
  for suppressor in suppressors:
    logger.addFilter(suppressor)
  absl_logging.set_verbosity(absl_logging.INFO)

  genai_api_keys = api.get_genai_api_keys()
  openai_api_keys = api.get_openai_api_keys()
  gcp_env = os.getenv("GCP_PROJECT_LOCATIONS", "")
  if gcp_env:
    gcp_project_and_locations = [
        tuple(project_location.split(":"))
        for project_location in gcp_env.split(",")
    ]
  else:
    gcp_project_and_locations = None

  creds = api.Credentials.from_lists(
      genai_api_keys=genai_api_keys,
      openai_api_keys=openai_api_keys,
      gcp_project_and_locations=gcp_project_and_locations,
  )

  models_cfg = _load_model_config(_MODEL.value, _CONFIG_DIR.value)
  print(
      f"Resolved models | answer={models_cfg.answer.model} judge={models_cfg.judge.model} embedding={models_cfg.embedding.model}"
  )


  for model_type, model_cfg in [
      ("answer", models_cfg.answer),
      ("judge", models_cfg.judge),
      ("embedding", models_cfg.embedding)
  ]:
    print(model_type, model_cfg)
    model_name = getattr(model_cfg, "model")
    backend_type = getattr(model_cfg, "backend")
    backend = api.get_api_client(model_name, creds, backend_type)

    if model_type == "answer" or model_type == "judge":
      try:
        _test_single_backend_generate(
            backend,
            model_cfg,
            DUMMY_PROMPT,
        )
      except Exception as e:  # pylint: disable=broad-except
        print(f"Generate check failed: {e}")
    elif model_type == "embedding":
      try:
        _test_single_backend_embed(
            backend,
            model_cfg,
            DUMMY_EMBED_TEXT,
        )
      except Exception as e:  # pylint: disable=broad-except
        print(f"Embedding check failed: {e}")
    else:
      raise ValueError(f"Unsupported model type: {model_type}. Cannot test.")


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
