r"""CLI entry point for running FHIR Agent Bench evaluations.

Supports comma-separated values for ``--strategy``, ``--dataset``, and
``--model`` flags, producing all combinations.
Each combination composes a config from YAML fragments, instantiates the
strategy, and runs the full evaluate-judge-save pipeline.

Runtime env (Vertex AI vs public GenAI, output directory) comes from CLI
flags rather than YAML fragments — pass ``--use_vertex --project-id ...
--location ...`` to use Vertex AI. Public Gemini runs use
``GENAI_API_KEYS``; OpenAI runs use ``OPENAI_API_KEYS``.

Example:
  python scripts/run_evals.py \\
    --strategy ontology_guided_retrieval,prefiltered \\
    --dataset fhiragentbench \\
    --model gemini_pro_3_1 \\
    --max_rows 10
"""

from collections.abc import Sequence
import itertools
import json
import logging
import os
import pathlib
import subprocess
import time

from absl import app
from absl import flags
from absl import logging as absl_logging

from fhir_retrieval_bench import config
from fhir_retrieval_bench.eval import runner
from fhir_retrieval_bench.utils import api
from fhir_retrieval_bench.utils import logging as logging_utils


def _expand_grid(
    datasets: Sequence[str],
    strategy_names: Sequence[str],
    models: Sequence[str],
) -> list[config.ExperimentConfig]:
  """Expand the strategy x dataset x model Cartesian grid into configs.

  Logs each (strategy, dataset, model) triple so the multiplicative
  expansion from comma-separated CLI flags is visible rather than
  implicit. The same ``env`` is attached to every produced config.

  Args:
    datasets: Sequence of dataset names.
    strategy_names: Sequence of strategy names.
    models: Sequence of model names.

  Returns:
    List of ExperimentConfig objects.
  """
  logging.info(
      "Expanding grid: %d dataset(s) x %d strategy(ies) x %d model(s) = %d"
      " run(s)",
      len(datasets),
      len(strategy_names),
      len(models),
      len(strategy_names) * len(datasets) * len(models),
  )
  configs: list[config.ExperimentConfig] = []
  for d, s, m in itertools.product(datasets, strategy_names, models):
    logging.info("  grid: dataset=%s strategy=%s model=%s", d, s, m)
    configs.append(
        config.load_composed_config(
            dataset=d,
            strategy=s,
            model=m,
            config_dir=_CONFIG_DIR.value,
        )
    )
  return configs


_STRATEGY = flags.DEFINE_string(
    "strategy",
    None,
    "Strategy config fragment name(s), comma-separated.",
    required=True,
)
_DATASET = flags.DEFINE_string(
    "dataset",
    None,
    "Dataset config fragment name(s), comma-separated.",
    required=True,
)
_MODEL = flags.DEFINE_string(
    "model",
    None,
    "Model config fragment name(s), comma-separated.",
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "/tmp/fhir_retrieval_bench_results",
    "Local directory for results and checkpoints "
    "(default: /tmp/fhir_retrieval_bench_results).",
)
_CACHE_DIR = flags.DEFINE_string(
    "cache_dir",
    "/tmp/fhir_retrieval_bench_cache",
    "Local directory for caching embeddings and other intermediate results "
    "(default: /tmp/fhir_retrieval_bench_cache).",
)
_CONFIG_DIR = flags.DEFINE_string(
    "config_dir",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    ),
    "Root config directory (default: <repository_root>/config).",
)
_NUM_WORKERS = flags.DEFINE_integer(
    "num_workers",
    1,
    "Parallel threads per evaluation (default: 10)",
)
_SHUFFLE = flags.DEFINE_bool(
    "shuffle",
    False,
    "Shuffle the order of evaluation runs (default: False).",
)
_MAX_ROWS = flags.DEFINE_integer(
    "max_rows",
    None,
    "Max records to process. None for all (default: None)",
)
_DRY_RUN = flags.DEFINE_bool(
    "dry_run",
    False,
    "Print configs without running evaluations.",
)


def main(argv: list[str]):
  """Entry point: parse args, compose configs, and run evaluations."""
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

  script_start = time.time()

  logger.info(
      "Starting | strategy=%s dataset=%s model=%s | output_dir=%s |"
      " num_workers=%s max_rows=%s shuffle=%s dry_run=%s",
      _STRATEGY.value,
      _DATASET.value,
      _MODEL.value,
      _OUTPUT_DIR.value,
      _NUM_WORKERS.value,
      _MAX_ROWS.value,
      _SHUFFLE.value,
      _DRY_RUN.value,
  )

  strategy_names = [s.strip() for s in _STRATEGY.value.split(",")]
  datasets = [d.strip() for d in _DATASET.value.split(",")]
  models = [m.strip() for m in _MODEL.value.split(",")]

  configs = _expand_grid(datasets, strategy_names, models)
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
  ).shuffled()

  if _DRY_RUN.value:
    logger.info("DRY RUN — printing configs without execution")
    for i, c in enumerate(configs):
      print(f"--- Config {i + 1}/{len(configs)} ---")
      print(c.model_dump_json(indent=2))
    logger.info("Dry run complete. %d config(s) printed.", len(configs))
    return

  output_dir = pathlib.Path(_OUTPUT_DIR.value)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  cache_dir = pathlib.Path(_CACHE_DIR.value)
  if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

  for i, experiment_config in enumerate(configs):
    run_start = time.time()
    logger.info(
        "Run %d/%d: %s / %s / %s",
        i + 1,
        len(configs),
        experiment_config.strategy.name,
        experiment_config.dataset.name,
        experiment_config.models.answer.model,
    )
    logger.info("Starting evaluation...")
    summary = runner.run_experiment(
        experiment_config,
        creds=creds,
        shuffle=_SHUFFLE.value,
        max_rows=_MAX_ROWS.value,
        num_workers=_NUM_WORKERS.value,
        output_dir=output_dir,
        cache_dir=cache_dir,
    )
    print(f"RUN_EVALS_SUMMARY: {json.dumps(summary)}")
    run_elapsed = time.time() - run_start
    logger.info("Run %d/%d finished in %.1fs", i + 1, len(configs), run_elapsed)

  total_elapsed = time.time() - script_start
  logger.info(
      "All %d run(s) complete. Total time: %.1fs", len(configs), total_elapsed
  )
  # Set execute permissions for all files in the output directory.
  chmod_command = ["/usr/bin/fileutil", "chmod", "755", "-R", _OUTPUT_DIR.value]
  results = subprocess.run(
      chmod_command,
      check=True,
      capture_output=True,
      text=True,
  )
  print(
      "Changing permissions to 755 for all files in the output directory: ",
      " ".join(chmod_command),
      " | stdout: ",
      results.stdout,
  )


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
