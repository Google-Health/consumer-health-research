"""Smoke-test dataset loaders.

For each dataset fragment under ``config/dataset/``, load a small number of
records and print a summary (count, a sample question/answer, and the
number of FHIR bundle entries).  Use this before launching a full run to
verify data paths and parsing logic.

Examples:
  # check every dataset fragment in config/dataset/
  python scripts/check_dataset.py

  # check one or two specific datasets with a custom limit
  python scripts/check_dataset.py --dataset fhiragentbench,ehrqa --limit 5
"""

import logging
import os

from absl import app
from absl import flags
from absl import logging as absl_logging
import yaml

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import loader
from fhir_retrieval_bench.utils import logging as logging_utils

_DATASET = flags.DEFINE_string(
    "dataset",
    None,
    "Comma-separated dataset config fragment name(s). Default: all in"
    " config/dataset/.",
)
_LIMIT = flags.DEFINE_integer(
    "limit",
    None,
    "Records to load per dataset (default: all).",
)
_CONFIG_DIR = flags.DEFINE_string(
    "config_dir",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    ),
    "Root config directory (default: <repository_root>/config).",
)


def _available_datasets(config_dir: str) -> list[str]:
  """Return a list of all dataset fragment names in config_dir."""
  dataset_dir = os.path.join(config_dir, "dataset")
  return sorted(
      f.removesuffix(".yaml")
      for f in os.listdir(dataset_dir)
      if f.endswith(".yaml")
  )


def _load_dataset_config(name: str, config_dir: str) -> config.DatasetConfig:
  """Load a dataset config fragment from YAML."""
  path = os.path.join(config_dir, "dataset", f"{name}.yaml")
  with open(path) as f:
    frag = yaml.safe_load(f)
  return config.DatasetConfig(**frag["dataset"])


def _check_one_dataset(name: str, limit: int, config_dir: str) -> bool:
  """Check a single dataset fragment."""
  logging.info("=" * 60)
  logging.info("Dataset: %s", name)
  try:
    ds_cfg = _load_dataset_config(name, config_dir)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("  FAILED to read fragment: %s", e)
    return False

  try:
    records = loader.load_qa_pairs(ds_cfg, limit=limit)
    bundles = loader.load_fhir_bundles(ds_cfg)
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.exception("  FAILED to load records: %s", e)
    return False

  if not records:
    print("  FAILED: loader returned 0 records")
    return False

  print(f" {name}  Loaded {len(records)} record(s).")
  for record in records[:3]:
    print("=" * 60)
    print(record)

    fhir_bundle = loader.get_fhir_bundle_for_instance(record, bundles)
    if fhir_bundle is None:
      print("  FAILED: no FHIR bundle found for patient %s", record.patient_id)
      # Consider this a failure for the overall check of this dataset.
      return False
    else:
      print(f"  Bundle keys: {fhir_bundle.keys()}")
      print(f"  Bundle: {str(fhir_bundle)[:500]}...")

  return True


def main(argv: list[str]) -> None:
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  logger = logging.getLogger()
  suppressors = [
      logging_utils.LogSuppressor(
          filename="caching.py",
          message_regex=None,
      ),
      logging_utils.LogSuppressor(
          filename="operation.cc",
          message_regex=None,
      ),
  ]
  for suppressor in suppressors:
    logger.addFilter(suppressor)
  absl_logging.set_verbosity(absl_logging.INFO)

  if _DATASET.value:
    dataset_names = [d.strip() for d in _DATASET.value.split(",") if d.strip()]
  else:
    dataset_names = _available_datasets(_CONFIG_DIR.value)

  print(f"Checking {len(dataset_names)} dataset(s): {dataset_names}")

  for dataset_name in dataset_names:
    _check_one_dataset(dataset_name, _LIMIT.value, _CONFIG_DIR.value)


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
