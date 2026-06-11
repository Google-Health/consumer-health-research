"""Pack per-patient FHIR bundles into the bench's parquet format.

Reads the per-patient ``$everything`` JSON bundles produced by the
upstream FHIRPath-QA pipeline (via HAPI), maps each bundle's
``patient_id_hash`` filename back to the MIMIC patient_id using the
staged QA pickle, validates each bundle, and writes a parquet with
``Patient ID`` + ``FHIR Bundle`` columns to CNS — the schema that
``data.loader.load_fhir_bundles`` (loader.py:73-82) expects.

Example:
  python scripts/fhirpathqa/build_bundles_parquet.py \
      --bundles_dir=data/fhirpathqa/patient_bundles
"""

import json
import os

from absl import app
from absl import flags
from absl import logging
import pandas as pd

from fhir_retrieval_bench.data import fhir_utils

_DEFAULT_BUNDLES_DIR = (
    "data/fhirpathqa/patient_bundles"
)
_DEFAULT_QA_PICKLE = (
    "data/fhirpathqa/qa.pickle"
)
_DEFAULT_OUTPUT = (
    "data/fhirpathqa/fhir_bundles.pq"
)

_BUNDLES_DIR = flags.DEFINE_string(
    "bundles_dir",
    _DEFAULT_BUNDLES_DIR,
    "Directory of per-patient JSON bundles (filenames are patient_id_hash).",
)
_QA_PICKLE_PATH = flags.DEFINE_string(
    "qa_pickle_path",
    _DEFAULT_QA_PICKLE,
    "Path to the QA pickle (used to map patient_id_hash -> patient_id).",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    _DEFAULT_OUTPUT,
    "Path for the output parquet.",
)
_OVERWRITE = flags.DEFINE_bool(
    "overwrite",
    False,
    "Overwrite the parquet if it already exists.",
)


def _load_hash_to_patient_id(qa_pickle_path: str) -> dict[str, str]:
  """Load the QA pickle and return a {patient_id_hash: patient_id} mapping."""
  import pickle

  with open(qa_pickle_path, "rb") as f:
    records = pickle.load(f)
  mapping: dict[str, str] = {}
  for rec in records:
    h = rec.get("patient_id_hash")
    pid = rec.get("patient_id")
    if not h or not pid:
      continue
    if h in mapping and mapping[h] != pid:
      raise ValueError(
          f"Inconsistent mapping: hash {h} maps to both {mapping[h]} and {pid}"
      )
    mapping[h] = pid
  logging.info("Built hash -> patient_id mapping for %d patients", len(mapping))
  return mapping


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_path = _OUTPUT_PATH.value
  if os.path.exists(output_path) and not _OVERWRITE.value:
    raise app.UsageError(
        f"Output already exists at {output_path}. Pass --overwrite to replace."
    )

  hash_to_pid = _load_hash_to_patient_id(_QA_PICKLE_PATH.value)

  bundle_files = sorted(
      f for f in os.listdir(_BUNDLES_DIR.value) if f.endswith(".json")
  )
  logging.info("Found %d bundle files in %s", len(bundle_files), _BUNDLES_DIR.value)

  rows: list[dict[str, str]] = []
  skipped_no_mapping: list[str] = []
  skipped_invalid: list[str] = []

  for fname in bundle_files:
    h = fname.removesuffix(".json")
    pid = hash_to_pid.get(h)
    if pid is None:
      skipped_no_mapping.append(h)
      continue

    with open(os.path.join(_BUNDLES_DIR.value, fname), "r") as f:
      bundle = json.loads(f.read())

    if not fhir_utils.verify_with_pydantic(bundle):
      skipped_invalid.append(pid)
      logging.warning("Bundle for patient %s failed pydantic validation", pid)
      continue

    rows.append({"Patient ID": pid, "FHIR Bundle": json.dumps(bundle)})

  logging.info("Validated bundles: %d", len(rows))
  if skipped_no_mapping:
    logging.warning(
        "Skipped %d bundles with no matching patient_id in QA: %s",
        len(skipped_no_mapping),
        skipped_no_mapping[:5],
    )
  if skipped_invalid:
    logging.warning(
        "Skipped %d bundles failing validation: %s",
        len(skipped_invalid),
        skipped_invalid[:5],
    )

  if not rows:
    raise RuntimeError("No bundles to write.")

  df = pd.DataFrame(rows)
  logging.info(
      "DataFrame shape: %s | columns: %s", df.shape, list(df.columns)
  )

  parent = os.path.dirname(output_path)
  if parent and not os.path.exists(parent):
    logging.info("Creating parent directory %s", parent)
    os.makedirs(parent)

  with open(output_path, "wb") as f:
    df.to_parquet(f, index=False)
  size = os.path.getsize(output_path)
  logging.info("Wrote parquet (%d bytes) to %s", size, output_path)


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
