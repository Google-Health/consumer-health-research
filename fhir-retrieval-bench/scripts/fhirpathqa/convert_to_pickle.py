"""Convert the staged FHIRPath-QA JSONL into a pickle of list[dict].

Reads the JSONL written by ``fetch.py``, validates that every record has
the fields the loader will need, and pickles the resulting ``list[dict]``
to a sibling CNS path. The ``load_fhirpathqa_qa`` loader consumes this
pickle directly (matching the ehrqa / medagentbench loader pattern).

When ``--fhir_path`` is provided, records whose patient_id is not in the
bundle parquet's ``Patient ID`` column are dropped before saving, so the
loader and runner never see records without a matching bundle.

Logs per-perspective, per-split, unique-patient, and unique-template
counts so the conversion run also serves as a quick sanity-check.

Example:
  python scripts/fhirpathqa/convert_to_pickle.py
  python scripts/fhirpathqa/convert_to_pickle.py \
      --input_path=data/fhirpathqa/qa.jsonl --output_path=data/fhirpathqa/qa.pickle \
      --fhir_path=data/fhirpathqa/fhir_bundles.pq --overwrite
"""

import os
import collections
import json
import pickle

from absl import app
from absl import flags
from absl import logging
import pandas as pd


_DEFAULT_INPUT = (
    "data/fhirpathqa/qa.jsonl"
)
_DEFAULT_OUTPUT = (
    "data/fhirpathqa/qa.pickle"
)

_REQUIRED_KEYS = (
    "patient_id",
    "question",
    "query",
    "now",
    "perspective",
    "split",
    "answer",
)

_INPUT_PATH = flags.DEFINE_string(
    "input_path",
    _DEFAULT_INPUT,
    "Path to the staged JSONL (output of fetch_fhirpathqa).",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    _DEFAULT_OUTPUT,
    "Path to write the pickle to.",
)
_OVERWRITE = flags.DEFINE_bool(
    "overwrite",
    False,
    "Overwrite the output pickle if it already exists.",
)
_FHIR_PATH = flags.DEFINE_string(
    "fhir_path",
    None,
    "Optional FHIR bundle parquet. If set, records whose patient_id is not"
    " in the parquet's `Patient ID` column are dropped before saving.",
)


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_path = _OUTPUT_PATH.value
  if os.path.exists(output_path) and not _OVERWRITE.value:
    raise app.UsageError(
        f"Output already exists at {output_path}. Pass --overwrite to replace."
    )

  logging.info("Reading %s", _INPUT_PATH.value)
  with open(_INPUT_PATH.value, "r") as f:
    records = [json.loads(line) for line in f if line.strip()]
  logging.info("Loaded %d records", len(records))

  for idx, rec in enumerate(records):
    missing = [k for k in _REQUIRED_KEYS if k not in rec]
    if missing:
      raise ValueError(
          f"Record {idx} (patient_id={rec.get('patient_id')}) missing"
          f" required keys: {missing}"
      )

  if _FHIR_PATH.value:
    with open(_FHIR_PATH.value, "rb") as f:
      bundle_pids = set(
          pd.read_parquet(f, columns=["Patient ID"])["Patient ID"].astype(str)
      )
    before = len(records)
    records = [r for r in records if str(r["patient_id"]) in bundle_pids]
    dropped = before - len(records)
    if dropped:
      logging.info(
          "Dropped %d records (of %d) for patients absent from %s",
          dropped, before, _FHIR_PATH.value,
      )

  perspective_counts = collections.Counter(r["perspective"] for r in records)
  split_counts = collections.Counter(r["split"] for r in records)
  unique_patients = len({r["patient_id"] for r in records})
  unique_templates = len({r.get("question_template_id") for r in records})

  logging.info("Perspective counts: %s", dict(perspective_counts))
  logging.info("Split counts: %s", dict(split_counts))
  logging.info("Unique patients: %d", unique_patients)
  logging.info("Unique question templates: %d", unique_templates)

  payload = pickle.dumps(records)
  logging.info("Writing %d bytes to %s", len(payload), output_path)
  with open(output_path, "wb") as f:
    f.write(payload)


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
