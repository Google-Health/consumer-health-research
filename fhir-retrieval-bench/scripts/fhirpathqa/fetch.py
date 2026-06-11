"""Fetch the upstream FHIRPath-QA Benchmark JSONL into CNS.

Downloads ``fhirpath-qa-benchmark.jsonl`` from the FHIRPath-QA GitHub repo
and writes it byte-for-byte to a CNS path. This is the first step of
integrating FHIRPath-QA into the bench: stage the raw upstream artifact so
it can be inspected and converted later (see ``convert_to_pickle.py``).

Example:
  python scripts/fhirpathqa/fetch.py
  python scripts/fhirpathqa/fetch.py --output_path=data/fhirpathqa/qa.jsonl --overwrite
"""

import json
import os
import urllib.request

from absl import app
from absl import flags
from absl import logging


_DEFAULT_URL = (
    "https://raw.githubusercontent.com/mooshifrew/fhirpath-qa/main/output/"
    "fhirpath-qa-benchmark.jsonl"
)
_DEFAULT_OUTPUT = (
    "data/fhirpathqa/qa.jsonl"
)

_SOURCE_URL = flags.DEFINE_string(
    "source_url",
    _DEFAULT_URL,
    "URL to fetch the FHIRPath-QA Benchmark JSONL from.",
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path",
    _DEFAULT_OUTPUT,
    "Output path to write the JSONL to.",
)
_OVERWRITE = flags.DEFINE_bool(
    "overwrite",
    False,
    "Overwrite the output file if it already exists.",
)


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  output_path = _OUTPUT_PATH.value
  if os.path.exists(output_path) and not _OVERWRITE.value:
    raise app.UsageError(
        f"Output already exists at {output_path}. Pass --overwrite to replace."
    )

  logging.info("Fetching %s", _SOURCE_URL.value)
  with urllib.request.urlopen(_SOURCE_URL.value) as resp:
    payload = resp.read()

  parent = os.path.dirname(output_path)
  if parent and not os.path.exists(parent):
    logging.info("Creating parent directory %s", parent)
    os.makedirs(parent)

  logging.info("Writing %d bytes to %s", len(payload), output_path)
  with open(output_path, "wb") as f:
    f.write(payload)

  text = payload.decode("utf-8")
  lines = [line for line in text.splitlines() if line.strip()]
  logging.info("Lines: %d", len(lines))
  if lines:
    first = json.loads(lines[0])
    logging.info("First record:\n%s", json.dumps(first, indent=2))


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
