"""Module for loading QA pairs and FHIR bundles for evaluation."""

import ast
import json
import pickle
import random
import re
import time
from typing import Any

from absl import logging
import numpy as np
import pandas as pd

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils


def load_qa_pairs(
    dataset_config: config.DatasetConfig,
    shuffle: bool = False,
    limit: int | None = None,
) -> list[data_base.EvalInstance]:
  """Load evaluation instances eagerly into a list.

  This is the convenience API for small utilities. For the evaluation runner,
  prefer :func:`iter_dataset` so large datasets do not need to be materialised
  in memory.

  Args:
    dataset_config: The configuration for the dataset.
    shuffle: Whether to shuffle the dataset.
    limit: The maximum number of instances to load.

  Returns:
    A list of evaluation instances.
  """
  logging.info(
      "Loading QA pairs for dataset=%s qa_path=%s limit=%s",
      dataset_config.name,
      dataset_config.qa_path,
      limit,
  )
  start = time.time()

  if dataset_config.name == "fhiragentbench":
    instances = load_fhiragentbench_qa(
        dataset_config.qa_path, shuffle=shuffle, limit=limit
    )
  elif dataset_config.name == "ehrqa":
    instances = load_ehrqa_qa(
        dataset_config.qa_path, shuffle=shuffle, limit=limit
    )
  elif dataset_config.name == "medagentbench":
    instances = load_medagentbench_qa(
        dataset_config.qa_path, shuffle=shuffle, limit=limit
    )
  elif dataset_config.name == "fhirpathqa":
    instances = load_fhirpathqa_qa(
        dataset_config.qa_path, shuffle=shuffle, limit=limit
    )
  else:
    raise ValueError(f"Unsupported dataset: {dataset_config.name}")

  elapsed = time.time() - start
  logging.info("Loaded %d instances in %.2fs", len(instances), elapsed)
  return instances


def load_fhir_bundles(
    dataset_config: config.DatasetConfig
) -> dict[str, dict[str, Any]]:
  """Load FHIR bundles from disk into a dictionary."""

  with open(dataset_config.fhir_path, "rb") as f:
    df_fhir = pd.read_parquet(f)
  df_fhir["FHIR Bundle"] = df_fhir["FHIR Bundle"].map(json.loads)
  df_fhir["FHIR Bundle"] = df_fhir["FHIR Bundle"].map(
      fhir_utils.decode_document_reference
  )
  if not df_fhir["FHIR Bundle"].map(fhir_utils.verify_with_pydantic).all():
    raise ValueError("Failed to parse FHIR bundles with pydantic validation.")

  fhir_dict = df_fhir.set_index("Patient ID")["FHIR Bundle"].to_dict()

  return fhir_dict


def get_fhir_bundle_for_instance(
    instance: data_base.EvalInstance, bundles: dict[str, Any]
) -> dict[str, Any] | None:
  """Get the FHIR bundle matching an evaluation instance's patient_id.

  Args:
    instance: The evaluation instance.
    bundles: A dictionary of FHIR bundles keyed by patient ID.

  Returns:
    The FHIR bundle for the instance, or None if not found.
  """
  return bundles.get(instance.patient_id)


def load_fhiragentbench_qa(
    qa_path: str, shuffle: bool = False, limit: int | None = None
) -> list[data_base.EvalInstance]:
  """Load FHIRAgentBench instances from disk into a list.

  Args:
    qa_path: Path to the QA CSV file.
    shuffle: Whether to shuffle the instances.
    limit: Maximum number of instances to load.

  Returns:
    A list of EvalInstance objects.
  """

  instances = []

  with open(qa_path, "r") as f:
    df_qa = pd.read_csv(f)
  df_qa["instance_id"] = [f"{idx:04d}" for idx in range(len(df_qa))]

  if shuffle:
    df_qa = df_qa.sample(frac=1, random_state=42).reset_index(drop=True)

  for _, row in df_qa.reset_index().iterrows():
    if limit is not None and len(instances) >= limit:
      break
    patient_id = row["patient_fhir_id"]
    question = row["question"]
    raw_answer = row["true_answer"]
    context = row["assumption"]
    if isinstance(context, float) and np.isnan(context):
      context = None
    elif isinstance(context, str):
      context = [context]
    else:
      raise ValueError(f"Unsupported additional context type: {type(context)}")

    # Parse Answer (String Rep of List -> Str)
    ground_truth = raw_answer

    if isinstance(raw_answer, str) and raw_answer.startswith("[["):
      try:
        parsed = ast.literal_eval(raw_answer)
      except (ValueError, SyntaxError):
        try:
          # Fix common missing comma issues
          corrected = re.sub(r"\]\s*\[", "], [", raw_answer)
          parsed = ast.literal_eval(corrected)
        except (ValueError, SyntaxError) as e:
          logging.warning("Error parsing answer '%s': %s", raw_answer, e)
          parsed = None

      if isinstance(parsed, list) and len(parsed) > 0:
        item = parsed[0]
        if isinstance(item, list) and len(item) > 0:
          ground_truth = str(item[0])
        else:
          ground_truth = str(item)
    if ground_truth == "[]" or ground_truth == "None":
      ground_truth = "Value not recorded or not applicable"

    instances.append(
        data_base.EvalInstance(
            instance_id=row["instance_id"],
            patient_id=patient_id,
            question=question,
            ground_truth=ground_truth,
            question_context=context,
            source_meta=row.to_dict(),
        )
    )
  return instances


def load_ehrqa_qa(
    qa_path: str, shuffle: bool = False, limit: int | None = None
) -> list[data_base.EvalInstance]:
  """Load EHRQA instances from disk into a list.

  Args:
    qa_path: Path to the QA pickle file.
    shuffle: Whether to shuffle the instances.
    limit: Maximum number of instances to load.

  Returns:
    A list of EvalInstance objects.
  """
  instances = []

  with open(qa_path, "rb") as f:
    qa_list = pickle.load(f)
  for idx, qa_item in enumerate(qa_list):
    qa_item["instance_id"] = f"{idx:04d}"

  if shuffle:
    random.Random(42).shuffle(qa_list)

  for _, qa_item in enumerate(qa_list):
    if limit is not None and len(instances) >= limit:
      break
    patient_id = qa_item["patient_id"]
    question = qa_item["question"]
    choices = qa_item["answer_choices"]

    # Format Question
    if choices:
      question += "\n\nOptions:\n" + "\n".join(choices)
      question += "\n\nAnswer with the correct option letter."

    gold_answer = qa_item.get("correct_answer", "")
    # Expand gold_answer to full string if possible
    if choices:
      for c in choices:
        if c.strip().startswith((f"({gold_answer})", f"{gold_answer}.")):
          gold_answer = c
          break

    instances.append(
        data_base.EvalInstance(
            instance_id=qa_item["instance_id"],
            patient_id=patient_id,
            question=question,
            ground_truth=str(gold_answer),
            source_meta=qa_item,
        )
    )
  return instances


def load_fhirpathqa_qa(
    qa_path: str, shuffle: bool = False, limit: int | None = None
) -> list[data_base.EvalInstance]:
  """Load FHIRPath-QA Benchmark instances from a pickle of dicts.

  The ``now`` field is injected into ``question_context`` so time-relative
  phrasings can be resolved by the answering model. Records without a
  matching FHIR bundle are filtered out at preprocessing time by
  ``scripts/fhirpathqa/convert_to_pickle --fhir_path=...``.

  Args:
    qa_path: Path to the QA pickle file.
    shuffle: Whether to shuffle the instances.
    limit: Maximum number of instances to load.

  Returns:
    A list of EvalInstance objects.
  """
  instances = []

  with open(qa_path, "rb") as f:
    records = pickle.load(f)
  for idx, rec in enumerate(records):
    rec["instance_id"] = f"{idx:04d}"

  if shuffle:
    random.Random(42).shuffle(records)

  for rec in records:
    if limit is not None and len(instances) >= limit:
      break
    patient_id = str(rec["patient_id"])
    question = rec["question"]
    raw_answer = rec["answer"]

    # Unwrap JSON-stringified list answers (e.g. '["Void"]'). The upstream
    # ships these as JSON, so use json.loads — ast.literal_eval would
    # choke on JSON-only literals like null/true/false.
    ground_truth = raw_answer
    if isinstance(raw_answer, str) and raw_answer.startswith("["):
      try:
        parsed = json.loads(raw_answer)
      except ValueError as e:
        logging.warning("Error parsing answer '%s': %s", raw_answer, e)
        parsed = None
      if isinstance(parsed, list):
        if len(parsed) == 0:
          ground_truth = "Value not recorded or not applicable"
        elif len(parsed) == 1:
          ground_truth = str(parsed[0])
        else:
          ground_truth = ", ".join(str(x) for x in parsed)

    now = rec.get("now")
    question_context = (
        [f"Assume current datetime is {now}"] if now else None
    )

    instances.append(
        data_base.EvalInstance(
            instance_id=rec["instance_id"],
            patient_id=patient_id,
            question=question,
            ground_truth=ground_truth,
            question_context=question_context,
            source_meta=rec,
        )
    )
  return instances


def load_medagentbench_qa(
    qa_path: str, shuffle: bool = False, limit: int | None = None
) -> list[data_base.EvalInstance]:
  """Load MedAgentBench instances from disk into a list.

  Args:
    qa_path: Path to the QA pickle file.
    shuffle: Whether to shuffle the instances.
    limit: Maximum number of instances to load.

  Returns:
    A list of EvalInstance objects.
  """
  instances = []

  with open(qa_path, "rb") as f:
    df_qa = pd.read_pickle(f)
  df_qa["instance_id"] = [f"{idx:04d}" for idx in range(len(df_qa))]
  if shuffle:
    df_qa = df_qa.sample(frac=1, random_state=42).reset_index(drop=True)

  for _, row in df_qa.reset_index().iterrows():
    if limit is not None and len(instances) >= limit:
      break
    patient_id = row["eval_MRN"]
    question = str(row["instruction"])
    # MedAgentBench's ground truth column was pre-computed in the dataset
    # preprocessing pipeline.
    ground_truth = str(row["ground_truth"])
    # -1 indicates the value is not recorded or not applicable
    if ground_truth == "-1":
      ground_truth = "Value not recorded or not applicable"
    context = row["context"]
    if not isinstance(context, list):
      context = [context]
    if context == [""]:
      context = None

    instances.append(
        data_base.EvalInstance(
            instance_id=row["instance_id"],
            patient_id=str(patient_id),
            question=question,
            ground_truth=ground_truth,
            question_context=context,
            source_meta=row.to_dict(),
        )
    )

  return instances
