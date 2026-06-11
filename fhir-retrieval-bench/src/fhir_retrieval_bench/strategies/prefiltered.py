"""Prefiltered strategy.

Pre-filters FHIR resources by clinical status and deduplicates
observations/conditions by concept code (keeping only the latest).
"""

import collections
import copy
import json
from typing import Any

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base
from fhir_retrieval_bench.utils import api


def _strip_useless_keys(obj: Any) -> Any:
  """Recursively strip id, system, reference, and fullUrl keys.

  Args:
    obj: The object to strip keys from.

  Returns:
    The object with keys stripped.
  """
  if isinstance(obj, dict):
    return {
        k: _strip_useless_keys(v)
        for k, v in obj.items()
        if k not in ("id", "system", "reference", "fullUrl")
    }
  elif isinstance(obj, list):
    return [_strip_useless_keys(item) for item in obj]
  return obj


def _extract_text_for_index(
    res: dict[str, Any], dataset_name: str = ""
) -> str:
  """Extract a human-readable display name for a FHIR resource.

  Args:
    res: The FHIR resource.
    dataset_name: The name of the dataset.

  Returns:
    A human-readable display name.
  """
  r_type = res.get("resourceType")

  if r_type == "Patient":
    names = res.get("name", [])
    if names:
      family = names[0].get("family", "")
      given = " ".join(names[0].get("given", []))
      return f"{given} {family}".strip() or "Anonymous"

  if "code" in res:
    text = res["code"].get("text")
    if text:
      return text
    codings = res["code"].get("coding", [])
    if codings:
      return codings[0].get("display", "Unnamed Resource")

  if r_type == "MedicationRequest":
    med_cc = res.get("medicationCodeableConcept", {})
    if med_cc.get("text"):
      return med_cc["text"]
    if med_cc.get("coding"):
      return med_cc["coding"][0].get("display", "Medication")
    return res.get("medicationReference", {}).get("display", "Medication")

  if dataset_name == "fhiragentbench" and r_type == "Medication":
    code = res.get("code", {})
    if code.get("text"):
      return code["text"]
    if code.get("coding"):
      return code["coding"][0].get("display", "Medication")
    if res.get("identifier"):
      for ident in res.get("identifier", []):
        if ident.get("value"):
          return ident["value"]

  return "Unknown"


def _filter_fhir_bundle(
    fhir_bundle: dict[str, Any], dataset_name: str = ""
) -> dict[str, Any]:
  """Filter a FHIR bundle by clinical status and deduplicate by concept.

  Keeps Medication/MedicationRequest only when status is active, completed,
  or on-hold and the category (if present) is valid. Observations and
  Conditions are grouped by concept code and only the latest entry per
  group is retained.  Patient and Encounter resources pass through.
  All other resource types are dropped.

  Args:
    fhir_bundle: The FHIR bundle to filter.
    dataset_name: The name of the dataset.

  Returns:
    The filtered FHIR bundle.
  """
  entries = fhir_bundle.get("entry", [])
  filtered_resources: list[dict[str, Any]] = []

  obs_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  cond_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)

  for entry in entries:
    res = entry.get("resource", {})
    r_type = res.get("resourceType")

    if r_type in ["Medication", "MedicationRequest"]:
      status = res.get("status", "").lower()
      valid_statuses = {"active", "completed", "on-hold"}
      categories = str(res.get("category", [])).lower()
      valid_cats = {"outpatient", "community", "office", "order"}
      is_valid_cat = not res.get("category") or any(
          cat in categories for cat in valid_cats
      )

      if status in valid_statuses and is_valid_cat:
        filtered_resources.append(res)

    elif r_type == "Observation":
      c = res.get("code", {}).get("coding", [{}])[0]
      concept_key = f"{c.get('system')}|{c.get('code')}"
      obs_groups[concept_key].append(res)

    elif r_type == "Condition":
      c = res.get("code", {}).get("coding", [{}])[0]
      concept_key = f"{c.get('system')}|{c.get('code')}"
      cond_groups[concept_key].append(res)

    elif r_type in ["Patient", "Encounter"]:
      filtered_resources.append(res)

    # All other types are DROPPED (not included)

  # Selection Logic (Latest Only)
  for group in obs_groups.values():
    group.sort(
        key=lambda r: fhir_utils.parse_fhir_date(
            r.get("effectiveDateTime") or r.get("issued")
        ),
        reverse=True,
    )
    filtered_resources.append(group[0])

  for group in cond_groups.values():
    group.sort(
        key=lambda r: fhir_utils.parse_fhir_date(
            r.get("onsetDateTime") or r.get("recordedDate")
        ),
        reverse=True,
    )
    filtered_resources.append(group[0])

  # Build Slim Bundle with index
  return {
      "resourceType": "Bundle",
      "type": "collection",
      "entry": [{"resource": r} for r in filtered_resources],
      "index": [
          {
              "resource_type": r.get("resourceType"),
              "display_name": _extract_text_for_index(r, dataset_name),
              "date": fhir_utils.get_resource_date(r),
          }
          for r in filtered_resources
      ],
  }


class PrefilteredStrategy(base.RAGStrategy):
  """Filters FHIR resources by clinical status and deduplicates by concept.

  Applies status-based filtering for medications, keeps only the latest
  observation/condition per concept code, and strips noisy keys before
  serialising the slim bundle as JSON context.
  """

  def __init__(
      self,
      creds: api.Credentials,
      answer_config: config.LLMConfig,
      dataset_name: str,
  ):
    super().__init__(
        creds=creds,
        answer_config=answer_config,
    )
    self.dataset_name = dataset_name

  def prepare_fhir_context(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> str:
    """Filter, deduplicate, strip noisy keys, and return JSON context.

    Args:
      record: The evaluation instance.
      fhir_bundle: The FHIR bundle.

    Returns:
      The JSON context string.
    """
    if not fhir_utils.verify_with_pydantic(fhir_bundle):
      raise ValueError("Failed to parse FHIR bundle with pydantic validation.")
    fhir_bundle = copy.deepcopy(fhir_bundle)
    slim_bundle = _filter_fhir_bundle(fhir_bundle, self.dataset_name)
    slim_bundle = _strip_useless_keys(slim_bundle)
    return json.dumps(slim_bundle, indent=2)
