"""FHIR-to-Text strategy.

Converts FHIR bundle resources into natural language text that replicates
the original Proto-based ``_format_patient_text_local`` output format,
adapted to work with JSON dicts instead of FHIR Protos.

Only Condition, MedicationRequest (with coding), Observation (with
effectiveDateTime), and Procedure resources are rendered.  All other
resource types are skipped.
"""

import copy
from typing import Any

from absl import logging
import immutabledict

from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base

_MED_STATUS_MAP = immutabledict.immutabledict({
    1: "ACTIVE",
    2: "ON_HOLD",
    3: "CANCELLED",
    4: "COMPLETED",
    5: "ENTERED_IN_ERROR",
    6: "STOPPED",
    7: "DRAFT",
    8: "UNKNOWN",
})


def _safe_get_code_display(concept_dict: dict[str, Any]) -> str:
  """Extract display text from a FHIR CodeableConcept dict.

  Tries ``text``, then ``coding[0].display``, then ``coding[0].code``.
  Returns ``"Unknown"`` if nothing is found.

  Args:
    concept_dict: A CodeableConcept-shaped dict.

  Returns:
    The best available display string.
  """
  if not concept_dict:
    return "Unknown"
  text = concept_dict.get("text")
  if text:
    return text
  display = concept_dict.get("coding", [{}])[0].get("display")
  if display:
    return display
  code = concept_dict.get("coding", [{}])[0].get("code")
  if code:
    return code
  return "Unknown"


def _format_observation_value(res: dict[str, Any]) -> str:
  """Format the value of an Observation resource.

  Tries ``valueQuantity``, then ``valueString``, then
  ``valueCodeableConcept``.  Falls back to ``"unknown value"``.

  Args:
    res: An Observation resource dict.

  Returns:
    A human-readable value string.
  """
  vq = res.get("valueQuantity")
  if vq:
    value = vq.get("value", "")
    unit = vq.get("unit", "")
    return f"{value} {unit}".strip()
  vs = res.get("valueString")
  if vs:
    return vs
  vcc = res.get("valueCodeableConcept")
  if vcc:
    return _safe_get_code_display(vcc)
  return "complex value"


def _get_med_status_string(status) -> str:
  """Convert a MedicationRequest status to an uppercased string.

  If *status* is numeric (or a string representation of an int), it is
  mapped through the Proto enum table.  Otherwise the raw string value
  is uppercased.

  Args:
    status: The ``status`` field from a MedicationRequest resource.

  Returns:
    An uppercased status string.
  """
  try:
    return _MED_STATUS_MAP.get(int(status), str(status).upper())
  except (ValueError, TypeError):
    return str(status).upper() if status else "UNKNOWN"


class Fhir2TextStrategy(base.RAGStrategy):
  """Converts FHIR resources to natural language.

  Matches the original Proto format.
  """

  def prepare_fhir_context(
      self,
      record: data_base.EvalInstance,
      fhir_bundle: dict[str, Any],
  ) -> str:
    """Transform the FHIR bundle into natural-language sentences.

    Only Condition, MedicationRequest (with coding), Observation (with
    effectiveDateTime), and Procedure resources are rendered.  All other
    resource types are silently skipped.

    Args:
      record: The evaluation record whose ``fhir_bundle`` will be processed.
      fhir_bundle: The FHIR bundle dict to process.

    Returns:
      Newline-joined natural-language sentences, or ``"No records found."``
      if no supported resources were present.
    """
    if not fhir_utils.verify_with_pydantic(fhir_bundle):
      raise ValueError("Failed to parse FHIR bundle with pydantic validation.")
    fhir_bundle = copy.deepcopy(fhir_bundle)
    entries = fhir_bundle.get("entry", [])
    logging.debug(
        "prepare_fhir_context: patient=%s entries=%d",
        record.patient_id,
        len(entries),
    )

    lines = []
    for entry in entries:
      res = entry.get("resource", {})
      r_type = res.get("resourceType")

      if r_type == "Condition":
        name = _safe_get_code_display(res.get("code", {}))
        status = (
            res.get("clinicalStatus", {})
            .get("coding", [{}])[0]
            .get("code", "unknown")
        )
        date = (
            res.get("onsetDateTime")
            or res.get("onsetString")
            or res.get("recordedDate")
            or "unknown date"
        )
        lines.append(
            f'The patient has a condition of "{name}" ({status}) recorded on'
            f" {date}."
        )

      elif r_type == "MedicationRequest":
        has_coding = bool(
            res.get("medicationCodeableConcept", {}).get("coding")
        )
        if has_coding:
          med_concept = res.get("medicationCodeableConcept", {})
          name = (
              med_concept.get("text")
              or med_concept.get("coding", [{}])[0].get("display")
              or med_concept.get("coding", [{}])[0].get("code")
              or res.get("medicationReference", {}).get("display")
              or "Unknown"
          )
          date = res.get("authoredOn") or "N/D"
          prescriber = res.get("requester", {}).get("display") or "Unknown"
          status = _get_med_status_string(res.get("status"))
          lines.append(
              f'The patient was prescribed "{name}" on {date} by {prescriber}.'
              f" This prescription is {status} as of now."
          )

      elif r_type == "Observation":
        has_date = bool(res.get("effectiveDateTime"))
        if has_date:
          cat = (
              res.get("category", [{}])[0]
              .get("coding", [{}])[0]
              .get("code", "unknown")
          )
          code = _safe_get_code_display(res.get("code", {}))
          val = _format_observation_value(res)
          date = res.get("effectiveDateTime")
          lines.append(
              f'The patient has an observation of category {cat} for "{code}"'
              f' with a value of "{val}" and an effective time of {date}.'
          )

      elif r_type == "Procedure":
        name = _safe_get_code_display(res.get("code", {}))
        date = (
            res.get("performedDateTime")
            or res.get("performedPeriod", {}).get("start")
            or "unknown"
        )
        lines.append(
            f'The patient had a procedure "{name}" performed on {date}.'
        )

    logging.debug("Generated %d text lines", len(lines))
    return "\n".join(lines) if lines else "No records found."
