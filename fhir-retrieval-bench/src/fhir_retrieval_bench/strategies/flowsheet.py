"""Flowsheet strategy.

Transforms FHIR resources into tabular views (Markdown tables) as context.
"""

import copy
from typing import Any

from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base


def get_code_display(concept_dict: dict[str, Any]) -> str:
  """Get human-readable display for a FHIR CodeableConcept.

  Tries to find the best available text:
    .text > .coding[].display > .coding[].code

  Args:
    concept_dict: A FHIR CodeableConcept dict (can be partial).

  Returns:
    A string representing the code/display, or 'Unknown' if nothing is found.
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
  vq = res.get("valueQuantity")
  if vq:
    val = vq.get("value")
    return str(val) if val is not None else "N/A"
  vs = res.get("valueString")
  if vs:
    return vs
  vcc = res.get("valueCodeableConcept")
  if vcc:
    return get_code_display(vcc)
  return "N/A"


def get_medication_name(res: dict[str, Any]) -> str:
  med_concept = res.get("medicationCodeableConcept", {})
  name = (
      med_concept.get("text")
      or med_concept.get("coding", [{}])[0].get("display")
      or med_concept.get("coding", [{}])[0].get("code")
      or res.get("medicationReference", {}).get("display")
      or "Unknown"
  )
  return name


def build_flowsheet(fhir_bundle: dict[str, Any]) -> str:
  """Transform the FHIR bundle into Markdown tables.

  Args:
    fhir_bundle: The FHIR bundle dict to process.

  Returns:
    Markdown tables representing flowsheet data.
  """
  if not fhir_bundle.get("entry"):
    return "No records found."

  if not fhir_utils.verify_with_pydantic(fhir_bundle):
    raise ValueError("Failed to parse FHIR bundle with pydantic validation.")

  fhir_bundle = copy.deepcopy(fhir_bundle)
  entries = fhir_bundle.get("entry", [])

  patient_info = {}
  obs_rows = []
  med_rows = []
  cond_rows = []

  for entry in entries:
    res = entry.get("resource", {})
    r_type = res.get("resourceType")

    date = fhir_utils.get_resource_date(res) or "N/A"

    if r_type == "Observation":
      code = get_code_display(res.get("code", {}))
      val = _format_observation_value(res)
      unit = res.get("valueQuantity", {}).get("unit", "")
      obs_rows.append((date, code, val, unit))

    elif r_type in ["MedicationRequest", "MedicationAdministration"]:
      name = get_medication_name(res)
      status = res.get("status", "unknown")
      med_rows.append((date, name, status))

    elif r_type == "Condition":
      name = get_code_display(res.get("code", {}))
      status = (
          res.get("clinicalStatus", {})
          .get("coding", [{}])[0]
          .get("code", "unknown")
      )
      cond_rows.append((date, name, status))

    elif r_type == "Patient":
      patient_info["ID"] = res.get("id")
      names = res.get("name", [])
      if names:
        name = names[0]
        given = " ".join(name.get("given", []))
        family = name.get("family", "")
        patient_info["Name"] = f"{given} {family}".strip()
      patient_info["DOB"] = res.get("birthDate")
      patient_info["Gender"] = res.get("gender")

  # Sort by date
  obs_rows.sort(key=lambda x: x[0])
  med_rows.sort(key=lambda x: x[0])
  cond_rows.sort(key=lambda x: x[0])

  context_parts = []

  if patient_info:
    context_parts.append("### Patient Information")
    for k, v in patient_info.items():
      if v:
        context_parts.append(f"- **{k}**: {v}")
    context_parts.append("")

  if obs_rows:
    context_parts.append("### Observations")
    context_parts.append("| Date | Observation | Value | Unit |")
    context_parts.append("| --- | --- | --- | --- |")
    for row in obs_rows:
      context_parts.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    context_parts.append("")

  if med_rows:
    context_parts.append("### Medications")
    context_parts.append("| Date | Medication | Status |")
    context_parts.append("| --- | --- | --- |")
    for row in med_rows:
      context_parts.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    context_parts.append("")

  if cond_rows:
    context_parts.append("### Conditions")
    context_parts.append("| Date | Condition | Status |")
    context_parts.append("| --- | --- | --- |")
    for row in cond_rows:
      context_parts.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    context_parts.append("")

  return "\n".join(context_parts) if context_parts else "No records found."


class FlowsheetStrategy(base.RAGStrategy):
  """Transforms FHIR resources into Markdown tables for context."""

  def prepare_fhir_context(
      self,
      record: data_base.EvalInstance,
      fhir_bundle: dict[str, Any],
  ) -> str:
    return build_flowsheet(fhir_bundle)
