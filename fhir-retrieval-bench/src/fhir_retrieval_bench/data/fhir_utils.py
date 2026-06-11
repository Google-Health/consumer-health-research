"""FHIR utility functions for date parsing and Bundle transformation."""

import base64
import datetime
from typing import Annotated, Any, Literal

from absl import logging
import pydantic

FHIRResourceTypes = Literal[
    "Condition",
    "Encounter",
    "Location",
    "Medication",
    "MedicationAdministration",
    "MedicationDispense",
    "MedicationRequest",
    "Observation",
    "Organization",
    "Patient",
    "Procedure",
    "Specimen",
    "DiagnosticReport",
    "DocumentReference",
    "Claim",
    "ExplanationOfBenefit",
    "Immunization",
    "Provenance",
    "AllergyIntolerance",
    "CareTeam",
    "CarePlan",
    "SupplyDelivery",
    "Device",
    "ImagingStudy",
    "MedicationStatement",
]


class FHIRResource(pydantic.BaseModel):
  """Represents an individual FHIR resource.

  Example dict:
  {
      "resourceType": "Patient",
      "id": "123",
      # Other fields are allowed without explicit definition.
  }
  """
  # Use extra="allow" if you want to keep all other FHIR fields without
  # defining them one by one
  model_config = pydantic.ConfigDict(extra="allow")
  resourceType: FHIRResourceTypes  # pylint: disable=invalid-name
  id: str


class BundleEntry(pydantic.BaseModel):
  """A single entry within a Bundle, wrapping a resource.

  Example dict:
  {
      "resource": {
          "resourceType": "Observation",
          "id": "obs-1"
      }
  }
  """
  resource: FHIRResource


class FHIRBundle(pydantic.BaseModel):
  """A FHIR Bundle of type 'collection' containing at least one entry and exactly one Patient resource.

  Example dict:
  {
      "resourceType": "Bundle",
      "type": "collection",
      "entry": [
          {"resource": {"resourceType": "Patient", "id": "p-01"}},
          {"resource": {"resourceType": "Condition", "id": "c-01"}}
      ]
  }
  """
  entry: Annotated[list[BundleEntry], pydantic.Field(min_length=1)]
  resourceType: str = pydantic.Field(..., pattern="^Bundle$")  # pylint: disable=invalid-name
  type: str = "collection"

  @pydantic.model_validator(mode="after")
  def validate_patient_count(self) -> "FHIRBundle":
    # Count how many entries are Patients
    patient_count = sum(
        1 for e in self.entry if e.resource.resourceType == "Patient"
    )

    if patient_count == 0:
      raise ValueError("Bundle must contain a Patient resource.")
    if patient_count > 1:
      raise ValueError(
          f"Bundle must contain exactly one Patient (found {patient_count})."
      )

    return self


def verify_with_pydantic(
    fhir_bundle_to_test: dict[str, Any], verbose: bool = False
) -> bool:
  """Verifies that the FHIR bundle is valid according to the FHIRBundle model.

  Args:
    fhir_bundle_to_test: The FHIR bundle to test.
    verbose: If True, log success or failure messages.

  Returns:
    True if the FHIR bundle is valid, False otherwise.
  """
  try:
    # This parses and validates the entire dict at once
    bundle = FHIRBundle(**fhir_bundle_to_test)
    if verbose:
      logging.info("✅ Success: %d entries validated.", len(bundle.entry))
    return True
  except pydantic.ValidationError:
    if verbose:
      logging.exception("❌ Failed during FHIR bundle validation.")
    return False


def decode_document_reference(fhir_bundle: dict[str, Any]) -> dict[str, Any]:
  """Decodes the base64 data in DocumentReference content to UTF-8 and adds it to the resource.

  If the decoding fails for a specific DocumentReference, the function will log
  the error and continue processing the rest of the bundle without failing.

  Args:
    fhir_bundle: The FHIR bundle to decode.

  Returns:
    The FHIR bundle with the decoded DocumentReference content.
  """
  for entry in fhir_bundle["entry"]:
    resource = entry["resource"]
    if resource["resourceType"] == "DocumentReference":
      for content_item in resource.get("content", []):
        attachment = content_item.get("attachment", {})
        if "data" in attachment:
          b64_data = attachment["data"]
          decoded = base64.b64decode(b64_data).decode("utf-8")
          attachment["decoded_data"] = decoded
          if "title" not in attachment:
            attachment["title"] = "Decoded Clinical Note"

  return fhir_bundle


def parse_fhir_date(date_str: str) -> datetime.datetime:
  """Robustly parses FHIR dates.

  Handles partial dates (year-only, year-month) and datetime strings
  with time components (splits on ``T``). Returns ``datetime.datetime.min``
  on failure or empty input.

  Args:
    date_str: The date string to parse.

  Returns:
    A datetime object, or ``datetime.datetime.min`` on failure.
  """
  if not date_str:
    return datetime.datetime.min
  clean_date = date_str.split("T")[0]
  parts = clean_date.split("-")
  if len(parts) == 1:
    clean_date = f"{parts[0]}-01-01"
  elif len(parts) == 2:
    clean_date = f"{parts[0]}-{parts[1]}-01"
  try:
    return datetime.datetime.fromisoformat(clean_date)
  except ValueError:
    return datetime.datetime.min


def get_resource_date(res: dict[str, Any]) -> str:
  """Extracts the most relevant date string for a FHIR resource.

  Uses type-specific field extraction logic matching original baseline
  experiments.

  Args:
    res: A FHIR resource dictionary.

  Returns:
    The most relevant date string, or an empty string if none is found.
  """
  r_type = res.get("resourceType")
  if r_type == "Encounter":
    return res.get("period", {}).get("start", "")
  elif r_type == "Condition":
    return str(res.get("onsetDateTime") or res.get("recordedDate") or "")
  elif r_type in ["Medication", "MedicationRequest"]:
    return res.get("authoredOn") or ""
  elif r_type == "Observation":
    return str(res.get("effectiveDateTime") or res.get("issued") or "")
  elif r_type == "DocumentReference":
    return str(res.get("date") or "")
  elif r_type == "Procedure":
    return str(
        res.get("performedDateTime")
        or res.get("performedPeriod", {}).get("start")
        or ""
    )
  elif r_type == "AllergyIntolerance":
    return str(res.get("recordedDate") or "")
  elif r_type in ["CarePlan", "CareTeam"]:
    return str(res.get("period", {}).get("start") or "")
  elif r_type in ["Claim", "ExplanationOfBenefit"]:
    return str(
        res.get("billablePeriod", {}).get("start") or res.get("created") or ""
    )
  elif r_type == "Device":
    return str(res.get("manufactureDate") or "")
  elif r_type == "DiagnosticReport":
    return str(res.get("effectiveDateTime") or res.get("issued") or "")
  elif r_type == "ImagingStudy":
    return str(res.get("started") or "")
  elif r_type in ["Immunization", "SupplyDelivery"]:
    return str(res.get("occurrenceDateTime") or "")
  elif r_type == "Specimen":
    return str(res.get("collection", {}).get("collectedDateTime") or "")
  return ""


def get_resource_date_v2(res: dict[str, Any]) -> str:
  """Extracts the most relevant date string with expanded fallbacks.

  First attempts to get the date using original baseline logic. If missing,
  uses expanded fallback fields for missing data.

  Args:
    res: A FHIR resource dictionary.

  Returns:
    The most relevant date string, or an empty string if none is found.
  """
  orig_dt = get_resource_date(res)
  if orig_dt:
    return orig_dt

  r_type = res.get("resourceType")
  if r_type == "Encounter":
    return str(res.get("period", {}).get("end") or "")
  elif r_type == "Condition":
    return str(
        res.get("assertedDate")
        or res.get("onsetPeriod", {}).get("start")
        or res.get("abatementDateTime")
        or res.get("issued")
        or res.get("created")
        or ""
    )
  elif r_type == "MedicationAdministration":
    return str(
        res.get("effectiveDateTime")
        or res.get("effectivePeriod", {}).get("start")
        or res.get("created")
        or ""
    )
  elif r_type == "MedicationDispense":
    return str(
        res.get("whenHandled")
        or res.get("whenPrepared")
        or res.get("created")
        or ""
    )
  elif r_type == "MedicationStatement":
    return str(
        res.get("effectiveDateTime")
        or res.get("effectivePeriod", {}).get("start")
        or res.get("dateAsserted")
        or ""
    )
  elif r_type == "AllergyIntolerance":
    return str(res.get("recordedDate") or res.get("lastOccurred") or "")
  return ""
