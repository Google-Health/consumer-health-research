"""Ontology-guided retrieval strategy.

Decomposes the user's question into tokens and expands them using a static
medical ontology expansion map, then scores each FHIR resource based on
keyword matches against the original and expanded tokens (plus the resource
type) and sends only the highest-scoring resources as context.
"""

import copy
import datetime
import re
from typing import Any

from absl import logging

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base
from fhir_retrieval_bench.utils import api

MANDATORY_SCORE = 9999


def _expand_query_terms(query: str) -> tuple[set[str], set[str]]:
  """Decomposes query into base tokens and medically expanded tokens."""
  tokens = set(re.findall(r"\w+", query.lower()))
  stop_words = {
      "what",
      "is",
      "my",
      "the",
      "a",
      "an",
      "of",
      "in",
      "on",
      "for",
      "to",
      "does",
      "have",
      "did",
      "check",
      "looking",
      "during",
      "time",
  }
  base_tokens = {t for t in tokens if t not in stop_words and len(t) > 2}

  expansion_map = {
      "heart": [
          "cardiac",
          "coronary",
          "valve",
          "aortic",
          "mitral",
          "ventricle",
          "atrium",
          "pulse",
          "rate",
          "troponin",
          "bnp",
      ],
      "sugar": ["glucose", "hba1c", "a1c", "insulin", "diabetes", "metformin"],
      "diabetes": ["glucose", "hba1c", "a1c", "insulin", "sugar"],
      "kidney": ["renal", "creatinine", "bun", "urea", "nephro", "gfr"],
      "liver": ["hepatic", "alt", "ast", "bilirubin", "cirrhosis", "alp"],
      "pressure": [
          "bp",
          "systolic",
          "diastolic",
          "hypertension",
          "hypotension",
      ],
      "breathing": [
          "respiratory",
          "lung",
          "pulmonary",
          "oxygen",
          "o2",
          "breath",
          "rate",
      ],
      "medication": [
          "prescription",
          "drug",
          "rx",
          "pill",
          "tablet",
          "capsule",
          "dose",
          "meds",
      ],
      "prescribed": ["prescription", "drug", "rx", "medication", "ordered"],
      "surgery": [
          "procedure",
          "operation",
          "incision",
          "graft",
          "mesh",
          "removal",
          "repair",
      ],
      "procedure": [
          "surgery",
          "operation",
          "cath",
          "catheterization",
          "ultrasound",
          "scan",
          "xray",
          "x-ray",
          "mri",
          "ct",
          "echo",
          "electro",
      ],
      "test": [
          "lab",
          "panel",
          "count",
          "level",
          "analysis",
          "culture",
          "blood",
          "serum",
      ],
      "lab": ["test", "panel", "count", "level", "analysis", "result"],
      "shot": ["immunization", "vaccine", "vax", "jab"],
      "vaccine": ["immunization", "shot", "vax"],
      "admitted": ["encounter", "admission", "hospital", "inpatient", "visit"],
      "hospital": [
          "encounter",
          "admission",
          "inpatient",
          "visit",
          "emergency",
          "discharge",
      ],
  }

  expanded_tokens = set()
  for token in base_tokens:
    if token in expansion_map:
      expanded_tokens.update(expansion_map[token])

  return base_tokens, expanded_tokens


def _score_resource(
    text: str,
    r_type: str,
    date_obj: datetime.datetime,
    base_tokens: set[str],
    expanded_tokens: set[str],
) -> int:
  """Calculates relevance score for a resource text against the query."""
  score = 0
  text_lower = text.lower()

  # 1. Type Boosting
  type_map = {
      "MedicationRequest": [
          "medication",
          "prescribed",
          "rx",
          "drug",
          "medicine",
      ],
      "Procedure": ["procedure", "surgery", "operation", "test", "scan"],
      "Observation": [
          "lab",
          "test",
          "result",
          "level",
          "vital",
          "rate",
          "pressure",
      ],
      "Condition": ["condition", "diagnosis", "problem", "disease", "history"],
      "Immunization": ["vaccine", "shot", "immunization"],
      "AllergyIntolerance": ["allergy", "intolerance", "reaction", "allergic"],
      "DiagnosticReport": ["report", "scan", "test"],
      "ImagingStudy": ["scan", "xray", "mri", "ct", "image", "imaging"],
  }

  keywords = type_map.get(r_type, [])
  if any(k in base_tokens for k in keywords):
    score += 5

  # 2. Text Matching
  matches = 0
  for token in base_tokens:
    if token in text_lower:
      score += 10
      matches += 1

  for token in expanded_tokens:
    if token in text_lower:
      score += 4
      matches += 1

  # 3. Recency Heuristic
  if matches > 0 and not any(
      t in base_tokens for t in ["history", "first", "oldest", "previous"]
  ):
    if date_obj != datetime.datetime.min:
      score += 2

  return score


class OntologyGuidedRetrievalStrategy(base.RAGStrategy):
  """Retrieves relevant resources via keyword-based scoring as context."""

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
    logging.info(
        "Initialized OntologyGuidedRetrievalStrategy: dataset_name=%s",
        dataset_name,
    )

  def prepare_fhir_context(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> str:
    """Keyword-based scoring RAG: keep Encounters/Active Conditions."""
    if not fhir_utils.verify_with_pydantic(fhir_bundle):
      raise ValueError("Failed to parse FHIR bundle with pydantic validation.")
    fhir_bundle = copy.deepcopy(fhir_bundle)
    base_tokens, expanded_tokens = _expand_query_terms(record.question)
    scored_items = []  # (Date, Text, Score)
    all_parsed_items = []  # (Date, Text, Score) - Backup for fallback

    entries = fhir_bundle.get("entry", [])

    for entry in entries:
      res = entry.get("resource", {})
      r_type = res.get("resourceType")

      # --- PARSING & FORMATTING ---
      date_obj = datetime.datetime.min
      is_mandatory = False
      score = 0

      if r_type == "Encounter":
        # MANDATORY SKELETON
        cls = res.get("class", {}).get("code", "VISIT").upper()
        enc_type_str = ""
        if res.get("type"):
          enc_types = []
          for t in res.get("type", []):
            if t.get("coding"):
              enc_types.append(t["coding"][0].get("display", ""))
          if enc_types:
            enc_type_str = f" - Type: {', '.join(filter(None, enc_types))}"

        start = res.get("period", {}).get("start", "")
        end = res.get("period", {}).get("end", "")
        date_obj = fhir_utils.parse_fhir_date(start)
        text = (
            f"[{start}] *** ENCOUNTER: {cls}{enc_type_str} (End:"
            f" {end or 'Ongoing'}) ***"
        )
        is_mandatory = True
        score = MANDATORY_SCORE

      elif r_type == "Condition":
        name = (
            res.get("code", {}).get("text")
            or res.get("code", {}).get("coding", [{}])[0].get("display")
            or "Unknown Condition"
        )
        # For fhiragentbench, clinicalStatus might be missing, don't default to
        # active to prevent context bloat
        status_default = (
            "unknown" if self.dataset_name == "fhiragentbench" else "active"
        )
        status = (
            res.get("clinicalStatus", {})
            .get("coding", [{}])[0]
            .get("code", status_default)
        )
        date_str = res.get("onsetDateTime") or res.get("recordedDate")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Condition: {name} ({status})"

        if status in ["active", "recurrence"]:
          is_mandatory = True
          score = MANDATORY_SCORE

      elif r_type in [
          "Medication",
          "MedicationRequest",
          "MedicationAdministration",
      ]:
        name = None
        if self.dataset_name == "fhiragentbench":
          if r_type == "Medication":
            name = res.get("code", {}).get("text")
            if not name and res.get("code", {}).get("coding"):
              name = res.get("code", {}).get("coding")[0].get("display")
            if not name and res.get("identifier"):
              ident_vals = [
                  ident.get("value")
                  for ident in res["identifier"]
                  if ident.get("value")
              ]
              if ident_vals:
                name = (name + " " if name else "") + " ".join(ident_vals)
          elif r_type in ["MedicationRequest", "MedicationAdministration"]:
            name = res.get("medicationCodeableConcept", {}).get("text")
            if not name and res.get("medicationCodeableConcept", {}).get(
                "coding"
            ):
              name = (
                  res.get("medicationCodeableConcept", {})
                  .get("coding")[0]
                  .get("display", "")
              )
              if not name:
                name = (
                    res.get("medicationCodeableConcept", {})
                    .get("coding")[0]
                    .get("code", "")
                )
        else:
          name = res.get("medicationCodeableConcept", {}).get("text")
          if not name:
            codings = res.get("medicationCodeableConcept", {}).get("coding", [])
            if codings:
              name = codings[0].get("display")
          if not name:
            name = res.get("medicationReference", {}).get("display")

        if not name:
          name = "Unknown Med"

        status = res.get("status", "unknown")
        date_str = res.get("authoredOn") or res.get("effectiveDateTime")
        date_obj = fhir_utils.parse_fhir_date(date_str)

        # Additional specific parsing for medagentbench (which might include
        # dosageInstruction)
        dosage = ""
        if r_type == "MedicationRequest" and res.get("dosageInstruction"):
          timing = (
              res.get("dosageInstruction", [{}])[0]
              .get("timing", {})
              .get("code", {})
              .get("text", "")
          )
          route = (
              res.get("dosageInstruction", [{}])[0]
              .get("route", {})
              .get("text", "")
          )
          if timing or route:
            dosage = f" - {timing} {route}".strip()

        prefix = (
            "Administered" if r_type == "MedicationAdministration" else "Rx"
        )
        text = (
            f"[{date_str or 'N/D'}] {prefix}: {name}{dosage} (Status: {status})"
        )

      elif r_type == "Procedure":
        name = (
            res.get("code", {}).get("text")
            or res.get("code", {}).get("coding", [{}])[0].get("display")
            or "Unknown Procedure"
        )
        date_str = res.get("performedDateTime") or res.get(
            "performedPeriod", {}
        ).get("start")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Procedure: {name}"

      elif r_type == "Observation":
        if "DiagnosticReport" in str(res.get("category", "")):
          continue
        name = (
            res.get("code", {}).get("text")
            or res.get("code", {}).get("coding", [{}])[0].get("display")
            or "Unknown Lab"
        )

        val = res.get("valueQuantity", {}).get("value")
        unit = res.get("valueQuantity", {}).get("unit", "")
        val_str = (
            f"{val} {unit}" if val is not None else res.get("valueString", "")
        )
        if not val_str:
          continue

        # Interpretation Flags
        interp = (
            res.get("interpretation", [{}])[0]
            .get("coding", [{}])[0]
            .get("code", "")
        )
        interp_map = {
            "H": " ***[HIGH]***",
            "HH": " ***[HIGH]***",
            "High": " ***[HIGH]***",
            "L": " ***[LOW]***",
            "LL": " ***[LOW]***",
            "Low": " ***[LOW]***",
            "A": " ***[ABNORMAL]***",
            "AA": " ***[ABNORMAL]***",
            "Abnormal": " ***[ABNORMAL]***",
        }
        flag = interp_map.get(interp, "")

        date_str = res.get("effectiveDateTime") or res.get("issued")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Lab: {name} = {val_str}{flag}"

      elif r_type == "DocumentReference":
        name = res.get("type", {}).get("coding", [{}])[0].get("display", "Note")
        date_str = res.get("date", "")
        date_obj = fhir_utils.parse_fhir_date(date_str)

        content_text = ""
        content = res.get("content", [])
        if content:
          attachment = content[0].get("attachment", {})
          if attachment.get("decoded_data"):
            content_text = attachment.get("decoded_data")[:200]
          elif attachment.get("title"):
            content_text = attachment.get("title")

        text = f"[{date_str or 'N/D'}] Note: {name} - {content_text}..."
        is_mandatory = True
        score = MANDATORY_SCORE

      elif r_type == "Location":
        name = res.get("name")
        if not name and res.get("identifier"):
          name = res.get("identifier", [{}])[0].get("value")
        text = f"Location: {name or 'Unknown Location'}"

      elif r_type == "Specimen":
        spec_type = (
            res.get("type", {}).get("coding", [{}])[0].get("code")
            or "Unknown Specimen Type"
        )
        date_str = res.get("collection", {}).get("collectedDateTime")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Specimen: {spec_type}"

      elif r_type == "AllergyIntolerance":
        name = (
            res.get("code", {}).get("text")
            or res.get("code", {}).get("coding", [{}])[0].get("display")
            or "Unknown Allergy"
        )
        status = (
            res.get("clinicalStatus", {})
            .get("coding", [{}])[0]
            .get("code", "unknown")
        )
        date_str = res.get("recordedDate")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Allergy: {name} (Status: {status})"

      elif r_type == "CarePlan":
        cat = (
            res.get("category", [{}])[0].get("text")
            or res.get("category", [{}])[0]
            .get("coding", [{}])[0]
            .get("display")
            or "Unknown CarePlan"
        )
        status = res.get("status", "unknown")
        date_str = res.get("period", {}).get("start")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] CarePlan: {cat} (Status: {status})"

      elif r_type == "CareTeam":
        status = res.get("status", "unknown")
        date_str = res.get("period", {}).get("start")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] CareTeam (Status: {status})"

      elif r_type == "Claim":
        claim_type = (
            res.get("type", {}).get("coding", [{}])[0].get("code", "unknown")
        )
        total = res.get("total", {}).get("value", "unknown")
        date_str = res.get("billablePeriod", {}).get("start") or res.get(
            "created"
        )
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Claim: {claim_type} - Total: {total}"

      elif r_type == "Device":
        name = res.get("type", {}).get("text") or res.get("deviceName", [{}])[
            0
        ].get("name", "Unknown Device")
        status = res.get("status", "unknown")
        date_str = res.get("manufactureDate")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Device: {name} (Status: {status})"

      elif r_type == "DiagnosticReport":
        name = (
            res.get("code", {}).get("text")
            or res.get("code", {}).get("coding", [{}])[0].get("display")
            or "Unknown DiagnosticReport"
        )
        status = res.get("status", "unknown")
        date_str = res.get("effectiveDateTime") or res.get("issued")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = (
            f"[{date_str or 'N/D'}] DiagnosticReport: {name} (Status: {status})"
        )

      elif r_type == "ExplanationOfBenefit":
        eob_type = (
            res.get("type", {}).get("coding", [{}])[0].get("code", "unknown")
        )
        outcome = res.get("outcome", "unknown")
        date_str = res.get("billablePeriod", {}).get("start") or res.get(
            "created"
        )
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = (
            f"[{date_str or 'N/D'}] ExplanationOfBenefit: {eob_type} (Outcome:"
            f" {outcome})"
        )

      elif r_type == "ImagingStudy":
        proc = (
            res.get("procedureCode", [{}])[0].get("text")
            or res.get("procedureCode", [{}])[0]
            .get("coding", [{}])[0]
            .get("display")
            or "Unknown ImagingStudy"
        )
        status = res.get("status", "unknown")
        date_str = res.get("started")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] ImagingStudy: {proc} (Status: {status})"

      elif r_type == "Immunization":
        name = (
            res.get("vaccineCode", {}).get("text")
            or res.get("vaccineCode", {}).get("coding", [{}])[0].get("display")
            or "Unknown Immunization"
        )
        status = res.get("status", "unknown")
        date_str = res.get("occurrenceDateTime")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = f"[{date_str or 'N/D'}] Immunization: {name} (Status: {status})"

      elif r_type == "SupplyDelivery":
        item = (
            res.get("suppliedItem", {})
            .get("itemCodeableConcept", {})
            .get("text")
            or res.get("suppliedItem", {})
            .get("itemCodeableConcept", {})
            .get("coding", [{}])[0]
            .get("display")
            or "Unknown SupplyDelivery"
        )
        status = res.get("status", "unknown")
        date_str = res.get("occurrenceDateTime")
        date_obj = fhir_utils.parse_fhir_date(date_str)
        text = (
            f"[{date_str or 'N/D'}] SupplyDelivery: {item} (Status: {status})"
        )

      elif r_type == "Patient":
        # Always include Patient demographics
        names = res.get("name", [])
        name_str = "Unknown"
        if names:
          family = names[0].get("family", "")
          given = " ".join(names[0].get("given", []))
          name_str = f"{given} {family}".strip()

        dob = res.get("birthDate", "Unknown")
        gender = res.get("gender", "Unknown")

        text = (
            f"*** PATIENT INFO ***\nName: {name_str}\nDOB: {dob}\nGender:"
            f" {gender}"
        )
        is_mandatory = True
        score = MANDATORY_SCORE

      else:
        continue

      # Score item if not mandatory
      if not is_mandatory:
        score = _score_resource(
            text, r_type, date_obj, base_tokens, expanded_tokens
        )

      all_parsed_items.append((date_obj, text, score))
      if score > 0:
        scored_items.append((date_obj, text, score))

    # Fallback if no match
    if not scored_items:
      if all_parsed_items:
        all_parsed_items.sort(key=lambda x: x[0], reverse=True)
        return "\n".join(x[1] for x in all_parsed_items[:50])

    # Sort by date (Newest First)
    scored_items.sort(key=lambda x: x[0], reverse=True)
    return "\n".join(x[1] for x in scored_items)
