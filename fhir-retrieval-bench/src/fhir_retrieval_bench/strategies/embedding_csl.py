"""Embedding CSL (Clinical Semantic Linking) strategy.

Uses per-resource embedding with type-specific serialization to find
clinically relevant resources, then retrieves those as context sorted
by relevance and date.
"""

import collections
import copy
import json
import logging
import os
from typing import Any

import numpy as np

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base
from fhir_retrieval_bench.utils import api
from fhir_retrieval_bench.utils import embeddings


logger = logging.getLogger(__name__)


def _serialize_resource_for_embedding(res: dict[str, Any]) -> str:
  """Produce a rich type-specific serialization for a FHIR resource.

  Port of reference lines 749-995.

  Args:
    res: The FHIR resource as a dictionary.

  Returns:
    A comma-joined string of descriptive parts for embedding and display.
  """
  r_type = res.get("resourceType", "Unknown")
  parts = [f"Type: {r_type}"]

  if r_type == "Observation":
    code = res.get("code", {}).get("text", "N/A")
    if code == "N/A" and res.get("code", {}).get("coding"):
      code = res.get("code", {}).get("coding")[0].get("display", "N/A")
    val = res.get("valueQuantity", {}).get("value")
    unit = res.get("valueQuantity", {}).get("unit", "")
    val_str = (
        f"{val} {unit}".strip()
        if val is not None
        else res.get("valueString", "N/A")
    )
    res_time = fhir_utils.get_resource_date(res) or "N/A"
    parts.extend([f"Code: {code}", f"Value: {val_str}", f"Time: {res_time}"])

  elif r_type == "Condition":
    code = res.get("code", {}).get("text", "N/A")
    if code == "N/A" and res.get("code", {}).get("coding"):
      code = res.get("code", {}).get("coding")[0].get("display", "N/A")
    status = (
        res.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "N/A")
    )
    onset = fhir_utils.get_resource_date(res) or "N/A"
    parts.extend([f"Condition: {code}", f"Status: {status}", f"Onset: {onset}"])

  elif r_type == "Procedure":
    code = res.get("code", {}).get("text", "N/A")
    if code == "N/A" and res.get("code", {}).get("coding"):
      code = res.get("code", {}).get("coding")[0].get("display", "N/A")
    res_time = fhir_utils.get_resource_date(res) or "N/A"
    parts.extend([f"Procedure: {code}", f"Time: {res_time}"])

  elif r_type == "Encounter":
    enc_class = res.get("class", {}).get("code", "N/A")
    enc_types = []
    if res.get("type"):
      for t in res.get("type", []):
        if t.get("coding"):
          enc_types.append(t["coding"][0].get("display", ""))
    enc_type_str = ", ".join(filter(None, enc_types))
    start = res.get("period", {}).get("start", "N/A")
    end = res.get("period", {}).get("end", "N/A")

    enc_parts = [
        f"Encounter Class: {enc_class}",
        f"Start: {start}",
        f"End: {end}",
    ]
    if enc_type_str:
      enc_parts.append(f"Type: {enc_type_str}")
    parts.extend(enc_parts)

  elif r_type in [
      "Medication",
      "MedicationRequest",
      "MedicationAdministration",
  ]:
    if r_type in ["MedicationRequest", "MedicationAdministration"]:
      name = res.get("medicationCodeableConcept", {}).get("text")
      if not name and res.get("medicationCodeableConcept", {}).get("coding"):
        name = (
            res.get("medicationCodeableConcept", {})
            .get("coding")[0]
            .get("display")
        )
        if not name:
          name = (
              res.get("medicationCodeableConcept", {})
              .get("coding")[0]
              .get("code")
          )
      if not name:
        name = res.get("medicationReference", {}).get("display")
    else:
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

    if not name:
      name = "Unknown Med"

    status = res.get("status", "unknown")
    date_str = fhir_utils.get_resource_date(res) or "N/A"

    dosage = ""
    if r_type == "MedicationRequest" and res.get("dosageInstruction"):
      timing = (
          res.get("dosageInstruction", [{}])[0]
          .get("timing", {})
          .get("code", {})
          .get("text", "")
      )
      route = (
          res.get("dosageInstruction", [{}])[0].get("route", {}).get("text", "")
      )
      if timing or route:
        dosage = f" {timing} {route}".strip()

    parts.extend([
        f"Medication: {name}{dosage}",
        f"Status: {status}",
        f"Date: {date_str}",
    ])

  elif r_type == "Location":
    name = res.get("name")
    if not name and res.get("identifier"):
      name = res.get("identifier", [{}])[0].get("value")
    parts.extend([f"Location: {name or 'Unknown Location'}"])

  elif r_type == "Specimen":
    spec_type = (
        res.get("type", {}).get("coding", [{}])[0].get("code")
        or "Unknown Specimen Type"
    )
    date_str = fhir_utils.get_resource_date(res) or "N/A"
    parts.extend([f"Specimen: {spec_type}", f"Time: {date_str}"])

  elif r_type == "AllergyIntolerance":
    name = res.get("code", {}).get("text")
    if not name and res.get("code", {}).get("coding"):
      name = res.get("code", {}).get("coding")[0].get("display")
    status = (
        res.get("clinicalStatus", {})
        .get("coding", [{}])[0]
        .get("code", "unknown")
    )
    parts.extend([
        f"Allergy: {name or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "CarePlan":
    cat = res.get("category", [{}])[0].get("text")
    if not cat and res.get("category", [{}])[0].get("coding"):
      cat = res.get("category", [{}])[0].get("coding")[0].get("display")
    status = res.get("status", "unknown")
    parts.extend([
        f"CarePlan: {cat or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "CareTeam":
    status = res.get("status", "unknown")
    parts.extend([
        f"CareTeam Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "Claim":
    claim_type = (
        res.get("type", {}).get("coding", [{}])[0].get("code", "unknown")
    )
    total = res.get("total", {}).get("value", "unknown")
    parts.extend([
        f"Claim Type: {claim_type}",
        f"Total: {total}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "Device":
    name = res.get("type", {}).get("text")
    if not name and res.get("deviceName"):
      name = res.get("deviceName", [{}])[0].get("name")
    status = res.get("status", "unknown")
    parts.extend([
        f"Device: {name or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "DiagnosticReport":
    name = res.get("code", {}).get("text")
    if not name and res.get("code", {}).get("coding"):
      name = res.get("code", {}).get("coding")[0].get("display")
    status = res.get("status", "unknown")
    parts.extend([
        f"DiagnosticReport: {name or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "DocumentReference":
    name = res.get("type", {}).get("coding", [{}])[0].get("display", "Note")
    status = res.get("status", "unknown")
    parts.extend([
        f"DocumentReference: {name}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "ExplanationOfBenefit":
    eob_type = res.get("type", {}).get("coding", [{}])[0].get("code", "unknown")
    outcome = res.get("outcome", "unknown")
    parts.extend([
        f"ExplanationOfBenefit Type: {eob_type}",
        f"Outcome: {outcome}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "ImagingStudy":
    proc = res.get("procedureCode", [{}])[0].get("text")
    if not proc and res.get("procedureCode", [{}])[0].get("coding"):
      proc = res.get("procedureCode", [{}])[0].get("coding")[0].get("display")
    status = res.get("status", "unknown")
    parts.extend([
        f"ImagingStudy: {proc or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "Immunization":
    name = res.get("vaccineCode", {}).get("text")
    if not name and res.get("vaccineCode", {}).get("coding"):
      name = res.get("vaccineCode", {}).get("coding")[0].get("display")
    status = res.get("status", "unknown")
    parts.extend([
        f"Immunization: {name or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  elif r_type == "SupplyDelivery":
    item = (
        res.get("suppliedItem", {}).get("itemCodeableConcept", {}).get("text")
    )
    if not item and res.get("suppliedItem", {}).get(
        "itemCodeableConcept", {}
    ).get("coding"):
      item = (
          res.get("suppliedItem", {})
          .get("itemCodeableConcept", {})
          .get("coding")[0]
          .get("display")
      )
    status = res.get("status", "unknown")
    parts.extend([
        f"SupplyDelivery: {item or 'Unknown'}",
        f"Status: {status}",
        f"Date: {fhir_utils.get_resource_date(res) or 'N/A'}",
    ])

  else:
    name = res.get("code", {}).get("text")
    if not name and res.get("code", {}).get("coding"):
      name = res.get("code", {}).get("coding")[0].get("display")
    if not name:
      name = "N/A"

    res_time = fhir_utils.get_resource_date(res) or "N/A"
    parts.extend([f"Name: {name}", f"Time: {res_time}"])

  return ", ".join(parts)


def serialize_fhir_bundle(
    fhir_bundle: dict[str, Any],
) -> tuple[list[tuple[dict[str, Any], str]], list[str]]:
  """Serializes a FHIR bundle into a list of text representations."""
  if not fhir_utils.verify_with_pydantic(fhir_bundle):
    raise ValueError("Failed to parse FHIR bundle with pydantic validation.")

  entries = fhir_bundle.get("entry", [])
  mandatory_texts = []
  serialized_resources = []

  for entry in entries:
    res = entry.get("resource", {})
    r_type = res.get("resourceType", "")

    if r_type == "Patient":
      name_parts = res.get("name", [{}])[0] if res.get("name") else {}
      given = " ".join(name_parts.get("given", []))
      family = name_parts.get("family", "")
      name = f"{given} {family}".strip() or "Unknown"
      dob = res.get("birthDate", "Unknown")
      gender = res.get("gender", "Unknown")
      mandatory_texts.append(
          f"*** PATIENT INFO ***\nName: {name}\nDOB: {dob}\nGender: {gender}"
      )

    elif r_type == "Encounter":
      cls = res.get("class", {}).get("code", "VISIT").upper()
      start = res.get("period", {}).get("start", "")
      end = res.get("period", {}).get("end", "")
      mandatory_texts.append(
          f"[{start}] *** ENCOUNTER: {cls} (End: {end or 'Ongoing'}) ***"
      )

    else:
      text_rep = _serialize_resource_for_embedding(res)
      serialized_resources.append((res, text_rep))
  return serialized_resources, mandatory_texts


def load_or_compute_node_embeddings(
    text_to_compute: list[str],
    embedder: embeddings.VectorCompass,
    cache_path: str | None = None,
) -> list[np.ndarray]:
  """Precompute embeddings for all nodes in the graph.

  Args:
    text_to_compute: The list of texts to compute embeddings for.
    embedder: The embedding model to use.
    cache_path: Optional path to load cached embeddings. We support *.parquet
      format files.

  Returns:
    A dictionary mapping node IDs to their embeddings.
  """
  if not text_to_compute:
    raise ValueError("No text to compute embeddings for.")

  if not cache_path:
    embedding_pool = {}
  else:
    embedding_pool = embeddings.load_embedding_pool(cache_path)
    if embedding_pool:
      logging.info("Loaded %d embeddings from cache.", len(embedding_pool))

  # Prepare text data for all nodes and identify missing embeddings.
  # Update embeddding_pool when necessary
  texts_not_in_cache = collections.OrderedDict()
  for text in text_to_compute:
    text_hash = embeddings.get_cache_hash(text, embedder.model_name)

    if text_hash not in embedding_pool:
      texts_not_in_cache[text_hash] = text
  if texts_not_in_cache:
    embeddings.update_embedding_pool(
        embedder, embedding_pool, texts_not_in_cache
    )
    if cache_path:
      embeddings.save_embedding_pool(embedding_pool, cache_path)

  # Construct the final result map.
  text_embeddings = [
      embedding_pool[embeddings.get_cache_hash(text, embedder.model_name)]
      for text in text_to_compute
  ]

  # Verify consistency of embedding dimensions.
  if text_embeddings:
    embedding_dims = {vec.shape[0] for vec in text_embeddings}
    if len(embedding_dims) > 1:
      raise ValueError(f"Inconsistent vector sizes found: {embedding_dims}")
    current_dim = list(embedding_dims)[0]
    if not hasattr(embedder, "vector_size"):
      embedder.vector_size = current_dim
  else:
    if not hasattr(embedder, "vector_size"):
      embedder.vector_size = embeddings.DEFAULT_VECTOR_SIZE

  return text_embeddings


class EmbeddingCSLStrategy(base.RAGStrategy):
  """Retrieves per-resource embeddings with type-specific serialization."""

  def __init__(
      self,
      creds: api.Credentials,
      answer_config: config.LLMConfig,
      embedding_config: config.EmbeddingConfig,
      saliency_threshold: float = 0.75,
      output_dir: str | None = None,
      cache_dir: str | None = None,
  ):
    super().__init__(creds=creds, answer_config=answer_config)
    # Embeddings only run through the public GenAI API in this build.
    if embedding_config.backend == "vertex":
      embedding_backend = api.GenAIBackend(
          use_vertex_ai=True,
          gcp_project_and_locations=list(creds.gcp_project_and_locations),
      )
    else:
      embedding_backend = api.GenAIBackend(
          genai_api_keys=list(creds.genai_api_keys),
          use_vertex_ai=False,
      )

    self.embedder = embeddings.GenAIEmbedder(
        backend=embedding_backend,
        model_name=embedding_config.model,
    )
    self.saliency_threshold = saliency_threshold
    self.similarities_dir = os.path.join(output_dir, "similarities")
    self.cache_dir = cache_dir
    os.makedirs(self.similarities_dir)
    logger.info(
        "Initialized: embedding_model=%s saliency_threshold=%.2f"
        " similarities_dir=%s",
        embedding_config.model,
        saliency_threshold,
        self.similarities_dir,
    )

  def prepare_fhir_context(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> str:
    """Embed per-resource serializations, filter by saliency, sort by date.

    Port of reference lines 1062-1208 (minus GCS caching).

    Args:
      record: The evaluation instance containing the question.
      fhir_bundle: The FHIR bundle containing patient data.

    Returns:
      The prepared context string.

    Steps:
      1. Extract entries from the FHIR bundle.
      2. Always include Patient and Encounter as mandatory context.
      3. Serialize all other resources with type-specific logic.
      4. Embed the question and all resource texts.
      5. Score each resource by cosine similarity to the question.
      6. Filter by saliency_threshold, sort by score then date.
      7. Return mandatory texts + filtered resource texts.
    """
    if not fhir_utils.verify_with_pydantic(fhir_bundle):
      raise ValueError("Failed to parse FHIR bundle with pydantic validation.")
    fhir_bundle = copy.deepcopy(fhir_bundle)

    logger.debug(
        "prepare_fhir_context: patient=%s",
        record.patient_id,
    )

    serialized_resources, mandatory_texts = serialize_fhir_bundle(fhir_bundle)
    if not serialized_resources:
      return "\n".join(mandatory_texts)

    resource_texts = [text for _, text in serialized_resources]

    query_emb = self.embedder.embed_text(
        record.question, task_type="SEMANTIC_SIMILARITY"
    )
    if query_emb is None:
      raise ValueError("Failed to embed the query.")

    cache_path = None
    if self.cache_dir:
      cache_path = os.path.join(
          self.cache_dir,
          f"embedding_csl_embeddings_{record.patient_id}.parquet",
      )
    res_embs = load_or_compute_node_embeddings(
        resource_texts, self.embedder, cache_path=cache_path
    )

    scored_items = []
    similarities_log: dict[str, dict[str, Any]] = {}
    for i, ((res, text_rep), res_emb) in enumerate(
        zip(serialized_resources, res_embs)
    ):
      score = self.embedder.similarity(query_emb, res_emb)
      date_obj = fhir_utils.parse_fhir_date(fhir_utils.get_resource_date(res))

      resource_id = res.get("id", f"unknown_{i}")
      resource_type = res.get("resourceType", "Unknown")
      similarities_log[f"{resource_type}/{resource_id}"] = {
          "score": float(score),
          "text": text_rep,
      }

      if score >= self.saliency_threshold:
        date_str = fhir_utils.get_resource_date(res) or "N/D"
        llm_text = f"[{date_str}] {text_rep} (Relevance Score: {score:.2f})"
        scored_items.append((score, date_obj, llm_text))

    # Save similarity scores locally for observability
    # if record.patient_id:
    #   query_hash = abs(hash(record.question)) % 100000000
    #   sim_path = os.path.join(
    #       self.similarities_dir, f"{record.patient_id}_{query_hash}.json"
    #   )
    #   try:
    #     with open(sim_path, "w") as f:
    #       json.dump(similarities_log, f, indent=2)
    #   except Exception as e:  # pylint: disable=broad-exception-caught
    #     logging.warning(
    #         "Failed to save similarities for %s: %s", record.patient_id, e
    #     )

    top_items = sorted(scored_items, key=lambda x: x[0], reverse=True)[:10000]
    top_items.sort(key=lambda x: x[1], reverse=True)

    return "\n".join(mandatory_texts + [x[2] for x in top_items])
