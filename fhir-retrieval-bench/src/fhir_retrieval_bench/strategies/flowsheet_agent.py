"""Flowsheet Agent strategy.

Builds an agent that interacts with the flowsheet tabular structure using full
object-oriented resource abstractions and querying tools.
"""

import abc
import collections
from collections.abc import Sequence
import datetime
import enum
import functools
import inspect
import json
import re
from typing import Any, TypeVar

from absl import logging
from google.adk import runners
from google.adk.agents import llm_agent
from google.adk.events import event as adk_event
from google.adk.sessions import in_memory_session_service
from google.adk.tools import function_tool
from google.genai import types
import pandas as pd
import pydantic

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.strategies import base
from fhir_retrieval_bench.strategies import flowsheet
from fhir_retrieval_bench.utils import api
from ns_agent_adk import engine as engine_module
from ns_agent_adk.config import config as ns_agent_config_module

_SYSTEM_INSTRUCTION = """
You are an expert clinician and FHIR semantic specialist answering a patient's
medical query. You have access to tabular flowsheet databases that map directly
to native FHIR resources and representations.

Operational Workflow & Core Principles:
1. Begin by invoking the metadata tool to view the full list of available FHIR
   resource types and medical concepts.
2. Based on the patient's query, efficiently pull relevant data tables.
3. Real-world FHIR resources and EHR data often contain noise, overlapping
   entries, auxiliary sub-encounters, and mapping imperfections. Synthesize data
   carefully across these resources to identify true clinical visits, filter out
   overlapping duplicates, and isolate valid measurements.
4. Missing Data on Totals/Sums: If a query asks for the sum or total
   output/volume on a specific requested date, but no measurements whatsoever
   exist on that requested date, you MUST NOT return 0. Instead, return exactly:
   `[ANSWER]Value not recorded or not applicable[/ANSWER]`.
5. Budget Management: You have a strict, limited total reasoning budget.
   Be exceptionally concise and direct in your thought turns. Minimize
   unnecessary tool executions, avoid writing redundant or verbose
   self-descriptions of what you plan to query next, and focus powerfully on
   achieving your final answer as quickly as possible.
6. Temporal Filtering: Pay close attention to any assumed 'current time'
   specified in the query. You MUST strictly ignore and filter out all records,
   encounters, observations, procedures, and medications that reside in the
   future relative to the assumed current time. If the queried data or patient
   is truly unrecorded or missing within the requested window, output:
   `Value not recorded or not applicable`.
7. Batching Requests: When you need to retrieve multiple dataframes (e.g.,
   pulling a specific clinical concept and the encounter list to cross-reference
   dates), you MUST request them all together in a single tool call. The tool
   accepts a list of identifiers; pass all necessary identifiers in that list
   at once rather than querying them sequentially.

CRITICAL OUTPUT FORMATTING: You MUST follow all formatting instructions
specified in the user query.
- At the very end of your final response, you MUST wrap the exact required
  formatted target answer inside standard [ANSWER]...[/ANSWER] XML-style tags
  (e.g., `[ANSWER]Yes[/ANSWER]` or `[ANSWER]No[/ANSWER]` or
  `[ANSWER]-1[/ANSWER]`). CRITICAL: The content inside the
  `[ANSWER]...[/ANSWER]` tags must be ONLY the direct raw answer string
  (no commentary, no preconditioned conversational sentences, no conversational
  reasoning, no Markdown formatting)
- When returning any date/time answers, you MUST return the full exact timestamp
  matching the database record format (including hours, minutes, and seconds, 
  e.g., `2116-12-27 16:27:00` or `2116-12-27T16:27:00-05:00`), never simplify or
  drop hours/minutes/seconds.
- You MUST make sure your response inside the `[ANSWER]...[/ANSWER]` tags aligns
  faithfully with the specific entity type asked for by the original query
  (e.g., if asked for a test name, output **ONLY** the exact clinical name
  string; if asked for a time, output ONLY the timestamp; if asked for a
  quantity, output ONLY the numeric value).
- If asked a binary/verification question (such as yes/no, true/false), you MUST
  output ONLY the exact single verdict word inside the tags (e.g.,
  `[ANSWER]Yes[/ANSWER]`, `[ANSWER]No[/ANSWER]`, `[ANSWER]True[/ANSWER]`,
  `[ANSWER]False[/ANSWER]`). It is explicitly FORBIDDEN to include surrounding
  explanatory sentences, conversational summaries, "Yes, the patient is...", or
  clinical narrative paragraphs inside these tags.
"""


_T = TypeVar("_T", bound="_BaseFHIRModel")


class _BaseFHIRModel(pydantic.BaseModel):
  """Base model providing automated camelCase parsing alias generator."""

  model_config = pydantic.ConfigDict(
      extra="ignore",
      populate_by_name=True,
      alias_generator=pydantic.alias_generators.to_camel,
  )

  @classmethod
  def safe_validate_json(cls: type[_T], json_str: str) -> _T | None:
    if not json_str or json_str == "{}":
      return None
    try:
      return cls.model_validate_json(json_str)
    except pydantic.ValidationError:
      return None


class _CodingModel(_BaseFHIRModel):
  """Coding model containing code, display, and system."""

  display: str | None = None
  code: str | None = None
  system: str | None = None


class _CodeableConceptModel(_BaseFHIRModel):
  """CodeableConcept parsing textual displays and list of codings."""

  text: Any | None = None
  coding: list[_CodingModel] = pydantic.Field(default_factory=list)

  @pydantic.model_validator(mode="before")
  @classmethod
  def _handle_string_input(cls, data: Any) -> Any:
    if isinstance(data, str):
      return {"text": data}
    return data

  def get_display(self) -> str:
    displays = (
        [str(self.text)] if self.text is not None and self.text != "" else []
    )
    displays.extend(
        c.display or c.code for c in self.coding if c.display or c.code
    )
    unique_displays = list(
        dict.fromkeys(d.strip() for d in displays if d.strip())
    )
    return " / ".join(unique_displays) if unique_displays else "N/A"


class _ReferenceModel(_BaseFHIRModel):
  """Reference resource mapping target paths and display titles."""

  reference: str | None = None
  display: str | None = None


class _QuantityModel(_BaseFHIRModel):
  """Quantity model conveying numeric value and string unit."""

  value: Any | None = None
  unit: str | None = None

  def __str__(self) -> str:
    if self.value is None:
      return "N/A"
    return f"{self.value} {self.unit or ''}".strip()


class _IdentifierModel(_BaseFHIRModel):
  """Identifier model representing system-scoped business keys."""

  system: str | None = None
  value: Any | None = None


class _PeriodModel(_BaseFHIRModel):
  """Period element representing start and end timestamps."""

  start: str | None = None
  end: str | None = None


class _EncounterClassModel(_BaseFHIRModel):
  """Encounter class outlining inpatient/outpatient classes."""

  display: str | None = None
  code: str | None = None


class _EncounterModel(_BaseFHIRModel):
  """Encounter resource specifying classes, type codes, and dates."""

  id: str | None = None
  status: str | None = None
  class_: _EncounterClassModel | None = pydantic.Field(
      default=None, alias="class"
  )
  type: list[_CodeableConceptModel] = pydantic.Field(default_factory=list)
  period: _PeriodModel | None = None

  def get_context_string(self) -> str:
    e_name = None
    ec = self.class_
    if ec is not None:
      e_name = ec.display or ec.code
    if not e_name:
      e_name = next(
          (c.display for t in self.type for c in t.coding if c.display),
          "Encounter",
      )

    e_date = ""
    ep = self.period
    if ep is not None and ep.start:
      e_date = str(ep.start)

    return f"{e_name} ({e_date[:10]})" if e_date else str(e_name)


class _IngredientStrengthModel(_BaseFHIRModel):
  """Ingredient strength measurement matching numerator ratios."""

  numerator: _QuantityModel | None = None


class _IngredientModel(_BaseFHIRModel):
  """Medication ingredient detailing referenced items and strengths."""

  item_reference: _ReferenceModel | None = None
  strength: _IngredientStrengthModel | None = None


class _MedicationModel(_BaseFHIRModel):
  """Medication resource specifying codes, ingredients, and amounts."""

  id: str | None = None
  code: _CodeableConceptModel | None = None
  identifier: list[_IdentifierModel] = pydantic.Field(default_factory=list)
  ingredient: list[_IngredientModel] = pydantic.Field(default_factory=list)
  amount: _IngredientStrengthModel | None = None


class _PatientNameModel(_BaseFHIRModel):
  """Patient name component parsing given names and family names."""

  given: list[str] = pydantic.Field(default_factory=list)
  family: str | None = None

  def get_full_name(self) -> str:
    g_str = " ".join(self.given)
    return f"{g_str} {self.family or ''}".strip()


class _PatientModel(_BaseFHIRModel):
  """Patient resource outlining identifiers, birth date, and gender."""

  id: str | None = None
  identifier: list[_IdentifierModel] = pydantic.Field(default_factory=list)
  birth_date: str | None = None
  gender: str | None = None
  name: list[_PatientNameModel] = pydantic.Field(default_factory=list)


class _TimingRepeatModel(_BaseFHIRModel):
  """Timing repeat options tracking frequencies and period durations."""

  frequency: Any | None = None
  period: Any | None = None
  period_unit: str | None = None

  def get_display(self) -> str:
    if (
        self.frequency is not None
        and self.period is not None
        and self.period_unit is not None
    ):
      return f"{self.frequency} per {self.period} {self.period_unit}"
    elif self.frequency is not None:
      return f"{self.frequency} times"
    return ""


class _TimingModel(_BaseFHIRModel):
  """Timing element parsing repetition schedules and codes."""

  code: _CodeableConceptModel | None = None
  repeat: _TimingRepeatModel | None = None


class _DoseAndRateModel(_BaseFHIRModel):
  """Dosage doseAndRate component keeping target dose quantities."""

  dose_quantity: _QuantityModel | None = None


class _DosageModel(_BaseFHIRModel):
  """Dosage containing texts, timing patterns, and quantities."""

  text: str | None = None
  dose_and_rate: list[_DoseAndRateModel] | _DoseAndRateModel | None = None
  timing: _TimingModel | None = None
  dose: _QuantityModel | None = None
  rate_quantity: _QuantityModel | None = None
  route: _CodeableConceptModel | None = None


class _ObservationValueModel(_BaseFHIRModel):
  """Observation value types supporting primitive and complex fields."""

  value_quantity: _QuantityModel | None = None
  value_string: str | None = None
  value_codeable_concept: _CodeableConceptModel | None = None
  value_date_time: str | None = None
  value_boolean: bool | None = None
  value_integer: int | None = None
  value_time: str | None = None

  def resolve_display(self) -> str:
    if (
        self.value_quantity is not None
        and self.value_quantity.value is not None
    ):
      return str(self.value_quantity.value)
    if self.value_codeable_concept is not None:
      return self.value_codeable_concept.get_display()
    primitives = [
        self.value_string,
        self.value_date_time,
        self.value_boolean,
        self.value_integer,
        self.value_time,
    ]
    val = next((v for v in primitives if v is not None), None)
    if val is not None:
      return str(val)
    return "N/A"


class _ComponentModel(_ObservationValueModel):
  """Observation component carrying nested values and codes."""

  code: _CodeableConceptModel | None = None


class _FHIRRegistry:
  """Centralized lookup registry."""

  def __init__(self, fhir_bundle: dict[str, Any]):
    self._encounters: dict[str, _EncounterModel] = {}
    self._medications: dict[str, _MedicationModel] = {}

    def _parse_logic(data_dict: dict[str, Any]) -> str:
      try:
        model = _CodeableConceptModel.model_validate(data_dict)
        return model.get_display() if model is not None else "N/A"
      except Exception:
        return "N/A"

    self.get_display_for_code = _parse_logic
    self._raw_entries: list[dict[str, Any]] = fhir_bundle.get("entry", [])

    for entry in self._raw_entries:
      res = entry.get("resource", {})
      r_type = res.get("resourceType")
      res_id = str(res.get("id", ""))

      if r_type == "Encounter" and res_id:
        norm_id = self._normalize_reference(res_id, "Encounter/")
        self._encounters[norm_id] = _EncounterModel.model_validate(res)
      elif r_type == "Medication" and res_id:
        norm_id = self._normalize_reference(res_id, "Medication/")
        self._medications[norm_id] = _MedicationModel.model_validate(res)

  def _normalize_reference(
      self, reference_str: str | None, resource_prefix: str
  ) -> str:
    """Refinement 5 helper normalising identifier paths clearly."""
    if not reference_str:
      return ""
    return (
        reference_str.replace("urn:uuid:", "")
        .replace(resource_prefix, "")
        .strip()
    )

  def get_encounter_context(self, reference_str: str | None) -> str:
    """Returns the formatted class/type and date string for an encounter reference."""
    norm_ref = self._normalize_reference(reference_str, "Encounter/")
    if not norm_ref:
      return "N/A"

    enc = self._encounters.get(norm_ref)
    if not enc:
      return "N/A"
    return enc.get_context_string()

  def get_encounter(self, reference_str: str | None) -> _EncounterModel | None:
    """Retrieves the parsed _EncounterModel in."""
    norm_ref = self._normalize_reference(reference_str, "Encounter/")
    if not norm_ref:
      return None
    return self._encounters.get(norm_ref)

  def get_medication(
      self, reference_str: str | None
  ) -> _MedicationModel | None:
    """Retrieves the parsed _MedicationModel."""
    norm_ref = self._normalize_reference(reference_str, "Medication/")
    if not norm_ref:
      return None
    return self._medications.get(norm_ref)

  @property
  def raw_entries(self) -> list[dict[str, Any]]:
    return self._raw_entries


class _DownsampleGranularity(enum.Enum):
  """Downsample granularities in increasing order of size."""

  MINUTE = "min"
  HOUR = "h"
  DAY = "D"
  WEEK = "W"
  MONTH = "M"


def _downsample(
    df: pd.DataFrame,
    *,
    item_id_cols: Sequence[str],
    datetime_col: str,
    granularity: _DownsampleGranularity,
) -> None:
  """Downsamples the given dataframe by retaining the last recorded row per group period."""
  if df.empty:
    return

  df.reset_index(drop=True, inplace=True)
  temp_bucket_col = "__downsample_bucket_temp__"
  datetimes = pd.to_datetime(df[datetime_col], errors="coerce", utc=True)

  if granularity == _DownsampleGranularity.MONTH:
    df[temp_bucket_col] = datetimes.dt.to_period("M")
  elif granularity == _DownsampleGranularity.WEEK:
    df[temp_bucket_col] = datetimes.dt.to_period("W")
  else:
    df[temp_bucket_col] = datetimes.dt.floor(granularity.value)

  def _to_hashable(x):
    if isinstance(x, (list, tuple, dict)):
      try:
        return json.dumps(x, sort_keys=True)
      except TypeError as e:
        logging.warning(
            "Failed JSON dumps serialization in _to_hashable", exc_info=e
        )
        return str(x)
    return str(x)

  subset = list(item_id_cols) + [temp_bucket_col]
  duplicates_mask = (
      df[subset].map(_to_hashable).duplicated(keep="last")
      & df[temp_bucket_col].notnull()
  )

  df.drop(index=df.index[duplicates_mask], inplace=True)
  df.drop(columns=[temp_bucket_col], inplace=True)


def _sort_by_datetime(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    ascending: bool = True,
    extra_sort_cols: Sequence[str] = (),
) -> None:
  """Sorts the given dataframe chronologically in place."""
  if df.empty:
    return

  df.reset_index(inplace=True, drop=True)
  temp_sort_col = "__sort_datetime_temp__"
  df[temp_sort_col] = pd.to_datetime(
      df[datetime_col], errors="coerce", utc=True
  )
  sort_cols = [temp_sort_col] + list(extra_sort_cols)
  df.sort_values(
      by=sort_cols, ascending=ascending, na_position="first", inplace=True
  )
  df.drop(columns=[temp_sort_col], inplace=True)
  df.reset_index(inplace=True, drop=True)


def _adaptively_downsample(
    df: pd.DataFrame,
    *,
    item_id_cols: Sequence[str],
    datetime_col: str,
    initial_granularity: _DownsampleGranularity,
    max_total_rows: int = 500,
) -> None:
  """Iteratively compresses records horizontally across expanding time periods."""
  if df.empty:
    return

  all_granularities = list(_DownsampleGranularity)
  start_idx = all_granularities.index(initial_granularity)

  for i in range(start_idx, len(all_granularities)):
    _downsample(
        df,
        item_id_cols=item_id_cols,
        datetime_col=datetime_col,
        granularity=all_granularities[i],
    )
    if len(df) <= max_total_rows:
      return

  _sort_by_datetime(df, datetime_col=datetime_col, ascending=True)
  if len(df) > max_total_rows:
    df.drop(df.index[:-max_total_rows], inplace=True)
    df.reset_index(drop=True, inplace=True)


class _FlowsheetResource(abc.ABC):
  """Abstract encapsulation of a FHIR resource type's flowsheet representation."""

  resource_type: str

  @property
  def name(self) -> str:
    return getattr(self, "resource_type", "Unknown")

  def __init__(
      self,
      resources: list[dict[str, Any]],
      registry: _FHIRRegistry | None = None,
  ):
    self._registry = registry or _FHIRRegistry({"entry": []})
    self._resources = resources

  @functools.cached_property
  def df(self) -> pd.DataFrame:
    rows = self._extract_rows(self._resources, self._registry)
    if rows:
      return pd.DataFrame(rows, columns=self._columns())
    return pd.DataFrame(columns=self._columns())

  @functools.cached_property
  def dfs_by_display(self) -> dict[str, pd.DataFrame]:
    res = {}
    if not self.df.empty:
      display_col = self._display_column()
      for display, group in self.df.groupby(display_col):
        res[str(display)] = group
    return res

  def _extract_rows(
      self, resources: list[dict[str, Any]], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    """Generalized extraction loop delegating clean payload extraction to subclasses."""
    rows = []
    for res in resources:
      rows.extend(self._parse_resource(res, registry))
    return rows

  def _extract_standard_metadata(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> tuple[str, str]:
    """Extracts standard date and encounter context, falling back to encounter dates if missing."""
    date = fhir_utils.get_resource_date_v2(res)
    enc_ref = res.get("encounter", {}).get("reference") or res.get(
        "context", {}
    ).get("reference")

    # fallback to encounter date if resource date is missing
    if not date and enc_ref:
      enc = registry.get_encounter(enc_ref)
      if enc and enc.period and enc.period.start:
        date = str(enc.period.start)

    date = date or "N/A"
    enc_ctx = registry.get_encounter_context(enc_ref)
    return date, enc_ctx

  @abc.abstractmethod
  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    """Parses a single matching resource into tabular tuple rows."""
    pass

  @abc.abstractmethod
  def _columns(self) -> list[str]:
    """Column names for the generated DataFrame."""
    pass

  @abc.abstractmethod
  def _display_column(self) -> str:
    """Column containing primary display names used for filtering."""
    pass

  @property
  def display_column(self) -> str:
    """Public read-only property exposing the primary display column."""
    return self._display_column()

  def get_metadata(self) -> list[str]:
    """Returns sorted list of unique display names exposed by this resource."""
    if self.df.empty:
      return []
    return sorted(list(self.df[self._display_column()].unique()))


class _ObservationResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Observation resources."""

  resource_type = "Observation"

  def __init__(
      self,
      resources: list[dict[str, Any]],
      registry: _FHIRRegistry | None = None,
  ):
    super().__init__(resources, registry)

  def _parse_observation_value(
      self, data_dict: dict[str, Any]
  ) -> tuple[str, str]:
    try:
      val_model = _ObservationValueModel.model_validate(data_dict)
      vq = val_model.value_quantity
      return (
          val_model.resolve_display(),
          (vq.unit or "") if vq is not None else "",
      )
    except Exception:
      return "N/A", ""

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    rows = []
    date, enc_ctx = self._extract_standard_metadata(res, registry)

    components = res.get("component", [])
    if components:
      for comp in components:
        comp_code = registry.get_display_for_code(comp.get("code", {}))
        comp_val, comp_unit = self._parse_observation_value(comp)
        rows.append((date, comp_code, comp_val, comp_unit, enc_ctx))
    else:
      val, unit = self._parse_observation_value(res)
      if val == "N/A" and "hasMember" in res:
        return []
      code = registry.get_display_for_code(res.get("code", {}))
      rows.append((date, code, val, unit, enc_ctx))
    return rows

  def _columns(self) -> list[str]:
    return ["Date", "Observation", "Value", "Unit", "EncounterContext"]

  def _display_column(self) -> str:
    return "Observation"


def _is_valid_med_name(name: Any) -> bool:
  """Validates that a extracted medication name is present, not a placeholder, and not just numbers."""
  if not name:
    return False
  name_str = str(name).strip()
  if name_str.lower() in {"unknown", "none", "n/a", ""}:
    return False
  if name_str.isdigit():
    return False
  return True


def _fallback_parse_strength_from_name(name_str: str | None) -> str | None:
  """Standalone helper parsing physical strength substring from medication name via regex."""
  if not _is_valid_med_name(name_str):
    return None
  m = re.search(
      r"(\d+(?:\.\d+)?)\s*((?:mg|g|ml|mcg|units|unit|meq|%)(?:\s*\/\s*(?:ml|mg|g|l))?|(?:\s*\/\s*)(?:ml|mg))",
      name_str,
      re.IGNORECASE,
  )
  if m:
    unit = re.sub(r"\s+", "", m.group(2))
    return f"{m.group(1)} {unit}".strip()
  return None


class _MedicationBaseResource(_FlowsheetResource):
  """Base Flowsheet class representing decomposed medication resource types."""

  def _resolve_med_name(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> str:
    """Resolves medication name via structured, flattened early-return fallbacks."""
    native_name = flowsheet.get_medication_name(res)
    if _is_valid_med_name(native_name):
      return str(native_name)

    med_ref = res.get("medicationReference", {}).get("reference")
    med = registry.get_medication(med_ref)
    if not med:
      return "Unknown"

    if med.code:
      c_name = med.code.get_display()
      if _is_valid_med_name(c_name) and c_name != "N/A":
        return c_name

    sys_name = self._try_system_identifier(med)
    if sys_name and _is_valid_med_name(sys_name):
      return sys_name

    ing_name = self._try_ingredient_fallback(med, registry)
    if ing_name and _is_valid_med_name(ing_name):
      return ing_name

    return "Unknown"

  def _try_system_identifier(self, med: _MedicationModel) -> str | None:
    """Extracts name from identifiers like MIMIC or medication-mix."""
    for ident in med.identifier:
      sys = ident.system or ""
      if "mimic-medication-name" in sys and ident.value:
        return str(ident.value)
    for ident in med.identifier:
      sys = ident.system or ""
      if "medication-mix" in sys and ident.value:
        raw_mix = str(ident.value)
        parts = []
        for seg in raw_mix.split("_"):
          subparts = seg.split("--")
          if subparts:
            parts.append(subparts[0])
        return " / ".join(parts) if parts else raw_mix
    return None

  def _get_name_from_medication_model(
      self, med: _MedicationModel
  ) -> str | None:
    """Helper: Extracts a valid name from a medication's code or identifiers."""
    if med.code:
      c_name = med.code.get_display()
      if _is_valid_med_name(c_name) and c_name != "N/A":
        return c_name

    for ident in med.identifier:
      if "mimic-medication-name" in (ident.system or "") and ident.value:
        sys_name = str(ident.value)
        if _is_valid_med_name(sys_name):
          return sys_name

    return None

  def _try_ingredient_fallback(
      self, med: _MedicationModel, registry: _FHIRRegistry
  ) -> str | None:
    """Resolves the first valid ingredient name from a medication."""
    for ing in med.ingredient:
      ref_str = ing.item_reference.reference if ing.item_reference else None
      if child_med := registry.get_medication(ref_str):
        child_name = self._get_name_from_medication_model(child_med)
        if child_name:
          return child_name
    return None

  def _get_dose_from_medication_model(
      self, med: _MedicationModel
  ) -> str | None:
    """Extracts dosage strictly from a _MedicationModel's ingredients or total amount."""
    # Check ingredients
    for ing in med.ingredient:
      num = ing.strength.numerator if ing.strength else None
      if num and num.value is not None:
        return f"{num.value} {num.unit or ''}".strip()

    # Check total amount
    num = med.amount.numerator if med.amount else None
    if num and num.value is not None:
      return f"{num.value} {num.unit or ''}".strip()

    return None

  def _extract_comprehensive_dosage(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> tuple[str, str, str]:
    """Extracts dose, frequency, and instructions for medication resources."""
    dose, freq, inst = "", "", ""
    r_type = res.get("resourceType")

    # Base extraction driven by resource type
    if r_type in ("MedicationRequest", "MedicationDispense"):
      dose, freq, inst = self._extract_instruction_dosage(
          res.get("dosageInstruction")
      )
    elif r_type == "MedicationStatement":
      dose, freq, inst = self._extract_instruction_dosage(res.get("dosage"))
    elif r_type == "MedicationAdministration":
      dose, inst = self._extract_administration_dosage(res.get("dosage", {}))

    # Sequential fallbacks for missing dose
    if not dose or dose == "N/A":
      med_ref = res.get("medicationReference", {}).get("reference")
      if med := registry.get_medication(med_ref):
        dose = self._get_dose_from_medication_model(med)

    if not dose or dose == "N/A":
      resolved_name = self._resolve_med_name(res, registry)
      dose = _fallback_parse_strength_from_name(resolved_name)

    return dose or "N/A", freq or "N/A", inst or "N/A"

  def _extract_instruction_dosage(self, dis: Any) -> tuple[str, str, str]:
    """Extracts dose, frequency, and instructions from a Dosage dictionary."""

    # Normalize input: safely grab the first item if it's a list
    first_di = dis[0] if isinstance(dis, list) and dis else dis
    if not isinstance(first_di, dict):
      return "", "", ""

    dm = _DosageModel.model_validate(first_di)

    # Extract Instructions
    inst = dm.text or ""

    # Extract Dose
    dose = ""
    first_dr = (
        dm.dose_and_rate[0]
        if isinstance(dm.dose_and_rate, list) and dm.dose_and_rate
        else dm.dose_and_rate
    )

    if (
        isinstance(first_dr, _DoseAndRateModel)
        and first_dr.dose_quantity
        and first_dr.dose_quantity.value is not None
    ):
      dose = (
          f"{first_dr.dose_quantity.value} {first_dr.dose_quantity.unit or ''}"
          .strip()
      )

    # Extract Frequency
    freq = ""
    if dm.timing:
      if dm.timing.code:
        freq = dm.timing.code.text or dm.timing.code.get_display()
        if freq == "N/A":
          freq = ""

      if not freq and dm.timing.repeat:
        freq = dm.timing.repeat.get_display()

    return dose, freq, inst

  def _extract_administration_dosage(
      self, ds: dict[str, Any]
  ) -> tuple[str, str]:
    """Extracts dose and instruction for medication administration."""
    dose = ""
    dm = _DosageModel.model_validate(ds)
    inst = dm.text or ""
    if dm.dose is not None and dm.dose.value is not None:
      dose = f"{dm.dose.value} {dm.dose.unit or ''}".strip()
    if (
        not dose
        and dm.rate_quantity is not None
        and dm.rate_quantity.value is not None
    ):
      dose = f"{dm.rate_quantity.value} {dm.rate_quantity.unit or ''}".strip()
    return dose, inst

  def _display_column(self) -> str:
    return self.name


class _MedicationRequestResource(_MedicationBaseResource):
  """Flowsheet abstraction for FHIR MedicationRequest resources."""

  resource_type = "MedicationRequest"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = self._resolve_med_name(res, registry)
    dose, freq, inst = self._extract_comprehensive_dosage(res, registry)

    intent = res.get("intent", "")
    status = res.get("status", "")

    req = res.get("requester") or {}
    perf = res.get("performer") or {}

    provider = req.get("display") or perf.get("display") or ""

    return [(date, name, dose, freq, inst, intent, provider, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return [
        "Date",
        "MedicationRequest",
        "Dose",
        "Frequency",
        "Instructions",
        "Intent",
        "Provider",
        "Status",
        "EncounterContext",
    ]


class _MedicationAdministrationResource(_MedicationBaseResource):
  """Flowsheet abstraction for FHIR MedicationAdministration resources."""

  resource_type = "MedicationAdministration"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = self._resolve_med_name(res, registry)
    dose, _, _ = self._extract_comprehensive_dosage(res, registry)

    ds = res.get("dosage", {})
    rate = str(_QuantityModel.model_validate(ds.get("rate_quantity", {})))

    route = flowsheet.get_code_display(ds.get("route", {}))
    status = res.get("status", "")
    return [(date, name, dose, rate, route, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return [
        "Date",
        "MedicationAdministration",
        "Dose",
        "Rate",
        "Route",
        "Status",
        "EncounterContext",
    ]


class _MedicationStatementResource(_MedicationBaseResource):
  """Flowsheet abstraction for FHIR MedicationStatement resources."""

  resource_type = "MedicationStatement"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = self._resolve_med_name(res, registry)
    dose, freq, inst = self._extract_comprehensive_dosage(res, registry)

    info_src = flowsheet.get_code_display(res.get("informationSource", {}))
    if info_src in ["", "N/A", "Unknown", "unknown"]:
      info_src = "Patient"

    status = res.get("status", "")
    return [(date, name, dose, freq, inst, info_src, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return [
        "Date",
        "MedicationStatement",
        "Dose",
        "Frequency",
        "Instructions",
        "InformationSource",
        "Status",
        "EncounterContext",
    ]


class _MedicationDispenseResource(_MedicationBaseResource):
  """Flowsheet abstraction for FHIR MedicationDispense resources."""

  resource_type = "MedicationDispense"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = self._resolve_med_name(res, registry)
    dose, freq, inst = self._extract_comprehensive_dosage(res, registry)

    q_disp = str(_QuantityModel.model_validate(res.get("quantity", {})))
    d_sup = str(_QuantityModel.model_validate(res.get("daysSupply", {})))

    status = res.get("status", "")
    return [(date, name, dose, freq, inst, q_disp, d_sup, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return [
        "Date",
        "MedicationDispense",
        "Dose",
        "Frequency",
        "Instructions",
        "QuantityDispensed",
        "DaysSupply",
        "Status",
        "EncounterContext",
    ]


class _ConditionResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Condition resources."""

  resource_type = "Condition"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = registry.get_display_for_code(res.get("code", {}))

    status = registry.get_display_for_code(res.get("clinicalStatus", {}))
    if status in ["", "unknown", "N/A", "None"]:
      status = registry.get_display_for_code(res.get("verificationStatus", {}))

    return [(date, name, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return ["Date", "Condition", "Status", "EncounterContext"]

  def _display_column(self) -> str:
    return "Condition"


class _ProcedureResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Procedure resources."""

  resource_type = "Procedure"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = registry.get_display_for_code(res.get("code", {}))
    status = res.get("status", "")
    return [(date, name, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return ["Date", "Procedure", "Status", "EncounterContext"]

  def _display_column(self) -> str:
    return "Procedure"


class _ImmunizationResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Immunization resources."""

  resource_type = "Immunization"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = registry.get_display_for_code(res.get("vaccineCode", {}))
    status = res.get("status", "")
    return [(date, name, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return ["Date", "Immunization", "Status", "EncounterContext"]

  def _display_column(self) -> str:
    return "Immunization"


class _AllergyIntoleranceResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR AllergyIntolerance resources."""

  resource_type = "AllergyIntolerance"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date = fhir_utils.get_resource_date_v2(res) or "N/A"
    name = registry.get_display_for_code(res.get("code", {}))
    status = registry.get_display_for_code(res.get("clinicalStatus", {}))
    if status in ["", "unknown", "N/A", "None"]:
      status = registry.get_display_for_code(res.get("verificationStatus", {}))
    return [(date, name, status)]

  def _columns(self) -> list[str]:
    return ["Date", "AllergyIntolerance", "Status"]

  def _display_column(self) -> str:
    return "AllergyIntolerance"


class _PatientResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Patient resources."""

  resource_type = "Patient"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    pm = _PatientModel.model_validate(res)
    ids = [str(pm.id)] if pm.id is not None else []
    for ident in pm.identifier:
      if ident.value is not None:
        v_str = str(ident.value)
        if v_str not in ids:
          ids.append(v_str)
    patient_id = " / ".join(ids)
    dob = str(pm.birth_date) if pm.birth_date is not None else ""
    gender = str(pm.gender) if pm.gender is not None else ""
    name_str = pm.name[0].get_full_name() if pm.name else ""
    return [(patient_id, name_str, dob, gender)]

  def _columns(self) -> list[str]:
    return ["ID", "Name", "DOB", "Gender"]

  def _display_column(self) -> str:
    return "Name"


class _DiagnosticReportResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR DiagnosticReport resources."""

  resource_type = "DiagnosticReport"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = registry.get_display_for_code(res.get("code", {}))
    val = res.get("conclusion") or "N/A"
    status = res.get("status", "")
    return [(date, name, val, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return ["Date", "DiagnosticReport", "Value", "Status", "EncounterContext"]

  def _display_column(self) -> str:
    return "DiagnosticReport"


class _SpecimenResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Specimen resources."""

  resource_type = "Specimen"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = registry.get_display_for_code(res.get("type", {}))
    status = res.get("status", "")
    return [(date, name, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return ["Date", "Specimen", "Status", "EncounterContext"]

  def _display_column(self) -> str:
    return "Specimen"


class _ImagingStudyResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR ImagingStudy resources."""

  resource_type = "ImagingStudy"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    name = (
        res.get("description")
        or (
            res.get("series", [{}])[0].get("description")
            if res.get("series")
            else None
        )
        or "Imaging Study"
    )
    val = (
        res.get("series", [{}])[0].get("modality", {}).get("code", "N/A")
        if res.get("series")
        else "N/A"
    )
    status = res.get("status", "")
    return [(date, name, val, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return ["Date", "ImagingStudy", "Value", "Status", "EncounterContext"]

  def _display_column(self) -> str:
    return "ImagingStudy"


def _calculate_duration_days(start_str: str | None, end_str: str | None) -> str:
  """Refinement 4 pure module helper abstracting datetime duration math safely."""
  if not start_str or not end_str:
    return "N/A"
  try:

    t_start = datetime.datetime.fromisoformat(start_str.replace("Z", "+00:00"))
    t_end = datetime.datetime.fromisoformat(end_str.replace("Z", "+00:00"))
    return f"{(t_end - t_start).total_seconds() / 86400:.8f}"
  except ValueError:
    return "N/A"


class _EncounterResource(_FlowsheetResource):
  """Flowsheet abstraction for FHIR Encounter resources."""

  resource_type = "Encounter"

  def _parse_resource(
      self, res: dict[str, Any], registry: _FHIRRegistry
  ) -> list[tuple[Any, ...]]:
    date, enc_ctx = self._extract_standard_metadata(res, registry)
    cl_display = (
        res.get("class", {}).get("display")
        or res.get("class", {}).get("code")
        or "N/A"
    )

    types_list = []
    for cc in res.get("type", []):
      t_disp = registry.get_display_for_code(cc)
      if t_disp and t_disp != "N/A":
        types_list.append(t_disp)
    type_str = " / ".join(types_list) if types_list else "N/A"

    period = res.get("period", {})
    duration_days = _calculate_duration_days(
        period.get("start"), period.get("end")
    )

    status = res.get("status", "")
    return [(date, cl_display, type_str, duration_days, status, enc_ctx)]

  def _columns(self) -> list[str]:
    return [
        "Date",
        "EncounterClass",
        "EncounterType",
        "DurationDays",
        "Status",
        "EncounterContext",
    ]

  def _display_column(self) -> str:
    return "EncounterClass"


class _FlowsheetDataStore:
  """Orchestrator mapping resources to centralized tool endpoints."""

  _METADATA_PREAMBLE = (
      "Available Medical Concept Identifiers (CRITICAL: pass the exact"
      " complete composite string including parentheses when querying, e.g."
      " 'ConceptName (ResourceType)'):\n"
  )

  def __init__(self, fhir_bundle: dict[str, Any]):
    self._registry = _FHIRRegistry(fhir_bundle)
    self._resources: dict[str, _FlowsheetResource] = {}

    resources_by_type = collections.defaultdict(list)
    for entry in fhir_bundle.get("entry", []):
      res = entry.get("resource", {})
      if r_type := res.get("resourceType"):
        resources_by_type[r_type].append(res)

    for cls in self._get_all_subclasses(_FlowsheetResource):
      if getattr(cls, "resource_type", None):
        type_str = getattr(cls, "resource_type")
        self._resources[type_str] = cls(
            resources_by_type[type_str], self._registry
        )

  @classmethod
  def _get_all_subclasses(cls, base_cls: type[Any]) -> list[type[Any]]:
    """Recursively finds all subclasses of a given base class."""
    subclasses = []
    for subclass in base_cls.__subclasses__():
      subclasses.append(subclass)
      subclasses.extend(cls._get_all_subclasses(subclass))
    return subclasses

  def get_available_resource_types(self) -> list[str]:
    """Returns a sorted list of registered resource types."""
    return sorted(list(self._resources.keys()))

  def get_metadata(
      self, include_dates: bool = False, search_term: str = ""
  ) -> str:
    """Returns available medical concepts as a readable summary string."""
    items = []
    for name, res in self._resources.items():
      keys = res.get_metadata()
      if not keys and res.df.empty:
        continue

      has_date = "Date" in res.df.columns
      for k in keys:
        if search_term:
          try:
            if not re.search(search_term, f"{k} ({name})", re.IGNORECASE):
              continue
          except re.error:
            if search_term.lower() not in f"{k} ({name})".lower():
              continue
        if not include_dates:
          items.append(f"* {k} ({name})")
          continue

        min_d, max_d = "N/A", "N/A"
        if has_date and k in res.dfs_by_display:
          valid_dates = res.dfs_by_display[k]["Date"].dropna().astype(str)
          valid_dates = valid_dates[
              ~valid_dates.isin(["N/A", "", "unknown", "None"])
          ]
          if not valid_dates.empty:
            min_d = str(valid_dates.min())
            max_d = str(valid_dates.max())
        items.append(f"* {k} ({name}) : {min_d} - {max_d}")
    return self._METADATA_PREAMBLE + "\n".join(items)

  def _find_concept(
      self, concept_identifier: str
  ) -> tuple[str, _FlowsheetResource] | None:
    """Finds the underlying display name and _FlowsheetResource matching the identifier."""
    cid_stripped = concept_identifier.strip()

    # Try exact match with resource type suffix (e.g. "Concept (Resource)")
    for name, res in self._resources.items():
      for k in res.dfs_by_display:
        if cid_stripped == f"{k} ({name})":
          return k, res

    # Fallback to matching just the concept string
    for res in self._resources.values():
      if cid_stripped in res.dfs_by_display:
        return cid_stripped, res

    return None

  def _get_dataframe_by_concept(
      self,
      concept_identifier: str,
      granularity: str = "h",
      max_rows: int = 200,
      start_date: str | None = None,
      end_date: str | None = None,
  ) -> str:
    """Returns markdown DataFrame for a single composite concept, optionally filtered and downsampled."""
    match = self._find_concept(concept_identifier)
    if match is None:
      return f"Unknown concept identifier: {concept_identifier}"

    k, found_res = match

    found_df = found_res.dfs_by_display[k].copy()

    # Date Filtering
    if (
        (start_date or end_date)
        and "Date" in found_df.columns
        and not found_df.empty
    ):
      if start_date:
        found_df = found_df[
            found_df["Date"].astype(str) >= str(start_date).strip()
        ]
      if end_date:
        norm_end = str(end_date).strip()
        if len(norm_end) == 10 and norm_end.count("-") == 2:
          norm_end += "T23:59:59.999999"
        found_df = found_df[found_df["Date"].astype(str) <= norm_end]
      found_df = found_df.reset_index(drop=True)

    # Downsampling
    if len(found_df) > max_rows:
      try:
        target_gran = _DownsampleGranularity(granularity)
      except ValueError:
        target_gran = _DownsampleGranularity.HOUR

      _adaptively_downsample(
          found_df,
          item_id_cols=[found_res.display_column],
          datetime_col="Date",
          initial_granularity=target_gran,
          max_total_rows=max_rows,
      )

      if "Date" in found_df.columns:
        _sort_by_datetime(found_df, datetime_col="Date", ascending=True)

      md_str = str(found_df.to_markdown(index=False))
      md_str += (
          f"\n\n* Note: The table for '{concept_identifier}' was filtered and"
          f" adaptively downsampled (target max_rows={max_rows},"
          f" granularity={granularity}) to fit context restrictions. *\n"
      )
      return md_str

    # Standard Return
    if "Date" in found_df.columns:
      _sort_by_datetime(found_df, datetime_col="Date", ascending=True)
    return str(found_df.to_markdown(index=False))

  def get_dataframes(
      self,
      concept_identifiers: list[str],
      granularity: str = "h",
      max_rows: int = 200,
      start_date: str = "",
      end_date: str = "",
  ) -> str:
    """Centralized tool endpoint to fetch tables for given concepts."""
    res_blocks = []
    for cid in concept_identifiers:
      df_str = self._get_dataframe_by_concept(
          cid, granularity, max_rows, start_date or None, end_date or None
      )
      res_blocks.append(f"### Concept: {cid}\n{df_str}")
    return "\n\n".join(res_blocks)


class FlowsheetAgentStrategy(base.Strategy):
  """Strategy using an ADK agent equipped with FHIR tabular tools."""

  def __init__(
      self,
      creds: api.Credentials,
      answer_config: config.LLMConfig,
      downsampling_enable_llm_control: bool = False,
      date_selection_enabled: bool = False,
  ):
    self._creds = creds
    self._answer_config = answer_config
    self._downsampling_enable_llm_control = downsampling_enable_llm_control
    self._date_selection_enabled = date_selection_enabled
    self._context_window_limit = api.get_context_limit(
        self._answer_config.model
    )
    logging.info("Initialized FlowsheetAgentStrategy")

  @property
  def context_window_limit(self) -> int:
    return self._context_window_limit

  def _get_agent_model(self) -> Any:
    shuffled_creds = self._creds.shuffled()
    llm_use_vertex_ai = self._answer_config.backend == "vertex"

    llm_gcp_project_and_locations = None
    if shuffled_creds.gcp_project_and_locations:
      pairs = []
      for project_id, location in shuffled_creds.gcp_project_and_locations:
        pairs.append(f"{project_id}:{location}")
      llm_gcp_project_and_locations = ",".join(pairs)

    if llm_use_vertex_ai:
      llm_api_key = None
    else:
      maker = api.model_maker_for(self._answer_config.model)
      if maker == "google":
        llm_api_key = (
            shuffled_creds.genai_api_keys[0]
            if shuffled_creds.genai_api_keys
            else None
        )
      elif maker == "openai":
        llm_api_key = (
            shuffled_creds.openai_api_keys[0]
            if shuffled_creds.openai_api_keys
            else None
        )
      elif maker == "anthropic":
        llm_api_key = None
      else:
        raise ValueError(f"Unsupported model maker: {maker}")

    config_obj = ns_agent_config_module.Config(
        llm_use_vertex_ai=llm_use_vertex_ai,
        llm_gcp_project_and_locations=llm_gcp_project_and_locations,
        llm_api_key=llm_api_key,
        llm_model_name=self._answer_config.model,
        llm_temperature=self._answer_config.temperature,
    )
    return config_obj.get_llm_model()

  def _create_metadata_tool(self, store: _FlowsheetDataStore) -> Any:
    """Creates the appropriate metadata tool endpoint based on configuration."""

    def get_metadata(search_term: str = "") -> str:
      return store.get_metadata(
          include_dates=self._date_selection_enabled, search_term=search_term
      )

    valid_types_str = ", ".join(store.get_available_resource_types())

    if self._date_selection_enabled:
      get_metadata.__doc__ = (
          "Returns composite clinical concepts and dates as a summary"
          " string.\n\nFormat: '* ConceptName (ResourceType) : MinDate -"
          " MaxDate'.\nUsage: The absolute priority is to MINIMIZE CONTEXT"
          " SIZE. Never pull the full metadata list. Always pass the optional"
          " `search_term` parameter to drastically reduce the volume of"
          " queries. We strongly recommend searching strictly by resource type"
          f" (Valid types: {valid_types_str}).\nExample:"
          " search_term='Observation' or search_term='Medication'.\nThe"
          " search_term executes as a case-insensitive regex. Avoid searching"
          " by narrow specific variable names, as this may miss critical data"
          " if the clinic's naming conventions differ from your expectation."
          " After filtering the metadata, pass the exact parenthesized"
          " composite identifiers (e.g. 'Platelet Count (Observation)') to the"
          " data extraction tools."
      )
    else:
      get_metadata.__doc__ = (
          "Returns composite clinical concepts as a summary string.\n\nFormat:"
          " '* ConceptName (ResourceType)'.\nUsage: The absolute priority is"
          " to MINIMIZE CONTEXT SIZE. Never pull the full metadata list."
          " Always pass the optional `search_term` parameter to drastically"
          " reduce the volume of queries. We strongly recommend searching"
          " strictly by resource type (Valid types:"
          f" {valid_types_str}).\nExample: search_term='Observation' or"
          " search_term='Medication'.\nThe search_term executes as a"
          " case-insensitive regex. Avoid searching by narrow specific"
          " variable names, as this may miss critical data if the clinic's"
          " naming conventions differ from your expectation. After filtering"
          " the metadata, pass the exact parenthesized composite identifiers"
          " (e.g. 'Platelet Count (Observation)') to the data extraction"
          " tools."
      )

    # Use a dynamic wrapper with custom signature because the built-in decorator
    # reflection struggles with some edge types when doing pure pass-throughs
    @functools.wraps(get_metadata)
    def custom_meta_func(*args, **kwargs):
      return get_metadata(*args, **kwargs)

    custom_meta_func.__signature__ = inspect.Signature([
        inspect.Parameter(
            "search_term",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default="",
            annotation=str,
        )
    ])
    return function_tool.FunctionTool(custom_meta_func)

  def _create_dataframes_tool(self, store: _FlowsheetDataStore) -> Any:
    """Creates the appropriate dataframe retrieval tool."""

    # Define the base logic once, accepting all possible parameters
    def get_dataframes_by_display_names(
        concept_identifiers: list[str],
        granularity: str = "h",
        max_rows: int = 50,
        start_date: str = "",
        end_date: str = "",
    ) -> str:
      return store.get_dataframes(
          concept_identifiers=concept_identifiers,
          granularity=granularity,
          max_rows=max_rows,
          start_date=start_date,
          end_date=end_date,
      )

    # Procedurally build the signature parameters and docstring
    params = [
        inspect.Parameter(
            "concept_identifiers",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=list[str],
        )
    ]
    doc_args = [
        "  concept_identifiers: List of exact string identifiers from metadata."
    ]

    if self._downsampling_enable_llm_control:
      params.extend([
          inspect.Parameter(
              "granularity",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              default="h",
              annotation=str,
          ),
          inspect.Parameter(
              "max_rows",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              default=50,
              annotation=int,
          ),
      ])
      doc_args.extend([
          (
              "  granularity: Downsampling bucket size ('min', 'h', 'D', 'W',"
              " 'M'). Only applied if we exceed the row limit. Otherwise, the"
              " latest within the time range is returned. Defaults to 'h'."
          ),
          (
              "  max_rows: Cap on total rows retained per concept table. The"
              " oldest records are dropped after this limit and granularity"
              " are applied. Use this to get a sample of the table and to"
              " minimize/optimize your context window. Defaults to 50."
          ),
      ])

    if self._date_selection_enabled:
      params.extend([
          inspect.Parameter(
              "start_date",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              default="",
              annotation=str,
          ),
          inspect.Parameter(
              "end_date",
              inspect.Parameter.POSITIONAL_OR_KEYWORD,
              default="",
              annotation=str,
          ),
      ])
      doc_args.extend([
          "  start_date: Optional inclusive start date bound ('YYYY-MM-DD').",
          "  end_date: Optional inclusive end date bound ('YYYY-MM-DD').",
      ])

    # Create a wrapper and patch its metadata so the LLM reads it correctly.
    @functools.wraps(get_dataframes_by_display_names)
    def custom_tool_func(*args, **kwargs):
      return get_dataframes_by_display_names(*args, **kwargs)

    # Override the signature the LLM framework will see
    custom_tool_func.__signature__ = inspect.Signature(params)
    # Stitch and override the docstring
    base_doc = (
        "Retrieves concatenated markdown DataFrames for multiple composite"
        " concepts.\n\nCRITICAL: Pass exact complete identifiers including"
        " parenthesized\nsuffixes (e.g., ['Platelet Count"
        " (Observation)']).\n\nArgs:\n"
    )
    custom_tool_func.__doc__ = base_doc + "\n".join(doc_args)

    return function_tool.FunctionTool(custom_tool_func)

  def process(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> tuple[str | None, dict[str, Any] | None, str | None, str, str, int]:
    logging.info(
        "FlowsheetAgentStrategy.process: patient=%s, question=%s",
        record.patient_id,
        record.question_for_answering or record.question,
    )
    if not fhir_utils.verify_with_pydantic(fhir_bundle):
      raise ValueError("Failed to parse FHIR bundle with pydantic validation.")

    store = _FlowsheetDataStore(fhir_bundle)

    meta_tool = self._create_metadata_tool(store)
    disp_tool = self._create_dataframes_tool(store)

    agent = llm_agent.Agent(
        name="FlowsheetAgent",
        model=self._get_agent_model(),
        instruction=_SYSTEM_INSTRUCTION,
        tools=[meta_tool, disp_tool],
    )

    runner = runners.Runner(
        agent=agent,
        app_name="flowsheet_agent_app",
        session_service=in_memory_session_service.InMemorySessionService(),
        auto_create_session=True,
    )

    query = record.question_for_answering or record.question
    events = []

    for event in runner.run(
        user_id="flowsheet_agent_user",
        session_id="flowsheet_agent_session",
        new_message=types.Content(
            role="user", parts=[types.Part.from_text(text=query)]
        ),
    ):
      events.append(event)

    final_response = ""
    final_prompt_tokens = 0

    found_response = False
    found_tokens = False

    for event in reversed(events):
      if not isinstance(event, adk_event.Event):
        raise TypeError(f"Expected ADK Event, got {type(event)}")

      if not found_tokens and event.usage_metadata:
        final_prompt_tokens = event.usage_metadata.prompt_token_count or 0
        found_tokens = True
      if (
          not found_response
          and event.author == agent.name
          and event.content
          and event.content.parts
      ):
        text_parts = [
            p.text for p in event.content.parts if p.text and not p.thought
        ]
        if text_parts:
          combined_text = "".join(text_parts)
          m_ans = re.search(
              r"\[ANSWER\](.*?)\[/ANSWER\]", combined_text, re.DOTALL
          )
          if m_ans:
            final_response = m_ans.group(1).strip()
          else:
            final_response = combined_text.strip()
          found_response = True
      if found_response and found_tokens:
        break

    logging.info(
        "FlowsheetAgentStrategy.process: final_response=%s,"
        " final_prompt_tokens=%s, events_count=%s",
        final_response,
        final_prompt_tokens,
        len(events),
    )
    agent_trace = engine_module.parse_events(events)
    if final_response:
      return (
          final_response,
          {"events": agent_trace},
          None,
          "",
          query,
          final_prompt_tokens,
      )
    else:
      return (
          None,
          {"events": agent_trace},
          "No answer returned by FlowsheetAgent",
          "",
          query,
          final_prompt_tokens,
      )
