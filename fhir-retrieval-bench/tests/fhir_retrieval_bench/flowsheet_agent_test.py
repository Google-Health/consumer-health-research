"""Unit tests for flowsheet_agent strategy focusing extensively on public interfaces, enhanced with varied examples for top-tier coverage."""

from typing import Any
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from google.adk.events import event as adk_event
import pandas as pd
import pydantic

from fhir_retrieval_bench.strategies import flowsheet_agent


class FlowsheetAgentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            # Required Patient resource
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-1",
                    "identifier": [{"value": "pat-alt-id"}],
                    "birthDate": "2130-09-10",
                    "gender": "female",
                    "name": [{"given": ["Jane"], "family": "Doe"}],
                }
            },
            # Encounter with type array to hit type string extraction
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-1",
                    "status": "finished",
                    "class": {"display": "ACUTE", "code": "IMP"},
                    "type": [{"text": "Inpatient Stay"}],
                    "period": {
                        "start": "2141-05-20T08:00:00Z",
                        "end": "2141-05-23T08:00:00Z",
                    },
                }
            },
            # Encounter with malformed date to naturally hit exception block
            {
                "resource": {
                    "resourceType": "Encounter",
                    "id": "enc-malformed",
                    "status": "planned",
                    "period": {
                        "start": "malformed-start-date",
                        "end": "malformed-end-date",
                    },
                }
            },
            # Multiple Observations to elegantly test adaptive downsampling and
            # date formatting
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-1",
                    "code": {"text": "Platelet Count"},
                    "valueQuantity": {"value": 250, "unit": "K/uL"},
                    "effectiveDateTime": "2141-05-29T05:50:00-04:00",
                    "encounter": {"reference": "Encounter/enc-1"},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-2",
                    "code": {"text": "Platelet Count"},
                    "valueQuantity": {"value": 255, "unit": "K/uL"},
                    "effectiveDateTime": "2141-05-29T05:51:00-04:00",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-3",
                    "code": {"text": "Platelet Count"},
                    "valueQuantity": {"value": 260, "unit": "K/uL"},
                    "effectiveDateTime": "2141-05-29T10:00:00-04:00",
                }
            },
            # Observation occurring in a different month to naturally trigger
            # tail slice dropping
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-diff-month",
                    "code": {"text": "Platelet Count"},
                    "valueQuantity": {"value": 270, "unit": "K/uL"},
                    "effectiveDateTime": "2141-06-15T10:00:00-04:00",
                }
            },
            # Observation with valueCodeableConcept to hit coded display
            # extraction
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-code",
                    "code": {"text": "Coded Test"},
                    "valueCodeableConcept": {"text": "Positive Result"},
                    "effectiveDateTime": "2141-05-29T11:00:00-04:00",
                }
            },
            # Observation with component list to hit component extraction
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-comp",
                    "code": {"text": "Blood Pressure Panel"},
                    "effectiveDateTime": "2141-05-29T12:00:00-04:00",
                    "component": [{
                        "code": {
                            "coding": [
                                {"code": "8480-6", "display": "Systolic"}
                            ]
                        },
                        "valueQuantity": {"value": 120, "unit": "mmHg"},
                    }],
                }
            },
            # Observation with hasMember and no direct value to hit parent panel
            # skipping
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-parent",
                    "code": {"text": "Parent Panel"},
                    "effectiveDateTime": "2141-05-29T13:00:00-04:00",
                    "hasMember": [{"reference": "Observation/obs-1"}],
                }
            },
            # Observation with tuple text to natively hit json.dumps
            # serialization in _to_hashable
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-tuple",
                    "code": {"text": ("TupleTest", "Code")},
                    "valueQuantity": {"value": 10, "unit": "mg"},
                    "effectiveDateTime": "2141-05-29T14:00:00-04:00",
                }
            },
            # Observation with unserializable mapping to natively hit TypeError
            # block in _to_hashable
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-unserializable",
                    "code": {"text": {"lambda": lambda: None}},
                    "valueQuantity": {"value": 20, "unit": "mg"},
                    "effectiveDateTime": "2141-05-29T15:00:00-04:00",
                }
            },
            # Observation pointing to a nonexistent encounter to natively hit
            # DA:267,0 return N/A
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs-no-enc",
                    "code": {"text": "Missing Enc Test"},
                    "valueQuantity": {"value": 5, "unit": "mg"},
                    "effectiveDateTime": "2141-05-29T16:00:00-04:00",
                    "encounter": {"reference": "Encounter/nonexistent-enc"},
                }
            },
            # Condition without code object to hit get_all_displays null shield
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-nocode",
                    "clinicalStatus": {"text": "active"},
                    "recordedDate": "2141-05-20T09:00:00-04:00",
                }
            },
            # Procedure with coding without display to hit get_all_displays code
            # fallback
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": "proc-code-only",
                    "code": {"coding": [{"code": "PROC-CODE"}]},
                    "status": "completed",
                    "performedDateTime": "2141-05-20T11:00:00-04:00",
                }
            },
            # Basic MedicationRequest and Administration
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medreq-1",
                    "status": "completed",
                    "intent": "order",
                    "medicationCodeableConcept": {"text": "Azithromycin 500mg"},
                    "authoredOn": "2141-05-20T10:00:00-04:00",
                    "requester": {"display": "Dr. Smith"},
                    "dosageInstruction": [{
                        "text": "Take 500mg daily",
                        "route": {"coding": [{"display": "PO", "code": "PO"}]},
                        "doseAndRate": [
                            {"doseQuantity": {"value": 500, "unit": "mg"}}
                        ],
                        "timing": {
                            "repeat": {
                                "frequency": 1,
                                "period": 1,
                                "periodUnit": "d",
                            }
                        },
                    }],
                }
            },
            {
                "resource": {
                    "resourceType": "MedicationAdministration",
                    "id": "medadm-1",
                    "status": "completed",
                    "medicationCodeableConcept": {"text": "Azithromycin 500mg"},
                    "effectiveDateTime": "2141-05-21T12:00:00-04:00",
                    "dosage": {
                        "dose": {"value": 500, "unit": "mg"},
                        "route": {"coding": [{"display": "Oral"}]},
                    },
                }
            },
            # MedicationRequest pointing to a Medication with
            # mimic-medication-name identifier
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medreq-mimic",
                    "status": "active",
                    "medicationReference": {
                        "reference": "Medication/med-mimic"
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Medication",
                    "id": "med-mimic",
                    "identifier": [{
                        "system": "mimic-medication-name",
                        "value": "MimicDrug",
                    }],
                    "amount": {"numerator": {"value": 10, "unit": "mg"}},
                }
            },
            # MedicationRequest pointing to a Medication with medication-mix
            # identifier
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medreq-mix",
                    "status": "active",
                    "medicationReference": {"reference": "Medication/med-mix"},
                }
            },
            {
                "resource": {
                    "resourceType": "Medication",
                    "id": "med-mix",
                    "identifier": [{
                        "system": "medication-mix",
                        "value": "Dextrose--5%_NaCl--0.9%",
                    }],
                }
            },
            # MedicationRequest pointing to a parent Medication with ingredients
            # referencing child Medication
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medreq-parent",
                    "status": "active",
                    "medicationReference": {
                        "reference": "Medication/med-parent"
                    },
                }
            },
            {
                "resource": {
                    "resourceType": "Medication",
                    "id": "med-parent",
                    "ingredient": [{
                        "itemReference": {"reference": "Medication/med-child"},
                        "strength": {"numerator": {"value": 50, "unit": "mg"}},
                    }],
                }
            },
            {
                "resource": {
                    "resourceType": "Medication",
                    "id": "med-child",
                    "code": {"text": "12345"},
                    "identifier": [{
                        "system": "mimic-medication-name",
                        "value": "NestedMimicChild",
                    }],
                }
            },
            # MedicationRequest with coded frequency to hit
            # timing.code.coding.display cascade
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medreq-coded-freq",
                    "status": "completed",
                    "medicationCodeableConcept": {"text": "CodedFreqDrug"},
                    "dosageInstruction": [
                        {"timing": {"code": {"coding": [{"display": "BID"}]}}}
                    ],
                }
            },
            # MedicationRequest with repeat frequency only to hit f_val times
            # cascade
            {
                "resource": {
                    "resourceType": "MedicationRequest",
                    "id": "medreq-rep-only",
                    "status": "completed",
                    "medicationCodeableConcept": {"text": "RepOnlyDrug"},
                    "dosageInstruction": [
                        {"timing": {"repeat": {"frequency": 3}}}
                    ],
                }
            },
            # MedicationAdministration with rateQuantity only to hit rate
            # fallback
            {
                "resource": {
                    "resourceType": "MedicationAdministration",
                    "id": "medadm-rate-only",
                    "status": "completed",
                    "medicationCodeableConcept": {"text": "RateOnlyDrug"},
                    "dosage": {"rateQuantity": {"value": 100, "unit": "mL/hr"}},
                }
            },
            # Condition with missing clinicalStatus and dict verificationStatus
            # to hit fallback
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "cond-1",
                    "code": {"text": "Pneumonia"},
                    "verificationStatus": {"text": "confirmed"},
                    "recordedDate": "2141-05-20T09:00:00-04:00",
                }
            },
            # Procedure
            {
                "resource": {
                    "resourceType": "Procedure",
                    "id": "proc-1",
                    "code": {"text": "Chest X-Ray"},
                    "status": "completed",
                    "performedDateTime": "2141-05-20T11:00:00-04:00",
                }
            },
            # MedicationStatement
            {
                "resource": {
                    "resourceType": "MedicationStatement",
                    "id": "medsta-1",
                    "status": "active",
                    "medicationCodeableConcept": {"text": "Aspirin 81mg"},
                    "effectiveDateTime": "2141-05-20T10:00:00Z",
                    "informationSource": {"text": "Patient"},
                    "dosage": [{"text": "Take 1 tablet daily"}],
                }
            },
            # MedicationDispense
            {
                "resource": {
                    "resourceType": "MedicationDispense",
                    "id": "meddsp-1",
                    "status": "completed",
                    "medicationCodeableConcept": {"text": "Lisinopril 10mg"},
                    "whenHandled": "2141-05-20T11:00:00Z",
                    "quantity": {"value": 30, "unit": "tab"},
                    "daysSupply": {"value": 30, "unit": "days"},
                }
            },
            # Immunization
            {
                "resource": {
                    "resourceType": "Immunization",
                    "id": "imm-1",
                    "status": "completed",
                    "vaccineCode": {"text": "COVID-19 Vaccine"},
                    "occurrenceDateTime": "2141-05-20T12:00:00Z",
                }
            },
            # AllergyIntolerance with dict verificationStatus fallback
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "alg-1",
                    "verificationStatus": {"text": "unconfirmed"},
                    "code": {"text": "Penicillin"},
                    "recordedDate": "2141-05-20T13:00:00Z",
                }
            },
            # AllergyIntolerance with string verificationStatus fallback
            {
                "resource": {
                    "resourceType": "AllergyIntolerance",
                    "id": "alg-str",
                    "verificationStatus": "refuted",
                    "code": {"text": "Sulfa"},
                    "recordedDate": "2141-05-20T13:30:00Z",
                }
            },
            # DiagnosticReport
            {
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "diag-1",
                    "status": "final",
                    "code": {"text": "Comprehensive Metabolic Panel"},
                    "effectiveDateTime": "2141-05-20T14:00:00Z",
                    "conclusion": "Normal results",
                }
            },
            # Specimen
            {
                "resource": {
                    "resourceType": "Specimen",
                    "id": "spec-1",
                    "status": "available",
                    "type": {"text": "Blood sample"},
                    "collection": {"collectedDateTime": "2141-05-20T14:10:00Z"},
                }
            },
            # ImagingStudy
            {
                "resource": {
                    "resourceType": "ImagingStudy",
                    "id": "img-1",
                    "status": "available",
                    "started": "2141-05-20T15:00:00Z",
                    "description": "CT Scan Brain",
                    "series": [{"modality": {"code": "CT"}}],
                }
            },
        ],
    }

  def test_get_metadata(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    meta_str = store.get_metadata()

    self.assertIn("Available Medical Concept Identifiers", meta_str)
    self.assertIn("Platelet Count (Observation)", meta_str)
    self.assertIn("Azithromycin 500mg (MedicationRequest)", meta_str)
    self.assertIn("MimicDrug (MedicationRequest)", meta_str)
    self.assertIn("Dextrose / NaCl (MedicationRequest)", meta_str)
    self.assertIn("NestedMimicChild (MedicationRequest)", meta_str)
    self.assertIn("Aspirin 81mg (MedicationStatement)", meta_str)
    self.assertIn("Lisinopril 10mg (MedicationDispense)", meta_str)
    self.assertIn("COVID-19 Vaccine (Immunization)", meta_str)
    self.assertIn("Penicillin (AllergyIntolerance)", meta_str)
    self.assertIn("Comprehensive Metabolic Panel (DiagnosticReport)", meta_str)
    self.assertIn("Blood sample (Specimen)", meta_str)
    self.assertIn("CT Scan Brain (ImagingStudy)", meta_str)

  def test_get_metadata_with_dates(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    meta_dates = store.get_metadata(include_dates=True)

    self.assertIn("Platelet Count (Observation) : 2141-05-29", meta_dates)

  @parameterized.named_parameters(
      (
          "exact_case_insensitive_substring",
          "platelet count",
          ["Platelet Count (Observation)"],
          ["Azithromycin 500mg (MedicationRequest)"],
      ),
      (
          "resource_type_filtering",
          "MedicationRequest",
          [
              "Azithromycin 500mg (MedicationRequest)",
              "Dextrose / NaCl (MedicationRequest)",
          ],
          ["Platelet Count (Observation)"],
      ),
      (
          "regex_variations",
          "(?i)azithromycin|dextrose",
          [
              "Azithromycin 500mg (MedicationRequest)",
              "Dextrose / NaCl (MedicationRequest)",
          ],
          [
              "Aspirin 81mg (MedicationStatement)",
              "Platelet Count (Observation)",
          ],
      ),
  )
  def test_get_metadata_with_regex_search(
      self, search_term, expected_in, expected_not_in
  ):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    meta = store.get_metadata(search_term=search_term)

    for item in expected_in:
      self.assertIn(item, meta)

    for item in expected_not_in:
      self.assertNotIn(item, meta)

  @parameterized.named_parameters(
      (
          "med_request_1",
          "Azithromycin 500mg (MedicationRequest)",
          ["Dr. Smith"],
      ),
      ("med_request_2", "MimicDrug (MedicationRequest)", ["10 mg"]),
      ("med_statement", "Aspirin 81mg (MedicationStatement)", ["Patient"]),
      (
          "med_dispense",
          "Lisinopril 10mg (MedicationDispense)",
          ["30 tab", "30 days"],
      ),
      ("immunization", "COVID-19 Vaccine (Immunization)", ["completed"]),
      ("condition", "Pneumonia (Condition)", ["confirmed"]),
      (
          "allergy_unconfirmed",
          "Penicillin (AllergyIntolerance)",
          ["unconfirmed"],
      ),
      ("allergy_refuted", "Sulfa (AllergyIntolerance)", ["refuted"]),
      (
          "diagnostic_report",
          "Comprehensive Metabolic Panel (DiagnosticReport)",
          ["Normal results"],
      ),
      ("specimen", "Blood sample (Specimen)", ["available"]),
      ("imaging_study", "CT Scan Brain (ImagingStudy)", ["CT"]),
      ("encounter", "ACUTE (Encounter)", ["Inpatient Stay"]),
      ("patient", "Jane Doe (Patient)", ["pat-alt-id"]),
      ("nested_child", "NestedMimicChild (MedicationRequest)", ["50 mg"]),
      ("coded_frequency", "CodedFreqDrug (MedicationRequest)", ["BID"]),
      ("repeat_only", "RepOnlyDrug (MedicationRequest)", ["3 times"]),
      ("rate_only", "RateOnlyDrug (MedicationAdministration)", ["100 mL/hr"]),
      (
          "tuple_obs",
          f"{str(('TupleTest', 'Code'))} (Observation)",
          ["('TupleTest', 'Code')"],
      ),
      ("no_encounter_obs", "Missing Enc Test (Observation)", ["N/A"]),
  )
  def test_get_dataframe_by_display_name_all_resources(
      self, concept, expected_substrings
  ):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    md = store._get_dataframe_by_concept(concept)
    for substring in expected_substrings:
      self.assertIn(substring, md)

  def test_get_dataframe_by_display_name_fallback(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    plain_md = store._get_dataframe_by_concept("Platelet Count")
    self.assertIn("260", plain_md)

    plain_adv_md = store._get_dataframe_by_concept("Platelet Count")
    self.assertIn("260", plain_adv_md)

  def test_get_dataframe_by_display_name_unknown(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    res = store._get_dataframe_by_concept("Nonexistent Concept (Observation)")
    self.assertEqual(
        res, "Unknown concept identifier: Nonexistent Concept (Observation)"
    )

    res_adv = store._get_dataframe_by_concept(
        "Nonexistent Concept (Observation)"
    )
    self.assertEqual(
        res_adv, "Unknown concept identifier: Nonexistent Concept (Observation)"
    )

  def test_adaptively_downsample_trigger(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    # Request max_rows=1 to naturally trigger adaptive downsampling compression
    # over the multiple Observation rows
    res_compressed = store._get_dataframe_by_concept(
        "Platelet Count (Observation)", max_rows=1
    )
    self.assertIn("adaptively downsampled", res_compressed)
    self.assertIn("270", res_compressed)

    # Test advanced retrieval exception block when invalid granularity string is
    # passed
    res_malformed_gran = store._get_dataframe_by_concept(
        "Platelet Count (Observation)", granularity="invalid-gran", max_rows=1
    )
    self.assertIn("adaptively downsampled", res_malformed_gran)

  def test_get_dataframe_by_display_name_advanced_date_slicing(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    res_full = store._get_dataframe_by_concept(
        "Platelet Count (Observation)",
        start_date="2141-05-28",
        end_date="2141-05-30",
    )
    self.assertIn("260", res_full)

  def test_get_dataframe_by_display_name_advanced_short_date_auto_expansion(
      self,
  ):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    res = store._get_dataframe_by_concept(
        "Platelet Count (Observation)", end_date="2141-05-29"
    )
    self.assertIn("260", res)

  def test_get_dataframes_by_display_names_advanced(self):
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    concat_res = store.get_dataframes(
        ["Pneumonia (Condition)", "Chest X-Ray (Procedure)"]
    )
    self.assertIn("### Concept: Pneumonia (Condition)", concat_res)
    self.assertIn("### Concept: Chest X-Ray (Procedure)", concat_res)

  def test_massive_table_downsampling(self):
    massive_bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-massive",
                }
            }
        ],
    }
    for i in range(605):
      massive_bundle["entry"].append({
          "resource": {
              "resourceType": "Observation",
              "id": f"obs-m-{i}",
              "code": {"text": "Massive Test"},
              "valueQuantity": {"value": i, "unit": "U"},
              "effectiveDateTime": f"2141-05-20T{i // 60:02d}:{i % 60:02d}:00Z",
          }
      })

    store = flowsheet_agent._FlowsheetDataStore(massive_bundle)

    # Hit protection block via store APIs
    group_md = store._get_dataframe_by_concept("Massive Test (Observation)")
    self.assertIn("adaptively downsampled", group_md)

  def test_downsample_public_edge_cases(self):

    # 1. Empty DataFrame hits DA:88,0 early return
    empty_df = pd.DataFrame()
    flowsheet_agent._downsample(
        empty_df,
        item_id_cols=["ID"],
        datetime_col="Date",
        granularity=flowsheet_agent._DownsampleGranularity.MINUTE,
    )
    flowsheet_agent._sort_by_datetime(empty_df, datetime_col="Date")
    flowsheet_agent._adaptively_downsample(
        empty_df,
        item_id_cols=["ID"],
        datetime_col="Date",
        initial_granularity=flowsheet_agent._DownsampleGranularity.MINUTE,
    )
    self.assertTrue(empty_df.empty)

    # 2. Custom DataFrame hitting nested JSON serialization blocks inside
    # _to_hashable
    custom_df = pd.DataFrame({
        "ID": [
            {"nested_key": "val"},
            ("tuple_k1", "tuple_k2"),
            {"lambda_key": lambda: None},
        ],
        "Date": [
            "2141-05-20T10:00:00Z",
            "2141-05-20T10:01:00Z",
            "2141-05-20T10:02:00Z",
        ],
    })
    flowsheet_agent._downsample(
        custom_df,
        item_id_cols=["ID"],
        datetime_col="Date",
        granularity=flowsheet_agent._DownsampleGranularity.MINUTE,
    )

    # 3. Hit non-downsampled rendering pathways (len <= 600) in full tables and
    # groups
    store = flowsheet_agent._FlowsheetDataStore(self.bundle)
    small_group_md = store._get_dataframe_by_concept(
        "Platelet Count (Observation)"
    )
    self.assertNotIn("adaptively downsampled", small_group_md)

  @parameterized.named_parameters(
      ("mg_with_space", "Acetaminophen 325 mg", "325 mg"),
      ("mg_no_space", "Lisinopril 10mg PO Tablet", "10 mg"),
      ("concentration", "Insulin 100 units/mL", "100 units/mL"),
      ("slash_ml_no_space", "Heparin 5000/mL", "5000 /mL"),
      ("slash_ml_with_spaces", "Heparin 5000 / ml", "5000 /ml"),
      ("percent", "Sodium Chloride 0.9%", "0.9 %"),
      ("no_dose", "No Dose Medication", None),
      ("none_input", None, None),
  )
  def test_fallback_parse_strength_from_name(self, name_str, expected):
    res = flowsheet_agent._fallback_parse_strength_from_name(name_str)
    self.assertEqual(res, expected)


class _MockCreds:

  def __init__(self):
    self.gcp_project_and_locations = [("proj", "loc")]
    self.genai_api_keys = ["key1"]
    self.openai_api_keys = ["okey1"]

  def shuffled(self):
    return self


class _MockLLMConfig:

  def __init__(self, backend="gemini", model="gemini-3-pro", temp=0.0):
    self.backend = backend
    self.model = model
    self.temperature = temp


class _MockEvalInstance:

  def __init__(self):
    self.patient_id = "pat-s-1"
    self.question = "Test Q"
    self.question_for_answering = "Test QA"


class _MockPart:

  def __init__(self, text, thought=False):
    self.text = text
    self.thought = thought


class _MockContent:

  def __init__(self, parts):
    self.parts = parts


class _MockUsage:

  def __init__(self, count):
    self.prompt_token_count = count
    self.total_token_count = count
    self.candidates_token_count = 0
    self.thoughts_token_count = 0


class _MockEvent(adk_event.Event):
  model_config = pydantic.ConfigDict(
      arbitrary_types_allowed=True,
      extra="allow",
  )

  def __init__(self, author, content, usage=None):
    super().__init__(
        author=author,
        content=content,
        usage_metadata=usage,
    )


class FlowsheetAgentStrategyTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "pat-s-1",
                }
            }
        ],
    }
    self.creds = _MockCreds()
    self.eval_instance = _MockEvalInstance()

  @parameterized.named_parameters(
      dict(
          testcase_name="vertex",
          backend="vertex",
          model_name="gemini-3-pro",
          maker=None,
      ),
      dict(
          testcase_name="google",
          backend="gemini",
          model_name="gemini-3-pro",
          maker="google",
      ),
      dict(
          testcase_name="openai",
          backend="gemini",
          model_name="gpt-4o",
          maker="openai",
      ),
      dict(
          testcase_name="anthropic",
          backend="gemini",
          model_name="claude-sonnet-4-6",
          maker="anthropic",
      ),
  )
  def test_get_agent_model_variations(self, backend, model_name, maker):
    self.enter_context(
        mock.patch.object(
            flowsheet_agent.ns_agent_config_module.Config,
            "get_llm_model",
            return_value="fake-model-obj",
        )
    )
    if maker:
      self.enter_context(
          mock.patch.object(
              flowsheet_agent.api, "model_maker_for", return_value=maker
          )
      )

    cfg = _MockLLMConfig(backend=backend, model=model_name)
    strat = flowsheet_agent.FlowsheetAgentStrategy(self.creds, cfg)
    self.assertEqual(strat._get_agent_model(), "fake-model-obj")

  def test_get_agent_model_unsupported_maker(self):
    self.enter_context(
        mock.patch.object(
            flowsheet_agent.api,
            "model_maker_for",
            return_value="unsupported_maker",
        )
    )
    cfg = _MockLLMConfig(backend="gemini", model="gemini-3-pro")
    strat = flowsheet_agent.FlowsheetAgentStrategy(self.creds, cfg)
    with self.assertRaises(ValueError):
      strat._get_agent_model()

  def _setup_process_mocks(self, events: list[Any]) -> None:
    self.enter_context(
        mock.patch.object(
            flowsheet_agent.ns_agent_config_module.Config,
            "get_llm_model",
            return_value="fake-model",
        )
    )
    mock_runner = self.enter_context(
        mock.patch.object(flowsheet_agent.runners.Runner, "run")
    )
    mock_runner.return_value = events
    self.enter_context(
        mock.patch.object(
            flowsheet_agent.api, "model_maker_for", return_value="google"
        )
    )

  def test_process_extracts_formatted_answer_and_tokens_when_all_flags_enabled(
      self,
  ):
    events = [
        _MockEvent(
            "FlowsheetAgent",
            _MockContent([_MockPart("Thinking...", thought=True)]),
        ),
        _MockEvent(
            "FlowsheetAgent",
            _MockContent(
                [_MockPart("Intro [ANSWER]Strict Answer[/ANSWER] Outro")]
            ),
            usage=_MockUsage(150),
        ),
    ]
    self._setup_process_mocks(events)
    cfg = _MockLLMConfig()
    strat = flowsheet_agent.FlowsheetAgentStrategy(
        self.creds,
        cfg,
        downsampling_enable_llm_control=True,
        date_selection_enabled=True,
    )
    res, _, _, _, _, tokens = strat.process(self.eval_instance, self.bundle)
    self.assertEqual(res, "Strict Answer")
    self.assertEqual(tokens, 150)

  def test_process_falls_back_to_plain_text_when_answer_tags_missing(self):
    events = [
        _MockEvent(
            "FlowsheetAgent",
            _MockContent([_MockPart("Plain unformatted text result")]),
        )
    ]
    self._setup_process_mocks(events)
    cfg = _MockLLMConfig()
    strat = flowsheet_agent.FlowsheetAgentStrategy(
        self.creds,
        cfg,
        downsampling_enable_llm_control=True,
        date_selection_enabled=False,
    )
    res, _, _, _, _, _ = strat.process(self.eval_instance, self.bundle)
    self.assertEqual(res, "Plain unformatted text result")

  def test_process_returns_none_error_tuple_when_no_agent_response_generated(
      self,
  ):
    events = [_MockEvent("User", _MockContent([_MockPart("Just user prompt")]))]
    self._setup_process_mocks(events)
    cfg = _MockLLMConfig()
    strat = flowsheet_agent.FlowsheetAgentStrategy(
        self.creds,
        cfg,
        downsampling_enable_llm_control=False,
        date_selection_enabled=True,
    )
    res, _, err_msg, _, _, _ = strat.process(self.eval_instance, self.bundle)
    self.assertIsNone(res)
    self.assertIn("No answer returned", err_msg)

  def test_process_executes_successfully_with_all_advanced_flags_disabled(self):
    self._setup_process_mocks([])
    cfg = _MockLLMConfig()
    strat = flowsheet_agent.FlowsheetAgentStrategy(
        self.creds,
        cfg,
        downsampling_enable_llm_control=False,
        date_selection_enabled=False,
    )
    strat.process(self.eval_instance, self.bundle)

  def test_process_invalid_bundle(self):
    cfg = _MockLLMConfig()
    strat = flowsheet_agent.FlowsheetAgentStrategy(self.creds, cfg)
    with self.assertRaises(ValueError):
      strat.process(self.eval_instance, {})


if __name__ == "__main__":
  absltest.main()
