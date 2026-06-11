"""Tests for flowsheet strategy."""

from absl.testing import absltest
from fhir_retrieval_bench.strategies import flowsheet


class FlowsheetTest(absltest.TestCase):

  def test_build_flowsheet(self):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "id": "p1",
                    "birthDate": "1990-01-01",
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "id": "o1",
                    "code": {"text": "Blood Pressure"},
                    "valueQuantity": {"value": 120, "unit": "mmHg"},
                    "effectiveDateTime": "2026-01-01T10:00:00Z",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "id": "c1",
                    "code": {"text": "Diabetes"},
                    "clinicalStatus": {
                        "coding": [{"code": "active", "system": "..."}]
                    },
                    "onsetDateTime": "2025-01-01",
                }
            },
        ],
    }

    result = flowsheet.build_flowsheet(bundle)

    self.assertIn("### Patient Information", result)
    self.assertIn("p1", result)
    self.assertIn("### Observations", result)
    self.assertIn("Blood Pressure", result)
    self.assertIn("### Conditions", result)
    self.assertIn("Diabetes", result)

  def test_build_flowsheet_empty(self):
    bundle = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [],
    }
    result = flowsheet.build_flowsheet(bundle)
    self.assertEqual(result, "No records found.")


if __name__ == "__main__":
  absltest.main()
