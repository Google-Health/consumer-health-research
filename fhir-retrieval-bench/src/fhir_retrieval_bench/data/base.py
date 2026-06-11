"""Base data structures for evaluation records.

Each evaluation record pairs a clinical question with its FHIR patient data
and the expected (gold) answer.  All dataset loaders produce lists of
``EvalInstance`` instances regardless of the underlying dataset format.
"""

import dataclasses
from typing import Any


@dataclasses.dataclass
class EvalInstance:
  """A single evaluation record: question + FHIR bundle + ground truth.

  Attributes:
    instance_id: Unique identifier for this evaluation instance.
    patient_id: Unique identifier for the patient within the dataset.
    question: The canonical clinical question from the dataset, used for
      judging, persisted outputs, and resume keys.
    ground_truth: The expected correct answer for scoring.
    question_context: Optional context to send to the answering model.
    question_for_answering: The question text to send to the answering model.
      Defaults to ``question`` but can include dataset-specific prompt
      augmentation when the source benchmark requires it.
    source_meta: Optional dataset-specific metadata (e.g. the original CSV row
      or JSONL task dict).
  """

  instance_id: str
  patient_id: str
  question: str
  ground_truth: str
  question_context: tuple[str, ...] | None = None
  question_for_answering: str | None = None
  source_meta: dict[str, Any] = dataclasses.field(default_factory=dict)

  def __post_init__(self):
    if self.question_for_answering is None:
      self.question_for_answering = self.question
    if self.question_context:
      self.question_for_answering = (
          "\n".join(self.question_context) + "\n" + self.question_for_answering
      )

  def __repr__(self) -> str:
    return (
        f"EvalInstance(instance_id={self.instance_id!r}, \n"
        f"patient_id={self.patient_id!r}, \n"
        f"question={self.question!r}, \n"
        f"ground_truth={self.ground_truth!r}, \n"
        f"question_for_answering={self.question_for_answering!r}, \n"
        f"source_meta={self.source_meta!r})"
    )
