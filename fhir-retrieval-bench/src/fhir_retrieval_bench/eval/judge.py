"""Evaluation judging — calls a Gemini model to score answers."""

import textwrap
import time

from absl import logging
import pydantic

from fhir_retrieval_bench import config
from fhir_retrieval_bench.utils import api as api_module


EVAL_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a clinical auditor. Evaluate the Model Response against the Ground Truth.
    Return a JSON object with two fields:
    1. "is_correct": boolean (true if the model answer accurately answer the question that aligns with the ground truth)
    2. "reason": string (brief explanation)
    
    **Patient Query**: {user_query}
    **Ground Truth Answer**: {ground_truth}
    **Model Response**: {model_response}
    
    JSON Output:
    """)


class JudgeResult(pydantic.BaseModel):
  """Structured output schema for the judge model."""

  is_correct: bool
  """Whether the model response conveys the same core clinical information as
  the ground truth."""
  reason: str
  """Brief explanation of the judgment."""


def evaluate_answer(
    question: str,
    ground_truth: str,
    prediction: str,
    experiment_config: config.ExperimentConfig,
    creds: api_module.Credentials,
) -> tuple[None | bool, str]:
  """Evaluate a prediction against the ground truth using the judge model.

  Formats the evaluation prompt, calls the judge model with structured JSON
  output, and parses the response into a ``(is_correct, reason)`` tuple.
  Returns ``(False, ...)`` if the call fails or the JSON cannot be parsed.

  Args:
    question: The original clinical question.
    ground_truth: The expected correct answer.
    prediction: The model's predicted answer to judge.
    experiment_config: Config containing judge model settings.
    creds: Provider credentials; used to build the judge-model backend.

  Returns:
    A ``(is_correct, reason)`` tuple.
  """
  prompt = EVAL_PROMPT_TEMPLATE.format(
      user_query=question, ground_truth=ground_truth, model_response=prediction
  )
  judge_model = experiment_config.models.judge.model
  judge_backend = experiment_config.models.judge.backend
  logging.debug(
      "Calling judge model=%s | prompt_len=%d",
      judge_model,
      len(prompt),
  )
  start = time.time()
  backend = api_module.get_api_client(judge_model, creds, judge_backend)

  judge_maker = api_module.model_maker_for(judge_model)
  if judge_maker == "google":
    extra_config = {
        "response_mime_type": "application/json",
        "response_schema": JudgeResult,
    }
  else:
    extra_config = None

  result_text = backend.generate(
      llm_config=experiment_config.models.judge,
      contents=prompt,
      extra_config=extra_config,
  )

  elapsed = time.time() - start

  if result_text is None:
    logging.warning("Judge call returned None after %.2fs", elapsed)
    return None, "Judge model call failed"

  logging.debug(
      "Judge call succeeded in %.2fs | response_len=%d",
      elapsed,
      len(result_text),
  )
  try:
    cleaned_text = result_text.strip()
    if cleaned_text.startswith("```json"):
      cleaned_text = cleaned_text[len("```json"):].strip()
    elif cleaned_text.startswith("```"):
      cleaned_text = cleaned_text[len("```"):].strip()
    if cleaned_text.endswith("```"):
      cleaned_text = cleaned_text[:-3].strip()

    result = JudgeResult.model_validate_json(cleaned_text)
    return result.is_correct, result.reason
  except pydantic.ValidationError as e:
    logging.exception(
        "Failed to parse judge response | raw=%r",
        result_text[:200],
    )
    return None, f"Judge parse error: {repr(e)}"

