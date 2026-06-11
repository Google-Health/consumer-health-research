"""Strategy base classes for the evaluation pipeline.

All strategies implement the two-phase pattern:
  1. Transform a FHIR bundle into a text context (strategy-specific).
  2. Call an LLM with the context + question to produce a prediction.

``Strategy`` is the abstract root.  ``RAGStrategy`` provides the shared
LLM-call boilerplate so that concrete strategies only need to implement
``prepare_fhir_context``.
"""

import abc
import time
from typing import Any

from absl import logging

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import base as data_base
from fhir_retrieval_bench.utils import api


class ContextTooLongError(Exception):
  """Raised when the context exceeds the model's context window limit."""


class Strategy(abc.ABC):
  """Abstract root for all retrieval/context strategies."""

  context_window_limit: int | None

  @abc.abstractmethod
  def process(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> tuple[str | None, dict[str, Any] | None, str | None, str, str, int]:
    """Given a record, return prediction, context, and the full prompt.

    Args:
      record: A ``EvalInstance`` containing the question, FHIR bundle, etc.
      fhir_bundle: The FHIR bundle to process.

    Returns:
      A ``(prediction, run_extra_data, run_error, context, prompt,
      prompt_token_count)`` tuple
      where *prediction* is the model's answer, *context* is the text sent to
      the LLM (used for token counting), and *prompt* is the fully-formatted
      prompt (or the question, for agent-style strategies without a
      traditional prompt template). Strategies without a traditional context
      (e.g. ns_agent) should return an empty string for *context*.
    """
    ...


class RAGStrategy(Strategy):
  """Base class for RAG-based strategies that prepare context then call an LLM.

  Subclasses implement ``prepare_fhir_context`` to extract/format relevant info
  from the FHIR bundle.  The LLM call with question + context is shared.

  Attributes:
    creds: Provider credentials; used to build the answer-model backend.
    answer_config: LLM config for the answerer model.
    context_window_limit: Limit of the model's context window in tokens.
  """

  QA_PROMPT_TEMPLATE = """
You are an expert clinician. Answer the patient's question based ONLY
 on the provided clinical context. If the Patient Query is such that it
 requires yes or no, answer 1 for yes and 0 for no.

**Patient Query**: {question}

**Clinical Context**:
{context}

**Answer**:
"""

  def __init__(
      self,
      creds: api.Credentials,
      answer_config: config.LLMConfig,
  ):
    self.creds = creds
    self.answer_config = answer_config
    self._answer_backend: api.BaseBackend = api.get_api_client(
        self.answer_config.model, self.creds, self.answer_config.backend
    )
    self.context_window_limit = api.get_context_limit(self.answer_config.model)
    logging.info(
        "RAGStrategy.__init__: model=%s",
        self.answer_config.model,
    )

  @abc.abstractmethod
  def prepare_fhir_context(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> str:
    """Extract and format relevant info from the FHIR bundle.

    Args:
      record: The evaluation record whose ``fhir_bundle`` will be processed.
      fhir_bundle: The FHIR bundle to process.

    Returns:
      A string to be injected as ``{context}`` in the QA prompt template.
    """
    ...

  def process(
      self, record: data_base.EvalInstance, fhir_bundle: dict[str, Any]
  ) -> tuple[str | None, dict[str, Any] | None, str | None, str, str, int]:
    logging.debug("RAGStrategy.process: patient=%s", record.patient_id)
    ctx_start = time.time()
    context = self.prepare_fhir_context(record, fhir_bundle)
    ctx_elapsed = time.time() - ctx_start
    logging.debug(
        "Context prepared in %.2fs | context_len=%d", ctx_elapsed, len(context)
    )
    prompt = self.QA_PROMPT_TEMPLATE.format(
        context=context,
        question=record.question_for_answering,
    )

    logging.debug("call_answer_model: prompt_len=%d", len(prompt))

    prompt_token_count = self._answer_backend.count_tokens(
        self.answer_config.model, prompt
    )
    if prompt_token_count is None:
      logging.warning(
          "Fallback to approximate token count for instance_id=%s |"
          " answer_model=%s",
          record.instance_id,
          self.answer_config.model,
      )
      prompt_token_count = len(prompt) // 4
    logging.debug(
        "Prompt token count: %d",
        prompt_token_count,
    )

    if prompt_token_count > self.context_window_limit:
      logging.warning(
          "Context too long for instance_id=%s | answer_model=%s |"
          " prompt_token_count=%d | context_window_limit=%d",
          record.instance_id,
          self.answer_config.model,
          prompt_token_count,
          self.context_window_limit,
      )
      return None, None, "Context too long", context, prompt, prompt_token_count
    else:
      result = self._answer_backend.generate(
          llm_config=self.answer_config,
          contents=prompt,
      )
      return result, None, None, context, prompt, prompt_token_count
