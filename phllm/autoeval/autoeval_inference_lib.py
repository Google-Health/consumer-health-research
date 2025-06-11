# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library for running AutoEval inference."""

import copy
import enum
import os
from typing import Any

import pandas as pd
from saxml.client.python import sax
import tensorflow as tf

import autoeval_prompt_lib


class Method(enum.Enum):
  """Denotes an inference method."""

  GENERATE = 'generate'
  SCORE = 'score'


def build_model_options(
    temperature: float | None,
    max_decode_steps: int | None,
) -> sax.ModelOptions:
  """Returns the Sax model options to be used for each query."""
  model_options = sax.ModelOptions()
  if temperature is not None:
    model_options.SetExtraInput('temperature', temperature)
  if max_decode_steps is not None:
    model_options.SetExtraInput(
        'per_example_max_decode_steps', max_decode_steps
    )
  return model_options


def convert_tensor_dict(tensor_dict: dict[str, tf.Tensor]) -> dict[str, Any]:
  """Converts a single tensor dictionary to a base value dictionary."""
  value_dict = {}
  for key, tensor in tensor_dict.items():
    value_np = tensor.numpy()
    if tensor.dtype == tf.string:
      value_np = str(value_np.decode('utf-8'))
    value_dict[key] = value_np
  return value_dict


def get_text_prompt_value(
    example_dict: dict[str, Any],
    text_prompt_feature_key: str,
) -> str:
  """Returns the text prompt value from the TF Example."""
  prompt_feature = example_dict.get(text_prompt_feature_key, None)
  if not prompt_feature:
    raise ValueError(
        f'No prompt feature found for key: {text_prompt_feature_key}'
    )
  return prompt_feature


def _score_one_example(
    tensor_dict: dict[str, tf.Tensor],
    model: sax.Model,
    model_options: sax.ModelOptions,
) -> dict[str, tf.Tensor]:
  """Runs inference with scoring for a single example.

  Args:
    tensor_dict: A tensor dictionary containing the target prompt.
    model: The Sax model used to run inference.
    model_options: Model options passed to the Sax model when running inference.

  Returns:
    A copy of `tensor_dict` with score fields appended for each Likert option.
  """
  principle = tensor_dict['principle'].numpy().decode('utf-8')
  new_tensor_dict = copy.deepcopy(tensor_dict)
  example_dict = convert_tensor_dict(tensor_dict)
  prompt = get_text_prompt_value(example_dict, 'inputs')
  for rating in range(1, 6):
    option = autoeval_prompt_lib.build_target(
        principle=principle, rating=rating
    )
    score = model.Score(prompt, [option], model_options)  # pytype: disable=attribute-error
    assert len(score) == 1
    key = f'inference_score_option_{rating}'
    assert key not in new_tensor_dict
    new_tensor_dict[key] = tf.constant(
        score[0],
        dtype=tf.float32,
    )
  return new_tensor_dict


def _run_score_for_tensor_dicts(
    tensor_dicts: list[dict[str, tf.Tensor]],
    model: sax.Model,
    model_options: sax.ModelOptions,
    num_parallel_workers: int,
) -> list[dict[str, tf.Tensor]]:
  """Runs parallel inference for examples, appending score outputs."""
  del num_parallel_workers  # Unused.
  # Note: The paralellism library used here is internal to google, so this is
  # done sequentially for the open-source version.
  # new_tfdicts = parallel.RunInParallel(
  #     function=_score_one_example,
  #     list_of_kwargs_to_function=[
  #         {
  #             'tensor_dict': tensor_dict,
  #             'model': model,
  #             'model_options': model_options,
  #         }
  #         for tensor_dict in tensor_dicts
  #     ],
  #     num_workers=num_parallel_workers,
  #     report_progress=True,
  # )
  new_tfdicts = []
  for tensor_dict in tensor_dicts:
    new_tensor_dict = _score_one_example(
        tensor_dict=tensor_dict,
        model=model,
        model_options=model_options,
    )
    new_tfdicts.append(new_tensor_dict)
  return new_tfdicts


def write_output_tsv(
    tensor_dicts: list[dict[str, tf.Tensor]],
    filepath: str,
) -> None:
  """Writes inference results to a TSV file."""
  dirpath = os.path.dirname(filepath)
  if not os.path.isfile(dirpath):
    os.makedirs(dirpath)
  example_dicts = []
  for tensor_dict in tensor_dicts:
    example_dict = {
        k: v.numpy() if v.dtype != tf.string else v.numpy().decode('utf-8')
        for k, v in tensor_dict.items()
    }
    example_dicts.append(example_dict)
  df = pd.DataFrame(example_dicts)
  print(f'Writing n={len(df)} inference results to {filepath=}')
  with open(filepath, 'w') as f:
    df.to_csv(f, sep='\t', index=False)


def _build_model(
    sax_model_name: str,
    temperature: float | None,
    max_decode_steps: int | None,
) -> tuple[sax.Model, sax.ModelOptions]:
  """Returns a Sax model and model options."""
  sax_model = f'/sax/phllm/{sax_model_name}'
  model = sax.Model(sax_model, sax.Options()).LM()
  model_options = build_model_options(
      temperature=temperature,
      max_decode_steps=max_decode_steps,
  )
  return model, model_options


def _run_one_example(
    tensor_dict: dict[str, tf.Tensor],
    model: sax.Model,
    model_options: sax.ModelOptions,
    prompt_feature_key: str,
    output_text_feature_key: str,
    output_score_feature_key: str,
) -> dict[str, tf.Tensor]:
  """Runs inference for a single example and appends the output to a copy.

  Args:
    tensor_dict: A tensor dictionary containing the target `prompt_feature_key`.
    model: The Sax model used to run inference.
    model_options: Model options passed to the Sax model when running inference.
    prompt_feature_key: The `tensor_dict` feature key containing the prompt.
    output_text_feature_key: The feature key to which the model output is
      written.
    output_score_feature_key: The feature key to which the model output score is
      written.

  Returns:
    A copy of `tensor_dict` with the output text and score fields appended.
  """
  new_tensor_dict = copy.deepcopy(tensor_dict)
  example_dict = convert_tensor_dict(tensor_dict)
  prompt = get_text_prompt_value(
      example_dict,
      prompt_feature_key,
  )
  generate_results = model.Generate(prompt, model_options)  # pytype: disable=attribute-error
  assert len(generate_results) == 1
  assert len(generate_results[0]) == 2
  (generate_text, generate_score) = generate_results[0]
  assert output_text_feature_key not in new_tensor_dict
  assert output_score_feature_key not in new_tensor_dict
  new_tensor_dict[output_text_feature_key] = tf.constant(
      generate_text,
      dtype=tf.string,
  )
  new_tensor_dict[output_score_feature_key] = tf.constant(
      generate_score,
      dtype=tf.float32,
  )
  return new_tensor_dict


def run_inference_for_message_tensor_dicts(
    tensor_dicts: list[dict[str, tf.Tensor]],
    model: sax.Model,
    model_options: sax.ModelOptions,
    prompt_feature_key: str,
    output_text_feature_key: str,
    output_score_feature_key: str,
    num_parallel_workers: int,
) -> list[dict[str, tf.Tensor]]:
  """Runs inference for a list of examples, appending outputs to examples."""
  del num_parallel_workers  # Unused.
  # Note: The paralellism library used here is internal to google, so this is
  # done sequentially for the open-source version.
  # new_tfdicts = parallel.RunInParallel(
  #     function=_run_one_example,
  #     list_of_kwargs_to_function=[
  #         {
  #             'tensor_dict': tensor_dict,
  #             'model': model,
  #             'model_options': model_options,
  #             'prompt_feature_key': prompt_feature_key,
  #             'output_text_feature_key': output_text_feature_key,
  #             'output_score_feature_key': output_score_feature_key,
  #         }
  #         for tensor_dict in tensor_dicts
  #     ],
  #     num_workers=num_parallel_workers,
  #     report_progress=True,
  # )
  new_tfdicts = []
  for tensor_dict in tensor_dicts:
    new_tensor_dict = _run_one_example(
        tensor_dict=tensor_dict,
        model=model,
        model_options=model_options,
        prompt_feature_key=prompt_feature_key,
        output_text_feature_key=output_text_feature_key,
        output_score_feature_key=output_score_feature_key,
    )
    new_tfdicts.append(new_tensor_dict)
  return new_tfdicts


def run_inference(
    original_tensor_dicts: list[dict[str, tf.Tensor]],
    method: Method,
    sax_model_name: str,
    num_parallel_workers: int,
    temperature: int | None,
    max_decode_steps: int | None,
) -> list[dict[str, tf.Tensor]]:
  """Runs AutoEval inference using the given Sax model."""
  model, model_options = _build_model(
      sax_model_name,
      temperature=temperature,
      max_decode_steps=max_decode_steps,
  )
  if method == Method.SCORE:
    inference_tensor_dicts = _run_score_for_tensor_dicts(
        tensor_dicts=original_tensor_dicts,
        model=model,
        model_options=model_options,
        num_parallel_workers=num_parallel_workers,
    )
  elif method == Method.GENERATE:
    inference_tensor_dicts = run_inference_for_message_tensor_dicts(
        tensor_dicts=original_tensor_dicts,
        model=model,
        model_options=model_options,
        prompt_feature_key='inputs',
        output_text_feature_key='inference_text_output',
        output_score_feature_key='inference_score_output',
        num_parallel_workers=num_parallel_workers,
    )
  else:
    raise NotImplementedError(f'Unsupported method: {method}')
  return inference_tensor_dicts
