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
r"""Runs AutoEval inference for a given Sax model and conversation sstable.

See run_autoeval_eval.py for AutoEval on a new candidate AutoEval model (i.e.,
we want to evaluate an AutoEval model on ground-truth expert ratings).

Output ratings are written to
.../{artifacts_dir}/case_studies/{vertical}/autoeval/inference_conversations.
Output filenames use the following format:

```
{sstable_basename}.{sax_model_name}.{method}.{date}.tsv
```

Example usage:

```bash
python -m run_autoeval_inference \
  --
  --conversation_filepath=/path/to/conversations.sstable \
  --sax_model_name=autoeval_sax_model \
  --vertical=sleep \
  --methods=score \
  --methods=generate \
  --num_parallel_workers=100 \
  --alsologtostderr
  # --temperature      # Optional flag that overrides Sax model defaults.
  # --max_decode_steps # Optional flag that overrides Sax model defaults.
  # --overwrite        # Optional flag used to overwrite files if they exist.
```
"""

import datetime
import os
from typing import Any, Sequence

from absl import app
from absl import flags
import tensorflow as tf

import autoeval_data_lib
import autoeval_inference_lib
import autoeval_prompt_lib
from ..protos import conversation_pb2
import sstable


_CONVERSATION_SSTABLE = flags.DEFINE_string(
    name='conversation_filepath',
    default=None,
    help='The filepath to the conversation sstable.',
    required=True,
)

_SAX_MODEL_NAME = flags.DEFINE_string(
    name='sax_model_name',
    default=None,
    help=(
        'The Sax model name used for inference. This model name should '
        ' correspond to an active Sax endpoint (i.e.,'
        ' /sax/phllm/{sax_model_name}).'
    ),
    required=True,
)

_VERTICAL = flags.DEFINE_enum_class(
    name='vertical',
    default=autoeval_data_lib.Vertical.SLEEP,
    enum_class=autoeval_data_lib.Vertical,
    help=(
        'The vertical to run inference on (should correspond to the'
        ' conversation sstable).'
    ),
)

_TEMPERATURE = flags.DEFINE_float(
    name='temperature',
    default=None,
    help=(
        'The optional tempature override used when calling "lm.generate". If'
        ' unset, we default to the value defined in the Sax serving template.'
    ),
    required=False,
)

_MAX_DECODE_STEPS = flags.DEFINE_integer(
    name='max_decode_steps',
    default=None,
    help=(
        'The optional maximum number of decoding steps override used when'
        'calling "lm.generate". If unset, we default to the value defined in'
        ' the Sax serving template.'
    ),
    required=False,
)

_NUM_PARALLEL_WORKERS = flags.DEFINE_integer(
    name='num_parallel_workers',
    default=100,
    help='The number of parallel workers used to query models.',
)

_METHODS = flags.DEFINE_multi_enum_class(
    name='methods',
    default=[
        autoeval_inference_lib.Method.GENERATE,
        autoeval_inference_lib.Method.SCORE,
    ],
    enum_class=autoeval_inference_lib.Method,
    help='The methods to use for inference.',
)

_OVERWRITE = flags.DEFINE_bool(
    name='overwrite',
    default=False,
    help='Whether to overwrite AutoEval results files if they already exist.',
)


def _sstable_to_conversation_dict(
    conversation_sstable: sstable.SSTable,
) -> dict[str, conversation_pb2.Conversation]:
  """Converts a conversation sstable to a dict keyed by case study id."""
  key_to_conversation = {}
  for conversation in conversation_sstable.values():
    key = conversation.case_study_id
    assert key not in key_to_conversation
    key_to_conversation[key] = conversation
  return key_to_conversation


def _build_tensor_dicts(
    conversation_sstable_filepath: str,
    vertical: autoeval_data_lib.Vertical,
    sax_model_name: str,
) -> list[dict[str, Any]]:
  """Builds AutoEval tensor dictionaries for inference from conversations."""
  target_conversation_dict = _sstable_to_conversation_dict(
      autoeval_data_lib.load_conversation_sstable(
          filename=conversation_sstable_filepath,
      )
  )
  expert_conversation_dict = _sstable_to_conversation_dict(
      autoeval_data_lib.load_graded_conversations(
          vertical=vertical,
          conversation_source='expert',
      )
  )
  tensor_dicts = []
  for conversation in target_conversation_dict.values():
    expert_conversation = expert_conversation_dict[conversation.case_study_id]
    for tag in autoeval_data_lib.VERTICAL_TO_EVAL_TAGS[vertical]:
      for principle in autoeval_data_lib.PRINCIPLES:
        prompt = autoeval_prompt_lib.build_prompt_for_conversation(
            vertical=vertical,
            tag=tag,
            principle=principle,
            target_conversation=conversation,
            expert_conversation=expert_conversation,
        )
        tensor_dict = {
            'vertical': vertical.value,
            'case_study_id': conversation.case_study_id,
            'conversation_id': conversation.conversation_id,
            'inputs': prompt,
            'autoeval_model': sax_model_name,
            'tag': tag,
            'principle': principle,
        }
        for key, value in tensor_dict.items():
          tensor_dict[key] = tf.constant(
              value,
              dtype=tf.string,
          )
        tensor_dicts.append(tensor_dict)
  return tensor_dicts


def _output_filepath(
    conversation_sstable_filepath: str,
    vertical: autoeval_data_lib.Vertical,
    sax_model_name: str,
    inference_tag: str | None = None,
    datestr: str = autoeval_data_lib.ASOF_DATESTR,
) -> str:
  """Returns a versioned AutoEval inference output filepath."""
  df_dir = os.path.join(
      '/path/to/artifacts/dir',
      'case_studies',
      vertical.value,
      'autoeval',
      'inference_conversations',
  )
  sstable_basename = os.path.splitext(
      os.path.basename(conversation_sstable_filepath)
  )[0]
  filestr = datetime.date.fromisoformat(datestr).strftime('%Y%m%d')
  filename_tag = f'{inference_tag}.' if inference_tag else ''
  filename = f'{sstable_basename}.{sax_model_name}.{filename_tag}{filestr}.tsv'
  return os.path.join(df_dir, filename)


def run_inference(
    conversation_sstable_filepath: str,
    vertical: autoeval_data_lib.Vertical,
    methods: list[autoeval_inference_lib.Method],
    sax_model_name: str,
    num_parallel_workers: int,
    temperature: int | None,
    max_decode_steps: int | None,
    overwrite: bool,
) -> None:
  """Runs AutoEval inference on conversations using the given AutoEval model."""
  original_tensor_dicts = _build_tensor_dicts(
      conversation_sstable_filepath=conversation_sstable_filepath,
      vertical=vertical,
      sax_model_name=sax_model_name,
  )
  for method in methods:
    output_filepath = _output_filepath(
        conversation_sstable_filepath=conversation_sstable_filepath,
        vertical=vertical,
        sax_model_name=sax_model_name,
        inference_tag=method.value,
    )
    if os.path.isfile(output_filepath) and not overwrite:
      raise FileExistsError(output_filepath)
    inference_tensor_dicts = autoeval_inference_lib.run_inference(
        original_tensor_dicts=original_tensor_dicts,
        method=method,
        sax_model_name=sax_model_name,
        num_parallel_workers=num_parallel_workers,
        temperature=temperature,
        max_decode_steps=max_decode_steps,
    )
    autoeval_inference_lib.write_output_tsv(
        inference_tensor_dicts,
        output_filepath,
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_inference(
      conversation_sstable_filepath=_CONVERSATION_SSTABLE.value,
      vertical=_VERTICAL.value,
      methods=_METHODS.value,
      sax_model_name=_SAX_MODEL_NAME.value,
      num_parallel_workers=_NUM_PARALLEL_WORKERS.value,
      temperature=_TEMPERATURE.value,
      max_decode_steps=_MAX_DECODE_STEPS.value,
      overwrite=_OVERWRITE.value,
  )


if __name__ == '__main__':
  app.run(main)
