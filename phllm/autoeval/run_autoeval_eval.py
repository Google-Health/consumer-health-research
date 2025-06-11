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
r"""Runs inference for AutoEval training and validation samples.

See run_autoeval_inference.py for AutoEval on a new candidate model (i.e., for a
conversation source for which we don't have ground truth expert ratings).

Example usage:

```bash
python -m run_autoeval_eval \
  --
  --sax_model_name=autoeval_sleep_all \
  --verticals=sleep \
  --verticals=fitness \
  --methods=score \
  --methods=generate \
  --splits=train \
  --splits=validation \
  --num_parallel_workers=100 \
  --alsologtostderr
  # --temperature      # Optional flag that overrides Sax model defaults.
  # --max_decode_steps # Optional flag that overrides Sax model defaults.
  # --overwrite        # Optional flag used to overwrite files if they exist.
```
"""

import os
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

import autoeval_data_lib
import autoeval_inference_lib


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

_VERTICALS = flags.DEFINE_multi_enum_class(
    name='verticals',
    default=[
        autoeval_data_lib.Vertical.SLEEP,
        autoeval_data_lib.Vertical.FITNESS,
    ],
    enum_class=autoeval_data_lib.Vertical,
    help='The verticals to run inference on.',
)

_SPLITS = flags.DEFINE_multi_enum_class(
    name='splits',
    default=[autoeval_data_lib.Split.VALIDATION],
    enum_class=autoeval_data_lib.Split,
    help='The splits to run inference on.',
)

_OVERWRITE = flags.DEFINE_bool(
    name='overwrite',
    default=False,
    help='Whether to overwrite AutoEval results files if they already exist.',
)


def run_eval(
    vertical: autoeval_data_lib.Vertical,
    split: autoeval_data_lib.Split,
    method: autoeval_inference_lib.Method,
    original_tensor_dicts: list[dict[str, tf.Tensor]],
    sax_model_name: str,
    num_parallel_workers: int,
    temperature: int | None,
    max_decode_steps: int | None,
    overwrite: bool,
) -> None:
  """Runs inference for AutoEval evaluation examples and writes outputs."""
  output_filepath = autoeval_data_lib.inference_output_filepath(
      vertical=vertical,
      split=split,
      sax_model_name=sax_model_name,
      datestr=autoeval_data_lib.ASOF_DATESTR,
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

  for vertical in _VERTICALS.value:
    for split in _SPLITS.value:
      original_tensor_dicts = autoeval_data_lib.load_tensor_dicts(
          vertical,
          split,
      )
      for method in _METHODS.value:
        run_eval(
            vertical=vertical,
            split=split,
            method=method,
            original_tensor_dicts=original_tensor_dicts,
            sax_model_name=_SAX_MODEL_NAME.value,
            num_parallel_workers=_NUM_PARALLEL_WORKERS.value,
            temperature=_TEMPERATURE.value,
            max_decode_steps=_MAX_DECODE_STEPS.value,
            overwrite=_OVERWRITE.value,
        )


if __name__ == '__main__':
  app.run(main)
