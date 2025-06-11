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
r"""Generates AutoEval LLM TFRecords for use in training and evaluation.

The resulting TFRecords contain the following keys:

  - `vertical`: The case study conversation vertical.
  - `case_study_id`: The case study ID.
  - `conversation_id`: The conversation ID.
  - `inputs`: The LLM prompt.
  - `targets`: The LLM targets.
  - `rater`: The expert rater.
  - `rating`: The ground truth expert rating.
  - `conversation_source`: The source of the rated conversation (i.e., one of
      expert, base Gemini, or finetuned Gemini).
  - `tag`: The rated category tag.
  - `principle`: The principle used when rating.

See autoeval_prompt_lib.py for details on prompt structure. Importantly, the
case study data provided to the model in the prompt is sourced from the
expert-generated case study inputs (rather than cascaded model inputs from prior
sections) to ensure consistency across conversation sources and to limit the
chances of early hallucinations impacting evaluation.

TFRecords filepath use the following directory structure:

```
/{artifact_dir}/{vertical}/autoeval/tfrecords/llm/{rater}/{conversation_source}/{tag}/{principle}
```

Filenames follow `llm.autoeval.{split}.{date}.tfrecord-00000-of-00001`. Given
the small number of ratings per directory criteria, each category consists of a
single shard (i.e., all files are `*.tfrecord-00000-of-00001`).
"""

import collections
import dataclasses
import os
from typing import Sequence

from absl import app
from absl import flags
import tensorflow as tf

import autoeval_data_lib
import autoeval_prompt_lib
from ..protos import case_study_pb2
from ..protos import conversation_pb2

_OVERWRITE = flags.DEFINE_bool(
    name='overwrite',
    default=False,
    help='Whether to overwrite existing split files if they already exist.',
)


@dataclasses.dataclass(frozen=True)
class ConversationRating:
  """Containiner class for all information related to a given rating."""

  vertical: autoeval_data_lib.Vertical
  case_study_id: str
  conversation_id: str
  rater: str
  tag: str
  principle: str
  conversation_source: str
  rating: int
  split: autoeval_data_lib.Split
  case_study: case_study_pb2.CaseStudy
  conversation: conversation_pb2.Conversation
  expert_conversation: conversation_pb2.Conversation

  def get_prompt(self) -> str:
    """Returns the prompt for the conversation rating."""
    prompt = autoeval_prompt_lib.build_prompt_for_conversation(
        vertical=self.vertical,
        tag=self.tag,
        principle=self.principle,
        target_conversation=self.conversation,
        expert_conversation=self.expert_conversation,
    )
    return prompt

  def get_target(self) -> str:
    """Returns the target for the conversation rating."""
    target = autoeval_prompt_lib.build_target(
        principle=self.principle,
        rating=self.rating,
    )
    return target

  def get_filepath_prefix(self) -> str:
    """Returns the filepath prefix for the conversation rating."""
    filepath_prefix = autoeval_data_lib.llm_tfrecords_filepath_prefix(
        vertical=self.vertical,
        split=self.split,
        rater=self.rater,
        conversation_source=self.conversation_source,
        tag=self.tag,
        principle=self.principle,
        datestr=autoeval_data_lib.ASOF_DATESTR,
    )
    return filepath_prefix

  def to_tf_example(self) -> tf.train.Example:
    """Returns the tf.train.Example for the conversation rating."""
    example_dict = {
        'vertical': self.vertical.value,
        'case_study_id': self.case_study_id,
        'conversation_id': self.conversation_id,
        'inputs': self.get_prompt(),
        'targets': self.get_target(),
        'rater': self.rater,
        'rating': self.rating,
        'conversation_source': self.conversation_source,
        'tag': self.tag,
        'principle': self.principle,
    }
    tf_example = tf.train.Example()
    for key, value in example_dict.items():
      if key == 'rating':
        tf_example.features.feature[key].int64_list.value.append(value)
      else:
        tf_example.features.feature[key].bytes_list.value.extend(
            [str(value).encode('utf-8')]
        )
    return tf_example


def build_split_dict(
    vertical: autoeval_data_lib.Vertical,
) -> dict[str, autoeval_data_lib.Split]:
  """Returns a mapping of case study ID to split."""
  split_dict = autoeval_data_lib.load_splits(vertical)
  case_study_id_to_split = {}
  for split, case_study_ids in split_dict.items():
    for case_study_id in case_study_ids:
      case_study_id_to_split[str(case_study_id)] = split
  return case_study_id_to_split


def build_case_study_dict(
    vertical: autoeval_data_lib.Vertical,
) -> dict[str, case_study_pb2.CaseStudy]:
  """Returns a mapping of case study ID to case study."""
  del vertical  # Unused.
  # Note: This function relied on internal Google storage systems and should
  # be implemented sich that a mapping of case study IDs to case studies is
  # returned.
  case_study_id_to_case_study = dict()
  return case_study_id_to_case_study


def build_conversation_dict(
    vertical: autoeval_data_lib.Vertical,
) -> dict[str, dict[str, conversation_pb2.Conversation]]:
  """Returns a mapping of source to case study ID to conversation."""
  source_to_conversation_sstable = (
      autoeval_data_lib.load_all_graded_conversations(vertical)
  )
  source_to_id_to_conversation = collections.defaultdict(dict)
  for source, conversation_sstable in source_to_conversation_sstable.items():
    for conversation in conversation_sstable.values():
      key = conversation.case_study_id
      assert key not in source_to_id_to_conversation[source]
      source_to_id_to_conversation[source][key] = conversation
  return source_to_id_to_conversation


def build_conversation_ratings(
    vertical: autoeval_data_lib.Vertical,
) -> list[ConversationRating]:
  """Returns a list of conversation ratings for all ratings in a vertical."""
  ratings_df = autoeval_data_lib.load_rater_responses(vertical)
  split_dict = build_split_dict(vertical)
  case_study_dict = build_case_study_dict(vertical)
  conversation_dict = build_conversation_dict(vertical)

  conversation_ratings = []
  for rating_record in ratings_df.to_dict('records'):
    case_study_id = rating_record['case_study_id']
    source = rating_record['conversation_source']
    conversation_rating = ConversationRating(
        vertical=vertical,
        case_study_id=case_study_id,
        conversation_id=rating_record['conversation_id'],
        rater=rating_record['rater'],
        tag=rating_record['tag'],
        principle=rating_record['principle'],
        conversation_source=source,
        rating=int(rating_record['rating']),
        split=split_dict[case_study_id],
        case_study=case_study_dict[case_study_id],
        conversation=conversation_dict[source][case_study_id],
        expert_conversation=conversation_dict[source][case_study_id],
    )
    conversation_ratings.append(conversation_rating)
  return conversation_ratings


def run_generate_for_vertical(
    vertical: autoeval_data_lib.Vertical, overwrite: bool
) -> None:
  """Generates AutoEval LLM TFRecords for a given vertical."""
  # Build conversation ratings.
  conversation_ratings = build_conversation_ratings(vertical)

  # Collect all conversations by the target TFRecord output filepath prefix.
  filepath_prefix_to_conversation_ratings = collections.defaultdict(list)
  for conversation_rating in conversation_ratings:
    filepath_prefix_to_conversation_ratings[
        conversation_rating.get_filepath_prefix()
    ].append(conversation_rating)

  # Convert conversation rating to tf examples and check if filepaths exists.
  filepath_to_tf_examples = {}
  for (
      filepath_prefix,
      ratings,
  ) in filepath_prefix_to_conversation_ratings.items():
    filepath = f'{filepath_prefix}-00000-of-00001'
    if os.path.isfile(filepath) and not overwrite:
      raise FileExistsError(filepath)
    filepath_to_tf_examples[filepath] = [
        rating.to_tf_example() for rating in ratings
    ]

  # Write all conversation ratings out as a single shard.
  for (
      filepath,
      tf_examples,
  ) in filepath_to_tf_examples.items():
    dirpath = os.path.dirname(filepath)
    if not os.path.isfile(dirpath):
      os.makedirs(dirpath)
    with tf.io.TFRecordWriter(str(filepath)) as writer:
      for example in tf_examples:
        writer.write(example.SerializeToString())  # pytype: disable=attribute-error


def run_generate(overwrite: bool) -> None:
  """Generates AutoEval LLM TFRecords for all verticals."""
  for vertical in [
      autoeval_data_lib.Vertical.SLEEP,
      autoeval_data_lib.Vertical.FITNESS,
  ]:
    run_generate_for_vertical(vertical, overwrite)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  run_generate(overwrite=_OVERWRITE.value)


if __name__ == '__main__':
  app.run(main)
