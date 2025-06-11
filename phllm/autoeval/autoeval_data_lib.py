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
"""A library for reading and writing with autoeval finetuning data."""

import datetime
import enum
import glob
import os
from typing import TypeVar

import immutabledict
import pandas as pd
import tensorflow as tf

from ..protos import conversation_pb2
import sstable


# ▼ Constants ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

ASOF_DATESTR = '2024-04-04'
ASOF_DATE = datetime.date.fromisoformat(ASOF_DATESTR)

# A base enumeration class type.
EnumBaseT = TypeVar('EnumBaseT', bound='EnumBase')


class EnumBase(enum.Enum):
  """A base enumeration with common classmethods."""

  @classmethod
  def from_name(cls: type[EnumBaseT], name: str) -> EnumBaseT:
    """Returns the enum value corresponding to the string name."""
    normalized_name = name.lower()
    for method in cls:
      if method.value == normalized_name:
        return method
    raise ValueError(f'Unknown enum: "{name}"')

  @classmethod
  def from_name_or_enum(
      cls: type[EnumBaseT],
      source: str | EnumBaseT,
  ) -> EnumBaseT:
    """Returns the enum value corresponding to the source."""
    if isinstance(source, str):
      return cls.from_name(source)
    if not isinstance(source, cls):
      raise ValueError(f'Unknown enum: "{source}"')
    return source


class Split(EnumBase):
  """Represents a dataset split (i.e., train, validation, test and holdout)."""

  TRAIN = 'train'
  VALIDATION = 'validation'
  TEST = 'test'
  HOLDOUT = 'holdout'


class Vertical(EnumBase):
  """Represents the vertical for case studies."""

  SLEEP = 'sleep'
  FITNESS = 'fitness'


#  Denotes the unique user ID.
USER_ID = 'user_id'

# Denotes the case study ID in case studies for data splitting.
CASE_STUDY_ID = 'case_study_id'

# Denotes the case study conversation ID.
CONVERSATION_ID = 'conversation_id'


# Mapping from vertical to id column name.
# Used for loading train/val/test/holdout splits based on the column name.
VERTICAL_TO_ID_COL = immutabledict.immutabledict({
    Vertical.SLEEP: CASE_STUDY_ID,
    Vertical.FITNESS: CASE_STUDY_ID,
})


# A mapping of vertical to the expected set of raters.
VERTICAL_TO_RATERS: immutabledict.immutabledict[Vertical, tuple[str, ...]] = (
    immutabledict.immutabledict({
        Vertical.SLEEP: (
            'Sleep Primary A',
            'Sleep Primary B',
            'Sleep Primary C',
            'Sleep Primary D',
            'Sleep Secondary A',
            'Sleep Secondary B',
        ),
        Vertical.FITNESS: (
            'Fitness Primary A',
            'Fitness Primary B',
            'Fitness Primary C',
            'Fitness Secondary A',
            'Fitness Secondary B',
            'Fitness Secondary C',
            'Fitness Secondary D',
        ),
    })
)

# A mapping of vertical to the evaluation tags/categories.
VERTICAL_TO_EVAL_TAGS: immutabledict.immutabledict[
    Vertical, tuple[str, ...]
] = immutabledict.immutabledict({
    Vertical.SLEEP: (
        'insight',
        'etiology',
        'recommendation',
        'overall',
    ),
    Vertical.FITNESS: (
        'health_metrics',
        'sleep',
        'training_load',
        'fitness_summary',
        'overall',
    ),
})

# A mapping of vertical to evaluation princples/criteria.
PRINCIPLES: tuple[str, ...] = (
    'important_domain_knowledge',
    'important_interpretations',
    'important_user_data',
    'no_assumptions',
    'no_hallucinations',
    'no_incorrect_domain_knowledge',
    'no_incorrect_important_interpretations',
    'no_incorrect_unimportant_interpretations',
    'no_incorrect_user_data',
    'no_unimportant_domain_knowledge',
    'no_unimportant_interpretations',
    'no_unimportant_user_data',
    'non_harmful',
    'overall_quality',
    'readable',
)

# A tuple of conversation sources.
CONVERSATION_SOURCES: tuple[str, ...] = (
    'expert',
    'base',
    'finetune',
)

# A mapping of vertical to conversation source to conversation sstable. These
# SSTables correspond to the conversations that were rated by experts.
_SLEEP_SSTABLE_DIR = '/path/to/sleep/sstable/dir'
_FITNESS_SSTABLE_DIR = '/path/to/fitness/sstable/dir'
VERTICAL_TO_SOURCE_TO_CONVERSATION_SSTABLE: immutabledict.immutabledict[
    Vertical, immutabledict.immutabledict[str, str]
] = immutabledict.immutabledict({
    Vertical.SLEEP: immutabledict.immutabledict({
        'expert': os.path.join(_SLEEP_SSTABLE_DIR, 'expert.sstable'),
        'base': os.path.join(_SLEEP_SSTABLE_DIR, 'base.sstable'),
        'finetune': os.path.join(_SLEEP_SSTABLE_DIR, 'finetune.sstable'),
    }),
    Vertical.FITNESS: immutabledict.immutabledict({
        'expert': os.path.join(_FITNESS_SSTABLE_DIR, 'expert.sstable'),
        'base': os.path.join(_FITNESS_SSTABLE_DIR, 'base.sstable'),
        'finetune': os.path.join(_FITNESS_SSTABLE_DIR, 'finetune.sstable'),
    }),
})


# ▼ Generic utilities ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼


def _tsv_to_df(filepath: str) -> pd.DataFrame:
  """Converts a TSV filepath to a pandas DataFrame."""
  with open(filepath, 'r') as f:
    df = pd.read_csv(f, sep='\t', index_col=False)
  return df


# ▼ Ratings ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼


def rater_responses_dir(vertical: Vertical) -> str:
  artifacts_dir = '/path/to/artifacts/dir'
  ratings_dir = os.path.join(
      artifacts_dir,
      'case_studies',
      vertical.value,
      'autoeval',
      'dataframes',
  )
  return ratings_dir


def rater_responses_filename(datestr: str) -> str:
  """Returns the ratings filename for the given date."""
  filestr = datetime.date.fromisoformat(datestr).strftime('%Y%m%d')
  return f'rater_responses.{filestr}.tsv'


def rater_responses_filepath(
    vertical: Vertical,
    datestr: str,
) -> str:
  """Returns the ratings filepath for the given vertical and date."""
  ratings_dir = rater_responses_dir(vertical)
  filename = rater_responses_filename(datestr)
  filepath = os.path.join(ratings_dir, filename)
  return filepath


def load_rater_responses(
    vertical: Vertical,
    datestr: str = ASOF_DATESTR,
) -> pd.DataFrame:
  """Returns the ratings dataframe for the given vertical and date."""
  filepath = rater_responses_filepath(vertical, datestr)
  return _tsv_to_df(filepath)


# ▼ Splits ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼


def split_dir(vertical: Vertical) -> str:
  """Returns the splits directory for the given vertical."""
  artifacts_dir = '/path/to/artifacts/dir'
  splits_dir = os.path.join(
      artifacts_dir,
      'case_studies',
      vertical.value,
      'autoeval',
      'splits',
  )
  return splits_dir


def split_filepath(
    vertical: Vertical,
    split: Split,
    datestr: str,
) -> str:
  """Returns the split filepath for the given vertical, split, and asof date."""
  output_dir = split_dir(vertical)
  split_enum = Split.from_name_or_enum(split)
  filestr = datetime.date.fromisoformat(datestr).strftime('%Y%m%d')
  filename = f'{split_enum.value.lower()}.{filestr}.csv'
  filepath = os.path.join(output_dir, filename)
  return filepath


def write_splits(
    splits: dict[Split, list[int]],
    vertical: Vertical,
    overwrite: bool = False,
) -> None:
  """Writes splits to a CSV file.

  Args:
    splits: A mapping of splits to ID sets.
    vertical: Vertical name.
    overwrite: Whether to overwrite the splits files if they already exist.

  Raises:
    ValueError: If not all splits are specified in `splits`.
    FileExistsError: The a splits file already exists and `overwrite` is False.
  """
  assert set(splits.keys()).issubset(set(Split))
  split_to_filepath = {
      split: split_filepath(
          vertical=vertical,
          split=split,
          datestr=ASOF_DATESTR,
      )
      for split in splits
  }
  # Note: We check whether files exist first to make this an atomic(ish) write.
  for split, filepath in split_to_filepath.items():
    if os.path.isfile(filepath) and not overwrite:
      raise FileExistsError(f'Split {split} already exists: {filepath}.')
  for split, filepath in split_to_filepath.items():
    df = pd.DataFrame({VERTICAL_TO_ID_COL[vertical]: sorted(splits[split])})
    with open(filepath, 'w') as f:
      df.to_csv(f, index=False)


def read_splits_file(filepath: str, id_col: str) -> list[int | str]:
  """Reads a list of splits from a CSV file."""
  with open(filepath, 'r') as f:
    df = pd.read_csv(f, sep='\t')
  return df[id_col].values.tolist()


def load_splits(
    vertical: Vertical,
    datestr: str = ASOF_DATESTR,
) -> dict[Split, list[int | str]]:
  """Returns a mapping of splits to IDs for the given vertical and date."""
  id_col = VERTICAL_TO_ID_COL[vertical]
  split_dict = {
      Split.TRAIN: read_splits_file(
          split_filepath(vertical, Split.TRAIN, datestr),
          id_col,
      ),
      Split.VALIDATION: read_splits_file(
          split_filepath(vertical, Split.VALIDATION, datestr),
          id_col,
      ),
  }
  return split_dict


# ▼ Conversations ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼


def load_conversation_sstable(filename: str) -> sstable.SSTable:
  """Loads a conversation sstable."""
  return sstable.SSTable(
      filename,
      wrapper=sstable.TableWrapper(conversation_pb2.Conversation.FromString),
  )


def load_graded_conversations(
    vertical: Vertical,
    conversation_source: str,
) -> sstable.SSTable:
  """Returns a conversations sstable for the given vertical and source."""
  filename = VERTICAL_TO_SOURCE_TO_CONVERSATION_SSTABLE[vertical][
      conversation_source
  ]
  return load_conversation_sstable(filename)


def load_all_graded_conversations(
    vertical: Vertical,
) -> dict[str, sstable.SSTable]:
  """Returns a mapping of conversation source to conversation sstables."""
  source_to_sstable = {
      source: load_graded_conversations(vertical, source)
      for source in CONVERSATION_SOURCES
  }
  return source_to_sstable


# ▼ TFRecords ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼


# AutoEval TFRecord TF example feature descriptors.
AUTOEVAL_TFRECORD_FEATURES = {
    'vertical': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'case_study_id': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'conversation_id': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'inputs': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'targets': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'rater': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'rating': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.int64,
    ),
    'conversation_source': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'tag': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
    'principle': tf.io.FixedLenFeature(
        shape=[],
        dtype=tf.string,
    ),
}


def tfrecord_dir(vertical: Vertical) -> str:
  """Returns the AutoEval TFrecords directory for the given vertical."""
  artifacts_dir = '/path/to/artifacts/dir'
  tfrecords_dir = os.path.join(
      artifacts_dir,
      'case_studies',
      vertical.value,
      'autoeval',
      'tfrecords',
  )
  return tfrecords_dir


def llm_tfrecords_filename_prefix(
    split: Split,
    datestr: str,
) -> str:
  """Returns the AutoEval TFrecords filename for the given split and date."""
  filestr = datetime.date.fromisoformat(datestr).strftime('%Y%m%d')
  return f'llm.autoeval.{split.value}.{filestr}.tfrecord'


def llm_tfrecords_filepath_prefix(
    vertical: Vertical,
    split: Split,
    rater: str,
    conversation_source: str,
    tag: str,
    principle: str,
    datestr: str,
) -> str:
  """Returns the AutoEval TFRecords filepath for the given descriptors."""
  if split not in [Split.TRAIN, Split.VALIDATION]:
    raise ValueError(f'Unsupported autoeval split: {split=}')
  if rater not in VERTICAL_TO_RATERS[vertical]:
    raise ValueError(f'Unknown {rater=} for {vertical=}')
  if conversation_source not in CONVERSATION_SOURCES:
    raise ValueError(f'Unknown {conversation_source=}')
  if tag not in VERTICAL_TO_EVAL_TAGS[vertical]:
    raise ValueError(f'Unknown {tag=} for {vertical=}')
  if principle not in PRINCIPLES:
    raise ValueError(f'Unknown {principle=}')
  tfrecords_dir = tfrecord_dir(vertical)
  filename_prefix = llm_tfrecords_filename_prefix(split, datestr)
  filepath = os.path.join(
      tfrecords_dir,
      'llm',
      rater,
      conversation_source,
      tag,
      principle,
      filename_prefix,
  )
  return filepath


def llm_tfrecords_filepaths(
    vertical: Vertical,
    split: Split,
    datestr: str = ASOF_DATESTR,
) -> list[str]:
  """Returns a list of all AutoEval TFRecord filepaths."""
  filepaths = []
  for rater in VERTICAL_TO_RATERS[vertical]:
    for conversation_source in CONVERSATION_SOURCES:
      for tag in VERTICAL_TO_EVAL_TAGS[vertical]:
        for principle in PRINCIPLES:
          filepath_prefix = llm_tfrecords_filepath_prefix(
              vertical=vertical,
              split=split,
              rater=rater,
              conversation_source=conversation_source,
              tag=tag,
              principle=principle,
              datestr=datestr,
          )
          filepaths.extend(glob.glob(f'{filepath_prefix}*'))
  return filepaths


def load_tensor_dicts(
    vertical: Vertical,
    split: Split,
    datestr: str = ASOF_DATESTR,
) -> list[dict[str, tf.Tensor]]:
  """Returns all tensor dictionaries for the given split."""
  filepaths = llm_tfrecords_filepaths(vertical, split, datestr)

  def _parse_function(tf_example: tf.train.Example) -> dict[str, tf.Tensor]:
    return tf.io.parse_single_example(tf_example, AUTOEVAL_TFRECORD_FEATURES)

  dataset = (
      tf.data.TFRecordDataset(
          filenames=filepaths,
          num_parallel_reads=tf.data.AUTOTUNE,
      )
      .map(
          _parse_function,
          num_parallel_calls=tf.data.AUTOTUNE,
      )
      .prefetch(
          buffer_size=tf.data.AUTOTUNE,
      )
  )
  tensor_dicts = [record for record in dataset]
  return tensor_dicts


def inference_output_filepath(
    vertical: Vertical,
    split: Split,
    sax_model_name: str,
    datestr: str,
    inference_tag: str | None = None,
) -> str:
  """Returns a versioned output filepath for the given descriptors."""
  df_dir = os.path.join(
      '/path/to/artifacts/dir',
      'case_studies',
      vertical.value,
      'autoeval',
      'inference',
  )
  filestr = datetime.date.fromisoformat(datestr).strftime('%Y%m%d')
  filename_tag = f'{inference_tag}.' if inference_tag else ''
  filename = f'{sax_model_name}.{split.value}.{filename_tag}{filestr}.tsv'
  return os.path.join(df_dir, filename)
