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
r"""Generates splits for the autoeval finetuning pipeline.

In order to ensure that each rater is evenly distributed across the autoeval
training and validation sets (80% and 20%, respectively), we first determine
aggregate all case studies so that they are grouped by the set of raters who
graded them. For each uninque rater group, we then split the case studies within
that group accroding to the target fractions and then merge the splits into
global training and validation sets. See the blocks below for concrete numbers
per vertical, split, and tag.

Note that this diverges from the existing split libraries for the following
reasons:

  1. We are only generating training and validation splits for the autoeval
     models (test and holdout sets are not needed).
  2. We are only splitting a subset of rated conversations (i.e., only a portion
     of the validation or holdout sets).
  3. We have additional logic ensuring that case studies are split in such a way
     that evenly distributes raters across the splits.

For sleep:

```
=================================
SLEEP
---------------------------------
TRAIN
                case_study_n=0038
                   ratings_n=4914
[             insight] tag_n=1512
[            etiology] tag_n=1512
[      recommendation] tag_n=1512
[             overall] tag_n=0378
---------------------------------
VALIDATION
                case_study_n=0012
                   ratings_n=1638
[             insight] tag_n=0504
[            etiology] tag_n=0504
[      recommendation] tag_n=0504
[             overall] tag_n=0126
=================================
```

For fitness:

```
=================================
FITNESS
---------------------------------
TRAIN
                case_study_n=0038
                   ratings_n=7140
[      health_metrics] tag_n=1680
[               sleep] tag_n=1680
[       training_load] tag_n=1680
[     fitness_summary] tag_n=1680
[             overall] tag_n=0420
---------------------------------
VALIDATION
                case_study_n=0012
                   ratings_n=2193
[      health_metrics] tag_n=0516
[               sleep] tag_n=0516
[       training_load] tag_n=0516
[     fitness_summary] tag_n=0516
[             overall] tag_n=0129
=================================
```

Output files are current written to the following directories:

  - sleep: /{artifacts_dir}/case_studies/sleep/autoeval/splits
  - fitness: /{artifacts_dir}/case_studies/fitness/autoeval/splits
"""

import collections
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
import pandas as pd

import autoeval_data_lib


_OVERWRITE = flags.DEFINE_bool(
    name='overwrite',
    default=False,
    help='Whether to overwrite existing split files if they already exist.',
)

# The random seed used when generating splits.
_SPLIT_RANDOM_SEED = 42

# Fractions determining the train/test/eval/holdout distributions.
_TRAIN_SPLIT_FRACTION = 0.8
_VALIDATION_SPLIT_FRACTION = 0.2
_TEST_SPLIT_FRACTION = 0.0
_HOLDOUT_SPLIT_FRACTION = 0.0


def group_case_studies_by_raters(
    ratings_df: pd.DataFrame,
    raters: tuple[str, ...],
) -> dict[tuple[str, ...], list[str]]:
  """Returns a mapping of raters to their case study IDs."""
  ratings_df_raters = set(ratings_df.rater)
  diff = ratings_df_raters.symmetric_difference(set(raters))
  if diff:
    raise ValueError(f'Unexpected mismatch of raters: {diff}')

  # Map each case study ID to the list of raters associated with that ID.
  id_to_raters = collections.defaultdict(set)
  for record in ratings_df.to_dict('records'):
    id_to_raters[record['case_study_id']].add(record['rater'])

  # Convert the list of raters to a tuple of raters so that it's hashable. Note
  # that the list of raters is first sorted so that pairs are ordered.
  id_to_raters_tuple = {
      case_study_id: tuple(sorted(raters))
      for case_study_id, raters in id_to_raters.items()
  }

  # Map each rater tuple to the set of IDs associated with that tuple.
  raters_to_ids = collections.defaultdict(set)
  for case_study_id, rater_tuple in id_to_raters_tuple.items():
    raters_to_ids[rater_tuple].add(case_study_id)
  sorted_raters_to_ids = {
      raters: sorted(ids)
      for raters, ids in sorted(
          raters_to_ids.items(),
          key=lambda x: len(x[1]),
          reverse=True,
      )
  }
  return sorted_raters_to_ids


def _assert_disjoint_split_dicts(
    split_dict_a: dict[autoeval_data_lib.Split, list[int | str]],
    split_dict_b: dict[autoeval_data_lib.Split, list[int | str]],
) -> None:
  """Asserts that two split dictionaries are disjoint."""
  key_superset = set(split_dict_a.keys()).union(set(split_dict_b.keys()))
  for key in key_superset:
    values_a = split_dict_a.get(key, [])
    values_b = split_dict_b.get(key, [])
    assert len(values_a) == len(set(values_a))
    assert len(values_b) == len(set(values_b))
    assert set(values_a).isdisjoint(set(values_b))


def _merge_split_dicts(
    split_dict_a: dict[autoeval_data_lib.Split, list[int | str]],
    split_dict_b: dict[autoeval_data_lib.Split, list[int | str]],
) -> dict[autoeval_data_lib.Split, list[int | str]]:
  """Merges two disjoint split dictionaries into one."""
  _assert_disjoint_split_dicts(split_dict_a, split_dict_b)
  merged_split_dict = collections.defaultdict(list)
  for key, values in split_dict_a.items():
    merged_split_dict[key].extend(values)
  for key, values in split_dict_b.items():
    merged_split_dict[key].extend(values)
  for key, values in merged_split_dict.items():
    merged_split_dict[key] = sorted(values)
  return merged_split_dict


def generate_splits(
    sample_ids: list[int | str],
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    holdout_fraction: float,
    seed: int,
) -> dict[autoeval_data_lib.Split, list[int | str]]:
  """Generates train, validation, and test splits for the given set of IDs.

  This process is deterministic for a given set of arguments. It first sorts the
  list of sample IDs, randomly shuffles them using `seed`, and then breaks up
  IDs according to the provided train, validation, and test fractions.

  Args:
    sample_ids: A list of unique IDs.
    train_fraction: The fraction of IDs to include in the training set.
    validation_fraction: The fraction of IDs to include in the validation set.
    test_fraction: The fraction of IDs to include in the test set.
    holdout_fraction: The fraction of IDs to include in the holdout set.
    seed: The random seed.

  Returns:
    A mapping of `autoeval_data_lib.Splits` to their corresponding ID set.

  Raises:
    ValueError: If elements in `sample_ids` are not unique.
    ValueError: If train, validation, and test fractions do not sum to 1.
    RuntimeError: If the split process fails.
  """
  if len(set(sample_ids)) != len(sample_ids):
    id_counts = collections.Counter(sample_ids)
    duplicate_ids = sorted([x for x, count in id_counts.items() if count > 1])
    raise ValueError(f'`sample_ids` must be unique: {duplicate_ids}.')

  fractions = [
      train_fraction,
      validation_fraction,
      test_fraction,
      holdout_fraction,
  ]
  if not np.isclose(sum(fractions), 1.0):
    raise ValueError(
        'Train, validation, test and holdout fractions must sum to 1.'
    )
  if min(fractions) < 0:
    raise ValueError(
        'Train, validation, test and holdout fractions must be >= 0:'
        f' {fractions}'
    )

  rng = np.random.default_rng(seed=seed)
  sample_ids = sorted(sample_ids)
  rng.shuffle(sample_ids)

  split_indices = np.cumsum((np.array(fractions) * len(sample_ids))[:-1])
  split_indices = split_indices.astype(np.int32)
  train_ids, validation_ids, test_ids, holdout_ids = np.split(
      sample_ids, split_indices
  )

  # Check that all individuals are in a split and that splits are unique.
  split_ids = [*train_ids, *validation_ids, *test_ids, *holdout_ids]
  if len(split_ids) != len(sample_ids):
    missing_ids = sorted(set(split_ids).difference(set(sample_ids)))
    raise RuntimeError(f'Not all IDs are in a split: {missing_ids}')
  if not (
      set(train_ids).isdisjoint(set(validation_ids))
      and set(train_ids).isdisjoint(set(test_ids))
      and set(train_ids).isdisjoint(set(holdout_ids))
      and set(validation_ids).isdisjoint(set(test_ids))
      and set(validation_ids).isdisjoint(set(holdout_ids))
      and set(holdout_ids).isdisjoint(set(test_ids))
  ):
    raise RuntimeError('Splits are not disjoint.')

  split_dict = {
      autoeval_data_lib.Split.TRAIN: sorted(train_ids),
      autoeval_data_lib.Split.VALIDATION: sorted(validation_ids),
      autoeval_data_lib.Split.TEST: sorted(test_ids),
      autoeval_data_lib.Split.HOLDOUT: sorted(holdout_ids),
  }
  return split_dict


def split_case_studies_by_raters(
    raters_to_ids: dict[tuple[str, ...], list[str]],
) -> dict[autoeval_data_lib.Split, list[int | str]]:
  """Returns a mapping of splits to case study IDs by merging per-rater splits.

  For each tuple of raters, which contains one rater for each rater that has
  evaluated a given case study, we generate a separate set of splits so that
  there is an even distribution of raters across the global split. We then merge
  these per-rate splits into our final split.

  Args:
    raters_to_ids: A mapping of rater tuples to a list of case study IDs.

  Returns:
    A global split dictionary that has been aggregated across per-rate splits.
  """
  split_dict = collections.defaultdict(list)
  for ids in raters_to_ids.values():
    rater_split_dict = generate_splits(
        sample_ids=ids,
        train_fraction=_TRAIN_SPLIT_FRACTION,
        validation_fraction=_VALIDATION_SPLIT_FRACTION,
        test_fraction=_TEST_SPLIT_FRACTION,
        holdout_fraction=_HOLDOUT_SPLIT_FRACTION,
        seed=_SPLIT_RANDOM_SEED,
    )
    split_dict = _merge_split_dicts(split_dict, rater_split_dict)
  return split_dict


def print_split_stats(
    vertical: autoeval_data_lib.Vertical,
    ratings_df: pd.DataFrame,
    split_dict: dict[autoeval_data_lib.Split, list[int | str]],
) -> None:
  """Prints out the number of case studies, ratings, and tags per split."""
  print('=' * 33)
  print(vertical.value.upper())
  for split, values in split_dict.items():
    if not values:
      continue
    print('-' * 33)
    case_study_mask = ratings_df.case_study_id.isin(values)
    ratings_n = case_study_mask.sum()
    print(f'{split.value.upper()}')
    empty = ''
    print(f'{empty:>16}case_study_n={len(values):0>4}')
    print(f'{empty:>19}ratings_n={ratings_n:0>4}')
    for tag in autoeval_data_lib.VERTICAL_TO_EVAL_TAGS[vertical]:
      tag_n = (case_study_mask & ratings_df.tag.isin([tag])).sum()
      print(f'[{tag:>20}] tag_n={tag_n:0>4}')
  print('=' * 33)
  print()


def run_split_for_vertical(
    vertical: autoeval_data_lib.Vertical,
) -> dict[autoeval_data_lib.Split, list[int | str]]:
  """Returns a mapping of splits to case study IDs for the given vertical."""
  ratings_df = autoeval_data_lib.load_rater_responses(vertical)
  raters = autoeval_data_lib.VERTICAL_TO_RATERS[vertical]
  raters_to_ids = group_case_studies_by_raters(ratings_df, raters)
  split_dict = split_case_studies_by_raters(raters_to_ids)
  print_split_stats(vertical, ratings_df, split_dict)
  return split_dict


def run_split(overwrite: bool):
  """Runs the splitting procedure for each vertical."""
  for vertical in [
      autoeval_data_lib.Vertical.SLEEP,
      autoeval_data_lib.Vertical.FITNESS,
  ]:
    split_dict = run_split_for_vertical(vertical)
    autoeval_data_lib.write_splits(split_dict, vertical, overwrite=overwrite)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  run_split(overwrite=_OVERWRITE.value)


if __name__ == '__main__':
  app.run(main)
