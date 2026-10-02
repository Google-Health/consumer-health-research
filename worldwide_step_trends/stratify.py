# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Determines mean physical activity by country, year, age and strata.

This module processes large-scale, sharded Fitbit daily step datasets and
aggregates them into multi-dimensional demographic grid strata: Country x
Calendar Year x Sex (self-reported at birth) x 5-Year Age Group (midage).
It merges raw device readings with user demographic profiles, filters outlier
step volumes, and calculates stratum-level running statistics (sample count,
sum, and sum-of-squares).

To support national and global public health comparisons, the module aligns
wearable aggregates with United Nations World Population Prospects (UN WPP)
census data. The UN WPP data is strictly for post-stratification reweighting
(adjusting for demographic distribution), as opposed to hierarchical
missing-data imputation (which imputes empty activity cells across neighboring
demographic strata).

Processing Pipeline:
  1. Demographic Ingestion: Standardizes country codes, timezones, and ages.
  2. Sharded Computation: Evaluates daily steps across N shards with
     automatic, resumable checkpointing to handle large datasets robustly.
  3. Stratum Reduction: Computes user-level annual means and aggregates into
     4D demographic cells (country, year, sex/gender, midage).
  4. Post-Stratification & Imputation: Applies UN census population proportions
     and provides stochastic or median imputation for sparse strata.

Typical usage example:

  import stratify

  # Initialize from precomputed or on-disk shards with checkpointing
  pipeline = stratify.StratifySteps.from_shards(
      path_base='/path/to/data',
      start_date='2015-01-01',
      end_date='2023-12-31',
      n_shards=100,
      un_pop_male_path='/path/to/un_male.xlsx',
      un_pop_female_path='/path/to/un_female.xlsx',
      min_days_per_user=1,
      min_age=20,
      max_age=80,
  )

  # Process all shards into the running results grid
  pipeline.process_shards()

  # Retrieve mean step counts with hierarchical imputation
  mean_grid, nan_grid = pipeline.get_mean_grid()

  # Obtain UN census-standardized population estimates
  reweighted_grid, _ = pipeline.get_reweighted_grid(
      groupby_cols=['country', 'year']
  )
"""

from collections.abc import Sequence
import hashlib
import os
import pickle as pkl
import time
from typing import Any, cast

from absl import app
from absl import logging
import numpy as np
import pandas as pd
import pycountry
import pytz

# Constants
_CKPT_FOLDER = 'stratify_steps'
_COUNTRY_ALIASES = {
    'United States': 'United States of America',
    'Korea, Republic of': 'South Korea',
    'Czechia': 'Czech Republic',
    'Viet Nam': 'Vietnam',
}
_UN_COUNTRY_ALIASES = {
    'China, Hong Kong SAR': 'Hong Kong',
    'Czechia': 'Czech Republic',
    'Republic of Korea': 'South Korea',
    'China, Taiwan Province of China': 'Taiwan',
    'Türkiye': 'Turkey',
    'Viet Nam': 'Vietnam',
}
_UN_AGE_GROUP_TO_MIDPOINT = {
    '20-24': 22.5,
    '25-29': 27.5,
    '30-34': 32.5,
    '35-39': 37.5,
    '40-44': 42.5,
    '45-49': 47.5,
    '50-54': 52.5,
    '55-59': 57.5,
    '60-64': 62.5,
    '65-69': 67.5,
    '70-74': 72.5,
    '75-79': 77.5,
    '80-84': 82.5,
    '85-89': 87.5,
    '90-94': 92.5,
    '95-99': 97.5,
    '100+': 102.5,
}
_TIMEZONE_MAP = {
    'America/Curacao': 'Curacao',
    'Europe/London': 'United Kingdom',
    'America/Marigot': 'Saint Martin',
    'America/New_York': 'United States of America',
    'Europe/Dublin': 'Ireland',
    'Africa/Lagos': 'Nigeria',
    'Europe/Madrid': 'Spain',
    'America/Toronto': 'Canada',
    'Europe/Kiev': 'Ukraine',
    'Europe/Brussels': 'Belgium',
    'Asia/Bangkok': 'Thailand',
    'America/Yellowknife': 'Canada',
    'America/Thunder_Bay': 'Canada',
    'America/Montserrat': 'Montserrat',
    'Atlantic/Stanley': 'Falkland Islands',
    'Antarctica/Palmer': 'Antarctica',
    'Asia/Nicosia': 'Cyprus',
    'America/Chicago': 'United States of America',
    'Asia/Tokyo': 'Japan',
    'Europe/Stockholm': 'Sweden',
    'Europe/Berlin': 'Germany',
    'Asia/Dubai': 'United Arab Emirates',
    'America/Winnipeg': 'Canada',
}
_DEMO_COLS = (
    'user_id',
    'birth_year',
    'gender',
    'country',
)
_GENDER_LABELS = {
    0: 'male',
    1: 'female',
}
# Generic names for the demographics input file and the raw daily step column.
_DEFAULT_DEMO_QUERY_NAME = 'demographics'
_DEFAULT_STEP_COL = 'daily_steps'


def _resolve_source_names() -> tuple[str, str]:
  """Returns the demographics file name and the raw daily step column name."""
  return _DEFAULT_DEMO_QUERY_NAME, _DEFAULT_STEP_COL


_DEMO_QUERY_NAME, _STEP_COL = _resolve_source_names()


# File handling abstraction layer


def _resolve_local_data_dir(subpath: str, default_dir: str | None) -> str:
  """Resolves a local data directory path."""
  if default_dir is not None:
    return default_dir
  local_dir = f'./data/{subpath}'
  if os.path.exists(local_dir):
    return local_dir
  repo_dir = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'data', subpath
  )
  return repo_dir if os.path.exists(repo_dir) else local_dir


def get_internal_data_dir(default_dir: str | None = None) -> str:
  """Returns the internal data directory (defaults to ./data/internal)."""
  env_dir = os.environ.get(
      'STEP_TRENDS_INTERNAL_DATA_DIR',
      os.environ.get('STEP_TRENDS_DATA_DIR'),
  )
  return env_dir or _resolve_local_data_dir('internal', default_dir)


def get_public_data_dir(default_dir: str | None = None) -> str:
  """Returns the public data directory (defaults to ./data/public)."""
  env_dir = os.environ.get('STEP_TRENDS_PUBLIC_DATA_DIR')
  return env_dir or _resolve_local_data_dir('public', default_dir)


def get_data_dir(default_dir: str | None = None) -> str:
  """Returns the default data directory path."""
  return get_internal_data_dir(default_dir)


def make_dirs(path: str) -> None:
  """Creates a directory and its parents if it doesn't exist."""
  if path:
    os.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
  """Checks whether a file exists."""
  return os.path.exists(path)


def open_file(path: str, mode: str = 'r') -> Any:
  """Opens a file with the given mode."""
  return open(path, mode)


def save_pickle(obj: Any, path: str) -> None:
  """Saves an object to a pickle file."""
  with open_file(path, 'wb') as f:
    pkl.dump(obj, f)


def load_pickle(path: str) -> Any:
  """Loads an object from a pickle file."""
  with open_file(path, 'rb') as f:
    return pkl.load(f)  # pylint: disable=g-unsafe-pickle-load


def rename_file(src: str, dst: str, overwrite: bool = False) -> None:
  """Renames a file."""
  if not overwrite and os.path.exists(dst):
    raise FileExistsError(f'Destination path {dst} already exists.')
  os.replace(src, dst)


def get_filename(
    query_name: str,
    path_base: str,
    start_date: str,
    end_date: str,
    one_in_n: int | None,
    modulo: int | None = 0,
) -> str:
  """Returns the filename to save the dataframe to.

  Args:
    query_name: Name of the query.
    path_base: Base path for the file.
    start_date: Start date for the data.
    end_date: End date for the data.
    one_in_n: Sampling rate (e.g., 1 in n).
    modulo: Modulo for sampling.

  Returns:
    The formatted filename.
  """
  if one_in_n is None:
    one_in_n = 1
  if modulo is None:
    modulo = 0
  return os.path.join(
      path_base,
      f'{query_name}_{start_date}_{end_date}_{one_in_n}_{modulo}.pkl',
  )


def pickle_write(dataframe: pd.DataFrame, filename: str) -> None:
  """Writes the dataframe to the supplied filename.

  Args:
    dataframe: The dataframe to write.
    filename: The filename to write to.
  """
  logging.info('Writing to %s', filename)
  t0 = time.time()
  dir_name = os.path.dirname(filename)
  if dir_name:
    make_dirs(dir_name)
  save_pickle(dataframe, filename)

  time_elapsed = time.time() - t0
  if time_elapsed < 60:
    logging.info('Total elapsed time: %.2f seconds', time_elapsed)
  else:
    logging.info('Total elapsed time: %.2f minutes', time_elapsed / 60)


def load_file(
    query_name: str,
    path_base: str,
    start_date: str,
    end_date: str,
    one_in_n: int | None,
    modulo: int | None = 0,
) -> pd.DataFrame:
  """Loads the dataframe from the file.

  Args:
    query_name: Name of the query.
    path_base: Base path for the file.
    start_date: Start date for the data.
    end_date: End_date for the data.
    one_in_n: Sampling rate (e.g., 1 in n).
    modulo: Modulo for sampling.

  Returns:
    The loaded dataframe.
  """
  file_name = get_filename(
      query_name=query_name,
      path_base=path_base,
      start_date=start_date,
      end_date=end_date,
      one_in_n=one_in_n,
      modulo=modulo,
  )
  return load_pickle(file_name)


def get_country_name_or_code(country_code: str | float | None) -> str | None:
  """Converts alpha2 country code to full name if known.

  Args:
    country_code: alpha2 country code, e.g. BB is Barbados.

  Returns:
    The full name if mapped, or the raw string fallback if the country code is
    not known. Returns None if missing, ensuring unmapped codes are not silently
    discarded.
  """
  if pd.isna(country_code) or country_code is None:
    return None
  code_str = str(country_code).strip()
  name = code_str
  try:
    match = pycountry.countries.get(alpha_2=code_str)
    if match:
      name = getattr(match, 'name', name)
  except (KeyError, ValueError, TypeError):
    pass
  return _COUNTRY_ALIASES.get(name, name)


def get_timezone_mapping() -> dict[str, str]:
  """Returns a mapping of timezone to country name.

  This map is built from both pytz.country_timezones and country aliases,
  namely, pytz indexes timezones by ISO-2 alpha country codes, which are then
  mapped to standard country names and augmented with manual alias overrides.
  """
  zone_to_country_name = {}
  for code, zones in pytz.country_timezones.items():
    for zone in zones:
      zone_to_country_name[zone] = get_country_name_or_code(code)
  for zone, country in _TIMEZONE_MAP.items():
    zone_to_country_name[zone] = _COUNTRY_ALIASES.get(country, country)
  return zone_to_country_name


def read_un_xlsx(
    path: str, gender_label: int, age_groups: Sequence[str]
) -> pd.DataFrame:
  """Reads and parses a single UN population XLSX file.

  Args:
    path: Path to the XLSX file.
    gender_label: Label to apply to the 'gender' column (0=male, 1=female).
    age_groups: Sequence of column names representing age brackets to extract.

  Returns:
    A melted DataFrame with columns: country, year, age_bracket, Count,
      gender.
  """
  with open_file(path, 'rb') as f:
    temp_df = pd.read_excel(f, skiprows=16, na_values='...')
    region_col = 'Region, subregion, country or area *'
    year_col = 'Year'
    cols_to_keep = [region_col, year_col] + list(age_groups)
    temp_df = temp_df[cols_to_keep].rename(
        columns={region_col: 'country', year_col: 'year'}
    )
    temp_df['year'] = pd.to_numeric(temp_df['year'], errors='coerce')
    temp_df = temp_df.dropna(subset=['country', 'year'])
    temp_df['year'] = temp_df['year'].astype(int)
    melted = temp_df.melt(
        id_vars=['country', 'year'],
        value_vars=list(age_groups),
        var_name='age_bracket',
        value_name='Count',
    )
    melted['Count'] = pd.to_numeric(melted['Count'], errors='coerce').fillna(0)
    melted['gender'] = gender_label
    return melted


def load_un_population_data(
    un_pop_male_path: str,
    un_pop_female_path: str,
) -> pd.DataFrame:
  """Loads UN population data and returns population counts for demographics.

  Args:
    un_pop_male_path: Path to the UN population data for males.
    un_pop_female_path: Path to the UN population data for females.

  Returns:
    A DataFrame with columns: country, year, gender, age (midpoint),
    and Count.
  """
  age_groups = list(_UN_AGE_GROUP_TO_MIDPOINT.keys())

  df_male = read_un_xlsx(
      un_pop_male_path, gender_label=0, age_groups=age_groups
  )
  df_female = read_un_xlsx(
      un_pop_female_path, gender_label=1, age_groups=age_groups
  )
  un_pop = pd.concat([df_male, df_female], ignore_index=True)

  un_pop['age'] = un_pop['age_bracket'].map(_UN_AGE_GROUP_TO_MIDPOINT)

  def map_country(country: str) -> str:
    country = _UN_COUNTRY_ALIASES.get(country, country)
    return _COUNTRY_ALIASES.get(country, country)

  un_pop['country'] = un_pop['country'].map(map_country)

  # Check and pad projection years independently (e.g. 2024 and 2025)
  for target_year in (2024, 2025):
    if target_year not in un_pop['year'].unique():
      pop_pad = un_pop[un_pop['year'] == 2023].copy()
      if not pop_pad.empty:
        pop_pad['year'] = target_year
        un_pop = pd.concat([un_pop, pop_pad], ignore_index=True)

  return cast(pd.DataFrame, un_pop)


def process_demo(
    path_base: str,
    start_date: str,
    end_date: str,
    min_days_per_user: int,
) -> pd.DataFrame:
  """Loads and standardizes user demographic attributes.

  Demographic data is loaded separately from daily steps because step records
  contain millions of daily time-series rows, while demographics are user-level
  static metadata. Loading them separately avoids redundant demographic columns
  across hundreds of step shards.

  Args:
    path_base: Base directory for user demographic file.
    start_date: Start date filter string.
    end_date: End date filter string.
    min_days_per_user: Minimum total number of recorded wear days required per
      user across the full study window (the paper's precomputed grids use
      min_days_per_user=1).

  Returns:
    Cleaned DataFrame with columns: user_id, birth_year, gender, country.
  """
  logging.info('Loading demographic data...')
  start = time.time()
  df_demo = load_file(
      query_name=_DEMO_QUERY_NAME,
      path_base=path_base,
      start_date=start_date,
      end_date=end_date,
      one_in_n=None,
      modulo=None,
  )

  zone_to_country_name = get_timezone_mapping()
  df_demo['country'] = df_demo['user_timezone'].map(zone_to_country_name)
  df_demo['country'] = df_demo['country'].map(
      lambda c: _COUNTRY_ALIASES.get(c, c) if pd.notna(c) else c
  )

  df_demo = df_demo[df_demo['gender'].isin([1, 2])]
  df_demo['gender'] = (df_demo['gender'] == 2).astype(int)

  df_demo = df_demo.dropna(subset=list(_DEMO_COLS))
  df_demo = df_demo[df_demo['n_days'] >= min_days_per_user]

  logging.info(
      'Time to process demographic data: %.2f seconds', time.time() - start
  )

  return df_demo[list(_DEMO_COLS)].copy()


def apply_demo(
    df_steps: pd.DataFrame,
    df_demo: pd.DataFrame,
    min_age: int,
    max_age: int,
) -> pd.DataFrame:
  """Filters step readings by joining demographics and restricting by age.

  Args:
    df_steps: DataFrame of user steps containing activity_dt and user_id.
    df_demo: Decoupled demographic details mappings.
    min_age: Minimum required boundary.
    max_age: Maximum required boundary.

  Returns:
    DataFrame with valid demographics mapped restricted to the age target.
  """
  df_merged = pd.merge(
      df_steps,
      df_demo,
      on='user_id',
      how='inner',
  )
  df_merged['activity_dt'] = pd.to_datetime(df_merged['activity_dt'])
  df_merged['year'] = df_merged['activity_dt'].dt.year
  df_merged['age'] = df_merged['year'] - df_merged['birth_year']

  df_merged = df_merged[
      (df_merged['age'] >= min_age) & (df_merged['age'] <= max_age)
  ]
  df_merged['country'] = df_merged['country'].map(
      lambda c: _COUNTRY_ALIASES.get(c, c) if pd.notna(c) else c
  )
  return df_merged


class StratifySteps:
  """Determines mean physical activity by demographic stratification.

  This class orchestrates the entire data processing pipeline:
  1. Ingests daily step shards and aggregates them into running statistics
     (sum, sum_sq, count) across (country, year, sex/gender, 5-year age group).
  2. Implements atomic checkpointing to reliably resume after preemptions.
  3. Applies hierarchical missing data imputation for sparse strata.
  4. Computes final, nationally representative step estimates using UN WPP
     census post-stratification reweighting.
  """

  def __init__(
      self,
      path_base: str,
      start_date: str,
      end_date: str,
      n_shards: int = 1,
      un_pop_male_path: str = '',
      un_pop_female_path: str = '',
      min_days_per_user: int = 30,
      min_age: int = 20,
      max_age: int = 80,
      exclude_outliers: bool = False,
      exclude_extreme_values: bool = False,
      min_steps: int | None = None,
      max_steps: int | None = None,
      exclude_days_of_week: Sequence[int] | None = None,
      target_country: str | None = None,
      start_month: int | None = None,
      stop_month: int | None = None,
      override_hash: str | None = None,
  ) -> None:
    """Initializes the StratifySteps instance.

    Args:
      path_base: Base path for reading and writing data files.
      start_date: Start date for the data analysis (e.g., 'YYYY-MM-DD').
      end_date: End date for the data analysis (e.g., 'YYYY-MM-DD').
      n_shards: Number of shards to split the processing into.
      un_pop_male_path: Path to the UN population data file for males.
      un_pop_female_path: Path to the UN population data file for females.
      min_days_per_user: Minimum total number of recorded wear days required per
        user across the full study window (the paper's precomputed grids use
        min_days_per_user=1).
      min_age: Minimum age boundary (inclusive).
      max_age: Maximum age boundary (inclusive).
      exclude_outliers: Whether to exclude outliers (1.5 * IQR, where IQR bounds
        are computed per shard when enabled).
      exclude_extreme_values: Whether to exclude extreme values (3.0 * IQR,
        where IQR bounds are computed per shard when enabled).
      min_steps: Minimum daily steps threshold (inclusive).
      max_steps: Maximum daily steps threshold (inclusive).
      exclude_days_of_week: Days of week to exclude (0=Monday, ..., 6=Sunday).
      target_country: Specific country to filter the analysis to, if any.
      start_month: Start month for seasonal filtering (inclusive).
      stop_month: Stop month for seasonal filtering (inclusive).
      override_hash: Optional hash to override the automatically generated
        params hash for checkpoints.
    """
    self.path_base = path_base or get_internal_data_dir()
    self.start_date = start_date
    self.end_date = end_date
    self.n_shards = n_shards
    pub_dir = get_public_data_dir()
    self.un_pop_male_path = (
        un_pop_male_path
        or f'{pub_dir}/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx'
    )
    female_default = (
        f'{pub_dir}/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx'
    )
    self.un_pop_female_path = un_pop_female_path or female_default
    self.min_days_per_user = min_days_per_user
    self.min_age = min_age
    self.max_age = max_age
    self.exclude_outliers = exclude_outliers
    self.exclude_extreme_values = exclude_extreme_values
    self.min_steps = min_steps
    self.max_steps = max_steps
    self.exclude_days_of_week = exclude_days_of_week
    self.target_country = target_country
    self.start_month = start_month
    self.stop_month = stop_month

    self._un_pop: pd.DataFrame | None = None
    self.demo: pd.DataFrame | None = None

    # Compute hash of parameters to make checkpoints unique
    self.params_hash = override_hash
    if self.params_hash is None:
      params_to_hash = (
          self.start_date,
          self.end_date,
          self.n_shards,
          self.min_days_per_user,
          self.min_age,
          self.max_age,
          self.exclude_outliers,
          self.exclude_extreme_values,
          self.min_steps,
          self.max_steps,
          list(self.exclude_days_of_week)
          if self.exclude_days_of_week is not None
          else None,
          self.target_country,
          self.start_month,
          self.stop_month,
      )
      params_str = str(params_to_hash).encode('utf-8')
      self.params_hash = hashlib.md5(params_str).hexdigest()

    self.checkpoint_dir = f'{self.path_base}/{_CKPT_FOLDER}'
    ckpt_prefix = f'{self.checkpoint_dir}/checkpoint'
    suffix = f'{self.start_date}_{self.end_date}_{self.params_hash}.pkl'
    self.checkpoint_completed_file = f'{ckpt_prefix}_completed_{suffix}'
    self.checkpoint_grid_file = f'{ckpt_prefix}_grid_{suffix}'
    self.checkpoint_processed_file = f'{ckpt_prefix}_processed_{suffix}'
    self.checkpoint_state_file = f'{ckpt_prefix}_state_{suffix}'

    make_dirs(self.checkpoint_dir)

    self.processed_shards: set[int] = set()
    self.results_grid: pd.DataFrame | None = None

  @property
  def un_pop(self) -> pd.DataFrame:
    """Returns the UN population DataFrame, loading it lazily if needed."""
    if self._un_pop is None:
      self._un_pop = self.load_un_population_data()
    return self._un_pop

  @un_pop.setter
  def un_pop(self, value: pd.DataFrame | None) -> None:
    self._un_pop = value

  @property
  def step_sum_grid(self) -> pd.DataFrame | None:
    """Returns the step sum grid derived from results_grid, or None."""
    if self.results_grid is None or 'steps' not in self.results_grid:
      return None
    return pd.DataFrame(self.results_grid['steps'])

  @step_sum_grid.setter
  def step_sum_grid(self, value: pd.DataFrame | None) -> None:
    if value is None:
      return
    if self.results_grid is None:
      self.results_grid = pd.DataFrame(index=value.index)
    self.results_grid['steps'] = value['steps']

  @property
  def count_grid(self) -> pd.DataFrame | None:
    if self.results_grid is None or 'readings' not in self.results_grid:
      return None
    return pd.DataFrame(self.results_grid['readings'])

  @count_grid.setter
  def count_grid(self, value: pd.DataFrame | None) -> None:
    if value is None:
      return
    if self.results_grid is None:
      self.results_grid = pd.DataFrame(index=value.index)
    self.results_grid['readings'] = value['readings']

  @classmethod
  def from_shards(
      cls,
      path_base: str,
      start_date: str,
      end_date: str,
      n_shards: int,
      un_pop_male_path: str,
      un_pop_female_path: str,
      min_days_per_user: int = 30,
      min_age: int = 20,
      max_age: int = 80,
      exclude_outliers: bool = False,
      exclude_extreme_values: bool = False,
      min_steps: int | None = None,
      max_steps: int | None = None,
      exclude_days_of_week: Sequence[int] | None = None,
      target_country: str | None = None,
      start_month: int | None = None,
      stop_month: int | None = None,
      override_hash: str | None = None,
  ) -> 'StratifySteps':
    """Initializes by fetching shards using stored checkpoints.

    This method allows resuming computation from interrupted runs. It
    calculates a parameter hash to bind checkpoints to unique parameter
    configurations, ensuring robust cache invalidation if execution arguments
    change. Checkpoint files are automatically loaded if they match this hash.

    Args:
      path_base: Base path for reading and writing data files.
      start_date: Start date for the data analysis (e.g., 'YYYY-MM-DD').
      end_date: End date for the data analysis (e.g., 'YYYY-MM-DD').
      n_shards: Number of shards to split the processing into.
      un_pop_male_path: Path to the UN population data file for males.
      un_pop_female_path: Path to the UN population data file for females.
      min_days_per_user: Minimum total number of recorded wear days required per
        user across the full study window (the paper's precomputed grids use
        min_days_per_user=1).
      min_age: Minimum age boundary (inclusive).
      max_age: Maximum age boundary (inclusive).
      exclude_outliers: Whether to exclude outliers (1.5 * IQR, where IQR bounds
        are computed per shard when enabled).
      exclude_extreme_values: Whether to exclude extreme values (3.0 * IQR,
        where IQR bounds are computed per shard when enabled).
      min_steps: Minimum daily steps threshold (inclusive).
      max_steps: Maximum daily steps threshold (inclusive).
      exclude_days_of_week: Days of week to exclude (0=Monday, ..., 6=Sunday).
      target_country: Specific country to filter the analysis to, if any.
      start_month: Start month for seasonal filtering (inclusive).
      stop_month: Stop month for seasonal filtering (inclusive).
      override_hash: Optional hash to override the automatically generated
        params hash for checkpoints.

    Returns:
      An initialized StratifySteps instance.
    """
    instance = cls(
        path_base=path_base,
        start_date=start_date,
        end_date=end_date,
        n_shards=n_shards,
        un_pop_male_path=un_pop_male_path,
        un_pop_female_path=un_pop_female_path,
        min_days_per_user=min_days_per_user,
        min_age=min_age,
        max_age=max_age,
        exclude_outliers=exclude_outliers,
        exclude_extreme_values=exclude_extreme_values,
        min_steps=min_steps,
        max_steps=max_steps,
        exclude_days_of_week=exclude_days_of_week,
        target_country=target_country,
        start_month=start_month,
        stop_month=stop_month,
        override_hash=override_hash,
    )

    is_completed = False
    if file_exists(instance.checkpoint_completed_file):
      is_completed = load_pickle(instance.checkpoint_completed_file)

    if is_completed:
      logging.info(
          'Process fully completed previously. Loading foundational grids from'
          ' checkpoint...'
      )
      if file_exists(instance.checkpoint_state_file):
        state = load_pickle(instance.checkpoint_state_file)
        instance.results_grid = state['results_grid']
        instance.processed_shards = set(range(instance.n_shards))
        return instance
      instance.results_grid = load_pickle(instance.checkpoint_grid_file)
      instance.processed_shards = set(range(instance.n_shards))
      return instance

    # Load existing checkpoints if available
    if file_exists(instance.checkpoint_state_file):
      logging.info('Loading state from combined checkpoint...')
      state = load_pickle(instance.checkpoint_state_file)
      instance.results_grid = state['results_grid']
      instance.processed_shards = state['processed_shards']
      return instance

    if file_exists(instance.checkpoint_grid_file) and file_exists(
        instance.checkpoint_processed_file
    ):
      logging.info('Loading grids and processed shards from checkpoint...')
      instance.results_grid = load_pickle(instance.checkpoint_grid_file)
      instance.processed_shards = load_pickle(
          instance.checkpoint_processed_file
      )

    return instance

  @classmethod
  def from_dataframe(
      cls,
      df_steps: pd.DataFrame,
      path_base: str,
      start_date: str,
      end_date: str,
      un_pop_male_path: str = '',
      un_pop_female_path: str = '',
      min_days_per_user: int = 30,
      min_age: int = 20,
      max_age: int = 80,
      exclude_outliers: bool = False,
      exclude_extreme_values: bool = False,
      min_steps: int | None = None,
      max_steps: int | None = None,
      exclude_days_of_week: Sequence[int] | None = None,
      target_country: str | None = None,
      start_month: int | None = None,
      stop_month: int | None = None,
  ) -> 'StratifySteps':
    """Initializes directly from a provided steps DataFrame."""
    instance = cls(
        path_base=path_base,
        start_date=start_date,
        end_date=end_date,
        un_pop_male_path=un_pop_male_path,
        un_pop_female_path=un_pop_female_path,
        min_days_per_user=min_days_per_user,
        min_age=min_age,
        max_age=max_age,
        exclude_outliers=exclude_outliers,
        exclude_extreme_values=exclude_extreme_values,
        min_steps=min_steps,
        max_steps=max_steps,
        exclude_days_of_week=exclude_days_of_week,
        target_country=target_country,
        start_month=start_month,
        stop_month=stop_month,
    )

    df_steps_grid = instance.summarize_dataframe(df_steps)

    # Initialize results_grid with full Cartesian demographic grid
    empty_grid = pd.DataFrame(index=instance.get_grid())
    empty_grid['steps'] = 0.0
    empty_grid['steps_sq'] = 0.0
    empty_grid['readings'] = 0
    instance.results_grid = empty_grid.add(
        df_steps_grid.rename(
            columns={'sum': 'steps', 'sum_sq': 'steps_sq', 'count': 'readings'}
        ),
        fill_value=0,
    )
    instance.processed_shards = set(range(instance.n_shards))
    return instance

  def process_shards(self) -> None:
    """Processes all unprocessed shards and checkpoints the progress.

    To handle datasets larger than memory, this method incrementally aggregates
    daily readings into sufficient statistics (`sum`, `sum_sq`, `count`) in the
    `results_grid`. It uses atomic file writes (`.tmp` to `.pkl`) to guarantee
    crash-safe recovery without data corruption, ensuring interrupted processing
    can safely resume from the last shard.
    """
    if len(self.processed_shards) == self.n_shards:
      logging.info('All grids have already been processed.')
      return

    if self.demo is None:
      self.demo = self.process_demo()

    if self.results_grid is None:
      self.results_grid = pd.DataFrame(index=self.get_grid())
      self.results_grid['steps'] = 0.0
      self.results_grid['steps_sq'] = 0.0
      self.results_grid['readings'] = 0

    unprocessed_shards = [
        m for m in range(self.n_shards) if m not in self.processed_shards
    ]
    for modulo in unprocessed_shards:
      shard = self.process_shard(modulo)
      shard_renamed = shard.rename(
          columns={'sum': 'steps', 'sum_sq': 'steps_sq', 'count': 'readings'}
      )

      self.results_grid = self.results_grid.add(shard_renamed, fill_value=0)
      self.processed_shards.add(modulo)

      # Save atomic combined checkpoint state
      checkpoint_state = {
          'results_grid': self.results_grid,
          'processed_shards': self.processed_shards,
      }
      tmp_state_file = f'{self.checkpoint_state_file}.tmp'
      save_pickle(checkpoint_state, tmp_state_file)
      rename_file(tmp_state_file, self.checkpoint_state_file, overwrite=True)

      # Save checkpoints for backwards compatibility
      save_pickle(self.results_grid, self.checkpoint_grid_file)
      save_pickle(self.processed_shards, self.checkpoint_processed_file)

    if len(self.processed_shards) == self.n_shards:
      save_pickle(True, self.checkpoint_completed_file)
      logging.info('All shards have been successfully processed.')

  def get_grid(self) -> pd.MultiIndex:
    """Returns the MultiIndex containing demographic strata.

    The returned index represents a Cartesian product of four dimensions:
    Country x Calendar Year x Sex (self-reported at birth) x 5-Year Age Group
    (midage). This forms the backbone of all aggregated demographic grids
    throughout the pipeline.
    """
    if self.target_country is not None:
      countries = [self.target_country]
    else:
      if self.demo is None:
        self.demo = self.process_demo()
      demo = self.demo
      assert demo is not None
      countries = demo['country'].unique()
    start_year = int(str(self.start_date)[:4])
    end_year = int(str(self.end_date)[:4])
    years = np.arange(start_year, end_year + 1)
    genders = list(_GENDER_LABELS.keys())
    midage = self._get_5_year_midages()

    grid = pd.MultiIndex.from_product(
        [countries, years, genders, midage],
        names=['country', 'year', 'gender', 'midage'],
    )
    return grid

  def _get_5_year_midages(self) -> np.ndarray:
    """Creates a numpy array of mid-ages for 5-year age groups."""
    min_age = self.min_age // 5 * 5
    max_age = self.max_age // 5 * 5
    return np.arange(min_age, max_age + 1, 5) + 2.5

  def process_demo(self) -> pd.DataFrame:
    """Processes demographic data."""
    return process_demo(
        path_base=self.path_base,
        start_date=self.start_date,
        end_date=self.end_date,
        min_days_per_user=self.min_days_per_user,
    )

  def load_un_population_data(self) -> pd.DataFrame:
    """Loads UN population data."""
    return load_un_population_data(
        un_pop_male_path=self.un_pop_male_path,
        un_pop_female_path=self.un_pop_female_path,
    )

  def summarize_dataframe(self, df_steps: pd.DataFrame) -> pd.DataFrame:
    """Processes a steps DataFrame, returning stratum running statistics.

    Args:
      df_steps: DataFrame containing user steps with columns: user_id,
        activity_dt, and the daily step count column (`_STEP_COL`).

    Returns:
      A DataFrame indexed by stratum (country, year, gender, midage) with
      columns: sum, sum_sq, and count.
    """
    if self.demo is None:
      self.demo = self.process_demo()

    df_steps = apply_demo(
        df_steps,
        self.demo,
        min_age=self.min_age,
        max_age=self.max_age,
    )

    if self.start_date is not None:
      start_dt = pd.to_datetime(self.start_date)
      if df_steps['activity_dt'].dt.tz is not None and start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize(df_steps['activity_dt'].dt.tz)
      df_steps = df_steps[df_steps['activity_dt'] >= start_dt]

    if self.end_date is not None:
      end_dt = pd.to_datetime(self.end_date)
      if df_steps['activity_dt'].dt.tz is not None and end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize(df_steps['activity_dt'].dt.tz)
      if end_dt.hour == 0 and end_dt.minute == 0 and end_dt.second == 0:
        df_steps = df_steps[
            df_steps['activity_dt'].dt.floor('D') <= end_dt.floor('D')
        ]
      else:
        df_steps = df_steps[df_steps['activity_dt'] <= end_dt]

    if (
        self.exclude_outliers or self.exclude_extreme_values
    ) and not df_steps.empty:
      q1 = df_steps[_STEP_COL].quantile(0.25)
      q3 = df_steps[_STEP_COL].quantile(0.75)
      iqr = q3 - q1

      factor = 1.5 if self.exclude_outliers else 3.0
      lower_bound = q1 - factor * iqr
      upper_bound = q3 + factor * iqr
      df_steps = df_steps[
          (df_steps[_STEP_COL] >= lower_bound)
          & (df_steps[_STEP_COL] <= upper_bound)
      ]

    if self.min_steps is not None:
      df_steps = df_steps[df_steps[_STEP_COL] >= self.min_steps]
    if self.max_steps is not None:
      df_steps = df_steps[df_steps[_STEP_COL] <= self.max_steps]

    if self.exclude_days_of_week is not None:
      df_steps = df_steps[
          ~df_steps['activity_dt'].dt.dayofweek.isin(self.exclude_days_of_week)
      ]

    if self.target_country is not None:
      df_steps = df_steps[df_steps['country'] == self.target_country]

    if self.start_month is not None or self.stop_month is not None:
      months = df_steps['activity_dt'].dt.month
      if self.start_month is not None and self.stop_month is not None:
        if self.start_month <= self.stop_month:
          mask = (months >= self.start_month) & (months <= self.stop_month)
        else:
          mask = (months >= self.start_month) | (months <= self.stop_month)
      elif self.start_month is not None:
        mask = months >= self.start_month
      else:
        mask = months <= self.stop_month
      df_steps = df_steps[mask]

    df_steps_avg = (
        df_steps.groupby(['user_id', 'year', 'gender', 'age', 'country'])[
            _STEP_COL
        ]
        .mean()
        .reset_index()
    )

    df_steps_avg['steps_sq'] = df_steps_avg[_STEP_COL] ** 2
    df_steps_avg['midage'] = df_steps_avg['age'] // 5 * 5 + 2.5

    df_steps_grid = df_steps_avg.groupby(
        ['country', 'year', 'gender', 'midage']
    ).agg(
        sum=(_STEP_COL, 'sum'),
        sum_sq=('steps_sq', 'sum'),
        count=(_STEP_COL, 'count'),
    )

    return df_steps_grid

  def process_shard(self, modulo: int) -> pd.DataFrame:
    """Processes a single step shard into stratum summary statistics.

    Args:
      modulo: Shard index modulo to load and process.

    Returns:
      A DataFrame of aggregated step statistics for the shard.
    """
    df_steps = load_file(
        query_name='daily',
        path_base=f'{self.path_base}/steps',
        start_date=self.start_date,
        end_date=self.end_date,
        one_in_n=self.n_shards,
        modulo=modulo,
    )
    return self.summarize_dataframe(df_steps)

  def get_mean_grid(
      self,
      groupby_cols: Sequence[str] | None = None,
      impute_hierarchy: Sequence[str] | None = None,
      seed: int = 42,
      imputation_method: str = 'stochastic',
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes mean steps for each stratum and patches gaps via imputation.

    Because fine-grained strata (like 90-94 year-olds in smaller countries)
    may lack Fitbit users, their cell mean yields NaN. This method resolves
    missing values by climbing the `impute_hierarchy`. At each step, it fills
    holes by sampling from non-NaN donors at that broader hierarchical level
    ('stochastic') or using the median fallback ('median').

    Args:
      groupby_cols: Columns to group by.
      impute_hierarchy: Hierarchy of columns to impute.
      seed: Seed for the random number generator.
      imputation_method: Method for filling NaNs ('stochastic', 'median',
        'zero').

    Returns:
      A tuple of (mean_grid, nan_grid).
    """
    if impute_hierarchy is None:
      impute_hierarchy = ['country', 'midage', 'gender']

    if imputation_method not in ['stochastic', 'median', 'zero']:
      raise ValueError(
          f'Unknown imputation method: {imputation_method}. Choose from'
          ' stochastic, median, or zero.'
      )

    results_grid = self.results_grid
    if results_grid is None:
      raise ValueError('Results grid is empty. Run process_shards() first.')

    if groupby_cols is not None:
      df_grouped = results_grid.groupby(list(groupby_cols)).sum()
      df_step_sum_grid = df_grouped['steps']
      df_count_grid = df_grouped['readings']
      impute_hierarchy = [
          col for col in impute_hierarchy if col in groupby_cols
      ]
    else:
      df_step_sum_grid = results_grid['steps']
      df_count_grid = results_grid['readings']
    df_avg = df_step_sum_grid / df_count_grid
    grid = df_avg.to_frame('mean_steps').reset_index()

    nan_grid = df_avg.isna().to_frame('nan').reset_index()

    if imputation_method == 'zero':
      grid['mean_steps'] = grid['mean_steps'].fillna(0)
    else:
      rng = np.random.default_rng(seed)
      for i in range(len(impute_hierarchy)):
        if not grid['mean_steps'].isna().any():
          break
        current_level = list(impute_hierarchy[i:])

        if imputation_method == 'stochastic':
          grid['mean_steps'] = grid.groupby(current_level, observed=True)[
              'mean_steps'
          ].transform(self.stochastic_fill, rng=rng)
        elif imputation_method == 'median':
          grid['mean_steps'] = grid.groupby(current_level, observed=True)[
              'mean_steps'
          ].transform(lambda x: x.fillna(x.median()))
        else:
          raise ValueError(f'Unknown imputation method: {imputation_method}')

      if grid['mean_steps'].isna().any():
        if imputation_method == 'stochastic':
          grid['mean_steps'] = self.stochastic_fill(grid['mean_steps'], rng=rng)
        elif imputation_method == 'median':
          grid['mean_steps'] = grid['mean_steps'].fillna(
              grid['mean_steps'].median()
          )

    return grid, nan_grid

  def get_variance_grid(
      self,
      groupby_cols: Sequence[str] | None = None,
      ddof: int = 1,
  ) -> pd.DataFrame:
    """Computes the variance of steps for each strata.

    Args:
      groupby_cols: Columns to group by.
      ddof: Delta Degrees of Freedom (1 for sample variance, 0 for population
        variance).

    Returns:
      A dataframe with the variance of steps for each strata.
    """
    results_grid = self.results_grid
    if results_grid is None:
      raise ValueError('Results grid is empty. Run process_shards() first.')

    if groupby_cols is not None:
      df_grouped = results_grid.groupby(list(groupby_cols)).sum()
      sum_grid = df_grouped['steps']
      sum_sq_grid = df_grouped['steps_sq']
      count_grid = df_grouped['readings']
    else:
      sum_grid = results_grid['steps']
      sum_sq_grid = results_grid['steps_sq']
      count_grid = results_grid['readings']

    valid_mask = count_grid > ddof
    variance = pd.Series(np.nan, index=count_grid.index, dtype=float)
    if valid_mask.any():
      safe_count = count_grid[valid_mask]
      safe_sum = sum_grid[valid_mask]
      safe_sum_sq = sum_sq_grid[valid_mask]
      var_values = (safe_sum_sq - (safe_sum**2) / safe_count) / (
          safe_count - ddof
      )
      variance[valid_mask] = var_values.clip(lower=0.0)

    return variance.to_frame('variance_steps').reset_index()

  def stochastic_fill(
      self, series: pd.Series, rng: np.random.Generator
  ) -> pd.Series:
    """Fills missing values by sampling from non-NaN peers in the group.

    Args:
      series: Series containing missing values to impute.
      rng: Random number generator used for stochastic donor selection.

    Returns:
      A copy of the series with NaN values populated from donor samples.
    """
    mask = series.isna()
    if not mask.any():
      return series

    donors = series[~mask]
    if donors.empty:
      return series
    result = series.copy()
    result[mask] = rng.choice(donors.to_numpy(), size=mask.sum(), replace=True)
    return result

  def get_reweighted_grid(
      self,
      groupby_cols: Sequence[str] | None = None,
      impute_hierarchy: Sequence[str] | None = None,
      seed: int = 42,
      imputation_method: str = 'stochastic',
  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies UN WPP post-stratification weighting adjustments.

    Fitbit users skew demographics. By taking the derived sample means (x_bar_h)
    and multiplying them by true population proportions from UN Census tables
    (N_h / N_total), this projects the cohort's step behavior onto the exact
    age-sex demographic pyramid of the target country.

    Formula: Representative National Mean = Sum_h [ (N_h / N_total) * x_bar_h ]

    Args:
      groupby_cols: Columns to group by.
      impute_hierarchy: Imputation column hierarchy.
      seed: Seed for stochastic imputation.
      imputation_method: Imputation method ('stochastic', 'median', 'zero').

    Returns:
      A tuple of (weighted_estimates, nan_summary).
    """
    if impute_hierarchy is None:
      impute_hierarchy = ['country', 'midage', 'gender']

    grid_strat, nan_grid = self.get_mean_grid(
        groupby_cols=None,
        impute_hierarchy=impute_hierarchy,
        seed=seed,
        imputation_method=imputation_method,
    )
    if self.un_pop is None:
      self.un_pop = self.load_un_population_data()
    weight_grid = grid_strat.merge(
        self.un_pop[['country', 'year', 'gender', 'age', 'Count']],
        left_on=['country', 'year', 'gender', 'midage'],
        right_on=['country', 'year', 'gender', 'age'],
        how='inner',
    )

    weight_grid = weight_grid.copy()
    if groupby_cols:
      total_counts = weight_grid.groupby(list(groupby_cols))['Count'].transform(
          'sum'
      )
    else:
      total_counts = weight_grid['Count'].sum()

    weight_grid['proportion'] = weight_grid['Count'] / total_counts
    weight_grid['mean_steps'] = (
        weight_grid['mean_steps'] * weight_grid['proportion']
    )

    if groupby_cols:
      weighted_estimates = (
          weight_grid.groupby(list(groupby_cols))[['mean_steps']]
          .sum()
          .reset_index()
      )
      nan_summary = (
          nan_grid.groupby(list(groupby_cols))[['nan']].any().reset_index()
      )
    else:
      weighted_estimates = pd.DataFrame(
          {'mean_steps': [weight_grid['mean_steps'].sum()]}
      )
      nan_summary = pd.DataFrame({'nan': [nan_grid['nan'].any()]})

    return weighted_estimates, nan_summary


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


if __name__ == '__main__':
  app.run(main)
