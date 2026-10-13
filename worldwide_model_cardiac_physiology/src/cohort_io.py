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

"""Cohort loader: CSV files in the public input schema, one interface.

The modelling and figure code works on a fixed set of column names for the
clean user-month frames (``user_id, activity_period, gender, age, lat_e7,
lng_e7, country_2_alpha_cd, un_region, dl_monthly, avg_temp_mean/min/max,
avg_resting_heart_rate_bpm | avg_hrv``).

Cohort inputs are CSV files with human-readable headers (``PUBLIC_COLUMNS``):
``<data_dir>/dummy_data.csv`` (RHR) and ``<data_dir>/dummy_data_hrv.csv``
(HRV). ``load_cohort`` reads them and converts the public columns and units to
the names used by the code so that every downstream step runs unchanged.

Public CSV schema (one row per user and calendar month)::

    id                      user identifier (any hashable)
    month                   calendar month, e.g. 2023-01 (parsed by pandas)
    age                     age in years
    sex                     'Male' | 'Female' (also accepts M/F, male/female)
    country                 ISO-3166 alpha-2 country code
    latitude, longitude     decimal degrees
    temperature             monthly mean air temperature, degrees C
    temperature min         monthly mean of the daily minimum, degrees C
    temperature max         monthly mean of the daily maximum, degrees C
    day light               mean day length of the month, hours
    resting heart rate      bpm  (RHR file)   |   heart rate variability  ms
    (HRV)
    region                  optional UN region used for grouped validation plots
"""

import os
from typing import Mapping, Optional, Sequence

import pandas as pd

import file_io as fio

# Public CSV header -> column name used by the code.
PUBLIC_COLUMNS: Mapping[str, str] = {
    'id': 'user_id',
    'month': 'activity_period',
    'age': 'age',
    'sex': 'gender',
    'country': 'country_2_alpha_cd',
    'region': 'un_region',
    'latitude': 'lat_e7',
    'longitude': 'lng_e7',
    'temperature': 'avg_temp_mean',
    'temperature min': 'avg_temp_min',
    'temperature max': 'avg_temp_max',
    'day light': 'dl_monthly',
    'resting heart rate': 'avg_resting_heart_rate_bpm',
    'heart rate variability': 'avg_hrv',
}
INTERNAL_TO_PUBLIC: Mapping[str, str] = {
    v: k for k, v in PUBLIC_COLUMNS.items()
}

# Columns that are optional in a public CSV and the value used when absent.
OPTIONAL_DEFAULTS: Mapping[str, str] = {'un_region': 'unknown'}

METRIC_TARGET_COLS: Mapping[str, str] = {
    'rhr': 'avg_resting_heart_rate_bpm',
    'hrv': 'avg_hrv',
}
PUBLIC_CSV_NAMES: Mapping[str, str] = {
    'rhr': 'dummy_data.csv',
    'hrv': 'dummy_data_hrv.csv',
}

_SEX_VALUES: Mapping[str, str] = {
    'male': 'Male',
    'm': 'Male',
    'female': 'Female',
    'f': 'Female',
}
_DEGREES_TO_E7 = 1e7
# pandas' default missing-value tokens (``pd.read_csv`` documentation).
_NA_TOKENS = (
    '',
    '#N/A',
    '#N/A N/A',
    '#NA',
    '-1.#IND',
    '-1.#QNAN',
    '-NaN',
    '-nan',
    '1.#IND',
    '1.#QNAN',
    '<NA>',
    'N/A',
    'NA',
    'NULL',
    'NaN',
    'None',
    'n/a',
    'nan',
    'null',
)


def _check_metric(metric: str) -> None:
  if metric not in PUBLIC_CSV_NAMES:
    raise ValueError(
        f'metric must be one of {sorted(PUBLIC_CSV_NAMES)}, got {metric!r}'
    )


# --- Paths -------------------------------------------------------------------


def public_csv_path(data_dir: str, metric: str) -> str:
  """Returns ``<data_dir>/dummy_data[_hrv].csv``."""
  _check_metric(metric)
  return os.path.join(data_dir, PUBLIC_CSV_NAMES[metric])




def resolve_input_path(data_dir: str, metric: str) -> str:
  """Returns the cohort file to read for ``metric``.

  Args:
    data_dir: Data root.
    metric: 'rhr' | 'hrv'.

  Returns:
    Path of the first existing candidate.

  Raises:
    FileNotFoundError: no cohort file exists under ``data_dir``.
  """
  candidates = [public_csv_path(data_dir, metric)]
  for path in candidates:
    if fio.file_exists(path):
      return path
  raise FileNotFoundError(
      f'No {metric} cohort under {data_dir!r}; looked for '
      + ' and '.join(repr(p) for p in candidates)
  )


# --- Public schema <-> code column names -------------------------------------


def _bad_rows(mask: pd.Series, limit: int = 10) -> str:
  rows = list(mask[mask].index[:limit])
  more = int(mask.sum()) - len(rows)
  return f'rows {rows}' + (f' and {more} more' if more > 0 else '')


def public_to_internal(df: pd.DataFrame) -> pd.DataFrame:
  """Converts a public-schema frame to the column names and units of the code.

  Header matching is case-insensitive and ignores surrounding blanks; unknown
  columns are kept untouched. Latitude/longitude are scaled to 1e-7 degrees
  (``lat_e7``/``lng_e7``), ``month`` is parsed to a timestamp, ``sex`` is
  normalised to 'Male'/'Female' and missing optional columns receive their
  default (``un_region='unknown'``).

  Args:
    df: Frame read from a public CSV.

  Returns:
    New frame with the code's column names.

  Raises:
    ValueError: a latitude/longitude value is missing or not numeric, or a sex
      label is not one of Male/Female/M/F (case-insensitive).
  """
  rename = {}
  for col in df.columns:
    key = str(col).strip().lower()
    if key in PUBLIC_COLUMNS:
      rename[col] = PUBLIC_COLUMNS[key]
  out = df.rename(columns=rename).copy()
  if 'activity_period' in out:
    out['activity_period'] = pd.to_datetime(out['activity_period'])
  for col in ('lat_e7', 'lng_e7'):
    if col in out:
      degrees = pd.to_numeric(out[col], errors='coerce')
      bad = degrees.isna()
      if bad.any():
        raise ValueError(
            f'{INTERNAL_TO_PUBLIC[col]!r} is missing or not numeric in'
            f' {_bad_rows(bad)}'
        )
      out[col] = (degrees * _DEGREES_TO_E7).round().astype('int64')
  if 'gender' in out:
    sex = out['gender'].astype(str).str.strip().str.lower().map(_SEX_VALUES)
    bad = sex.isna()
    if bad.any():
      raise ValueError(
          "'sex' must be Male/Female (or M/F); found"
          f' {sorted(set(out["gender"][bad].astype(str)))} in {_bad_rows(bad)}'
      )
    out['gender'] = sex
  for col, default in OPTIONAL_DEFAULTS.items():
    if col not in out:
      out[col] = default
  return out


def internal_to_public(df: pd.DataFrame) -> pd.DataFrame:
  """Inverse of ``public_to_internal`` (used to write dummy / example CSVs)."""
  out = df.copy()
  for col in ('lat_e7', 'lng_e7'):
    if col in out:
      out[col] = out[col] / _DEGREES_TO_E7
  if 'activity_period' in out:
    out['activity_period'] = pd.to_datetime(out['activity_period']).dt.strftime(
        '%Y-%m'
    )
  return out.rename(
      columns={
          c: INTERNAL_TO_PUBLIC[c]
          for c in out.columns
          if c in INTERNAL_TO_PUBLIC
      }
  )


# --- Loading -----------------------------------------------------------------


def read_public_csv(
    path: str, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
  """Reads a public CSV and returns it with the code's column names.

  Args:
    path: CSV path (read through ``file_io``).
    columns: Optional column names (code convention) to keep; missing optional
      columns are filled with their default, any other missing column raises.

  Returns:
    Frame in the code's column convention.

  Raises:
    KeyError: a requested non-optional column is absent from the file.
  """
  with fio.open_file(path, 'r') as f:
    header = list(pd.read_csv(f, nrows=0).columns)
    f.seek(0)
    # Only an empty country field is missing: pandas' default tokens would
    # turn Namibia's 'NA' into NaN. Other columns keep the default tokens.
    na_values = {}
    for col in header:
      norm_col = str(col).strip().lower()
      is_country = (
          PUBLIC_COLUMNS.get(norm_col, norm_col) == 'country_2_alpha_cd'
      )
      na_values[col] = [''] if is_country else list(_NA_TOKENS)
    df = public_to_internal(
        pd.read_csv(f, keep_default_na=False, na_values=na_values)
    )
  if columns is not None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
      public = [INTERNAL_TO_PUBLIC.get(c, c) for c in missing]
      raise KeyError(f'{path} lacks the column(s) {public}')
    df = df[list(columns)]
  return df




def load_cohort(
    data_dir: str,
    metric: str,
    columns: Optional[Sequence[str]] = None,
    path: Optional[str] = None,
) -> pd.DataFrame:
  """Loads the clean user-month cohort of ``metric``.

  Args:
    data_dir: Data root (``file_io.get_data_dir()``).
    metric: 'rhr' | 'hrv'.
    columns: Optional column names (code convention) to read / keep.
    path: Explicit file to read instead of ``resolve_input_path``.

  Returns:
    The cohort frame in the code's column convention.
  """
  _check_metric(metric)
  path = path or resolve_input_path(data_dir, metric)
  return read_public_csv(path, columns)
