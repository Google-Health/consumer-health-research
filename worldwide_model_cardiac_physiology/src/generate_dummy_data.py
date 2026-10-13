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

r"""Writes small synthetic cohorts in the public CSV schema (``data/dummy_data*.csv``).

The files exist only to exercise loading and the end-to-end notebooks
(01 -> 02 -> 03) without access to the Fitbit cohort; the numbers carry a
crude seasonal / age / sex structure so that model fits do not degenerate, but
they are NOT meant for any analysis.

Usage::

    python generate_dummy_data.py --out_dir=data --n_users=300
"""

import argparse
from collections.abc import Sequence
import os
from typing import Optional

import numpy as np
import pandas as pd

import cohort_io as cio
import file_io as fio
import seasonality_lib as sl

# ISO-2 -> (latitude, longitude, UN region) of a handful of countries spanning
# both hemispheres and most latitude bands.
COUNTRY_CENTROIDS = {
    'US': (38.0, -97.0, 'americas'),
    'CA': (56.0, -106.0, 'americas'),
    'MX': (23.0, -102.0, 'americas'),
    'BR': (-10.0, -55.0, 'americas'),
    'AR': (-34.0, -64.0, 'americas'),
    'GB': (54.0, -2.0, 'europe'),
    'DE': (51.0, 10.0, 'europe'),
    'FR': (46.0, 2.0, 'europe'),
    'ES': (40.0, -4.0, 'europe'),
    'IT': (42.0, 12.0, 'europe'),
    'SE': (62.0, 15.0, 'europe'),
    'NO': (61.0, 8.0, 'europe'),
    'IN': (21.0, 78.0, 'asia'),
    'JP': (36.0, 138.0, 'asia'),
    'KR': (36.0, 128.0, 'asia'),
    'SG': (1.0, 103.0, 'asia'),
    'ZA': (-29.0, 24.0, 'africa'),
    'AU': (-25.0, 134.0, 'oceania'),
    'NZ': (-41.0, 174.0, 'oceania'),
}
# Column order of the written CSVs (the metric column and ``region`` follow).
PUBLIC_HEADER_ORDER = [
    'id',
    'month',
    'age',
    'sex',
    'country',
    'latitude',
    'longitude',
    'temperature',
    'temperature min',
    'temperature max',
    'day light',
]
_MONTHS = pd.date_range('2023-01-01', periods=12, freq='MS')


def _make_dummy_users(n_users: int, seed: int = 0) -> pd.DataFrame:
  """Returns one row per synthetic user: demographics, location, user effect.

  Args:
    n_users: Number of users.
    seed: Random seed.

  Returns:
    Frame with the demographic columns (code convention) plus the hidden helper
    columns ``_lat``, ``_male`` and ``_u`` used by ``make_dummy_cohort``.
  """
  rng = np.random.default_rng(seed)
  codes = list(COUNTRY_CENTROIDS)
  country = rng.choice(codes, size=n_users, p=_country_weights(codes))
  lat0 = np.array([COUNTRY_CENTROIDS[c][0] for c in country])
  lng0 = np.array([COUNTRY_CENTROIDS[c][1] for c in country])
  region = np.array([COUNTRY_CENTROIDS[c][2] for c in country])
  lat = np.clip(lat0 + rng.normal(0, 3.0, n_users), -60, 70)
  lng = ((lng0 + rng.normal(0, 4.0, n_users) + 180) % 360) - 180
  age = rng.integers(18, 80, n_users)
  male = rng.random(n_users) < 0.5
  user_effect = rng.normal(0, 1.0, n_users)
  return pd.DataFrame({
      'user_id': np.arange(1, n_users + 1),
      'gender': np.where(male, 'Male', 'Female'),
      'age': age,
      'lat_e7': np.round(lat * 1e7).astype('int64'),
      'lng_e7': np.round(lng * 1e7).astype('int64'),
      'country_2_alpha_cd': country,
      'un_region': region,
      '_lat': lat,
      '_male': male.astype(float),
      '_u': user_effect,
  })


def make_dummy_cohort(
    metric: str,
    n_users: int,
    seed: int = 0,
    drop_frac: float = 0.15,
    users: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
  """Returns a synthetic user-month frame in the code's column convention.

  Args:
    metric: 'rhr' | 'hrv'.
    n_users: Number of users (ignored when ``users`` is given).
    seed: Random seed.
    drop_frac: Fraction of user-months removed at random (irregular follow-up).
    users: Optional user table from ``_make_dummy_users``; drawn with ``seed``
      when omitted.

  Returns:
    Frame with the columns (code convention) expected by the notebooks.
  """
  if users is None:
    users = _make_dummy_users(n_users, seed)
  rng = np.random.default_rng([seed, 0 if metric == 'rhr' else 1])
  df = users.merge(
      pd.DataFrame({'activity_period': _MONTHS, '_m': np.arange(12)}),
      how='cross',
  )
  m = df['_m'].to_numpy()
  lat_arr = df['_lat'].to_numpy()
  hemisphere = np.where(lat_arr >= 0, 1.0, -1.0)
  season = np.cos(2 * np.pi * (m - 6) / 12.0) * hemisphere  # +1 in local summer
  df['dl_monthly'] = sl.mid_month_day_length_hours(lat_arr, m + 1)
  temp = (
      27.0
      - 0.45 * np.abs(lat_arr)
      + 12.0 * season
      + rng.normal(0, 1.5, len(df))
  )
  df['avg_temp_mean'] = temp
  df['avg_temp_min'] = temp - 4.0 - np.abs(rng.normal(0, 1.0, len(df)))
  df['avg_temp_max'] = temp + 4.0 + np.abs(rng.normal(0, 1.0, len(df)))
  age_c = df['age'].to_numpy() - 45.0
  if metric == 'rhr':
    value = (
        63.0
        + 0.08 * age_c
        - 3.0 * df['_male'].to_numpy()
        - 1.2 * season
        + 5.0 * df['_u'].to_numpy()
        + rng.normal(0, 2.0, len(df))
    )
    df['avg_resting_heart_rate_bpm'] = np.clip(value, 35, 120)
  elif metric == 'hrv':
    log_value = (
        np.log(42.0)
        - 0.012 * age_c
        + 0.08 * df['_male'].to_numpy()
        + 0.03 * season
        + 0.30 * df['_u'].to_numpy()
        + rng.normal(0, 0.08, len(df))
    )
    df['avg_hrv'] = np.clip(np.exp(log_value), 6, 240)
  else:
    raise ValueError(f'metric must be rhr or hrv, got {metric!r}')
  keep = rng.random(len(df)) >= drop_frac
  df = df[keep].drop(columns=['_lat', '_male', '_u', '_m'])
  return df.reset_index(drop=True)


def _country_weights(codes: Sequence[str]) -> np.ndarray:
  weights = np.array([6.0 if c == 'US' else 1.0 for c in codes])
  return weights / weights.sum()


def write_dummy_data(
    out_dir: str, n_users: int = 300, seed: int = 0
) -> dict[str, str]:
  """Writes ``dummy_data.csv`` (RHR) and ``dummy_data_hrv.csv`` to ``out_dir``.

  The HRV cohort is a random ~2/3 subset of the RHR users (same demographics
  for the same id), as fewer users have HRV.

  Args:
    out_dir: Destination directory (created if needed).
    n_users: RHR users; HRV gets ``2 * n_users // 3``.
    seed: Random seed.

  Returns:
    Mapping metric -> written path.
  """
  users = _make_dummy_users(n_users, seed)
  hrv_users = users.sample(2 * n_users // 3, random_state=seed).sort_values(
      'user_id'
  )
  written = {}
  for metric, table in (('rhr', users), ('hrv', hrv_users)):
    df = cio.internal_to_public(
        make_dummy_cohort(metric, len(table), seed, users=table)
    )
    target = cio.INTERNAL_TO_PUBLIC[cio.METRIC_TARGET_COLS[metric]]
    df = df[PUBLIC_HEADER_ORDER + [target, 'region']]
    path = os.path.join(out_dir, cio.PUBLIC_CSV_NAMES[metric])
    with fio.open_file(path, 'w') as f:
      df.to_csv(f, index=False, float_format='%.4f')
    written[metric] = path
    print(
        f'{metric}: {len(df):,} rows / {df["id"].nunique():,} users -> {path}'
    )
  return written


def main(argv: Optional[Sequence[str]] = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
  parser.add_argument(
      '--out_dir',
      default=None,
      help='Directory receiving the CSVs (default: <repo>/data).',
  )
  parser.add_argument(
      '--n_users', type=int, default=300, help='Users in the RHR cohort.'
  )
  parser.add_argument('--seed', type=int, default=0, help='Random seed.')
  args = parser.parse_args(argv)
  out_dir = args.out_dir or os.path.join(
      os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'
  )
  write_dummy_data(out_dir, args.n_users, args.seed)


if __name__ == '__main__':
  main()
