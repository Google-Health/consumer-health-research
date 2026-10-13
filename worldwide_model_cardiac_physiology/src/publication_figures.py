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

"""Publication-quality figures for the seasonal RHR and HRV models.

Every figure is produced from the seasonal mixed-effects models of resting
heart rate (RHR) and heart rate variability (HRV) and the project's data
layout (see `file_io.py`):

  clean user-month cohort frames per metric       via `cohort_io.load_cohort`
  data_dir/results/{run_tag}/                     02_modelling.ipynb outputs
      lme-models-results-{metric}.csv             all candidate models
      model_{metric}_coefficients_interpreted.csv REML winner coefficients
      {timestamp}_{model}/model_coefficients.csv  per-model fixed effects
      {timestamp}_{model}/random_effects.parquet  per-model BLUPs (winner)
  extra_dir/                                      reference tables
      ne_110_admin_0_countries.geojson, unsd-countries.csv

Sections:
  1-2    Nature style, palettes, latitude bands, user registry.
  3-4    Demographic distributions figure, thinning-validation figures.
  5-9    Shared I/O, model constants, artefact loaders, prediction helpers
         (feature engineering itself lives in seasonality_lib).
  10-16  Model-based figures: lat-bin / country seasonal fits, within-user
         sensitivity, world contour residuals, joint demographics calibration,
         residual and random-intercept diagnostics.
  17-19  World maps: observed seasonal 2x2 maps (+ their validation-CSV
         producer).

Tables are produced by `paper_tables.py`; demographics tables use
`seasonality_lib.get_user_summary` + `paper_tables.demographics_table`.

All figures are saved as 300-DPI PNG plus editable SVG/PDF via `_save_fig`.
"""

import gc
import io
import os
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
import scipy.interpolate
import scipy.ndimage
import seaborn as sns
import statsmodels.formula.api as smf

import cohort_io as cio
import file_io as fio
import seasonality_lib as sl

try:
  import geopandas as gpd  # pylint: disable=g-import-not-at-top
except ImportError:
  gpd = None


# --- 1. Publication Style Setup ---


def setup_nature_style() -> None:
  """Configures Matplotlib rcParams to comply with Nature / npj guidelines."""
  plt.rc('font', family='sans-serif', size=8.0)
  plt.rcParams['font.sans-serif'] = [
      'Arial',
      'Helvetica',
      'DejaVu Sans',
      'Liberation Sans',
  ]
  plt.rcParams['axes.labelsize'] = 8.0
  plt.rcParams['axes.titlesize'] = 8.5
  plt.rcParams['xtick.labelsize'] = 7.0
  plt.rcParams['ytick.labelsize'] = 7.0
  plt.rcParams['legend.fontsize'] = 7.0
  plt.rcParams['legend.title_fontsize'] = 7.5
  plt.rcParams['figure.titlesize'] = 9.5
  plt.rcParams['figure.dpi'] = 300
  plt.rcParams['savefig.dpi'] = 300
  plt.rcParams['savefig.bbox'] = 'tight'
  plt.rcParams['savefig.pad_inches'] = 0.05
  plt.rcParams['axes.spines.top'] = False
  plt.rcParams['axes.spines.right'] = False
  plt.rcParams['axes.linewidth'] = 0.7
  plt.rcParams['xtick.major.width'] = 0.7
  plt.rcParams['ytick.major.width'] = 0.7
  plt.rcParams['xtick.minor.width'] = 0.5
  plt.rcParams['ytick.minor.width'] = 0.5
  plt.rcParams['lines.linewidth'] = 1.2
  plt.rcParams['lines.markersize'] = 3.5
  plt.rcParams['svg.fonttype'] = (
      'none'  # Keep text as editable vector glyphs in SVGs
  )
  plt.rcParams['pdf.fonttype'] = 42
  plt.rcParams['ps.fonttype'] = 42


# Colorblind-safe palette (Wong / Okabe-Ito inspired, avoiding red-green
# collision)
BAND_COLORS = {
    'Overall': '#111111',  # Black dashed
    'Mid South': '#2c7bb6',  # Deep steel blue
    'S. Tropics': '#80cdc1',  # Muted aqua / teal
    'N. Tropics': '#e66101',  # Warm amber / terracotta
    'Mid North': '#b2abd2',  # Lavender purple
    'Far North': '#5e3c99',  # Deep purple
}

GENDER_COLORS = {
    'Female': '#d95f02',  # Burnt Orange
    'Male': '#2c7bb6',  # Cool Blue
}

LATITUDE_BANDS = [
    ('Mid South', -60, -30),
    ('S. Tropics', -30, 0),
    ('N. Tropics', 0, 30),
    ('Mid North', 30, 60),
    ('Far North', 60, 90),
]


# --- 2. Helper Functions for Latitude Mapping & Statistics ---


def assign_latitude_band(lat_deg: pd.Series) -> pd.Series:
  """Maps latitude in degrees [-90, 90] to the 5 standard 30-degree study bands."""
  bins = [-90, -60, -30, 0, 30, 60, 90]
  labels = [
      'South Polar',
      'Mid South',
      'S. Tropics',
      'N. Tropics',
      'Mid North',
      'Far North',
  ]
  categorized = pd.cut(lat_deg, bins=bins, labels=labels, right=False)
  return pd.Series(categorized, index=lat_deg.index).astype(str)


def build_user_registry(
    df: pd.DataFrame,
    val_col: str,
    lat_col: str = 'lat_e7',
    year_cohort: int = 2022,
) -> pd.DataFrame:
  """Builds user-level static summary and assigns single most common latitude band."""
  df_work = df.copy()
  if 'latitude' not in df_work.columns and lat_col in df_work.columns:
    df_work['latitude'] = df_work[lat_col] / 1e7

  df_work['lat_band'] = assign_latitude_band(df_work['latitude'])

  if 'age' not in df_work.columns and 'birth_year' in df_work.columns:
    df_work['age'] = year_cohort - df_work['birth_year']

  # Per-user lifetime aggregations
  user_modes = df_work.groupby('user_id')['lat_band'].agg(
      lambda s: s.mode()[0] if not s.empty else 'Unknown'
  )
  user_counts = df_work.groupby('user_id')['activity_period'].count()
  user_means = df_work.groupby('user_id')[val_col].mean()
  user_ages = df_work.groupby('user_id')['age'].first()
  user_genders = df_work.groupby('user_id')['gender'].first()

  user_registry = pd.DataFrame({
      'user_id': user_ages.index,
      'gender': user_genders.values,
      'age': user_ages.values,
      'primary_band': user_modes.values,
      'user_months': user_counts.values,
      'mean_val': user_means.values,
  })
  return user_registry


def _registry_or_build(
    registry: Optional[pd.DataFrame],
    df: pd.DataFrame,
    val_col: str,
    lat_col: str,
    year_cohort: int,
) -> pd.DataFrame:
  """Returns ``registry`` if given, else `build_user_registry(df, ...)`."""
  if registry is not None:
    return registry
  return build_user_registry(
      df, val_col=val_col, lat_col=lat_col, year_cohort=year_cohort
  )


# --- 3. Population Demographic Distributions (panels A - F) ---


def _plot_age_curves(
    ax: plt.Axes,
    user_df: pd.DataFrame,
    gender: str,
    y_label: str,
    y_lim: Optional[Tuple[float, float]] = None,
) -> None:
  """Draws mean user value vs age (18-80) per latitude band plus overall.

  Used for the age-trend panels (A-D) of the demographic distributions
  figure. Each latitude band is drawn as a line with a 95 % CI band (mean
  ± 1.96 SEM across users of the same age); bands with fewer than 50 users
  and ages with fewer than 10 users are skipped. The overall curve (black
  dashed) requires at least 20 users per age.

  Args:
    ax: Target axes.
    user_df: User registry from `build_user_registry` (columns ``gender``,
      ``age``, ``primary_band``, ``mean_val``).
    gender: ``'Female'`` or ``'Male'``; selects the users to plot.
    y_label: Y-axis label.
    y_lim: Optional (low, high) y-axis limits.
  """
  sub = user_df[
      (user_df['gender'] == gender)
      & (user_df['age'] >= 18)
      & (user_df['age'] <= 80)
  ]

  # 1. Plot individual latitude bands
  for band_name, _, _ in LATITUDE_BANDS:
    b_df = sub[sub['primary_band'] == band_name]
    if len(b_df) < 50:
      continue
    stats_by_age = b_df.groupby('age')['mean_val'].agg(['mean', 'sem', 'count'])
    stats_by_age = stats_by_age[stats_by_age['count'] >= 10]
    ages = stats_by_age.index.values
    means = stats_by_age['mean'].values
    sems = stats_by_age['sem'].values

    color = BAND_COLORS.get(band_name, '#555555')
    ax.plot(ages, means, label=band_name, color=color, linewidth=1.1, alpha=0.9)
    ax.fill_between(
        ages,
        means - 1.96 * sems,
        means + 1.96 * sems,
        color=color,
        alpha=0.15,
        edgecolor='none',
    )

  # 2. Plot Overall (black dashed)
  all_stats = sub.groupby('age')['mean_val'].agg(['mean', 'sem', 'count'])
  all_stats = all_stats[all_stats['count'] >= 20]
  ages_all = all_stats.index.values
  means_all = all_stats['mean'].values
  sems_all = all_stats['sem'].values
  ax.plot(
      ages_all,
      means_all,
      label='Overall',
      color=BAND_COLORS['Overall'],
      linestyle='--',
      linewidth=1.5,
      zorder=10,
  )
  ax.fill_between(
      ages_all,
      means_all - 1.96 * sems_all,
      means_all + 1.96 * sems_all,
      color=BAND_COLORS['Overall'],
      alpha=0.12,
      edgecolor='none',
      zorder=9,
  )

  ax.set_xlabel('Age (years)')
  ax.set_ylabel(y_label)
  ax.set_xlim(18, 80)
  if y_lim:
    ax.set_ylim(y_lim)
  ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc')


def _plot_density(
    ax: plt.Axes,
    user_df: pd.DataFrame,
    x_label: str,
    title: str,
    x_lim: Tuple[float, float],
) -> None:
  """Draws the KDE of user mean values per latitude band plus overall.

  Used for the density panels (E-F) of the demographic distributions figure.
  Gaussian KDEs are evaluated on 300 points over ``x_lim``; bands with fewer
  than 100 users are skipped.

  Args:
    ax: Target axes.
    user_df: User registry from `build_user_registry` (columns ``primary_band``,
      ``mean_val``).
    x_label: X-axis label.
    title: Panel title (left-aligned, bold).
    x_lim: (low, high) range of the x axis and of the KDE evaluation grid.
  """
  xs = np.linspace(x_lim[0], x_lim[1], 300)

  for band_name, _, _ in LATITUDE_BANDS:
    vals = user_df[user_df['primary_band'] == band_name]['mean_val'].dropna()
    if len(vals) < 100:
      continue
    kde = stats.gaussian_kde(vals)
    color = BAND_COLORS.get(band_name, '#555555')
    ax.plot(xs, kde(xs), label=band_name, color=color, linewidth=1.1, alpha=0.9)

  # Overall density
  all_vals = user_df['mean_val'].dropna()
  kde_all = stats.gaussian_kde(all_vals)
  ax.plot(
      xs,
      kde_all(xs),
      label='Overall',
      color=BAND_COLORS['Overall'],
      linestyle='--',
      linewidth=1.5,
      zorder=10,
  )

  ax.set_xlabel(x_label)
  ax.set_ylabel('Density')
  ax.set_xlim(x_lim)
  ax.set_title(title, loc='left', fontsize=8.5, fontweight='bold')
  ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
  ax.legend(
      title='Latitude Band',
      loc='upper right',
      frameon=False,
      prop={'size': 6.5},
  )


def generate_figure_1(
    df_rhr: pd.DataFrame,
    df_hrv: pd.DataFrame,
    output_dir: Optional[str] = None,
    file_prefix: str = 'figure_1',
    rhr_col: str = 'avg_resting_heart_rate_bpm',
    hrv_col: str = 'avg_hrv',
    lat_col: str = 'lat_e7',
    year_cohort: int = 2022,
    rhr_registry: Optional[pd.DataFrame] = None,
    hrv_registry: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, Optional[str], Optional[str]]:
  """Generates the demographic distributions figure (panels A-F) at A4 width.

  Panels:
    (A) Women RHR vs Age (95% CI shaded)
    (B) Men RHR vs Age (95% CI shaded)
    (C) Women HRV vs Age (95% CI shaded)
    (D) Men HRV vs Age (95% CI shaded)
    (E) RHR Population Density Distribution by Latitude Band
    (F) HRV Population Density Distribution by Latitude Band

  Args:
    df_rhr: RHR user-month frame.
    df_hrv: HRV user-month frame.
    output_dir: Directory for png/svg/pdf; None = do not save.
    file_prefix: Output file stem.
    rhr_col: RHR value column.
    hrv_col: HRV value column.
    lat_col: Latitude (1e-7 degrees) column.
    year_cohort: Reference year used to derive age from birth_year if needed.
    rhr_registry: Optional pre-built `build_user_registry(df_rhr, ...)`; saves
      one pass over the frame when several figures share it.
    hrv_registry: Same for the HRV cohort.

  Returns:
    (figure, png path, svg path).
  """
  setup_nature_style()

  # A4 width = 210mm (~8.27 inches), printable width ~7.2 - 8.0 inches
  fig = plt.figure(figsize=(7.4, 9.6))
  gs = GridSpec(3, 2, figure=fig, hspace=0.32, wspace=0.25)

  ax_a = fig.add_subplot(gs[0, 0])
  ax_b = fig.add_subplot(gs[0, 1])
  ax_c = fig.add_subplot(gs[1, 0])
  ax_d = fig.add_subplot(gs[1, 1])
  ax_e = fig.add_subplot(gs[2, 0])
  ax_f = fig.add_subplot(gs[2, 1])

  # Prepare user-level data
  rhr_user = _registry_or_build(
      rhr_registry, df_rhr, rhr_col, lat_col, year_cohort
  )
  hrv_user = _registry_or_build(
      hrv_registry, df_hrv, hrv_col, lat_col, year_cohort
  )

  # Row 1: RHR Age curves
  _plot_age_curves(
      ax_a, rhr_user, 'Female', 'Resting Heart Rate (bpm)', y_lim=(61, 75)
  )
  ax_a.set_title(
      'A. Women: Mean Monthly RHR by Age and Latitude Band',
      loc='left',
      fontsize=8.5,
      fontweight='bold',
  )
  ax_a.legend(
      title='Latitude Band',
      loc='upper right',
      frameon=False,
      prop={'size': 6.5},
  )

  _plot_age_curves(
      ax_b, rhr_user, 'Male', 'Resting Heart Rate (bpm)', y_lim=(57, 73)
  )
  ax_b.set_title(
      'B. Men: Mean Monthly RHR by Age and Latitude Band',
      loc='left',
      fontsize=8.5,
      fontweight='bold',
  )
  ax_b.legend(
      title='Latitude Band',
      loc='upper right',
      frameon=False,
      prop={'size': 6.5},
  )

  # Row 2: HRV Age curves
  _plot_age_curves(
      ax_c, hrv_user, 'Female', 'Heart Rate Variability (ms)', y_lim=(24, 66)
  )
  ax_c.set_title(
      'C. Women: Mean Monthly HRV by Age and Latitude Band',
      loc='left',
      fontsize=8.5,
      fontweight='bold',
  )
  ax_c.legend(
      title='Latitude Band',
      loc='upper right',
      frameon=False,
      prop={'size': 6.5},
  )

  _plot_age_curves(
      ax_d, hrv_user, 'Male', 'Heart Rate Variability (ms)', y_lim=(24, 72)
  )
  ax_d.set_title(
      'D. Men: Mean Monthly HRV by Age and Latitude Band',
      loc='left',
      fontsize=8.5,
      fontweight='bold',
  )
  ax_d.legend(
      title='Latitude Band',
      loc='upper right',
      frameon=False,
      prop={'size': 6.5},
  )

  # Row 3: Population Densities
  _plot_density(
      ax_e,
      rhr_user,
      'Resting Heart Rate (bpm)',
      'E. Monthly RHR Distribution by Latitude Band',
      (35, 115),
  )
  _plot_density(
      ax_f,
      hrv_user,
      'Heart Rate Variability (ms)',
      'F. Monthly HRV Distribution by Latitude Band',
      (15, 130),
  )

  png_path, svg_path, _ = _save_fig(
      fig, output_dir, file_prefix, bbox_inches=None
  )
  return fig, png_path, svg_path


# --- 4. Thinning-Validation Figures ---


def generate_figure_a1(
    df_raw: pd.DataFrame,
    df_thinned: pd.DataFrame,
    output_dir: Optional[str] = None,
    file_prefix: str = 'figure_a1',
    rhr_col: str = 'avg_resting_heart_rate_bpm',
    lat_col: str = 'lat_e7',
    year_cohort: int = 2022,
    raw_registry: Optional[pd.DataFrame] = None,
    thinned_registry: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, Optional[str], Optional[str]]:
  """Plots mean RHR vs age and the RHR density by sex, before vs after thinning.

  Thinning-validation figure: a 2x2 grid whose rows are the raw and the
  thinned cohort and whose columns are the mean-RHR-vs-age curves (95 % CI)
  and the per-sex RHR KDE.

  Args:
    df_raw: RHR user-month frame before thinning.
    df_thinned: RHR user-month frame after thinning.
    output_dir: Directory for png/svg/pdf; None = do not save.
    file_prefix: Output file stem.
    rhr_col: RHR value column.
    lat_col: Latitude (1e-7 degrees) column.
    year_cohort: Reference year used to derive age from birth_year if needed.
    raw_registry: Optional pre-built `build_user_registry(df_raw, ...)`.
    thinned_registry: Optional pre-built `build_user_registry(df_thinned, ...)`.

  Returns:
    (figure, png path, svg path).
  """
  setup_nature_style()

  fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
  (ax_a, ax_b), (ax_c, ax_d) = axes

  user_raw = _registry_or_build(
      raw_registry, df_raw, rhr_col, lat_col, year_cohort
  )
  user_thin = _registry_or_build(
      thinned_registry, df_thinned, rhr_col, lat_col, year_cohort
  )

  def plot_lifespan(ax, user_df, title_label):
    for sex, color in GENDER_COLORS.items():
      sub = user_df[
          (user_df['gender'] == sex)
          & (user_df['age'] >= 18)
          & (user_df['age'] <= 80)
      ]
      stats_by_age = sub.groupby('age')['mean_val'].agg(
          ['mean', 'sem', 'count']
      )
      stats_by_age = stats_by_age[stats_by_age['count'] >= 10]
      ages = stats_by_age.index.values
      means = stats_by_age['mean'].values
      sems = stats_by_age['sem'].values

      ax.plot(ages, means, label=sex, color=color, linewidth=1.2)
      ax.fill_between(
          ages,
          means - 1.96 * sems,
          means + 1.96 * sems,
          color=color,
          alpha=0.18,
          edgecolor='none',
      )

    ax.set_xlabel('Age (years)')
    ax.set_ylabel('Mean RHR (bpm)')
    ax.set_xlim(18, 80)
    ax.set_ylim(61, 71)
    ax.set_title(title_label, loc='left', fontsize=8.5, fontweight='bold')
    ax.legend(title='Sex', frameon=False, loc='upper right', prop={'size': 6.5})
    ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc')

  def plot_sex_density(ax, user_df, title_label):
    xs = np.linspace(40, 120, 300)
    for sex, color in GENDER_COLORS.items():
      vals = user_df[user_df['gender'] == sex]['mean_val'].dropna()
      kde = stats.gaussian_kde(vals)
      ax.plot(xs, kde(xs), label=sex, color=color, linewidth=1.2)

    ax.set_xlabel('RHR (bpm)')
    ax.set_ylabel('Density')
    ax.set_xlim(40, 120)
    ax.set_title(title_label, loc='left', fontsize=8.5, fontweight='bold')
    ax.legend(title='Sex', frameon=False, loc='upper right', prop={'size': 6.5})
    ax.grid(True, linestyle=':', alpha=0.4, color='#cccccc')

  # Top row: Raw (Before thinning)
  plot_lifespan(ax_a, user_raw, 'a)')
  plot_sex_density(ax_b, user_raw, 'b)')

  # Bottom row: Thinned (After thinning)
  plot_lifespan(ax_c, user_thin, 'c)')
  plot_sex_density(ax_d, user_thin, 'd)')

  plt.tight_layout()
  png_path, svg_path, _ = _save_fig(
      fig, output_dir, file_prefix, bbox_inches=None
  )
  return fig, png_path, svg_path


def generate_figure_a2(
    df_raw: pd.DataFrame,
    df_thinned: pd.DataFrame,
    output_dir: Optional[str] = None,
    file_prefix: str = 'figure_a2',
    rhr_col: str = 'avg_resting_heart_rate_bpm',
    lat_col: str = 'lat_e7',
    year_cohort: int = 2022,
    raw_registry: Optional[pd.DataFrame] = None,
    thinned_registry: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, Optional[str], Optional[str]]:
  """Plots RHR densities for ages 21, 22 and 23 before/after thinning.

  Thinning-validation figure: one row per latitude band (plus overall), with
  the raw cohort on the left and the thinned cohort on the right.

  Args:
    df_raw: RHR user-month frame before thinning.
    df_thinned: RHR user-month frame after thinning.
    output_dir: Directory for png/svg/pdf; None = do not save.
    file_prefix: Output file stem.
    rhr_col: RHR value column.
    lat_col: Latitude (1e-7 degrees) column.
    year_cohort: Reference year used to derive age from birth_year if needed.
    raw_registry: Optional pre-built `build_user_registry(df_raw, ...)`.
    thinned_registry: Optional pre-built `build_user_registry(df_thinned, ...)`.

  Returns:
    (figure, png path, svg path).
  """
  setup_nature_style()

  fig, axes = plt.subplots(6, 2, figsize=(7.2, 10.0), sharex=True, sharey=True)

  user_raw = _registry_or_build(
      raw_registry, df_raw, rhr_col, lat_col, year_cohort
  )
  user_thin = _registry_or_build(
      thinned_registry, df_thinned, rhr_col, lat_col, year_cohort
  )

  row_specs = [('Overall', None)] + [
      (name, name) for name, _, _ in LATITUDE_BANDS
  ]
  age_colors = {
      21: '#2ca02c',  # Green
      22: '#d62728',  # Red
      23: '#1f77b4',  # Blue
  }

  xs = np.linspace(40, 115, 200)

  for idx, (band_label, band_filter) in enumerate(row_specs):
    ax_left = axes[idx, 0]
    ax_right = axes[idx, 1]

    # Filter data
    sub_raw = (
        user_raw
        if band_filter is None
        else user_raw[user_raw['primary_band'] == band_filter]
    )
    sub_thin = (
        user_thin
        if band_filter is None
        else user_thin[user_thin['primary_band'] == band_filter]
    )

    # Before (left)
    for age in [21, 22, 23]:
      v_raw = sub_raw[sub_raw['age'] == age]['mean_val'].dropna()
      if len(v_raw) > 20:
        kde = stats.gaussian_kde(v_raw)
        ax_left.plot(
            xs, kde(xs), label=str(age), color=age_colors[age], linewidth=1.1
        )

    # After (right)
    for age in [21, 22, 23]:
      v_thin = sub_thin[sub_thin['age'] == age]['mean_val'].dropna()
      if len(v_thin) > 20:
        kde = stats.gaussian_kde(v_thin)
        ax_right.plot(
            xs, kde(xs), label=str(age), color=age_colors[age], linewidth=1.1
        )

    ax_left.set_ylabel(f'{band_label}\nDensity')
    ax_left.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
    ax_right.grid(True, linestyle=':', alpha=0.4, color='#cccccc')

    if idx == 0:
      ax_left.set_title('Before', fontsize=8.5)
      ax_right.set_title('After', fontsize=8.5)
      ax_left.legend(
          title='Age', loc='upper right', frameon=False, prop={'size': 6.0}
      )

    if idx == 5:
      ax_left.set_xlabel('RHR (bpm)')
      ax_right.set_xlabel('RHR (bpm)')

  plt.tight_layout()
  png_path, svg_path, _ = _save_fig(
      fig, output_dir, file_prefix, bbox_inches=None
  )
  return fig, png_path, svg_path


def generate_figure_a3(
    df_raw: pd.DataFrame,
    df_thinned: pd.DataFrame,
    output_dir: Optional[str] = None,
    file_prefix: str = 'figure_a3',
    rhr_col: str = 'avg_resting_heart_rate_bpm',
    lat_col: str = 'lat_e7',
    year_cohort: int = 2022,
    raw_registry: Optional[pd.DataFrame] = None,
    thinned_registry: Optional[pd.DataFrame] = None,
) -> Tuple[plt.Figure, Optional[str], Optional[str]]:
  """Plots user counts and mean RHR vs age before/after thinning per band.

  Thinning-validation figure: one row per latitude band (plus overall); the
  left column shows unique users per age and the right column mean RHR per
  age, each with the raw and the thinned cohort overlaid.

  Args:
    df_raw: RHR user-month frame before thinning.
    df_thinned: RHR user-month frame after thinning.
    output_dir: Directory for png/svg/pdf; None = do not save.
    file_prefix: Output file stem.
    rhr_col: RHR value column.
    lat_col: Latitude (1e-7 degrees) column.
    year_cohort: Reference year used to derive age from birth_year if needed.
    raw_registry: Optional pre-built `build_user_registry(df_raw, ...)`.
    thinned_registry: Optional pre-built `build_user_registry(df_thinned, ...)`.

  Returns:
    (figure, png path, svg path).
  """
  setup_nature_style()

  fig, axes = plt.subplots(6, 2, figsize=(7.2, 10.0), sharex=True)

  user_raw = _registry_or_build(
      raw_registry, df_raw, rhr_col, lat_col, year_cohort
  )
  user_thin = _registry_or_build(
      thinned_registry, df_thinned, rhr_col, lat_col, year_cohort
  )

  row_specs = [('Overall', None)] + [
      (name, name) for name, _, _ in LATITUDE_BANDS
  ]

  color_orig = '#d95f02'  # Orange-Red for Before (Original)
  color_thinned = '#2c7bb6'  # Blue for After (Thinned)

  for idx, (band_label, band_filter) in enumerate(row_specs):
    ax_u = axes[idx, 0]  # Left: Unique Users vs Age
    ax_r = axes[idx, 1]  # Right: Mean RHR vs Age

    sub_raw = (
        user_raw
        if band_filter is None
        else user_raw[user_raw['primary_band'] == band_filter]
    )
    sub_thin = (
        user_thin
        if band_filter is None
        else user_thin[user_thin['primary_band'] == band_filter]
    )

    # Filter to ages 18 to 30
    sub_raw = sub_raw[(sub_raw['age'] >= 18) & (sub_raw['age'] <= 30)]
    sub_thin = sub_thin[(sub_thin['age'] >= 18) & (sub_thin['age'] <= 30)]

    stats_raw = sub_raw.groupby('age')['mean_val'].agg(['mean', 'sem', 'count'])
    stats_thin = sub_thin.groupby('age')['mean_val'].agg(
        ['mean', 'sem', 'count']
    )

    # Left: User counts
    ax_u.plot(
        stats_raw.index,
        stats_raw['count'],
        label='Before (Original)',
        color=color_orig,
        linestyle='--',
        marker='o',
        markersize=3.0,
    )
    ax_u.plot(
        stats_thin.index,
        stats_thin['count'],
        label='After (Thinned)',
        color=color_thinned,
        linestyle='-',
        marker='s',
        markersize=3.0,
    )

    # Right: Mean RHR with 95% CI
    ax_r.plot(
        stats_raw.index,
        stats_raw['mean'],
        label='Before (Original)',
        color=color_orig,
        linestyle='--',
        marker='o',
        markersize=3.0,
    )
    ax_r.fill_between(
        stats_raw.index,
        stats_raw['mean'] - 1.96 * stats_raw['sem'],
        stats_raw['mean'] + 1.96 * stats_raw['sem'],
        color=color_orig,
        alpha=0.15,
        edgecolor='none',
    )

    ax_r.plot(
        stats_thin.index,
        stats_thin['mean'],
        label='After (Thinned)',
        color=color_thinned,
        linestyle='-',
        marker='s',
        markersize=3.0,
    )
    ax_r.fill_between(
        stats_thin.index,
        stats_thin['mean'] - 1.96 * stats_thin['sem'],
        stats_thin['mean'] + 1.96 * stats_thin['sem'],
        color=color_thinned,
        alpha=0.15,
        edgecolor='none',
    )

    ax_u.set_ylabel(f'{band_label}\nUnique Users')
    ax_r.set_ylabel(f'{band_label}\nMean RHR (bpm)')
    ax_u.grid(True, linestyle=':', alpha=0.4, color='#cccccc')
    ax_r.grid(True, linestyle=':', alpha=0.4, color='#cccccc')

    if idx == 0:
      ax_u.legend(loc='upper left', frameon=False, prop={'size': 6.5})
      ax_r.legend(loc='upper right', frameon=False, prop={'size': 6.5})

    if idx == 5:
      ax_u.set_xlabel('Age (years)')
      ax_r.set_xlabel('Age (years)')

  plt.tight_layout()
  png_path, svg_path, _ = _save_fig(
      fig, output_dir, file_prefix, bbox_inches=None
  )
  return fig, png_path, svg_path


# --- 5. Shared I/O helpers for figures ---


def _save_fig(
    fig: plt.Figure,
    out_dir: Optional[str],
    stem: str,
    bbox_inches: Optional[str] = 'tight',
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
  """Writes `{stem}.png` (300 dpi), `{stem}.svg` and `{stem}.pdf` via file_io.

  Args:
    fig: Matplotlib figure to save.
    out_dir: Destination directory (created if missing). `None` disables
      saving: the notebooks render the figures inline and only write the files
        when `SAVE_PAPER_ASSETS` is on.
    stem: File name without extension.
    bbox_inches: Passed to `fig.savefig`; `None` keeps the figure's own layout
      (used by the demographic distributions and thinning-validation figures).

  Returns:
    The three written paths (png, svg, pdf), or `(None, None, None)` when
    `out_dir` is None.
  """
  if out_dir is None:
    return None, None, None
  fio.make_dirs(out_dir)
  written = []
  for fmt in ('png', 'svg', 'pdf'):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches=bbox_inches)
    path = os.path.join(out_dir, f'{stem}.{fmt}')
    with fio.open_file(path, 'wb') as f:
      f.write(buf.getvalue())
    written.append(path)
  return written[0], written[1], written[2]


def _read_geojson(path: str) -> Any:
  """Reads a GeoJSON file through file_io into a `gpd.GeoDataFrame`."""
  if gpd is None:
    raise ImportError('geopandas is required to render choropleth maps.')
  with fio.open_file(path, 'rb') as f:
    return gpd.read_file(io.BytesIO(f.read()))


def _write_csv(df: pd.DataFrame, path: str, index: bool = False) -> None:
  """Writes a DataFrame as CSV through file_io."""
  with fio.open_file(path, 'w') as f:
    df.to_csv(f, index=index)


# --- 6. Model-based figures: constants ---

METRIC_TARGET_COLS = sl.METRIC_TARGET_COLS
# Columns read from the clean user-month cohort frame for every model-based
# figure (the target column of the metric is appended at load time).
MODEL_INPUT_COLS: List[str] = [
    'user_id',
    'activity_period',
    'gender',
    'age',
    'lat_e7',
    'lng_e7',
    'country_2_alpha_cd',
    'avg_temp_mean',
    'avg_temp_min',
    'avg_temp_max',
    'dl_monthly',
]
# pd.cut edges used by the lat-bin fit figure (legend shows the intervals,
# e.g. "(-60, -30]", which is why assign_latitude_band is *not* used here).
MODEL_LAT_BINS: List[int] = [-60, -30, 0, 30, 60, 90]
MONTH_LABELS: List[str] = [
    'Jan',
    'Feb',
    'Mar',
    'Apr',
    'May',
    'Jun',
    'Jul',
    'Aug',
    'Sep',
    'Oct',
    'Nov',
    'Dec',
]
# Ordered: plotting order of the country-fit figure.
FIT_COUNTRY_NAMES: Dict[str, str] = {
    'US': 'United States',
    'GB': 'United Kingdom',
    'AU': 'Australia',
    'CN': 'China',
    'BR': 'Brazil',
    'IN': 'India',
    'ZA': 'South Africa',
    'NG': 'Nigeria',
}
FIT_COUNTRY_COLORS: Dict[str, str] = {
    'United States': '#e31a1c',
    'United Kingdom': '#1f78b4',
    'Australia': '#33a02c',
    'China': '#984ea3',
    'Brazil': '#ff7f00',
    'India': '#e3a802',
    'South Africa': '#a65628',
    'Nigeria': '#f781bf',
}
# Age groups are left-closed intervals [lo, hi) with an open top edge, so the
# youngest (18) and the oldest users are kept; see ``assign_age_group``.
SENSITIVITY_AGE_BINS: List[float] = [18, 30, 45, 60, np.inf]
SENSITIVITY_AGE_LABELS: List[str] = ['18-29', '30-44', '45-59', '60+']
SENSITIVITY_AGE_COLORS: Dict[str, str] = {
    '18-29': '#1f77b4',
    '30-44': '#ff7f0e',
    '45-59': '#2ca02c',
    '60+': '#d62728',
}
DEMOGRAPHIC_AGE_BINS: List[float] = [18, 28, 38, 48, 58, 68, np.inf]
DEMOGRAPHIC_AGE_LABELS: List[str] = [
    '18-27',
    '28-37',
    '38-47',
    '48-57',
    '58-67',
    '68+',
]


def assign_age_group(
    age: pd.Series, bins: Sequence[float], labels: Sequence[str]
) -> pd.Series:
  """Bins ages into left-closed groups ``[bins[i], bins[i + 1])``.

  Args:
    age: Ages in years.
    bins: Increasing edges; the last may be ``np.inf`` for an open top group.
    labels: One label per group (``len(bins) - 1``).

  Returns:
    Categorical series of labels (NaN below the first edge).
  """
  return pd.cut(age, bins=list(bins), labels=list(labels), right=False)


DEMOGRAPHIC_AGE_COLORS: Dict[str, str] = dict(
    zip(
        DEMOGRAPHIC_AGE_LABELS,
        ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b'],
    )
)
DEMOGRAPHIC_SEX_STYLES: Dict[str, Dict[str, str]] = {
    'Female': {'linestyle': '-', 'marker': 'o'},
    'Male': {'linestyle': '--', 'marker': 's'},
}
# Desaturated latitude-band colours: pastel blues -> white -> amber/red.
CONTOUR_CMAP = LinearSegmentedColormap.from_list(
    'residual_cmap',
    ['#8faed4', '#d2e6f1', '#ffffff', '#ffbf7f', '#f18d8e'],
    N=256,
)
WORLD_GRID_STEP_DEG = 2.0
WORLD_GRID_COLS: List[str] = [
    'user_id',
    'lat_e7',
    'lng_e7',
    'activity_period',
    'avg_temp_mean',
    'avg_temp_max',
    'avg_temp_min',
    'dl_monthly',
]
# Cohort column -> suffix of its per-cell mean `{metric}_{suffix}`.
WORLD_GRID_ENV_COLS: Dict[str, str] = {
    'avg_temp_mean': 'temp_mean',
    'avg_temp_max': 'temp_max',
    'avg_temp_min': 'temp_min',
    'dl_monthly': 'day_length',
}
RESIDUAL_PLOT_CFG: Dict[str, Dict[str, Any]] = {
    'rhr': dict(
        n_users=20000,
        temp_residual_method='harmonic',
        coef_source='model_dir',
        point_size=1.0,
        xlabel='Predicted Resting Heart Rate (bpm)',
        ylabel='Residuals (bpm)',
        stem='rhr_residuals',
    ),
    'hrv': dict(
        n_users=None,
        temp_residual_method='harmonic',
        coef_source='interpreted',
        point_size=36.0,
        xlabel='Predicted log(HRV)',
        ylabel='Residuals (Error)',
        stem='hrv_residuals',
    ),
}
RANDOM_EFFECTS_PLOT_CFG: Dict[str, Dict[str, str]] = {
    'rhr': dict(
        xlabel='Deviation from Mean RHR (bpm)',
        qq_ylabel='Ordered Values (bpm)',
        stem='rhr_qq',
    ),
    'hrv': dict(
        xlabel='Deviation from Mean log(HRV)',
        qq_ylabel='Ordered Values (log(HRV))',
        stem='hrv_qq',
    ),
}


# --- 7. Small I/O and styling helpers ---


def _read_parquet(
    path: str, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
  """Reads a parquet file through file_io (optionally a column subset)."""
  with fio.open_file(path, 'rb') as f:
    return pd.read_parquet(
        io.BytesIO(f.read()), columns=list(columns) if columns else None
    )


def _read_csv(path: str, **kwargs: Any) -> pd.DataFrame:
  """Reads a CSV file through file_io."""
  with fio.open_file(path, 'r') as f:
    return pd.read_csv(f, **kwargs)


def _load_cohort(
    data_dir: str, metric: str, columns: Optional[Sequence[str]] = None
) -> pd.DataFrame:
  """Loads the clean cohort of `metric` through `cohort_io.load_cohort`."""
  return cio.load_cohort(data_dir, metric, columns)


def _check_metric(metric: str) -> None:
  if metric not in METRIC_TARGET_COLS:
    raise ValueError(
        f'metric must be one of {sorted(METRIC_TARGET_COLS)}, got {metric!r}'
    )


def _resolve_per_metric(
    value: Union[str, Mapping[str, str]], metric: str
) -> str:
  """Returns `value` itself if it is a str, else `value[metric]`."""
  return value if isinstance(value, str) else value[metric]


def _style_like_source(seaborn_style: Optional[str]) -> None:
  """Applies the Nature rcParams plus an optional seaborn theme.

  The model-based figures either use a seaborn theme (`sns.set_theme(
  style=...)`) or plain matplotlib defaults with all four spines visible;
  every font size is passed explicitly by the plotting functions, so only
  spines/ticks/line defaults differ from `setup_nature_style`.  Remove the
  branches below to render the figures in pure Nature style instead.

  Args:
    seaborn_style: seaborn style name ('ticks' / 'white') or None to keep
      matplotlib defaults (all spines on, thicker lines/markers).
  """
  setup_nature_style()
  if seaborn_style is not None:
    sns.set_theme(style=seaborn_style)
  else:
    plt.rcParams.update({
        'axes.spines.top': True,
        'axes.spines.right': True,
        'lines.linewidth': 1.5,
        'lines.markersize': 6.0,
    })


# --- 8. Model artefact loaders ---


def find_model_dir(
    metric_results_dir: str,
    required_file: str = 'random_effects.parquet',
    metric: Optional[str] = None,
) -> str:
  """Returns the per-model directory of the winner.

  02_modelling.ipynb writes `{timestamp}_{formula-name}/` folders; only the
  final winner refit saves `random_effects.parquet`. When several folders
  contain that file (e.g. because extra models were refitted after the
  winner) the folder whose `model_coefficients.csv` has the same fixed-effect
  terms as `model_{metric}_coefficients_interpreted.csv` is chosen.

  Args:
    metric_results_dir: `{data_dir}/results/{run_tag}`.
    required_file: file that must exist inside the folder.
    metric: 'rhr' | 'hrv'; enables the term matching described above.

  Returns:
    Absolute path of the winner's per-model directory (no trailing slash).

  Raises:
    FileNotFoundError: if no sub-directory contains ``required_file``, or if
      several do and none matches the winner's terms (pass ``model_dir``
      explicitly in that case).
  """
  candidates = sorted(
      d.rstrip('/')
      for d in fio.list_dir(metric_results_dir)
      if fio.file_exists(os.path.join(metric_results_dir, d, required_file))
  )
  if not candidates:
    raise FileNotFoundError(
        f'No sub-directory of {metric_results_dir} contains {required_file}'
    )
  if len(candidates) == 1:
    return os.path.join(metric_results_dir, candidates[0])
  coef_path = (
      sl.coefficients_csv_path(metric_results_dir, metric) if metric else ''
  )
  if coef_path and fio.file_exists(coef_path):
    winner_terms = set(_read_csv(coef_path, index_col=0).index.astype(str))
    for d in reversed(candidates):
      path = os.path.join(metric_results_dir, d, 'model_coefficients.csv')
      if fio.file_exists(path):
        terms = set(_read_csv(path, index_col=0).index.astype(str))
        if {t for t in terms if not t.endswith(' Var')} == {
            t for t in winner_terms if not t.endswith(' Var')
        }:
          return os.path.join(metric_results_dir, d)
  raise FileNotFoundError(
      f'{len(candidates)} sub-directories of {metric_results_dir} contain'
      f' {required_file} and none matches the winner coefficients'
      f' ({coef_path or "no metric given"}); choose one explicitly:'
      f' {candidates}'
  )


def resolve_model_dir(
    data_dir: str,
    metric: str,
    run_tag: str,
    model_dir: Optional[str] = None,
    required_file: str = 'model-row.csv',
) -> str:
  """Returns the per-model folder the result figures/tables should be read from.

  Either an explicit ``model_dir`` (absolute path, or a folder name inside
  ``results/{run_tag}``) or - when ``None`` - the best-performing model of the
  run located with `find_model_dir` (the overall winner's folder is the one
  whose fixed-effect terms match ``model_{metric}_coefficients_interpreted.csv``
  when several candidates qualify).

  Args:
    data_dir: Root data directory.
    metric: 'rhr' | 'hrv'.
    run_tag: ``results/`` sub-folder of the run.
    model_dir: Explicit per-model folder (absolute, or relative to the run) or
      None to pick the winner.
    required_file: File that must exist in the chosen folder.

  Returns:
    Absolute path of the per-model folder (no trailing slash).

  Raises:
    FileNotFoundError: if the explicit folder does not contain
      ``required_file`` or no candidate folder exists.
  """
  _check_metric(metric)
  run_dir = fio.get_results_dir(run_tag, data_dir)
  if model_dir:
    path = model_dir.rstrip('/')
    if not fio.file_exists(os.path.join(path, required_file)):
      path = os.path.join(run_dir, model_dir.strip('/'))
    if not fio.file_exists(os.path.join(path, required_file)):
      raise FileNotFoundError(
          f'{model_dir!r} is not a model folder (no {required_file}) under'
          f' {run_dir}'
      )
    return path
  return find_model_dir(run_dir, required_file=required_file, metric=metric)


def extract_winning_coefficients(
    results_df: pd.DataFrame, model_name: Optional[str] = None
) -> pd.Series:
  """Standardised fixed-effect coefficients of one candidate-model row.

  Reads the `beta_std_*` columns of one row of
  `lme-models-results-{metric}.csv`.  The same numbers (REML refit instead of
  the ML fit) are in `model_{metric}_coefficients_interpreted.csv`, see
  `load_model_coefficients`.

  Args:
    results_df: `lme-models-results-{metric}.csv` (columns model, formula, BIC,
      beta_std_<term>, ...).
    model_name: row to use; defaults to the lowest-BIC row (the BIC winner).

  Returns:
    Series indexed by term name (prefix `beta_std_` stripped), NaN entries
    dropped.
  """
  if model_name is None:
    row = results_df.loc[results_df['BIC'].idxmin()]
  else:
    row = results_df.loc[results_df['model'] == model_name].iloc[0]
  beta_cols = [c for c in results_df.columns if c.startswith('beta_std_')]
  betas = row[beta_cols].dropna().astype(float)
  betas.index = [c[len('beta_std_') :] for c in betas.index]
  return betas


def load_model_coefficients(
    metric_results_dir: str,
    metric: str,
    source: str = 'interpreted',
    model_dir: Optional[str] = None,
) -> pd.Series:
  """Loads the winner's fixed-effect coefficients (term -> standardised beta).

  Args:
    metric_results_dir: `{data_dir}/results/{run_tag}`.
    metric: 'rhr' | 'hrv'.
    source: 'interpreted' -> `model_{metric}_coefficients_interpreted.csv`
      (index=term, column `coef`; REML refit).  'model_dir' ->
      `{model_dir}/model_coefficients.csv` (index=term, column `Coef.`; ML fit
      of the final winner).  'lme_results' -> lowest-BIC row of
      `lme-models-results-{metric}.csv` (`beta_std_*`).
    model_dir: per-model directory for source='model_dir' (default:
      `find_model_dir`).

  Returns:
    Series of coefficients; variance-component rows (`* Var`) are dropped.
  """
  _check_metric(metric)
  if source == 'lme_results':
    results_df = _read_csv(sl.results_csv_path(metric_results_dir, metric))
    return extract_winning_coefficients(results_df)
  if source == 'interpreted':
    path = sl.coefficients_csv_path(metric_results_dir, metric)
    column = 'coef'
  elif source == 'model_dir':
    path = os.path.join(
        model_dir or find_model_dir(metric_results_dir, metric=metric),
        'model_coefficients.csv',
    )
    column = 'Coef.'
  else:
    raise ValueError(
        "source must be 'interpreted', 'model_dir' or 'lme_results', got"
        f' {source!r}'
    )
  coef_df = _read_csv(path, index_col=0)
  coef = coef_df[column].astype(float)
  keep = [not str(term).endswith(' Var') for term in coef.index]
  return coef[keep]


def load_random_effects(model_dir: str) -> pd.Series:
  """Loads `{model_dir}/random_effects.parquet` (columns user_id, random_intercept).

  Args:
    model_dir: Directory containing the saved model's `random_effects.parquet`.

  Returns:
    Series indexed by user_id with the BLUP random intercept (model scale).
  """
  re_df = _read_parquet(os.path.join(model_dir, 'random_effects.parquet'))
  return pd.Series(
      re_df['random_intercept'].values,
      index=re_df['user_id'].values,
      name='random_intercept',
  )


# --- 9. Feature engineering + prediction from saved coefficients ---


def add_model_predictions(
    df: pd.DataFrame,
    metric: str,
    coef: pd.Series,
    random_effects: pd.Series,
    mode: str = 'accuracy',
) -> pd.DataFrame:
  """Adds predicted_val, re, fitted_values, plot_pred_final and plot_actual_final in place.

  Args:
    df: output of `prepare_model_features`, with `user_id`.
    metric: 'rhr' | 'hrv' (HRV is modelled as log(avg_hrv) and back-transformed
      with Duan's smearing factor computed on `df`).
    coef: fixed-effect coefficients.
    random_effects: user_id -> random intercept (users not in the fit get 0).
    mode: 'accuracy'   - prediction includes the random intercept, observation
      is raw (lat-bin and country figures, "Accuracy Mode / Raw Trends").
      'calibration'- prediction is fixed effects only, the random intercept is
      removed from the observation (joint demographics figure).

  Returns:
    The same frame.
  """
  _check_metric(metric)
  if mode not in ('accuracy', 'calibration'):
    raise ValueError(f"mode must be 'accuracy' or 'calibration', got {mode!r}")
  df['predicted_val'] = sl.predict_from_coefficients(df, coef)
  df['re'] = df['user_id'].map(random_effects).fillna(0.0)
  df['fitted_values'] = df['predicted_val'] + df['re']
  if metric == 'hrv':
    smearing = float(np.mean(np.exp(df['target_log'] - df['fitted_values'])))
    if mode == 'accuracy':
      df['plot_pred_final'] = np.exp(df['fitted_values']) * smearing
      df['plot_actual_final'] = df['avg_hrv']
    else:
      df['plot_pred_final'] = np.exp(df['predicted_val']) * smearing
      df['plot_actual_final'] = np.exp(df['target_log'] - df['re'])
  else:
    target = METRIC_TARGET_COLS['rhr']
    if mode == 'accuracy':
      df['plot_pred_final'] = df['fitted_values']
      df['plot_actual_final'] = df[target]
    else:
      df['plot_pred_final'] = df['predicted_val']
      df['plot_actual_final'] = df[target] - df['re']
  return df


def summarise_monthly_fit(
    df: pd.DataFrame, group_cols: Sequence[str]
) -> pd.DataFrame:
  """Monthly mean/std/count/SE of plot_actual_final and plot_pred_final per group.

  Args:
    df: Output of `add_model_predictions`, with a `month` column.
    group_cols: Columns to group by in addition to `month`.

  Returns:
    Columns: *group_cols, month, actual_mean, actual_std, n, predicted_mean,
    predicted_std, pred_count, actual_se, predicted_se.
  """
  keys = list(group_cols) + ['month']
  summary = (
      df.groupby(keys, observed=True)
      .agg(
          actual_mean=('plot_actual_final', 'mean'),
          actual_std=('plot_actual_final', 'std'),
          n=('plot_actual_final', 'count'),
          predicted_mean=('plot_pred_final', 'mean'),
          predicted_std=('plot_pred_final', 'std'),
          pred_count=('plot_pred_final', 'count'),
      )
      .reset_index()
  )
  summary['actual_se'] = summary['actual_std'] / (summary['n'] ** 0.5)
  summary['predicted_se'] = summary['predicted_std'] / (
      summary['pred_count'] ** 0.5
  )
  return summary


def _subsample_users(
    df: pd.DataFrame, n_users: Optional[int], random_state: int
) -> pd.DataFrame:
  """Keeps all rows of `n_users` randomly chosen users (np.random.seed + np.random.choice)."""
  if n_users is None:
    return df
  users = df['user_id'].unique()
  if len(users) <= n_users:
    return df
  rng = np.random.RandomState(random_state)
  keep = rng.choice(users, size=n_users, replace=False)
  return df[df['user_id'].isin(keep)].copy()


THINNED_USERS_FILE = 'thinning_dropped_user_ids.csv'


def load_thinned_user_ids(metric_results_dir: str) -> Optional[List[str]]:
  """User ids that 02_modelling.ipynb removed by thinning; None if not thinned."""
  path = os.path.join(metric_results_dir, THINNED_USERS_FILE)
  if not fio.file_exists(path):
    return None
  return _read_csv(path, dtype=str)['user_id'].tolist()


def drop_users(df: pd.DataFrame, user_ids: Iterable[Any]) -> pd.DataFrame:
  """Rows of ``df`` whose ``user_id`` is not in ``user_ids``.

  Ids are compared as strings: ``load_thinned_user_ids`` reads them as text
  while the cohort loaders keep the id dtype of the source (integers in the
  public CSV release), so a dtype-sensitive ``isin`` would match nothing.

  Args:
    df: Frame with a ``user_id`` column.
    user_ids: Ids to drop (any scalar type).

  Returns:
    A copy of ``df`` without the listed users.
  """
  wanted = {str(u) for u in user_ids}
  return df[~df['user_id'].astype(str).isin(wanted)].copy()


def prepare_run_features(
    df: pd.DataFrame,
    metric: str,
    metric_results_dir: str,
    temp_residual_method: str = 'harmonic',
) -> pd.DataFrame:
  """`sl.prepare_model_features` with the statistics of the fitted run.

  02_modelling.ipynb stores the standardisation means/SDs, the temperature
  OLS parameters and the male share in `feature_stats.json`; applying them
  reproduces the predictors the coefficients were estimated on. Without that
  file the statistics are recomputed on `df` (a message says so).

  Args:
    df: Clean user-month rows (modified in place).
    metric: 'rhr' | 'hrv'.
    metric_results_dir: `{data_dir}/results/{run_tag}`.
    temp_residual_method: Fallback residualisation when the run has no
      `feature_stats.json`.

  Returns:
    The same frame with the model predictors added.
  """
  run_stats = sl.read_feature_stats(metric_results_dir)
  if run_stats is None:
    print(
        f'  {sl.FEATURE_STATS_FILE} not found in {metric_results_dir}:'
        ' feature statistics recomputed on the loaded cohort'
    )
  return sl.prepare_model_features(df, metric, temp_residual_method, run_stats)


def load_fitted_frame(
    data_dir: str,
    metric_results_dir: str,
    metric: str,
    mode: str = 'accuracy',
    n_users: Optional[int] = None,
    row_filter: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    temp_residual_method: str = 'harmonic',
    coef_source: str = 'interpreted',
    model_dir: Optional[str] = None,
    random_state: int = 42,
    exclude_user_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
  """Loads a clean cohort, rebuilds the model features and attaches predictions.

  Order of operations: load -> drop excluded users -> `prepare_run_features`
  (run statistics, or recomputed on the full remaining cohort) -> user
  subsample -> `row_filter` -> `add_model_predictions`.

  Inputs:
    the clean user-month cohort frame of `metric` (MODEL_INPUT_COLS + target;
    see `cohort_io.load_cohort`),
    `{metric_results_dir}/model_{metric}_coefficients_interpreted.csv` (or the
    alternative given by `coef_source`),
    `{metric_results_dir}/feature_stats.json` and
    `thinning_dropped_user_ids.csv` when present, and
    `{model_dir}/random_effects.parquet`.

  Args:
    data_dir: dataset root.
    metric_results_dir: `{data_dir}/results/{run_tag}`.
    metric: 'rhr' | 'hrv'.
    mode: see `add_model_predictions`.
    n_users: optional user subsample size (seeded with `random_state`).
    row_filter: optional callable applied after the user subsample.
    temp_residual_method: fallback, see `prepare_run_features`.
    coef_source: see `load_model_coefficients`.
    model_dir: winner's per-model directory (default `find_model_dir`).
    random_state: seed for the user subsample.
    exclude_user_ids: users to drop; default: the users thinned out of the run
      (`thinning_dropped_user_ids.csv`), if any.

  Returns:
    Frame with features, predicted_val, re, fitted_values, plot_pred_final,
    plot_actual_final.
  """
  _check_metric(metric)
  df = _load_cohort(
      data_dir, metric, MODEL_INPUT_COLS + [METRIC_TARGET_COLS[metric]]
  )
  if exclude_user_ids is None:
    exclude_user_ids = load_thinned_user_ids(metric_results_dir)
  if exclude_user_ids is not None:
    df = drop_users(df, exclude_user_ids)
  prepare_run_features(df, metric, metric_results_dir, temp_residual_method)
  df = _subsample_users(df, n_users, random_state)
  if row_filter is not None:
    df = row_filter(df)
  model_dir = model_dir or find_model_dir(metric_results_dir, metric=metric)
  coef = load_model_coefficients(
      metric_results_dir, metric, coef_source, model_dir
  )
  random_effects = load_random_effects(model_dir)
  return add_model_predictions(df, metric, coef, random_effects, mode)


def extend_random_effects_to_cohort(
    data_dir: str,
    metric_results_dir: str,
    metric: str,
    model_dir: Optional[str] = None,
    temp_residual_method: str = 'harmonic',
    coef_source: str = 'interpreted',
    write: bool = True,
    overwrite: bool = False,
) -> pd.DataFrame:
  """Empirical-Bayes random intercepts for every user of the clean cohort.

  02_modelling.ipynb fits the LME on a random fraction of users
  (``SUBSAMPLE_FRAC``) to bound runtime and memory, so the fitted model only
  carries random intercepts for those users. The "accuracy mode" figures
  (lat-band and country fits, residual maps, random-intercept histogram) need
  an intercept for every plotted user. For user i with n_i rows and
  fixed-effect residuals r_ij, the BLUP under a random-intercept model is

      u_i = sigma_u^2 * sum_j r_ij / (sigma_u^2 * n_i + sigma_e^2),

  with sigma_u^2 / sigma_e^2 taken from the winner's `model-row.csv`
  (`random_intercept_var`, `residual_noise_var`). Features are rebuilt on the
  cohort of the run (thinned users excluded) with `prepare_run_features`.

  Writes `{model_dir}/random_effects.parquet` (user_id, random_intercept); the
  fit-sample version written by the notebook is kept as
  `random_effects_fit_sample.parquet`. If that backup already exists the folder
  was extended before and the stored full-cohort BLUPs are returned unchanged
  unless ``overwrite`` is set.

  Args:
    data_dir: Root data directory.
    metric_results_dir: ``results/{run_tag}`` of the run.
    metric: 'rhr' | 'hrv'.
    model_dir: Winner folder (default: `find_model_dir`).
    temp_residual_method: Fallback, see `prepare_run_features`.
    coef_source: 'interpreted' | 'model_dir' | 'lme_results' (see
      `load_model_coefficients`).
    write: Write the parquet into ``model_dir``.
    overwrite: Recompute even if the folder was already extended.

  Returns:
    DataFrame (user_id, random_intercept, n_obs).
  """
  _check_metric(metric)
  model_dir = model_dir or find_model_dir(metric_results_dir, metric=metric)
  out = os.path.join(model_dir, 'random_effects.parquet')
  backup = os.path.join(model_dir, 'random_effects_fit_sample.parquet')
  if not overwrite and fio.file_exists(backup):
    # A previous call already extended this folder (the fit-sample BLUPs were
    # backed up); `random_effects.parquet` therefore holds the full cohort.
    print(f'Random intercepts already extended in {model_dir}; reusing them.')
    return _read_parquet(out)
  row = _read_csv(os.path.join(model_dir, 'model-row.csv'))
  var_re = float(row['random_intercept_var'].iloc[0])
  var_resid = float(row['residual_noise_var'].iloc[0])
  coef = load_model_coefficients(
      metric_results_dir, metric, coef_source, model_dir
  )
  df = _load_cohort(
      data_dir, metric, MODEL_INPUT_COLS + [METRIC_TARGET_COLS[metric]]
  )
  thinned = load_thinned_user_ids(metric_results_dir)
  if thinned is not None:
    df = drop_users(df, thinned)
  prepare_run_features(df, metric, metric_results_dir, temp_residual_method)
  resid = df['target_log'].to_numpy() - sl.predict_from_coefficients(df, coef)
  grouped = pd.DataFrame(
      {'user_id': df['user_id'].to_numpy(), 'resid': resid}
  ).groupby('user_id')['resid']
  res_sum, res_count = grouped.sum(), grouped.count()
  re_df = pd.DataFrame({
      'user_id': res_sum.index,
      'random_intercept': (
          (var_re * res_sum / (var_re * res_count + var_resid)).to_numpy()
      ),
      'n_obs': res_count.to_numpy(),
  })
  if write:
    if fio.file_exists(out) and not fio.file_exists(backup):
      fio.copy_file(out, backup)
    buf = io.BytesIO()
    re_df.to_parquet(buf, index=False)
    with fio.open_file(out, 'wb') as f:
      f.write(buf.getvalue())
    print(f'Random intercepts for {len(re_df):,} users written to {out}')
  return re_df


# --- 10. fig:lat_bin_fits - seasonal fit per latitude band (RHR | HRV) ---


def plot_combined_seasonal_fit_lat_bin(
    rhr_summary: pd.DataFrame,
    hrv_summary: pd.DataFrame,
    suptitle: Optional[str] = (
        'Seasonal Physiological Trends across Latitude Bands (Accuracy Mode /'
        ' Raw Trends)'
    ),
) -> plt.Figure:
  """Plots the seasonal fit per latitude band (fig:lat_bin_fits).

  Observed monthly means with the model prediction overlaid, one line per
  latitude band, for RHR and HRV side by side.

  Args:
    rhr_summary: `summarise_monthly_fit(df_rhr, ['lat_group'])`, lat_group being
      the `pd.cut(lat_deg, MODEL_LAT_BINS)` intervals.
    hrv_summary: same for HRV (plot_pred_final in ms, smearing-corrected).
    suptitle: figure title; None omits it.

  Returns:
    1x2 figure (18 x 8 in): observed monthly means +-1.96 SE (points) and model
    prediction +-1.96 SE (line + band) per band, husl palette.
  """
  _style_like_source('ticks')
  fig, axes = plt.subplots(1, 2, figsize=(18, 8))
  groups = sorted(rhr_summary['lat_group'].unique())
  color_map = dict(zip(groups, sns.color_palette('husl', len(groups))))
  panels = [
      (
          axes[0],
          rhr_summary,
          'Resting Heart Rate (RHR)',
          'Resting Heart Rate (bpm)',
      ),
      (
          axes[1],
          hrv_summary,
          'Heart Rate Variability (HRV)',
          'Heart Rate Variability (ms)',
      ),
  ]
  for ax, summary, title, ylabel in panels:
    for grp in groups:  # observed means
      g = summary[summary['lat_group'] == grp]
      if g.empty:
        continue
      ax.errorbar(
          g['month'],
          g['actual_mean'],
          yerr=g['actual_se'] * 1.96,
          fmt='o',
          color=color_map[grp],
          capsize=4,
          markersize=7,
          alpha=0.4,
          label=str(grp),
      )
    for grp in groups:  # model trajectory
      g = summary[summary['lat_group'] == grp].sort_values('month')
      if not g.empty:
        ax.plot(
            g['month'], g['predicted_mean'], color=color_map[grp], linewidth=2.5
        )
    for grp in groups:  # prediction band
      g = summary[summary['lat_group'] == grp].sort_values('month')
      if not g.empty:
        ax.fill_between(
            g['month'],
            g['predicted_mean'] - 1.96 * g['predicted_se'],
            g['predicted_mean'] + 1.96 * g['predicted_se'],
            color=color_map[grp],
            alpha=0.15,
        )
    ax.set_xticks(range(1, 13))
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel('Month of Year', fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(
        title='Latitude Band',
        loc='upper right',
        frameon=True,
        fontsize=13,
        title_fontsize=13,
    )
    ax.grid(False)
  if suptitle:
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)
  fig.tight_layout()
  return fig


def make_combined_seasonal_fit_lat_bin(
    data_dir: str,
    results_dir: str,
    out_dir: Optional[str] = None,
    *,
    rhr_subdir: str,
    hrv_subdir: str,
    coef_source: Union[str, Mapping[str, str]] = 'interpreted',
    stem: str = 'rhr_hrv_combined_seasonal_prediction_fit_lat_bin',
    model_dirs: Optional[Mapping[str, str]] = None,
) -> plt.Figure:
  """Loads both full cohorts, predicts with random effects and saves the figure.

  Produces fig:lat_bin_fits.  Inputs: see `load_fitted_frame` (per metric).
  Features are standardised / residualised on the full clean cohort.

  Args:
    data_dir: dataset root (clean cohort frames via `cohort_io.load_cohort`).
    results_dir: `{data_dir}/results`.
    out_dir: output directory for `{stem}.{png,svg,pdf}`; None = do not save.
    rhr_subdir: RHR run tag (sub-directory of `results_dir` written by notebook
      02); keyword-only, no default.
    hrv_subdir: HRV run tag; keyword-only, no default.
    coef_source: see `load_model_coefficients` (str or per-metric mapping).
    stem: output file stem.
    model_dirs: optional {metric: per-model folder} overriding `find_model_dir`
      (random effects, and coefficients when `coef_source='model_dir'`).

  Returns:
    The 1x2 latitude-bin seasonal-fit figure (RHR left, HRV right).
  """
  summaries: Dict[str, pd.DataFrame] = {}
  for metric, subdir in (('rhr', rhr_subdir), ('hrv', hrv_subdir)):
    df = load_fitted_frame(
        data_dir,
        os.path.join(results_dir, subdir),
        metric,
        mode='accuracy',
        coef_source=_resolve_per_metric(coef_source, metric),
        model_dir=(model_dirs or {}).get(metric),
    )
    df['lat_group'] = pd.cut(df['lat_e7'] / 1e7, bins=MODEL_LAT_BINS)
    summaries[metric] = summarise_monthly_fit(df, ['lat_group'])
    del df
    gc.collect()
  fig = plot_combined_seasonal_fit_lat_bin(summaries['rhr'], summaries['hrv'])
  _save_fig(fig, out_dir, stem)
  return fig


# --- 11. fig:countries_prediction_fit - seasonal fit for eight countries ---


def plot_combined_seasonal_fit_country(
    rhr_summary: pd.DataFrame, hrv_summary: pd.DataFrame
) -> plt.Figure:
  """Plots the seasonal fit for eight countries (fig:countries_prediction_fit).

  Observed monthly means (markers) with the model prediction (lines) for the
  `FIT_COUNTRY_NAMES` countries, RHR and HRV side by side.

  Args:
    rhr_summary: `summarise_monthly_fit(df_rhr, ['country_name'])` restricted to
      `FIT_COUNTRY_NAMES.values()`.
    hrv_summary: same for HRV.

  Returns:
    1x2 figure (22 x 9 in); legend (left panel only) lists the countries plus
    black 'Predicted' line / 'Observed' marker handles.
  """
  _style_like_source(None)
  pred_handle = mlines.Line2D(
      [], [], color='black', linestyle='-', linewidth=2.5, label='Predicted'
  )
  obs_handle = mlines.Line2D(
      [],
      [],
      color='black',
      marker='o',
      linestyle='',
      markersize=6,
      label='Observed',
  )
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
  panels = [
      (ax1, rhr_summary, 'Resting Heart Rate (BPM)'),
      (ax2, hrv_summary, 'Heart Rate Variability (ms)'),
  ]
  for ax, summary, ylabel in panels:
    for country in FIT_COUNTRY_NAMES.values():
      color = FIT_COUNTRY_COLORS[country]
      g = summary[summary['country_name'] == country].sort_values('month')
      if g.empty:
        continue
      ax.errorbar(
          g['month'],
          g['actual_mean'],
          yerr=1.96 * g['actual_se'],
          fmt='o',
          color=color,
          alpha=0.3,
          markersize=5,
          capsize=2,
          label='_nolegend_',
      )
      ax.plot(
          g['month'],
          g['predicted_mean'],
          color=color,
          linestyle='-',
          linewidth=2.5,
          label=country,
      )
      ax.fill_between(
          g['month'],
          g['predicted_mean'] - 1.96 * g['predicted_se'],
          g['predicted_mean'] + 1.96 * g['predicted_se'],
          color=color,
          alpha=0.12,
      )
    ax.set_xlabel('Month of Year', fontsize=16, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=10)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(False)
  handles, _ = ax1.get_legend_handles_labels()
  handles.extend([pred_handle, obs_handle])
  ax1.legend(handles=handles, fontsize=13, loc='upper right', ncol=2)
  fig.tight_layout()
  return fig


def make_combined_seasonal_fit_country(
    data_dir: str,
    results_dir: str,
    out_dir: Optional[str] = None,
    *,
    rhr_subdir: str,
    hrv_subdir: str,
    coef_source: Union[str, Mapping[str, str]] = 'interpreted',
    stem: str = 'rhr_hrv_combined_seasonal_prediction_fit_country',
    model_dirs: Optional[Mapping[str, str]] = None,
) -> plt.Figure:
  """Loads both cohorts, keeps the `FIT_COUNTRY_NAMES`, predicts and saves.

  Produces fig:countries_prediction_fit. The predictors are rebuilt with the
  run's statistics before the country filter is applied (see
  `load_fitted_frame`); the HRV smearing factor is estimated on the kept
  countries.

  Inputs: see `load_fitted_frame`; additionally needs `country_2_alpha_cd`.
  `model_dirs` ({metric: folder}) optionally overrides `find_model_dir`.

  Args:
    data_dir: dataset root (clean cohort frames via `cohort_io.load_cohort`).
    results_dir: `{data_dir}/results`.
    out_dir: output directory for `{stem}.{png,svg,pdf}`; None = do not save.
    rhr_subdir: RHR run tag (sub-directory of `results_dir` written by notebook
      02); keyword-only, no default.
    hrv_subdir: HRV run tag; keyword-only, no default.
    coef_source: see `load_model_coefficients` (str or per-metric mapping).
    stem: output file stem.
    model_dirs: optional {metric: per-model folder} overriding `find_model_dir`
      (random effects, and coefficients when `coef_source='model_dir'`).

  Returns:
    The 1x2 per-country seasonal-fit figure (RHR left, HRV right).
  """
  codes = list(FIT_COUNTRY_NAMES)

  def _keep_countries(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d['country_2_alpha_cd'].isin(codes)].copy()
    d['country_name'] = d['country_2_alpha_cd'].map(FIT_COUNTRY_NAMES)
    return d

  summaries: Dict[str, pd.DataFrame] = {}
  for metric, subdir in (('rhr', rhr_subdir), ('hrv', hrv_subdir)):
    df = load_fitted_frame(
        data_dir,
        os.path.join(results_dir, subdir),
        metric,
        mode='accuracy',
        row_filter=_keep_countries,
        coef_source=_resolve_per_metric(coef_source, metric),
        model_dir=(model_dirs or {}).get(metric),
    )
    summaries[metric] = summarise_monthly_fit(df, ['country_name'])
    del df
    gc.collect()
  fig = plot_combined_seasonal_fit_country(summaries['rhr'], summaries['hrv'])
  _save_fig(fig, out_dir, stem)
  return fig


# --- 12. fig:within_user_sensitivity - within-subject temperature response ---


def compute_within_user_sensitivity(
    df: pd.DataFrame,
    metric: str,
    sample_size: int = 3_000_000,
    random_state: int = 42,
    min_user_obs: int = 5,
    temp_range: Tuple[float, float] = (-10.0, 10.0),
    min_count: int = 150,
) -> pd.DataFrame:
  """Within-user change of the metric per 1 degC of residualised temperature.

  Pipeline: users with >= `min_user_obs` months -> `sample_size` random rows
  -> temperature residualised with OLS
  `avg_temp_mean ~ abs_lat_deg_c + harmonics * lat_deg_c` (latitude in
  degrees, standardised on the sample) -> user-demeaned target (log for HRV)
  and temperature residual -> rounded temperature deviation in `temp_range`
  -> mean per (age group, deviation) with >= `min_count` rows.

  Args:
    df: clean user-month cohort frame columns user_id, activity_period, age,
      lat_e7, avg_temp_mean and the metric target.
    metric: 'rhr' (bpm change) | 'hrv' (percent change `(exp(d log) - 1) *
      100`).
    sample_size: number of rows randomly sampled after the `min_user_obs` filter
      (capped at the number of available rows).
    random_state: seed for the row sample.
    min_user_obs: minimum number of monthly observations a user must have to be
      included.
    temp_range: inclusive (min, max) of the rounded within-user temperature
      deviation (degC) kept in the output.
    min_count: minimum rows per (age group, deviation) cell to report a mean.

  Returns:
    Columns age_group, temp_res_within_round, mean, count.
  """
  _check_metric(metric)
  target = METRIC_TARGET_COLS[metric]
  counts = df['user_id'].value_counts()
  df = df[df['user_id'].isin(counts[counts >= min_user_obs].index)]
  df = df.sample(n=min(sample_size, len(df)), random_state=random_state).copy()
  sl.add_month_harmonics(df)
  df['lat_deg'] = df['lat_e7'] / 1e7
  df['abs_lat_deg'] = df['lat_deg'].abs()
  df['abs_lat_deg_c'] = (df['abs_lat_deg'] - df['abs_lat_deg'].mean()) / (
      2 * df['abs_lat_deg'].std()
  )
  df['lat_deg_c'] = (df['lat_deg'] - df['lat_deg'].mean()) / (
      2 * df['lat_deg'].std()
  )
  ols = smf.ols(
      'avg_temp_mean ~ abs_lat_deg_c + (month_sin + month_cos + month_sin_2 +'
      ' month_cos_2) * lat_deg_c',
      data=df,
  ).fit()
  df['temp_res'] = ols.resid
  df['target_val'] = np.log(df[target]) if metric == 'hrv' else df[target]
  df['target_within'] = df['target_val'] - df.groupby('user_id')[
      'target_val'
  ].transform('mean')
  df['temp_res_within'] = df['temp_res'] - df.groupby('user_id')[
      'temp_res'
  ].transform('mean')
  df['age_group'] = assign_age_group(
      df['age'], SENSITIVITY_AGE_BINS, SENSITIVITY_AGE_LABELS
  )
  df['temp_res_within_round'] = df['temp_res_within'].round()
  df = df[
      (df['temp_res_within_round'] >= temp_range[0])
      & (df['temp_res_within_round'] <= temp_range[1])
  ].copy()
  if metric == 'hrv':
    df['within_plot'] = (np.exp(df['target_within']) - 1) * 100
  else:
    df['within_plot'] = df['target_within']
  agg = (
      df.groupby(['age_group', 'temp_res_within_round'], observed=True)[
          'within_plot'
      ]
      .agg(['mean', 'count'])
      .reset_index()
  )
  return agg[agg['count'] >= min_count]


def plot_within_user_sensitivity(
    agg_rhr: pd.DataFrame, agg_hrv: pd.DataFrame
) -> plt.Figure:
  """Plots within-user temperature sensitivity (fig:within_user_sensitivity).

  Mean within-user change of RHR (bpm) and HRV (%) against the within-user
  deviation in residualised temperature, one line per age group.

  Args:
    agg_rhr: `compute_within_user_sensitivity(df_rhr, 'rhr')`.
    agg_hrv: `compute_within_user_sensitivity(df_hrv, 'hrv')`.

  Returns:
    1x2 figure (14 x 5.5 in) with one line per age group and a shared legend
    below.
  """
  _style_like_source('ticks')
  fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)
  panels = [
      (
          axes[0],
          agg_rhr,
          'o',
          '(a) Resting Heart Rate (RHR) Sensitivity',
          'Within-Subject Change (bpm)',
      ),
      (
          axes[1],
          agg_hrv,
          's',
          '(b) Heart Rate Variability (HRV) Sensitivity',
          'Within-Subject Change (%)',
      ),
  ]
  for ax, agg, marker, title, ylabel in panels:
    for age in SENSITIVITY_AGE_LABELS:
      g = agg[agg['age_group'] == age]
      ax.plot(
          g['temp_res_within_round'],
          g['mean'],
          label=age,
          color=SENSITIVITY_AGE_COLORS[age],
          marker=marker,
          markersize=4,
          linewidth=2,
      )
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(
        'Within-Subject Deviation in Residualised Temperature (°C)', fontsize=12
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(False)
    ax.tick_params(axis='both', which='major', labelsize=11)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      title='Age Group',
      loc='lower center',
      bbox_to_anchor=(0.5, -0.08),
      ncol=4,
      fontsize=11,
      title_fontsize=12,
  )
  fig.tight_layout()
  sns.despine(fig=fig)
  return fig


def make_within_user_sensitivity(
    data_dir: str,
    out_dir: Optional[str] = None,
    stem: str = 'within_user_sensitivity_combined',
) -> plt.Figure:
  """Loads both clean cohorts (no model outputs needed) and saves the figure.

  Produces fig:within_user_sensitivity.  Inputs: the clean user-month cohort
  frames of both metrics (see `cohort_io.load_cohort`), columns user_id,
  activity_period, age, lat_e7, avg_temp_mean, avg_resting_heart_rate_bpm |
  avg_hrv.

  Args:
    data_dir: dataset root.
    out_dir: output directory for `{stem}.{png,svg,pdf}`; None = do not save.
    stem: output file stem.

  Returns:
    The 1x2 within-user temperature-sensitivity figure (RHR left, HRV right).
  """
  aggs: Dict[str, pd.DataFrame] = {}
  for metric in ('rhr', 'hrv'):
    cols = [
        'user_id',
        'activity_period',
        'age',
        'lat_e7',
        'avg_temp_mean',
        METRIC_TARGET_COLS[metric],
    ]
    df = _load_cohort(data_dir, metric, cols)
    aggs[metric] = compute_within_user_sensitivity(df, metric)
    del df
    gc.collect()
  fig = plot_within_user_sensitivity(aggs['rhr'], aggs['hrv'])
  _save_fig(fig, out_dir, stem)
  return fig


# --- 13. fig:global_residuals_grid - world contour of model residuals ---


def aggregate_world_grid(
    df: pd.DataFrame, metric: str, step_deg: float = WORLD_GRID_STEP_DEG
) -> pd.DataFrame:
  """Per (lat_bin, lng_bin, month) means of the metric and the environment.

  Bins are `round(deg / step) * step`.

  Args:
    df: clean user-month cohort frame (user_id, lat_e7, lng_e7, activity_period,
      the metric target and any `WORLD_GRID_ENV_COLS`).
    metric: 'rhr' | 'hrv' (HRV is aggregated on the log scale, plus
      `hrv_actual_ms`).
    step_deg: grid cell size in degrees for latitude/longitude binning.

  Returns:
    Columns lat_bin, lng_bin, month, {metric}_actual, [hrv_actual_ms],
    {metric}_{suffix} per environmental column present and {metric}_users.
  """
  _check_metric(metric)
  work = pd.DataFrame({
      'lat_bin': (df['lat_e7'] / 1e7 / step_deg).round() * step_deg,
      'lng_bin': (df['lng_e7'] / 1e7 / step_deg).round() * step_deg,
      'month': df['activity_period'].dt.month,
      'user_id': df['user_id'],
  })
  aggs: Dict[str, Tuple[str, str]] = {}
  if metric == 'hrv':
    work['actual'] = np.log(df['avg_hrv'])
    work['actual_ms'] = df['avg_hrv']
    aggs['hrv_actual'] = ('actual', 'mean')
    aggs['hrv_actual_ms'] = ('actual_ms', 'mean')
  else:
    work['actual'] = df[METRIC_TARGET_COLS['rhr']]
    aggs['rhr_actual'] = ('actual', 'mean')
  for col, suffix in WORLD_GRID_ENV_COLS.items():
    if col in df.columns:
      work[suffix] = df[col]
      aggs[f'{metric}_{suffix}'] = (suffix, 'mean')
  aggs[f'{metric}_users'] = ('user_id', 'nunique')
  return work.groupby(['lat_bin', 'lng_bin', 'month']).agg(**aggs).reset_index()


def extract_contour_metadata(
    df: pd.DataFrame, n_sample: int = 200_000, random_state: int = 42
) -> Dict[str, Any]:
  """Standardisation statistics and temperature-residual OLS fits on a sample.

  The sample is `df.sample(n=n_sample, random_state=random_state)` of the FULL
  cohort (row order dependent); `predict_grid_from_coefficients` uses the
  result to rebuild the model features on grid cells.

  Args:
    df: full clean cohort with activity_period, lat_e7 and any of
      avg_temp_mean/max/min, dl_monthly (derived from latitude and month when
      missing).
    n_sample: number of rows sampled (capped at `len(df)`).
    random_state: seed for the row sample.

  Returns:
    {'stats': {feature}_mean/std and {feature}_sq_mean/std for latitude,
    |latitude|, day length and each temperature present; 'ols_models':
    `sl.residualise_temperature(sample, 'harmonic')`}.
  """
  sample = df.sample(n=min(n_sample, len(df)), random_state=random_state).copy()
  sl.add_month_harmonics(sample)
  sample['abs_lat_e7'] = sample['lat_e7'].abs()
  if 'dl_monthly' not in sample.columns:
    sample['dl_monthly'] = sl.mid_month_day_length_hours(
        sample['lat_e7'] / 1e7, sample['month']
    )
  summary = sl.standardise_model_features(
      sample,
      [
          'lat_e7',
          'abs_lat_e7',
          'dl_monthly',
          'avg_temp_mean',
          'avg_temp_max',
          'avg_temp_min',
      ],
  )
  ols_models = sl.residualise_temperature(sample, 'harmonic')
  return {'stats': summary, 'ols_models': ols_models}


def _standardise_grid_feature(
    g: pd.DataFrame, name: str, raw: pd.Series, summary: Mapping[str, float]
) -> None:
  """Adds `{name}_c` and `{name}_sq_c` to `g` from the cohort statistics."""
  g[f'{name}_c'] = (raw - summary[f'{name}_mean']) / (
      2 * summary[f'{name}_std']
  )
  g[f'{name}_sq_c'] = (raw**2 - summary[f'{name}_sq_mean']) / (
      2 * summary[f'{name}_sq_std']
  )


def predict_grid_from_coefficients(
    grid: pd.DataFrame,
    coef: pd.Series,
    metadata: Mapping[str, Any],
    metric: str,
) -> pd.Series:
  """Population-level prediction of the winning model on grid cells.

  Spatial, seasonal, day-length and temperature features (incl. squares and
  the latitude x season temperature residuals) are rebuilt from the cell
  centre, month and cell-mean environment using `metadata`; day length falls
  back to latitude and month when the grid has no cell mean. Demographic
  predictors are set to their centred mean (zero) and no random effects are
  added.

  Args:
    grid: rows with lat_bin, lng_bin, month and the `{metric}_{suffix}` cell
      means of `aggregate_world_grid`.
    coef: winner coefficients (e.g. `load_model_coefficients(...,
      'lme_results')`).
    metadata: output of `extract_contour_metadata`.
    metric: 'rhr' | 'hrv' (HRV prediction is on the log scale).

  Returns:
    Series aligned with `grid.index`.

  Raises:
    ValueError: if the coefficients use a predictor that cannot be rebuilt on
      the grid (missing cell-mean column or metadata).
  """
  _check_metric(metric)
  s = metadata['stats']
  ols = metadata['ols_models']
  g = pd.DataFrame(index=grid.index)
  lat_e7 = grid['lat_bin'] * 1e7
  _standardise_grid_feature(g, 'lat_e7', lat_e7, s)
  g['abs_lat_e7_c'] = (lat_e7.abs() - s['abs_lat_e7_mean']) / (
      2 * s['abs_lat_e7_std']
  )
  m = grid['month']
  g['month_sin'] = np.sin(2 * np.pi * m / 12)
  g['month_cos'] = np.cos(2 * np.pi * m / 12)
  g['month_sin_2'] = np.sin(2 * np.pi * m / 6)
  g['month_cos_2'] = np.cos(2 * np.pi * m / 6)
  lng_rad = grid['lng_bin'] * np.pi / 180
  g['sin_lng'] = np.sin(lng_rad)
  g['cos_lng'] = np.cos(lng_rad)
  day_length_col = f'{metric}_{WORLD_GRID_ENV_COLS["dl_monthly"]}'
  if 'dl_monthly_mean' in s:
    if day_length_col in grid.columns:
      day_length = grid[day_length_col]
    else:
      day_length = pd.Series(
          sl.mid_month_day_length_hours(grid['lat_bin'], m), index=grid.index
      )
    _standardise_grid_feature(g, 'dl_monthly', day_length, s)
  for base in ('avg_temp_mean', 'avg_temp_max', 'avg_temp_min'):
    col = f'{metric}_{WORLD_GRID_ENV_COLS[base]}'
    if col not in grid.columns or f'{base}_mean' not in s:
      continue
    _standardise_grid_feature(g, base, grid[col], s)
    if f'{base}_c' in ols:
      expected = sl.predict_from_coefficients(
          g, pd.Series(ols[f'{base}_c']['params'])
      )
      g[f'{base}_c_res'] = g[f'{base}_c'] - expected
      g[f'{base}_sq_c_res'] = g[f'{base}_c_res'] ** 2
  for col in ('age_c', 'age_sq_c', 'gender_n'):
    g[col] = 0.0
  try:
    return sl.predict_from_coefficients(g, coef)
  except KeyError as err:
    raise ValueError(
        'The winner formula uses predictors that cannot be rebuilt on the'
        f' world grid: {err}. Check the cell-mean columns of the grid and the'
        ' statistics in metadata.'
    ) from err


def compute_grid_residuals(
    grid: pd.DataFrame,
    coef: pd.Series,
    metadata: Mapping[str, Any],
    metric: str,
    min_months: int = 8,
) -> pd.DataFrame:
  """Mean residual per grid cell for cells observed in >= `min_months` months.

  RHR residual = observed - predicted (bpm); HRV residual =
  `(exp(observed_log - predicted_log) - 1) * 100` (percent), averaged over
  months.

  Args:
    grid: `aggregate_world_grid` output.
    coef: winner coefficients; see `predict_grid_from_coefficients`.
    metadata: output of `extract_contour_metadata`.
    metric: 'rhr' | 'hrv'.
    min_months: minimum months with data per cell.

  Returns:
    Columns lat_bin, lng_bin, resid.
  """
  actual_col = f'{metric}_actual'
  cells = grid.dropna(subset=[actual_col])
  counts = cells[['lat_bin', 'lng_bin']].value_counts()
  valid = counts[counts >= min_months].reset_index()[['lat_bin', 'lng_bin']]
  cells = cells.merge(valid, on=['lat_bin', 'lng_bin'], how='inner')
  pred = predict_grid_from_coefficients(cells, coef, metadata, metric)
  if metric == 'hrv':
    resid = (np.exp(cells[actual_col] - pred) - 1) * 100
  else:
    resid = cells[actual_col] - pred
  return (
      cells.assign(resid=resid)
      .groupby(['lat_bin', 'lng_bin'])['resid']
      .mean()
      .reset_index()
  )


def _interpolate_residual_surface(
    points: np.ndarray,
    values: np.ndarray,
    lng_grid: np.ndarray,
    lat_grid: np.ndarray,
    sigma: float = 1.5,
) -> np.ndarray:
  """Linear griddata interpolation, NaN-filled with the mean, Gaussian-smoothed, re-masked."""
  z = scipy.interpolate.griddata(
      points, values, (lng_grid, lat_grid), method='linear'
  )
  nan_mask = np.isnan(z)
  z_smoothed = scipy.ndimage.gaussian_filter(
      np.where(nan_mask, np.nanmean(values), z), sigma=sigma
  )
  z_smoothed[nan_mask] = np.nan
  return z_smoothed


def plot_world_contour_residuals(
    rhr_res: pd.DataFrame, hrv_res: pd.DataFrame, world_gdf: Any
) -> plt.Figure:
  """Plots world contour maps of model residuals (fig:global_residuals_grid).

  Observed-minus-predicted residuals per grid cell, interpolated to a smooth
  surface and drawn as filled contours over country outlines, for RHR (bpm)
  and HRV (%) side by side.

  Args:
    rhr_res: `compute_grid_residuals(..., 'rhr')` (bpm).
    hrv_res: `compute_grid_residuals(..., 'hrv')` (percent).
    world_gdf: GeoDataFrame of country polygons
      (`_read_geojson(ne_110_admin_0_countries.geojson)`); an empty frame skips
      the boundaries.

  Returns:
    1x2 figure (14 x 3.8 in).  RHR colour range is +-clip(p95(|resid|), 0.5,
    3.0) bpm
    (25 linear levels); HRV range is [max(p2, -30), 20] % with 0 centred
    (TwoSlopeNorm, reversed colormap).  Surfaces are linearly interpolated on a
    360 x 200 lon/lat mesh and Gaussian-smoothed (sigma 1.5).
  """
  _style_like_source('white')
  rhr_limit = float(
      min(
          max(np.percentile(np.abs(rhr_res['resid'].dropna().values), 95), 0.5),
          3.0,
      )
  )
  rhr_clipped = np.clip(rhr_res['resid'].values, -rhr_limit, rhr_limit)
  hrv_min = float(
      max(np.percentile(hrv_res['resid'].dropna().values, 2), -30.0)
  )
  hrv_max = 20.0
  hrv_clipped = np.clip(hrv_res['resid'].values, hrv_min, hrv_max)

  lng_grid, lat_grid = np.meshgrid(
      np.linspace(-180, 180, 360), np.linspace(-60, 85, 200)
  )
  z_rhr = _interpolate_residual_surface(
      rhr_res[['lng_bin', 'lat_bin']].values, rhr_clipped, lng_grid, lat_grid
  )
  z_hrv = _interpolate_residual_surface(
      hrv_res[['lng_bin', 'lat_bin']].values, hrv_clipped, lng_grid, lat_grid
  )
  rhr_levels = np.linspace(-rhr_limit, rhr_limit, 25)
  hrv_levels = np.concatenate(
      [np.linspace(hrv_min, 0, 13)[:-1], np.linspace(0, hrv_max, 13)]
  )

  fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
  ax_rhr, ax_hrv = axes
  contour_rhr = ax_rhr.contourf(
      lng_grid,
      lat_grid,
      z_rhr,
      levels=rhr_levels,
      cmap=CONTOUR_CMAP,
      alpha=0.85,
      extend='both',
  )
  contour_hrv = ax_hrv.contourf(
      lng_grid,
      lat_grid,
      z_hrv,
      levels=hrv_levels,
      cmap=CONTOUR_CMAP.reversed(),
      norm=TwoSlopeNorm(vcenter=0, vmin=hrv_min, vmax=hrv_max),
      alpha=0.85,
      extend='both',
  )
  for ax in axes:
    if world_gdf is not None and not world_gdf.empty:
      world_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.6)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-56, 85)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude (°)', fontsize=13, labelpad=2)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(False)
    sns.despine(ax=ax, left=True, bottom=True)
  ax_rhr.set_ylabel('Latitude (°)', fontsize=13)
  ax_hrv.set_ylabel('', fontsize=13)

  cbar_rhr = fig.colorbar(
      contour_rhr, ax=ax_rhr, orientation='horizontal', pad=0.22, shrink=0.6
  )
  cbar_rhr.set_label(
      'Residual: Actual - Predicted (bpm)', fontsize=12, labelpad=5
  )
  cbar_rhr.ax.tick_params(labelsize=10)
  cbar_rhr.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
  cbar_hrv = fig.colorbar(
      contour_hrv, ax=ax_hrv, orientation='horizontal', pad=0.22, shrink=0.6
  )
  cbar_hrv.set_label(
      'Residual: Observed vs Predicted (%)', fontsize=12, labelpad=5
  )
  cbar_hrv.ax.tick_params(labelsize=10)
  cbar_hrv.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))
  fig.subplots_adjust(bottom=0.15, top=0.95, left=0.05, right=0.95, wspace=0.02)
  return fig


def make_world_contour_residuals(
    data_dir: str,
    results_dir: str,
    extra_dir: str,
    out_dir: Optional[str] = None,
    *,
    rhr_subdir: str,
    hrv_subdir: str,
    coef_source: Union[str, Mapping[str, str]] = 'lme_results',
    geojson_name: str = 'ne_110_admin_0_countries.geojson',
    stem: str = 'world_contour_combined_residuals',
) -> plt.Figure:
  """Runs grid -> metadata -> coefficients -> residuals -> figure, then saves.

  Produces fig:global_residuals_grid.  Chain (one metric at a time to bound
  memory):
    `aggregate_world_grid` (thinned users excluded) ->
    feature statistics of the run (`feature_stats.json`, else
    `extract_contour_metadata`) ->
    `load_model_coefficients(source=coef_source)` (by default the `beta_std_*`
    of the lowest-BIC row of lme-models-results-{metric}.csv) ->
    `compute_grid_residuals` + `plot_world_contour_residuals`.

  Inputs:
    the clean user-month cohort frame of each metric (WORLD_GRID_COLS +
    target; see `cohort_io.load_cohort`),
    `{results_dir}/{subdir}/lme-models-results-{metric}.csv` (or the alternative
    chosen by `coef_source`), `{extra_dir}/{geojson_name}`.

  Args:
    data_dir: dataset root (clean cohort frames via `cohort_io.load_cohort`).
    results_dir: `{data_dir}/results`.
    extra_dir: directory holding the Natural Earth country GeoJSON.
    out_dir: output directory for `{stem}.{png,svg,pdf}`; None = do not save.
    rhr_subdir: RHR run tag (sub-directory of `results_dir` written by notebook
      02); keyword-only, no default.
    hrv_subdir: HRV run tag; keyword-only, no default.
    coef_source: see `load_model_coefficients` (str or per-metric mapping).
    geojson_name: file name of the country-outline GeoJSON in `extra_dir`.
    stem: output file stem.

  Returns:
    The 1x2 world residual contour figure (RHR bpm left, HRV percent right).
  """
  world_gdf = _read_geojson(os.path.join(extra_dir, geojson_name))
  residuals: Dict[str, pd.DataFrame] = {}
  for metric, subdir in (('rhr', rhr_subdir), ('hrv', hrv_subdir)):
    metric_results_dir = os.path.join(results_dir, subdir)
    cols = WORLD_GRID_COLS + [METRIC_TARGET_COLS[metric]]
    try:
      df = _load_cohort(data_dir, metric, cols)
    except KeyError:
      # Day length is derived from latitude and month when the cohort lacks it
      # (`compute_grid_residuals`); any other missing column raises here.
      df = _load_cohort(
          data_dir, metric, [c for c in cols if c != 'dl_monthly']
      )
    thinned = load_thinned_user_ids(metric_results_dir)
    if thinned is not None:
      df = drop_users(df, thinned)
    grid = aggregate_world_grid(df, metric)
    run_stats = sl.read_feature_stats(metric_results_dir)
    if run_stats is None:
      print(
          f'  {sl.FEATURE_STATS_FILE} not found in {metric_results_dir}:'
          ' feature statistics estimated on a cohort sample'
      )
      metadata = extract_contour_metadata(df)
    else:
      metadata = {
          'stats': run_stats['standardisation'],
          'ols_models': run_stats['temperature_ols'],
      }
    del df
    gc.collect()
    coef = load_model_coefficients(
        metric_results_dir,
        metric,
        _resolve_per_metric(coef_source, metric),
    )
    residuals[metric] = compute_grid_residuals(grid, coef, metadata, metric)
  fig = plot_world_contour_residuals(
      residuals['rhr'], residuals['hrv'], world_gdf
  )
  _save_fig(fig, out_dir, stem)
  return fig


# --- 14. fig:demographic_fit - calibration by sex x age bracket (RHR | HRV) ---


def plot_joint_demographics_calibration(
    rhr_summary: pd.DataFrame, hrv_summary: pd.DataFrame
) -> plt.Figure:
  """Plots model calibration by sex x age bracket (fig:demographic_fit).

  Observed monthly means (markers) and model predictions (lines) per
  sex x age-bracket cohort, for RHR and HRV side by side.

  Args:
    rhr_summary: `summarise_monthly_fit(df_rhr, ['gender', 'age_group'])` from a
      `mode='calibration'` frame (age_group = DEMOGRAPHIC_AGE_LABELS).
    hrv_summary: same for HRV (ms, smearing-corrected).

  Returns:
    1x2 figure (22 x 10 in): colour = age bracket, solid/'o' = Female,
    dashed/'s' = Male;
    shared legend "Cohorts (Gender: Age Bracket)" below the panels.
  """
  _style_like_source(None)
  fig, axes = plt.subplots(1, 2, figsize=(22, 10), sharex=True)
  panels = [
      (
          axes[0],
          rhr_summary,
          'Resting Heart Rate (bpm)',
          '(a) Resting Heart Rate (RHR) Model Calibration',
      ),
      (
          axes[1],
          hrv_summary,
          'Heart Rate Variability (ms)',
          '(b) Heart Rate Variability (HRV) Model Calibration',
      ),
  ]
  for ax, summary, ylabel, title in panels:
    for age in DEMOGRAPHIC_AGE_LABELS:
      for sex, style in DEMOGRAPHIC_SEX_STYLES.items():
        g = summary[
            (summary['gender'] == sex) & (summary['age_group'] == age)
        ].sort_values('month')
        if g.empty:
          continue
        color = DEMOGRAPHIC_AGE_COLORS[age]
        ax.errorbar(
            g['month'],
            g['actual_mean'],
            yerr=g['actual_se'] * 1.96,
            fmt=style['marker'],
            color=color,
            capsize=2,
            markersize=4,
            alpha=0.2,
            label='_nolegend_',
        )
        ax.plot(
            g['month'],
            g['predicted_mean'],
            color=color,
            linestyle=style['linestyle'],
            linewidth=1.0,
            label=f'{sex}: {age}',
        )
        ax.fill_between(
            g['month'],
            g['predicted_mean'] - 1.96 * g['predicted_se'],
            g['predicted_mean'] + 1.96 * g['predicted_se'],
            color=color,
            alpha=0.45,
        )
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS)
    ax.set_xlabel('Month', fontsize=14, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(False)
    ax.set_title(title, fontsize=15, pad=15)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(
      handles,
      labels,
      loc='lower center',
      ncol=6,
      bbox_to_anchor=(0.5, -0.05),
      fontsize=12,
      title='Cohorts (Gender: Age Bracket)',
      title_fontsize=13,
  )
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.12)
  return fig


def _assign_demographic_age_group(df: pd.DataFrame) -> pd.DataFrame:
  df['age_group'] = assign_age_group(
      df['age'], DEMOGRAPHIC_AGE_BINS, DEMOGRAPHIC_AGE_LABELS
  )
  return df.dropna(subset=['age_group']).copy()


def make_joint_demographics_calibration(
    data_dir: str,
    results_dir: str,
    out_dir: Optional[str] = None,
    *,
    rhr_subdir: str,
    hrv_subdir: str,
    n_users: Optional[int] = 20000,
    random_state: int = 42,
    coef_source: Optional[Union[str, Mapping[str, str]]] = None,
    temp_residual_method: str = 'harmonic',
    stem: str = 'demographic_fit',
    model_dirs: Optional[Mapping[str, str]] = None,
) -> plt.Figure:
  """Loads a user subsample per metric, predicts in calibration mode and saves.

  Produces fig:demographic_fit.  Predictors are rebuilt with the run's own
  statistics (see `load_fitted_frame`); the HRV smearing factor is estimated
  on the subsample after dropping ages outside (18, 78].  By default RHR
  coefficients are read from the winner directory's `model_coefficients.csv`
  and HRV from the interpreted CSV (`coef_source` default
  `{'rhr': 'model_dir', 'hrv': 'interpreted'}`).

  Inputs: see `load_fitted_frame`. `model_dirs` ({metric: folder}) optionally
  overrides `find_model_dir`.

  Args:
    data_dir: dataset root (clean cohort frames via `cohort_io.load_cohort`).
    results_dir: `{data_dir}/results`.
    out_dir: output directory for `{stem}.{png,svg,pdf}`; None = do not save.
    rhr_subdir: RHR run tag (sub-directory of `results_dir` written by notebook
      02); keyword-only, no default.
    hrv_subdir: HRV run tag; keyword-only, no default.
    n_users: number of users subsampled per metric; None = full cohort.
    random_state: seed for the user subsample.
    coef_source: see `load_model_coefficients` (str or per-metric mapping); None
      = `{'rhr': 'model_dir', 'hrv': 'interpreted'}`.
    temp_residual_method: fallback residualisation passed to
      `load_fitted_frame`.
    stem: output file stem.
    model_dirs: optional {metric: per-model folder} overriding `find_model_dir`
      (random effects, and coefficients when `coef_source='model_dir'`).

  Returns:
    The 1x2 sex x age-bracket calibration figure (RHR left, HRV right).
  """
  if coef_source is None:
    coef_source = {'rhr': 'model_dir', 'hrv': 'interpreted'}
  summaries: Dict[str, pd.DataFrame] = {}
  for metric, subdir in (('rhr', rhr_subdir), ('hrv', hrv_subdir)):
    df = load_fitted_frame(
        data_dir,
        os.path.join(results_dir, subdir),
        metric,
        mode='calibration',
        n_users=n_users,
        row_filter=_assign_demographic_age_group,
        temp_residual_method=temp_residual_method,
        coef_source=_resolve_per_metric(coef_source, metric),
        model_dir=(model_dirs or {}).get(metric),
        random_state=random_state,
    )
    summaries[metric] = summarise_monthly_fit(df, ['gender', 'age_group'])
    del df
    gc.collect()
  fig = plot_joint_demographics_calibration(summaries['rhr'], summaries['hrv'])
  _save_fig(fig, out_dir, stem)
  return fig


# --- 15. fig:rhr_residuals / fig:hrv_residuals - residuals vs fitted ---


def plot_model_residuals(
    fitted: np.ndarray,
    resid: np.ndarray,
    xlabel: str,
    ylabel: str = 'Residuals (Error)',
    n_points: int = 50000,
    random_state: int = 42,
    point_size: Optional[float] = None,
) -> plt.Figure:
  """Residuals-vs-fitted scatter of a seeded subsample (no title).

  Delta w.r.t. `seasonality_lib.plot_residuals`: takes arrays instead of a
  results object, seeded subsample (`RandomState(random_state).choice`),
  custom axis labels and sizes, no title, seaborn 'ticks' style, no
  `plt.show()`.

  Args:
    fitted: fitted values (fixed effects + random intercept).
    resid: observed - fitted (log scale for HRV).
    xlabel: e.g. 'Predicted Resting Heart Rate (bpm)' / 'Predicted log(HRV)'.
    ylabel: 'Residuals (bpm)' (RHR) / 'Residuals (Error)' (HRV).
    n_points: subsample size.
    random_state: seed.
    point_size: scatter `s`; None keeps the matplotlib default (36).

  Returns:
    Single-axes figure (8 x 5 in) with the scatter and a red dashed zero line.
  """
  _style_like_source('ticks')
  fitted = np.asarray(fitted)
  resid = np.asarray(resid)
  if len(fitted) > n_points:
    idx = np.random.RandomState(random_state).choice(
        len(fitted), n_points, replace=False
    )
    fitted, resid = fitted[idx], resid[idx]
  fig = plt.figure(figsize=(8, 5))
  ax = fig.gca()
  scatter_kwargs: Dict[str, Any] = {'alpha': 0.1}
  if point_size is not None:
    scatter_kwargs['s'] = point_size
  ax.scatter(fitted, resid, **scatter_kwargs)
  ax.axhline(0, color='red', linestyle='--')
  ax.set_xlabel(xlabel, fontsize=15)
  ax.set_ylabel(ylabel, fontsize=15)
  ax.tick_params(axis='both', which='major', labelsize=13)
  ax.grid(False)
  fig.tight_layout()
  return fig


def make_model_residuals_plot(
    data_dir: str,
    results_dir: str,
    metric: str,
    out_dir: Optional[str] = None,
    subdir: Optional[str] = None,
    stem: Optional[str] = None,
    **overrides: Any,
) -> plt.Figure:
  """Rebuilds the winner's conditional residuals and saves the scatter plot.

  Produces fig:{metric}_residuals.  Defaults (RESIDUAL_PLOT_CFG):
    RHR: 20k-user subsample (seed 42), coefficients from the winner
      directory (`coef_source='model_dir'`), s=1.
    HRV: full cohort, interpreted coefficients, default marker size;
      residual = target_log - fitted.
  Predictors are rebuilt with the run's statistics (see `load_fitted_frame`).
  Any key of RESIDUAL_PLOT_CFG plus `model_dir`, `random_state`, `n_points` may
  be overridden through `overrides`.

  Inputs: see `load_fitted_frame`.

  Args:
    data_dir: dataset root (clean cohort frames via `cohort_io.load_cohort`).
    results_dir: `{data_dir}/results`.
    metric: 'rhr' | 'hrv'; selects the RESIDUAL_PLOT_CFG defaults.
    out_dir: output directory for the figure files; None = do not save.
    subdir: results sub-directory; defaults to `metric`.
    stem: output file stem; defaults to the RESIDUAL_PLOT_CFG stem.
    **overrides: RESIDUAL_PLOT_CFG keys (n_users, temp_residual_method,
      coef_source, xlabel, ylabel, point_size, stem) plus `model_dir`,
      `random_state`, `n_points`.

  Returns:
    The residuals-vs-fitted scatter figure.
  """
  _check_metric(metric)
  cfg = dict(RESIDUAL_PLOT_CFG[metric])
  cfg.update(overrides)
  metric_results_dir = os.path.join(results_dir, subdir or metric)
  df = load_fitted_frame(
      data_dir,
      metric_results_dir,
      metric,
      mode='accuracy',
      n_users=cfg['n_users'],
      temp_residual_method=cfg['temp_residual_method'],
      coef_source=cfg['coef_source'],
      model_dir=cfg.get('model_dir'),
      random_state=cfg.get('random_state', 42),
  )
  target = 'target_log' if metric == 'hrv' else METRIC_TARGET_COLS['rhr']
  fig = plot_model_residuals(
      df['fitted_values'].values,
      (df[target] - df['fitted_values']).values,
      xlabel=cfg['xlabel'],
      ylabel=cfg['ylabel'],
      point_size=cfg['point_size'],
      n_points=cfg.get('n_points', 50000),
      random_state=cfg.get('random_state', 42),
  )
  _save_fig(fig, out_dir, stem or cfg['stem'])
  return fig


# --- 16. fig:rhr_qq / fig:hrv_qq - random-intercept histogram + Q-Q ---


def plot_random_intercepts(
    random_effects: np.ndarray,
    xlabel: str,
    qq_ylabel: str,
    n_hist: int = 50000,
    n_qq: int = 10000,
    random_state: int = 42,
) -> plt.Figure:
  """Histogram (+KDE) and normal Q-Q plot of the user random intercepts.

  Delta w.r.t. `seasonality_lib.plot_random_effects`: input is the saved
  `random_effects.parquet` values instead of a results object, figsize 15x6
  (not 15x5), no titles, custom axis labels/sizes, both subsamples drawn from
  one `RandomState(random_state)` stream.

  Args:
    random_effects: random intercepts (model scale).
    xlabel: histogram x label ('Deviation from Mean RHR (bpm)' / 'Deviation from
      Mean log(HRV)').
    qq_ylabel: Q-Q y label ('Ordered Values (bpm)' / 'Ordered Values
      (log(HRV))').
    n_hist: histogram subsample size (all values if fewer are available).
    n_qq: Q-Q plot subsample size (all values if fewer are available).
    random_state: seed of the shared `RandomState` used for both subsamples.

  Returns:
    1x2 figure (15 x 6 in): histogram with KDE (left), normal Q-Q plot (right).
  """
  _style_like_source('ticks')
  values = np.asarray(random_effects)
  rng = np.random.RandomState(random_state)
  fig, ax = plt.subplots(1, 2, figsize=(15, 6))
  hist_sample = (
      rng.choice(values, n_hist, replace=False)
      if len(values) > n_hist
      else values
  )
  sns.histplot(hist_sample, kde=True, ax=ax[0], color='skyblue')
  ax[0].set_xlabel(xlabel, fontsize=15)
  ax[0].set_ylabel('Count', fontsize=15)
  ax[0].tick_params(axis='both', which='major', labelsize=13)
  qq_sample = (
      rng.choice(values, n_qq, replace=False) if len(values) > n_qq else values
  )
  stats.probplot(qq_sample, dist='norm', plot=ax[1])
  ax[1].set_xlabel('Theoretical Quantiles', fontsize=15)
  ax[1].set_ylabel(qq_ylabel, fontsize=15)
  ax[1].tick_params(axis='both', which='major', labelsize=13)
  ax[1].set_title('')
  fig.tight_layout()
  return fig


def make_random_intercepts_plot(
    results_dir: str,
    metric: str,
    out_dir: Optional[str] = None,
    subdir: Optional[str] = None,
    model_dir: Optional[str] = None,
    stem: Optional[str] = None,
) -> plt.Figure:
  """Loads the winner's `random_effects.parquet` and saves fig:{metric}_qq.

  Inputs: `{results_dir}/{subdir}/{model_dir}/random_effects.parquet`
  (columns user_id, random_intercept); `model_dir` defaults to `find_model_dir`.

  Args:
    results_dir: `{data_dir}/results`.
    metric: 'rhr' | 'hrv'; selects RANDOM_EFFECTS_PLOT_CFG labels and stem.
    out_dir: output directory for the figure files; None = do not save.
    subdir: results sub-directory; defaults to `metric`.
    model_dir: per-model folder with `random_effects.parquet`; defaults to
      `find_model_dir(results_dir/subdir, metric)`.
    stem: output file stem; defaults to the RANDOM_EFFECTS_PLOT_CFG stem.

  Returns:
    The 1x2 histogram + normal Q-Q figure of the user random intercepts.
  """
  _check_metric(metric)
  cfg = RANDOM_EFFECTS_PLOT_CFG[metric]
  metric_results_dir = os.path.join(results_dir, subdir or metric)
  random_effects = load_random_effects(
      model_dir or find_model_dir(metric_results_dir, metric=metric)
  )
  fig = plot_random_intercepts(
      random_effects.values, cfg['xlabel'], cfg['qq_ylabel']
  )
  _save_fig(fig, out_dir, stem or cfg['stem'])
  return fig


# --- 17. Map constants and geographic reference helpers ---

# (season label, month-range label, months) used by the 2x2 observed maps.
MAP_SEASONS: List[Tuple[str, str, List[int]]] = [
    ('Winter', 'January–March', [1, 2, 3]),
    ('Spring', 'April–June', [4, 5, 6]),
    ('Summer', 'July–September', [7, 8, 9]),
    ('Autumn', 'October–December', [10, 11, 12]),
]

MAP_XLIM = (-180.0, 180.0)
MAP_YLIM = (-56.0, 85.0)

# Blue -> light blue -> light orange -> orange -> red.
OBSERVED_MAP_CMAP_COLORS = [
    '#1f78b4',
    '#a6cee3',
    '#fdbf6f',
    '#ff7f00',
    '#e31a1c',
]

OBSERVED_MAP_CBAR_LABELS = {
    'rhr': 'Observed Resting Heart Rate (BPM)',
    'hrv': 'Observed Heart Rate Variability (ms)',
}

# Fallback colour range when a metric has no data.
OBSERVED_MAP_FALLBACK_RANGE = {'rhr': (60.0, 72.0), 'hrv': (30.0, 55.0)}

WORLD_GEOJSON_FILE = 'ne_110_admin_0_countries.geojson'
UNSD_COUNTRIES_FILE = 'unsd-countries.csv'
# ISO-2 codes that the UNSD M49 table does not list but Natural Earth draws
# (ISO_A3_EH). Applied on top of the table by ``_load_iso2_to_iso3``.
ISO2_TO_ISO3_SUPPLEMENT: Dict[str, str] = {'TW': 'TWN'}


def _load_iso2_to_iso3(extra_dir: str) -> Dict[str, str]:
  """Loads the ISO-alpha2 -> ISO-alpha3 mapping.

  Input: `extra_dir/unsd-countries.csv` with columns 'ISO-alpha2 Code',
  'ISO-alpha3 Code', supplemented with ``ISO2_TO_ISO3_SUPPLEMENT`` for codes
  the UNSD table lacks (the table never overrides the supplement).

  Args:
    extra_dir: directory containing `unsd-countries.csv`.

  Returns:
    Dict mapping ISO-alpha2 country code to ISO-alpha3 country code.
  """
  unsd = _read_csv(
      os.path.join(extra_dir, UNSD_COUNTRIES_FILE),
      keep_default_na=False,
      na_values=[''],
      dtype=str,
  )
  mapping = unsd.set_index('ISO-alpha2 Code')['ISO-alpha3 Code'].to_dict()
  mapping.update(ISO2_TO_ISO3_SUPPLEMENT)
  return mapping


# Natural Earth's ISO_A3 is -99 for a few countries (e.g. France, Norway);
# ISO_A3_EH / ADM0_A3 are populated for all of them.
_ISO3_COLUMN_CANDIDATES = (
    'ISO_A3_EH',
    'iso_a3_eh',
    'ADM0_A3',
    'adm0_a3',
    'ISO_A3',
    'iso_a3',
    'ISO3',
    'iso3',
)


def _find_iso_col(world: Any) -> str:
  """Returns the ISO-3 column name of a Natural Earth GeoDataFrame."""
  for candidate in _ISO3_COLUMN_CANDIDATES:
    if candidate in world.columns:
      return candidate
  raise KeyError('No ISO-3 column found in world boundaries.')


def _load_world(extra_dir: str) -> Any:
  """Loads Natural Earth 110m admin-0 boundaries from `extra_dir/ne_110_admin_0_countries.geojson`."""
  return _read_geojson(os.path.join(extra_dir, WORLD_GEOJSON_FILE))


# --- 18. Producer: {metric}_validation_country_gender.csv ---


def aggregate_country_gender_validation(
    df: pd.DataFrame, metric: str
) -> pd.DataFrame:
  """Aggregates observed / predicted values by country x gender x month.

  Args:
    df: Frame with 'country_2_alpha_cd', 'gender', 'month', 'target_log',
      'user_id' and optionally 'predicted_val' (model scale).
    metric: 'rhr' or 'hrv' (HRV cell means are exponentiated back to ms, i.e. a
      geometric mean per cell).

  Returns:
    Columns: country_2_alpha_cd, gender, month, actual_mean, predicted_mean,
    n_users.
  """
  work = (
      df if 'predicted_val' in df.columns else df.assign(predicted_val=np.nan)
  )
  agg = (
      work.groupby(['country_2_alpha_cd', 'gender', 'month'])
      .agg(
          actual_mean=('target_log', 'mean'),
          predicted_mean=('predicted_val', 'mean'),
          n_users=('user_id', 'nunique'),
      )
      .reset_index()
  )
  if metric == 'hrv':
    agg['actual_mean'] = np.exp(agg['actual_mean'])
    agg['predicted_mean'] = np.exp(agg['predicted_mean'])
  return agg


def build_country_gender_validation(
    data_dir: str,
    metric: str,
    out_dir: Optional[str] = None,
    results_dir: Optional[str] = None,
    run_tag: Optional[str] = None,
    model_dir: Optional[str] = None,
    with_predictions: bool = True,
    temp_residual_method: str = 'harmonic',
    coef_source: str = 'interpreted',
    df: Optional[pd.DataFrame] = None,
    exclude_user_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
  """Builds `{metric}_validation_country_gender.csv`.

  Chain: the clean user-month cohort frame of `metric` (columns
  MODEL_INPUT_COLS + METRIC_TARGET_COLS[metric]; see `cohort_io.load_cohort`)
  -> `prepare_model_features` -> optional fixed-effects prediction using the
  winner coefficients (`load_model_coefficients` +
  `predict_from_coefficients`) -> `aggregate_country_gender_validation`.
  Only the fixed-effect coefficient table of the winning model is needed;
  neither the pickled model nor random effects are read. The observed 2x2
  maps use only `actual_mean` and `n_users`, so `with_predictions=False`
  suffices for them (and skips the feature engineering).

  Args:
    data_dir: Root data directory.
    metric: 'rhr' or 'hrv'.
    out_dir: If given, writes
      `{out_dir}/{metric}_validation_country_gender.csv`.
    results_dir: Model results root; defaults to `data_dir/results`.
    run_tag: Results sub-folder (run tag); defaults to metric.
    model_dir: Optional per-model folder (uses its model_coefficients.csv).
    with_predictions: If False, `predicted_mean` is NaN and no coefficients are
      read.
    temp_residual_method: Fallback residualisation, see `prepare_run_features`.
    coef_source: See `load_model_coefficients`.
    df: Optional in-memory cohort (e.g. the thinned RHR frame) used instead of
      loading the clean cohort; it is copied, not modified.
    exclude_user_ids: Users to drop so the validation matches the fitted cohort.
      When the cohort is loaded here the default is the run's
      `thinning_dropped_user_ids.csv` (if any); an in-memory `df` is used as is.

  Returns:
    The aggregated validation frame.
  """
  value_col = METRIC_TARGET_COLS[metric]
  metric_results_dir = os.path.join(
      results_dir or os.path.join(data_dir, 'results'), run_tag or metric
  )
  if df is None:
    if with_predictions:
      cols = MODEL_INPUT_COLS + [value_col]
    else:
      cols = [
          'user_id',
          'activity_period',
          'gender',
          'country_2_alpha_cd',
          value_col,
      ]
    df = _load_cohort(data_dir, metric, cols)
    if exclude_user_ids is None:
      exclude_user_ids = load_thinned_user_ids(metric_results_dir)
  else:
    df = df.copy()
  if exclude_user_ids is not None:
    df = drop_users(df, exclude_user_ids)
  if with_predictions:
    df = prepare_run_features(
        df, metric, metric_results_dir, temp_residual_method
    )
    coefs = load_model_coefficients(
        metric_results_dir,
        metric,
        source='model_dir' if model_dir else coef_source,
        model_dir=model_dir,
    )
    df['predicted_val'] = sl.predict_from_coefficients(df, coefs)
  else:
    # Observed means only: no standardisation / residualisation needed.
    df['month'] = df['activity_period'].dt.month
    df['target_log'] = (
        np.log(df[value_col])
        if metric == 'hrv'
        else df[value_col].astype(float)
    )
  agg = aggregate_country_gender_validation(df, metric)
  if out_dir:
    _write_csv(
        agg, os.path.join(out_dir, f'{metric}_validation_country_gender.csv')
    )
  return agg


# --- 19. Observed seasonal 2x2 maps (fig:{rhr,hrv}_observed_map) ---


def _month_to_season(month: int) -> str:
  for season, _, months in MAP_SEASONS:
    if month in months:
      return season
  return 'Other'


def aggregate_seasonal_country_values(
    val_df: pd.DataFrame,
    iso2_to_iso3: Dict[str, str],
    user_threshold: int = 100,
    value_col: str = 'actual_mean',
) -> pd.DataFrame:
  """Collapses country x gender x month validation rows to country x season.

  ISO-2 -> ISO-3 mapping, calendar-quarter seasons, `n_users`-weighted mean of
  `value_col` over the gender x month cells of each season (plain mean if the
  weights sum to zero), `n_users` summed over cells, then countries with
  summed n_users < threshold dropped. ISO-2 codes absent from `iso2_to_iso3`
  are dropped and listed on stdout.

  Args:
    val_df: Output of `build_country_gender_validation` (country_2_alpha_cd,
      gender, month, actual_mean, predicted_mean, n_users).
    iso2_to_iso3: Mapping from `_load_iso2_to_iso3`.
    user_threshold: Minimum summed n_users per country-season (default 100).
    value_col: Column to average ('actual_mean' for the observed maps).

  Returns:
    Columns: country_3_alpha_cd, season, val_col, n_users.
  """
  df = val_df.copy()
  df['country_3_alpha_cd'] = df['country_2_alpha_cd'].map(iso2_to_iso3)
  unmapped = df['country_3_alpha_cd'].isna()
  if unmapped.any():
    codes = sorted(df.loc[unmapped, 'country_2_alpha_cd'].astype(str).unique())
    print(f'  [maps] dropped {len(codes)} unmapped ISO-2 code(s): {codes}')
    df = df[~unmapped]
  df['season'] = df['month'].astype(int).map(_month_to_season)
  df['n_users'] = df['n_users'].fillna(0)
  df['_wv'] = df[value_col] * df['n_users']
  grp = df.groupby(['country_3_alpha_cd', 'season']).agg(
      _wv=('_wv', 'sum'), n_users=('n_users', 'sum'), _mean=(value_col, 'mean')
  )
  grp['val_col'] = np.where(
      grp['n_users'] > 0,
      grp['_wv'] / grp['n_users'].replace(0, np.nan),
      grp['_mean'],
  )
  out = grp.reset_index()[
      ['country_3_alpha_cd', 'season', 'val_col', 'n_users']
  ]
  return out[out['n_users'] >= user_threshold]


def plot_seasonal_map_2x2_figure(
    season_df: pd.DataFrame, world: Any, metric: str
) -> plt.Figure:
  """Draws the 2x2 observed seasonal choropleth for one metric.

  Args:
    season_df: Output of `aggregate_seasonal_country_values`.
    world: Natural Earth GeoDataFrame (ISO-3 column auto-detected).
    metric: 'rhr' or 'hrv' (selects colorbar label and fallback range).

  Returns:
    16x8.5in figure; panels Winter/Spring/Summer/Autumn with a shared horizontal
    colorbar spanning the 2nd-98th percentile of the plotted values.
  """
  setup_nature_style()
  iso_col = _find_iso_col(world)
  cmap = LinearSegmentedColormap.from_list(
      'google_cmap', OBSERVED_MAP_CMAP_COLORS, N=256
  )

  vals = season_df['val_col'].dropna()
  if vals.empty:
    vmin, vmax = OBSERVED_MAP_FALLBACK_RANGE[metric]
  else:
    vmin, vmax = float(np.percentile(vals, 2)), float(np.percentile(vals, 98))

  fig, axes = plt.subplots(
      2, 2, figsize=(16, 8.5), gridspec_kw={'hspace': 0.02, 'wspace': -0.03}
  )
  for idx, (season, months_label, _) in enumerate(MAP_SEASONS):
    ax = axes[idx // 2, idx % 2]
    merged = world.merge(
        season_df[season_df['season'] == season],
        left_on=iso_col,
        right_on='country_3_alpha_cd',
        how='left',
    )
    merged.plot(ax=ax, color='#e0e0e0', edgecolor='#ffffff', linewidth=0.4)
    merged.dropna(subset=['val_col']).plot(
        column='val_col',
        ax=ax,
        cmap=cmap,
        edgecolor='#ffffff',
        linewidth=0.4,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(
        f'{season} ({months_label})', fontsize=15, fontweight='bold', pad=8
    )
    ax.set_xlim(*MAP_XLIM)
    ax.set_ylim(*MAP_YLIM)
    ax.set_axis_off()

  sm = plt.cm.ScalarMappable(
      cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
  )
  sm.set_array([])
  cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
  cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
  cbar.set_label(
      OBSERVED_MAP_CBAR_LABELS[metric],
      fontsize=13,
      fontweight='bold',
      labelpad=8,
  )
  cbar.ax.tick_params(labelsize=12)
  return fig


def plot_seasonal_maps_2x2(
    data_dir: str,
    extra_dir: str,
    out_dir: Optional[str] = None,
    metrics: Sequence[str] = ('rhr', 'hrv'),
    validation_dir: Optional[str] = None,
    user_threshold: int = 100,
    validation_frames: Optional[Mapping[str, pd.DataFrame]] = None,
) -> Dict[str, plt.Figure]:
  """Generates the observed seasonal 2x2 choropleth maps for each metric.

  Produces fig:{rhr,hrv}_observed_map, saved as
  `{metric}_actual_all_seasonal_map_2x2`.

  Inputs:
    * The country x sex x month means of the observed values: taken from
      `validation_frames[metric]` when given, else read from
      `{validation_dir}/{metric}_validation_country_gender.csv`, else built on
      the fly with `build_country_gender_validation(...,
      with_predictions=False)`
      (`validation_dir` defaults to `data_dir/results/{metric}`).
    * `extra_dir/unsd-countries.csv`,
    `extra_dir/ne_110_admin_0_countries.geojson`.

  Args:
    data_dir: Root data directory (for the clean-cohort fallback).
    extra_dir: Directory with the reference tables / geojson.
    out_dir: Output directory for png/svg/pdf; None = do not save.
    metrics: Subset of ('rhr', 'hrv').
    validation_dir: Folder holding the validation CSVs (see above).
    user_threshold: Minimum summed users per country-season (default 100).
    validation_frames: metric -> in-memory output of
      `build_country_gender_validation` (skips the CSV round trip).

  Returns:
    {metric: figure}.
  """
  world = _load_world(extra_dir)
  iso2_to_iso3 = _load_iso2_to_iso3(extra_dir)
  figs = {}
  for metric in metrics:
    vdir = validation_dir or os.path.join(data_dir, 'results', metric)
    csv_path = os.path.join(vdir, f'{metric}_validation_country_gender.csv')
    if validation_frames is not None and metric in validation_frames:
      val_df = validation_frames[metric]
    elif fio.file_exists(csv_path):
      val_df = _read_csv(
          csv_path,
          keep_default_na=False,
          na_values=[''],
          dtype={'country_2_alpha_cd': str},
      )
    else:
      val_df = build_country_gender_validation(
          data_dir,
          metric,
          out_dir=vdir if validation_dir else None,
          with_predictions=False,
      )
    season_df = aggregate_seasonal_country_values(
        val_df, iso2_to_iso3, user_threshold=user_threshold
    )
    fig = plot_seasonal_map_2x2_figure(season_df, world, metric)
    _save_fig(fig, out_dir, f'{metric}_actual_all_seasonal_map_2x2')
    figs[metric] = fig
  return figs
