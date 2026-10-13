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

"""Helper functions for seasonality modeling, evaluation, and visualization."""

import datetime as dt
import gc
import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from scipy import stats
import scipy.linalg
import seaborn as sns
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf

import file_io as fio

# UN regions / countries with fewer distinct users are left out of the grouped
# seasonal-trend plots (``plot_seasonal_trends``).
MIN_USERS_PER_GROUP = 500


def compute_r2(result, df):
  """Prints and returns variance-decomposition fit metrics of a mixed model.

  Following Nakagawa & Schielzeth, the fixed-effects variance is the variance
  of the fixed-effects linear predictor, the random-intercept variance is read
  from ``result.cov_re`` and the residual variance from ``result.scale``.
  Also prints AIC, BIC, log-likelihood and the residual SD (``sqrt(scale)``).

  Args:
    result: Fitted statsmodels ``MixedLMResults`` with a single random
      intercept.
    df: Unused; kept for call-site compatibility with the notebooks.

  Returns:
    A tuple ``(r2_marginal, r2_conditional, icc)`` where the marginal R2 is
    the share of total variance explained by fixed effects, the conditional
    R2 the share explained by fixed plus random effects, and the ICC the
    intraclass correlation of the random intercept.
  """
  del df  # Unused.
  # 1. Get predicted values from Fixed Effects only
  # We use the design matrix from the result to be more robust
  fe_values = np.dot(result.model.exog, result.fe_params)
  var_fixed = np.var(fe_values)

  # 2. Extract Variance Components
  if hasattr(result.cov_re, 'iloc'):
    var_random = float(result.cov_re.iloc[0, 0])
  else:
    var_random = float(result.cov_re[0, 0])
  var_resid = result.scale

  # 3. Total Variance
  total_var = var_fixed + var_random + var_resid

  # 4. Calculations
  r2_marginal = var_fixed / total_var
  r2_conditional = (var_fixed + var_random) / total_var
  icc = var_random / (var_random + var_resid)
  resid_sd = np.sqrt(var_resid)

  print('--- R-Squared & ICC Report ---')
  print(f'Fixed Effect Var:  {var_fixed:.4f}')
  print(f'Random Intercept:  {var_random:.4f}')
  print(f'Residual (Noise):  {var_resid:.4f}')
  print(f'AIC:               {result.aic:.4f}')
  print(f'BIC:               {result.bic:.4f}')
  print(f'Log-Likelihood:    {result.llf:.4f}')
  print('-' * 30)
  print(f'Marginal R2:       {r2_marginal:.4f}')
  print(f'Conditional R2:    {r2_conditional:.4f}')
  print(f'ICC:               {icc:.4f}')
  print(f'Residual SD:       {resid_sd:.4f}')

  return r2_marginal, r2_conditional, icc


def plot_residuals(result, col_name, df=None):
  """Plots residuals vs fitted values to check for homoscedasticity."""
  # Looking for an egg shape.

  fig = plt.figure(figsize=(8, 5))

  has_cached_resid = 'resid' in result.__dict__ or (
      'resid' in getattr(result, '_cache', {})
  )

  if has_cached_resid:
    preds = result.predict()
    resid = result.resid
  elif df is not None:
    print(
        '  [Plot Residuals] Computing residuals on provided DataFrame to avoid'
        ' full recomputation...'
    )
    preds = result.predict(exog=df)
    actual_col = result.model.endog_names
    actual = df[actual_col]

    if 'user_id' in df.columns:
      user_col = 'user_id'
    elif 'user_id_n' in df.columns:
      user_col = 'user_id_n'
    else:
      raise ValueError(
          'Could not find user_id or user_id_n in DataFrame for mapping random'
          ' effects.'
      )

    unique_df_users = df[user_col].unique()
    re_dict = {}
    for user_id in unique_df_users:
      if user_id in result.random_effects:
        v = result.random_effects[user_id]
        if isinstance(v, (list, np.ndarray)):
          re_dict[user_id] = v[0]
        elif hasattr(v, 'iloc'):
          re_dict[user_id] = v.iloc[0]
        elif hasattr(v, '__getitem__'):
          re_dict[user_id] = v[0]
        else:
          re_dict[user_id] = v

    re_val = df[user_col].map(re_dict).fillna(0.0)
    fitted = preds + re_val
    resid = actual - fitted
  else:
    print(
        '  [Plot Residuals] WARNING: No cached residuals and no DataFrame'
        ' provided. This might be slow!'
    )
    preds = result.predict()
    resid = result.resid

  if len(preds) > 50000:
    preds_arr = np.asarray(preds)
    resid_arr = np.asarray(resid)
    idx = np.random.choice(len(preds_arr), 50000, replace=False)
    plt.scatter(preds_arr[idx], resid_arr[idx], alpha=0.1)
  else:
    plt.scatter(preds, resid, alpha=0.1)
  plt.axhline(0, color='red', linestyle='--')
  plt.xlabel(f'Predicted {col_name}')
  plt.ylabel('Residuals (Error)')
  plt.title('Residuals vs. Fitted Values')
  plt.show()

  return fig


def plot_random_effects(result, col_name):
  """Plots the random effects of the model.

  We are checking whether the random intercept is normally distributed.
  If not, that means there is a subgroup or an effect that we are missing.
  Draws a histogram with KDE and a normal Q-Q plot of the per-user random
  intercepts (subsampled to 50,000 and 10,000 users respectively).

  Args:
    result: Fitted statsmodels mixed-model results object whose
      ``random_effects`` maps each user to its random-intercept estimate.
    col_name: Human-readable name of the modelled metric, used in axis labels.

  Returns:
    A matplotlib figure with the two subplots.
  """
  # 1. Extract the random effects (it returns a dict of {user_id: [value]})
  random_effects = [val[0] for val in result.random_effects.values()]

  # 2. Create a side-by-side plot: Histogram and Q-Q Plot
  fig, ax = plt.subplots(1, 2, figsize=(15, 5))

  # Subsample for histogram as well if there are too many to avoid slow plotting
  if len(random_effects) > 50000:
    print(
        '  [Plot Random Effects] Subsampling random effects to 50,000 for'
        ' histogram plotting...'
    )
    np.random.seed(42)
    hist_sample = np.random.choice(random_effects, 50000, replace=False)
  else:
    hist_sample = random_effects

  # Histogram
  sns.histplot(hist_sample, kde=True, ax=ax[0], color='skyblue')
  ax[0].set_title('Distribution of Random Intercepts (Users)')
  ax[0].set_xlabel(f'Deviation from Mean {col_name}')

  # Q-Q Plot (Points should follow the red line)
  if len(random_effects) > 10000:
    re_sample = np.random.choice(random_effects, 10000, replace=False)
    stats.probplot(re_sample, dist='norm', plot=ax[1])
  else:
    stats.probplot(random_effects, dist='norm', plot=ax[1])
  ax[1].set_title('Normal Q-Q Plot of Random Effects')

  plt.tight_layout()
  plt.show()

  return fig


def coefficient_table(result: Any) -> pd.DataFrame:
  """Coefficients, standard errors, z, p and 95 % CI of a fitted model.

  Fixed-effect rows are taken from the result as is. Random-effect rows (the
  entries of ``result.params`` after the fixed effects, e.g. ``user_id Var``)
  hold the variance *relative* to the residual variance; as in statsmodels'
  summary they are rescaled by ``result.scale`` so the table shows the
  variance itself, and ``z``, ``P>|z|`` and the CI are left blank for them
  (a Wald test of a variance against zero is not meaningful).

  Args:
    result: A fitted ``MixedLMResults`` (statsmodels or binned).

  Returns:
    Frame indexed by parameter name with columns ``coef``, ``std_err``, ``z``,
    ``P>|z|``, ``[0.025`` and ``0.975]``.
  """
  ci = np.asarray(result.conf_int(), dtype=float)
  table = pd.DataFrame(
      {
          'coef': np.asarray(result.params, dtype=float),
          'std_err': np.asarray(result.bse, dtype=float),
          'z': np.asarray(result.tvalues, dtype=float),
          'P>|z|': np.asarray(result.pvalues, dtype=float),
          '[0.025': ci[:, 0],
          '0.975]': ci[:, 1],
      },
      index=list(result.params.index),
  )
  k_fe = len(getattr(result, 'fe_params', result.params))
  if k_fe < len(table):
    scale = float(result.scale)
    table.iloc[k_fe:, table.columns.get_loc('coef')] *= scale
    table.iloc[k_fe:, table.columns.get_loc('std_err')] *= scale
    table.iloc[k_fe:, 2:] = np.nan
  return table


def summary_report(result, df, col_name):
  """Prints the coefficient table plus fit metrics for ``result``."""
  print(coefficient_table(result).to_string(float_format='{:.4f}'.format))

  compute_r2(result, df)

  fig1 = plot_random_effects(result, col_name)

  fig2 = plot_residuals(result, col_name, df)

  return fig1, fig2


def get_gvif_groups(column_names):
  """Groups regression columns into distinct mathematical term families.

  Month harmonics (1st and 2nd order) are grouped together into a single block
  of size 4. Other harmonic pairs (sin/cos) are bundled together into their
  respective term blocks, dummy-encoded levels of one categorical term (patsy
  ``term[T.level]`` names) form one block, and every remaining column is its
  own block. The 'Intercept' column is dropped.

  Args:
    column_names: Design-matrix column names (e.g. ``result.model.exog_names``).

  Returns:
    A list of lists of column names, one inner list per term family.
  """
  # Clean out intercept if present
  cols = [c for c in column_names if c != 'Intercept']

  groups = []
  processed = set()

  # 1. Identify and group monthly harmonics (1st and 2nd order sin/cos) together
  month_harmonics = [
      c
      for c in cols
      if 'month' in c.lower() and ('sin' in c.lower() or 'cos' in c.lower())
  ]
  if month_harmonics:
    groups.append(month_harmonics)
    processed.update(month_harmonics)

  # 2. Identify and group other isolated harmonic pairs (e.g. sin/cos_lng)
  for col in cols:
    if col in processed:
      continue

    if 'sin' in col.lower():
      counterpart = col.replace('sin', 'cos').replace('Sin', 'Cos')
      if counterpart in cols and counterpart not in processed:
        pair = [col, counterpart]
        groups.append(pair)
        processed.update(pair)
    elif 'cos' in col.lower():
      counterpart = col.replace('cos', 'sin').replace('Cos', 'Sin')
      if counterpart in cols and counterpart not in processed:
        pair = [col, counterpart]
        groups.append(pair)
        processed.update(pair)

  # 3. Group everything else by its exact term string structure
  #    (handling dummy encodings)
  for col in cols:
    if col in processed:
      continue

    if '[T.' in col:
      # Example: 'un_region[T.Europe]' -> base term 'un_region'
      base_term = col.split('[')[0]
      term_group = []
      for c in cols:
        if c in processed:
          continue
        if c.split('[')[0] == base_term:
          term_group.append(c)

      if term_group:
        groups.append(term_group)
        processed.update(term_group)
    else:
      groups.append([col])
      processed.add(col)

  return groups


def compute_model_gvif(result):
  """Computes GVIF for main effects directly from the fit result.

  Excludes interaction terms and squares of variables to capture the underlying
  multicollinearity of independent main predictors.

  Args:
    result: The fitted model results object.

  Returns:
    A pandas DataFrame containing 'feature', 'GVIF_Adj', and 'Family_Size'.
  """
  cols_all = result.model.exog_names
  cols_vif = [
      c
      for c in cols_all
      if c != 'Intercept'
      and ':' not in c
      and 'sq' not in c.lower()
      and '**2' not in c
  ]
  if not cols_vif:
    return pd.DataFrame(columns=['feature', 'GVIF_Adj', 'Family_Size'])

  col_indices = [cols_all.index(c) for c in cols_vif]

  # Sample for speed on large datasets
  sample_n = min(len(result.model.exog), 100000)
  rng = np.random.default_rng(42)
  sample_idx = rng.choice(len(result.model.exog), size=sample_n, replace=False)

  # Slice directly on numpy array, then build the small DataFrame
  x_sample = pd.DataFrame(
      result.model.exog[sample_idx][:, col_indices], columns=cols_vif
  ).astype(float)

  groups = get_gvif_groups(cols_vif)

  gvif_list = []
  # Add a tiny amount of jitter (1e-9) to the diagonal for numerical stability
  corr_matrix = x_sample.corr().values + np.eye(x_sample.shape[1]) * 1e-9

  try:
    det_full = scipy.linalg.det(corr_matrix)
  except (ValueError, np.linalg.LinAlgError):
    det_full = 1e-10

  for group in groups:
    indices = [cols_vif.index(c) for c in group]
    other_indices = [i for i in range(len(cols_vif)) if i not in indices]

    det_sub = scipy.linalg.det(
        x_sample[group].corr().values + np.eye(len(group)) * 1e-9
    )
    if other_indices:
      det_others = scipy.linalg.det(
          x_sample.iloc[:, other_indices].corr().values
          + np.eye(len(other_indices)) * 1e-9
      )
    else:
      det_others = 1.0

    gvif = (det_sub * det_others) / det_full
    df = len(group)
    gvif_adj = gvif ** (1 / (2 * df)) if gvif > 0 else np.nan

    for col in group:
      gvif_list.append(
          {'feature': col, 'GVIF_Adj': gvif_adj, 'Family_Size': df}
      )

  return pd.DataFrame(gvif_list).drop_duplicates(subset=['feature'])


def extract_model_stats(
    result, model_name, formula, data, target_col, scaling_factors=None
):
  """Extracts model stats and GVIF for a given model result.

  Args:
    result: The model result to extract stats from.
    model_name: The name of the model.
    formula: The formula used for the model.
    data: The data used for the model.
    target_col: The target column of the model.
    scaling_factors: A dictionary of scaling factors for the model.

  Returns:
    A tuple of (model_row, vif_df). ``log_likelihood``, ``AIC`` and ``BIC``
    are stored with their true signs.
  """

  # 1. Basic Metrics & Variance Components
  aic, bic, llf = result.aic, result.bic, result.llf
  var_resid = result.scale

  re_matrix = np.asarray(result.cov_re)
  var_random_effect = float(re_matrix[0, 0])

  icc = var_random_effect / (var_random_effect + var_resid)

  # 2. Fixed Effects Variance & Nakagawa R-squared
  fixed_pred = result.predict()
  var_fixed = np.var(fixed_pred)

  total_var = var_fixed + var_random_effect + var_resid
  marginal_r2 = var_fixed / total_var
  conditional_r2 = (var_fixed + var_random_effect) / total_var

  # RMSE (Absolute Accuracy)
  rmse = np.sqrt((result.resid**2).mean())

  # 3. Calibration Metric: Within-Person R2 (vectorised via np.bincount)

  group_codes = data['user_id_n'].values.astype(np.int64)
  counts = np.bincount(group_codes)

  actual_vals = data[target_col].values
  actual_sums = np.bincount(group_codes, weights=actual_vals)
  actual_means = actual_sums / counts
  actual_centered = actual_vals - actual_means[group_codes]
  del actual_sums, actual_means
  gc.collect()

  pred_sums = np.bincount(group_codes, weights=fixed_pred)
  pred_means = pred_sums / counts
  preds_centered = fixed_pred - pred_means[group_codes]
  del pred_sums, pred_means
  gc.collect()

  r_within = np.corrcoef(actual_centered, preds_centered)[0, 1]
  r2_within = r2_score(actual_centered, preds_centered)
  del actual_centered, preds_centered, group_codes, counts
  gc.collect()

  # 4. Parameter Dictionary Preparation
  params = result.params.to_dict()
  hessian_skipped = getattr(result, 'hessian_skipped', False)
  conf_int = None if hessian_skipped else result.conf_int()

  row = {
      'model': model_name,
      'formula': formula,
      'converged': bool(getattr(result, 'converged', True)),
      'fixed_effect_var': var_fixed,
      'random_intercept_var': var_random_effect,
      'residual_noise_var': var_resid,
      'rmse': rmse,
      'AIC': aic,
      'BIC': bic,
      'ICC': icc,
      'log_likelihood': llf,
      'marginal_R2': marginal_r2,
      'conditional_R2': conditional_r2,
      'r_within': r_within,
      'r2_within': r2_within,
  }

  # 5. Fixed Effects & Gelman Back-calculation
  for name, val in params.items():
    if hessian_skipped or conf_int is None:
      ci_low, ci_high = np.nan, np.nan
    else:
      ci_low, ci_high = conf_int.loc[name, 0], conf_int.loc[name, 1]
    denom = raw_scale_divisor(name, scaling_factors or {})
    row[f'beta_std_{name}'] = val
    row[f'ci_low_std_{name}'] = ci_low
    row[f'ci_high_std_{name}'] = ci_high
    row[f'beta_raw_{name}'] = val / denom
    row[f'ci_low_raw_{name}'] = ci_low / denom
    row[f'ci_high_raw_{name}'] = ci_high / denom

  # 6. GVIF Analysis (Collinearity Check)
  vif_df = compute_model_gvif(result)

  return row, vif_df


def plot_seasonal_trends(
    df,
    result,
    col='hrv_log',
    col_name='HRV',
    group_col=None,  # Now defaults to None
    use_random_effects=False,
    boxcox_lambda=None,
    title_fontsize=14,
    label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    min_users_per_group=None,
):
  """Plots seasonal trends for a given column.

  Args:
    df: The dataframe to plot.
    result: The model result to plot.
    col: The column to plot.
    col_name: The name of the column to plot.
    group_col: The column to group by.
    use_random_effects: Whether to use random effects.
    boxcox_lambda: Lambda parameter for inverse Box-Cox transform.
    title_fontsize: Font size of the figure title.
    label_fontsize: Font size of the x- and y-axis labels.
    tick_fontsize: Font size of the major tick labels.
    legend_fontsize: Font size of the legend entries and legend title.
    min_users_per_group: UN regions / countries with fewer distinct users are
      not plotted (only applies when ``group_col`` is ``un_region`` or
      ``country_2_alpha_cd``). ``None`` uses the module-level
      ``MIN_USERS_PER_GROUP`` as it is at call time.

  Returns:
    A matplotlib figure.
  """

  df = df.copy()
  if min_users_per_group is None:
    min_users_per_group = MIN_USERS_PER_GROUP

  if (
      group_col in ['un_region', 'country_2_alpha_cd']
      and 'user_id_n' in df.columns
  ):
    user_counts = df.groupby(group_col)['user_id_n'].nunique()
    keep_groups = user_counts[user_counts >= min_users_per_group].index
    df = df[df[group_col].isin(keep_groups)]

  # --- 1. Handle Population Level Logic ---
  # If group_col is None, create a dummy group for the whole population
  actual_group_col = group_col if group_col is not None else 'Population'
  if group_col is None:
    df[actual_group_col] = 'All Users'

  # --- 2. Logic for extracting values and back-transformation ---
  is_log = (boxcox_lambda == 0) or ('log' in col)

  if is_log and result is not None:
    # Ensure we have predicted_val and fitted_values
    if 'predicted_val' not in df.columns:
      df['predicted_val'] = result.predict(df)
    if 'fitted_values' not in df.columns:
      df['fitted_values'] = (
          result.fittedvalues
          if hasattr(result, 'fittedvalues')
          else df['predicted_val']
      )

    df['re'] = df['fitted_values'] - df['predicted_val']

    # Calculate residuals: e = y - fitted_values
    log_residuals = df[col] - df['fitted_values']
    # Duan's Smearing Estimator: mean(exp(residuals))
    smearing_factor = np.mean(np.exp(log_residuals))
    print(
        "  [Duan's Smearing Estimator] Calculated correction factor:"
        f' {smearing_factor:.6f}'
    )

    if use_random_effects:
      print(
          f'Using random effects (Total Accuracy Mode) - {actual_group_col}'
          ' Level (Duan Smearing Corrected)'
      )
      df['plot_pred_final'] = np.exp(df['fitted_values']) * smearing_factor
      df['plot_actual_final'] = np.exp(df[col])  # Raw HRV
      y_label = f'{col_name} (ms)'
    else:
      print(
          f'Using fixed effects only (Calibration Mode) - {actual_group_col}'
          ' Level (Duan Smearing Corrected, Population-Adjusted)'
      )
      # Remove random effects from actuals to show clean population
      # calibration in absolute units.
      df['plot_actual_adj'] = df[col] - df['re']
      df['plot_actual_final'] = np.exp(df['plot_actual_adj'])
      df['plot_pred_final'] = np.exp(df['predicted_val']) * smearing_factor
      y_label = f'{col_name} (ms)'
  else:
    # Original logic for non-log or if result is None
    if use_random_effects:
      print(
          f'Using random effects (Total Accuracy Mode) - {actual_group_col}'
          ' Level'
      )
      if 'fitted_values' in df.columns:
        df['plot_pred'] = df['fitted_values']
      else:
        df['plot_pred'] = result.fittedvalues if result is not None else df[col]
      df['plot_actual'] = df[col]
    else:
      print(
          f'Using fixed effects only (Calibration Mode) - {actual_group_col}'
          ' Level'
      )
      if 'predicted_val' in df.columns:
        raw_preds = df['predicted_val']
      else:
        raw_preds = (
            pd.Series(np.asarray(result.predict(df)), index=df.index)
            if result is not None
            else df[col]
        )

      # Centre predictions and observations on the same key (per user when
      # available, else per group) so that partial-year panels do not shrink
      # the observed swing relative to the predicted one.
      if 'user_id_n' in df.columns:
        centre_key = df['user_id_n']
      else:
        centre_key = df[actual_group_col]
      df['plot_pred'] = raw_preds - raw_preds.groupby(centre_key).transform(
          'mean'
      )
      df['plot_actual'] = df[col] - df[col].groupby(centre_key).transform(
          'mean'
      )

    # Handle Back-Transformation (Log or Box-Cox)
    if boxcox_lambda is not None:
      if boxcox_lambda == 0:
        df['plot_pred_final'] = np.exp(df['plot_pred'])
        df['plot_actual_final'] = np.exp(df['plot_actual'])
      else:
        df['plot_pred_final'] = (df['plot_pred'] * boxcox_lambda + 1) ** (
            1 / boxcox_lambda
        )
        df['plot_actual_final'] = (df['plot_actual'] * boxcox_lambda + 1) ** (
            1 / boxcox_lambda
        )
      y_label = f'{col_name} (Back-Transformed)'
    elif 'log' in col:
      df['plot_pred_final'] = np.exp(df['plot_pred'])
      df['plot_actual_final'] = np.exp(df['plot_actual'])
      y_label = f'{col_name} (Ratio to Mean)'
    else:
      df['plot_pred_final'] = df['plot_pred']
      df['plot_actual_final'] = df['plot_actual']
      y_label = f'Δ {col_name}'

  # --- 3. Summarize for Plotting ---
  # observed=True: with pd.cut groups and a small cohort, unobserved age bins
  # would otherwise appear in the summary but not in the colour map.
  monthly_summary = (
      df.groupby([actual_group_col, 'month'], observed=True)
      .agg({
          'plot_actual_final': ['mean', 'std', 'count'],
          'plot_pred_final': ['mean', 'std', 'count'],
      })
      .reset_index()
  )

  monthly_summary.columns = [
      actual_group_col,
      'month',
      'actual_mean',
      'actual_std',
      'n',
      'predicted_mean',
      'predicted_std',
      'pred_count',
  ]
  monthly_summary['actual_se'] = monthly_summary['actual_std'] / (
      monthly_summary['n'] ** 0.5
  )
  monthly_summary['predicted_se'] = monthly_summary['predicted_std'] / (
      monthly_summary['pred_count'] ** 0.5
  )

  # --- 4. Setup Plot ---
  fig = plt.figure(figsize=(12, 7))
  if monthly_summary.empty:
    # No group reached min_users_per_group: return an annotated empty figure.
    msg = f'No {actual_group_col} group with enough users to plot.'
    print(f'  [plot_seasonal_trends] {msg}')
    plt.text(0.5, 0.5, msg, ha='center', va='center')
    plt.axis('off')
    return fig
  unique_groups = df[actual_group_col].unique()
  palette = sns.color_palette('husl', len(unique_groups))
  color_map = dict(zip(unique_groups, palette))

  # A. Observed Means
  for grp in unique_groups:
    grp_data = monthly_summary[monthly_summary[actual_group_col] == grp]
    if not grp_data.empty:
      plt.errorbar(
          grp_data['month'],
          grp_data['actual_mean'],
          yerr=grp_data['actual_se'] * 1.96,
          fmt='o',
          color=color_map[grp],
          capsize=5,
          markersize=8,
          alpha=0.4,
          label=f'{grp} n={int(grp_data["n"].mean())}',
      )

  # B. Model Trajectory
  # Plot using the aggregated data to avoid slow bootstrapping on full data
  sns.lineplot(
      data=monthly_summary,
      x='month',
      y='predicted_mean',
      hue=actual_group_col,
      hue_order=list(unique_groups),  # categorical hue: only observed levels
      palette=color_map,
      linewidth=3,
      errorbar=None,
      legend=False,
  )

  # Add CI band manually
  for grp in unique_groups:
    grp_data = monthly_summary[monthly_summary[actual_group_col] == grp]
    if not grp_data.empty:
      plt.fill_between(
          grp_data['month'],
          grp_data['predicted_mean'] - 1.96 * grp_data['predicted_se'],
          grp_data['predicted_mean'] + 1.96 * grp_data['predicted_se'],
          color=color_map[grp],
          alpha=0.15,
      )

  # --- 5. Final Formatting ---
  mode_title = (
      'Fixed Effects (Calibration)'
      if not use_random_effects
      else 'Mixed Effects (Accuracy)'
  )
  level_title = 'Population' if group_col is None else f'Group: {group_col}'

  plt.xticks(range(1, 13))
  plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
  plt.title(
      f'Seasonal {col_name} Trends [{mode_title}]\n{level_title} Level'
      ' Trajectory',
      fontsize=title_fontsize,
      pad=15,
  )
  plt.xlabel('Month of Year', fontsize=label_fontsize)
  plt.ylabel(y_label, fontsize=label_fontsize)

  if group_col is not None:
    plt.legend(
        title=group_col,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
    )

  plt.grid(axis='y', linestyle='--', alpha=0.3)
  plt.tight_layout()

  return fig


def save_aggregate_validation(
    df,
    result,
    group_col,
    target_col,
    title_prefix,
    include_month=True,
    boxcox_lambda=None,
):
  """Plots observed vs. fixed-effects-predicted group means for calibration.

  Observations are aggregated by ``group_col`` (and month, if requested),
  groups with too few users are dropped (10 for country codes, 50
  otherwise), and the means are back-transformed to the original scale via
  the inverse Box-Cox or exponential transform where applicable. The Pearson
  correlation and the calibration slope (OLS of observed on predicted) are
  reported in the figure title alongside a 45-degree ideal line.

  Args:
    df: Observation-level frame with ``target_col``, ``user_id_n`` and
      ``group_col`` (plus ``month`` if ``include_month``). If it lacks a
      ``predicted_val`` column, fixed-effects predictions are computed from
      ``result``.
    result: Fitted statsmodels mixed-model results object.
    group_col: Column defining the aggregation groups (e.g. 'un_region',
      'country_2_alpha_cd').
    target_col: Name of the modelled target column (on the model scale).
    title_prefix: Text prefixed to the figure title.
    include_month: Whether to also aggregate by calendar month.
    boxcox_lambda: Lambda of the Box-Cox transform applied to the target (0
      means log); None falls back to exp() if 'log' is in ``target_col``.

  Returns:
    A tuple ``(fig, pearson_r, calibration_slope, agg_df)``, or
    ``(None, 0, 0, None)`` if no group passes the user-count threshold.
  """
  df = df.copy()
  # Ensure we use fixed effects only for calibration testing
  if 'predicted_val' not in df.columns:
    df['predicted_val'] = result.predict(df)

  # 1. Flexible Grouping
  grouping_dims = [group_col]
  if include_month and 'month' in df.columns:
    grouping_dims.append('month')

  agg_df = (
      df.groupby(grouping_dims)
      .agg(
          {target_col: 'mean', 'predicted_val': 'mean', 'user_id_n': 'nunique'}
      )
      .reset_index()
  )

  agg_df.columns = grouping_dims + ['actual_mean', 'predicted_mean', 'n_users']
  threshold = 10 if group_col == 'country_2_alpha_cd' else 50
  agg_df = agg_df[agg_df['n_users'] >= threshold]

  if agg_df.empty:
    return None, 0, 0, None  # Added extra returns for slope and df

  if boxcox_lambda is not None:
    if boxcox_lambda == 0:
      agg_df['actual_mean'] = np.exp(agg_df['actual_mean'])
      agg_df['predicted_mean'] = np.exp(agg_df['predicted_mean'])
    else:
      agg_df['actual_mean'] = (agg_df['actual_mean'] * boxcox_lambda + 1) ** (
          1 / boxcox_lambda
      )
      agg_df['predicted_mean'] = (
          agg_df['predicted_mean'] * boxcox_lambda + 1
      ) ** (1 / boxcox_lambda)
  elif 'log' in target_col:
    agg_df['actual_mean'] = np.exp(agg_df['actual_mean'])
    agg_df['predicted_mean'] = np.exp(agg_df['predicted_mean'])

  # 2. Metrics: Pearson R AND Calibration Slope
  # Correlation (how tight the dots are)
  r_val, _ = stats.pearsonr(agg_df['predicted_mean'], agg_df['actual_mean'])

  # Calibration Slope (the angle of the dots)
  # y = actual_mean, x = predicted_mean
  slope, _, _, _, _ = stats.linregress(
      agg_df['predicted_mean'], agg_df['actual_mean']
  )

  # 3. Plotting
  fig = plt.figure(figsize=(8, 6))
  sns.scatterplot(
      data=agg_df,
      x='predicted_mean',
      y='actual_mean',
      hue=group_col,
      palette='viridis',
      alpha=0.6,
  )

  # 45-degree Line (Perfect Calibration)
  limit_min = min(agg_df['actual_mean'].min(), agg_df['predicted_mean'].min())
  limit_max = max(agg_df['actual_mean'].max(), agg_df['predicted_mean'].max())
  plt.plot(
      [limit_min, limit_max],
      [limit_min, limit_max],
      'r--',
      label='Ideal (Slope=1)',
  )

  plt.title(
      f'{title_prefix} Validation\nPearson R={r_val:.3f}, Calib'
      f' Slope={slope:.3f}'
  )
  plt.xlabel('Model Prediction (Fixed Effects)')
  plt.ylabel('Observed Mean')
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()

  return fig, r_val, slope, agg_df


def plot_global_residuals(
    df_validation,
    iso_col='country_3_alpha_cd',
    obs_col='actual_mean',
    pred_col='predicted_mean',
    is_hrv=False,
):
  """Plots a choropleth of per-country mean residuals (observed - predicted).

  If a ``month`` column is present the residuals are first averaged over
  months so that each country contributes one value; countries with fewer
  than 10 users (average per month) are dropped. The diverging colour scale
  is symmetric, with its limit set by the 99th percentile of the absolute
  residuals, clipped to [5, 15] ms for HRV and [2, 5] bpm for RHR.

  Args:
    df_validation: Aggregated validation frame (e.g. the ``agg_df`` returned by
      ``save_aggregate_validation``) with one row per country[/month].
    iso_col: Column holding ISO-3166 alpha-3 country codes.
    obs_col: Column with the observed group mean.
    pred_col: Column with the model-predicted group mean.
    is_hrv: Whether the metric is HRV (ms) rather than RHR (bpm); controls
      labels and colour-scale limits.

  Returns:
    A plotly Figure with the choropleth map.
  """
  df_plot = df_validation.copy()

  # Calculate Residuals
  df_plot['residual'] = df_plot[obs_col] - df_plot[pred_col]

  # Aggregate over months if present to have one row per country
  if 'month' in df_plot.columns:
    df_plot = (
        df_plot.groupby(iso_col)
        .agg({
            'residual': 'mean',
            obs_col: 'mean',
            pred_col: 'mean',
            'n_users': 'mean',  # Average users per month
        })
        .reset_index()
    )

  # Keep only countries with at least 10 users (average per month).
  if 'n_users' in df_plot.columns:
    df_plot = df_plot[df_plot['n_users'] >= 10]

  var_name = 'HRV' if is_hrv else 'RHR'
  unit_name = 'ms' if is_hrv else 'BPM'

  # 3. Determine symmetric color scale limits (robust to extreme outliers)
  if not df_plot['residual'].empty:
    max_val = np.percentile(df_plot['residual'].abs(), 99)
    if is_hrv:
      limit = min(max(max_val, 5.0), 15.0)
    else:
      limit = min(max(max_val, 2.0), 5.0)
  else:
    limit = 5.0 if is_hrv else 2.0

  hover_dict = {obs_col: ':.1f', pred_col: ':.1f', 'residual': ':.2f'}
  if 'n_users' in df_plot.columns:
    hover_dict['n_users'] = ':,d'

  # 4. Create the Choropleth
  title = 'Global Residuals: Observed - Predicted'
  fig = px.choropleth(
      df_plot,
      locations=iso_col,
      color='residual',
      hover_name=iso_col,
      hover_data=hover_dict,
      color_continuous_scale='RdBu_r',  # Red = Underpred, Blue = Overpred
      range_color=[-limit, limit],
      labels={
          'residual': f'{unit_name} Offset',
          obs_col: f'Observed {var_name}',
          pred_col: f'Predicted {var_name}',
      },
      title=title,
  )

  # 5. Refine Layout & Map Projection
  fig.update_geos(
      showframe=False,
      showcoastlines=True,
      coastlinecolor='#d9c2af',
      showland=True,
      landcolor='#ebd8c8',  # Warm beige for countries with no data
      showocean=True,
      oceancolor='#ffffff',
      projection_type='equirectangular',
  )

  fig.update_layout(
      margin={'r': 10, 't': 60, 'l': 10, 'b': 10},
      title_font_size=18,
      coloraxis_colorbar=dict(
          title=f'{unit_name} Offset (Obs - Pred)',
          thickness=15,
          len=0.4,
          x=0.5,
          xanchor='center',
          y=-0.05,
          yanchor='top',
          orientation='h',
      ),
  )

  return fig


def get_create_output_path(formula, r_folder, colname, varname):
  """Builds and creates a timestamped output directory for a model formula.

  The formula is compressed into a short model identifier (operators and
  long predictor names are abbreviated, ``colname`` is replaced by
  ``varname``) and prefixed with the current ``YYYYmmdd_HHMMSS`` timestamp.

  Args:
    formula: The statsmodels/patsy formula string of the fitted model.
    r_folder: Root results folder under which the run directory is created.
    colname: Target column name in the formula (e.g. 'hrv_log').
    varname: Short metric label that replaces ``colname`` in the identifier.

  Returns:
    The created directory path, with a trailing path separator.
  """
  clean_name = (
      formula.replace(' ', '')
      .replace('~', '_v_')
      .replace('+', '_')
      .replace('*', 'x')
      .replace('(', '')
      .replace(')', '')
      .replace(',', '')
  )

  model_id = (
      clean_name.replace('age_c', 'age')
      .replace('Cgender_n', 'gen')
      .replace('avg_temp_mean_c', 'temp')
      .replace('latitude_c', 'lat')
      .replace('longitude_c', 'lon')
      .replace('month_sin', 'sin')
      .replace('month_cos', 'cos')
      .replace('month_sin_2', 'sin2')
      .replace('month_cos_2', 'cos2')
      .replace('dl_monthly_c', 'dl')
      .replace('lat_e7_c', 'lat')
      .replace('lng_e7_c', 'lon')
      .replace('sin_lng', 'sin_lng')
      .replace('cos_lng', 'cos_lng')
      .replace('sleep_hrs_c', 'sleep')
      .replace('avg_temp_max_c', 'tmax')
      .replace('avg_temp_min_c', 'tmin')
      .replace('avg_temp_delta_c', 'tdelta')
      .replace('abs(lat_e7_c)', 'abs_lat')
      .replace('I(age_c**2)', 'age2')
      .replace('cc(month, df=5)', 'month_spline_5')
      .replace(colname, varname)
  )

  model_id = f'{dt.datetime.now().strftime("%Y%m%d_%H%M%S")}_{model_id}'
  output_path = os.path.join(r_folder, model_id, '')  # keeps a trailing '/'

  # Create folder
  fio.make_dirs(output_path)

  return output_path


# -----------------------------------------------------------------------------
# Density-based thinning of the age-22 reporting artefact
# -----------------------------------------------------------------------------
THINNING_LAT_BANDS = [
    'Mid South',
    'S. Tropics',
    'N. Tropics',
    'Mid North',
    'Far North',
]


def assign_thinning_lat_band(lat: pd.Series) -> pd.Series:
  """Maps latitude (degrees or E7) to five 30-degree latitude bands."""
  deg = lat / 1e7 if lat.abs().max() > 1000 else lat
  conds = [
      (deg >= -60) & (deg < -30),
      (deg >= -30) & (deg < 0),
      (deg >= 0) & (deg < 30),
      (deg >= 30) & (deg < 60),
      (deg >= 60) & (deg <= 90),
  ]
  return pd.Series(
      np.select(conds, THINNING_LAT_BANDS, default='Other'), index=lat.index
  )


def get_user_summary(
    df: pd.DataFrame,
    target_col: str,
    age_col: str = 'age',
    gender_col: str = 'gender',
    lat_col: str = 'lat_e7',
) -> pd.DataFrame:
  """One row per user: mean target, mean age, sex, primary latitude band, months.

  The primary latitude band is the band in which the user contributed the most
  months.

  Args:
    df: User-month DataFrame.
    target_col: Physiological column to summarise (e.g.
      'avg_resting_heart_rate_bpm' or 'avg_hrv').
    age_col: Age column (years).
    gender_col: Sex column ('Male'/'Female').
    lat_col: Latitude column (degrees or E7).

  Returns:
    DataFrame with columns user_id, val, age, gender, months_contributed,
    lat_band, age_int. Users with any missing field are dropped.
  """
  band = assign_thinning_lat_band(df[lat_col])
  band_months = (
      pd.DataFrame({'user_id': df['user_id'], 'lat_band': band})
      .value_counts()
      .rename('band_months')
      .reset_index()
      .sort_values(
          ['user_id', 'band_months', 'lat_band'], ascending=[True, False, True]
      )
      .drop_duplicates('user_id', keep='first')
      .set_index('user_id')['lat_band']
  )
  summary = df.groupby('user_id').agg(
      val=(target_col, 'mean'),
      age=(age_col, 'mean'),
      gender=(gender_col, 'first'),
      months_contributed=('user_id', 'count'),
  )
  summary['lat_band'] = band_months
  summary = summary.dropna(subset=['val', 'age', 'gender', 'lat_band'])
  summary['age_int'] = summary['age'].round().astype(int)
  return summary.reset_index()


def thin_histogram_matching(
    user_summary: pd.DataFrame,
    target_age: int = 22,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Set[Any]]:
  """Local histogram-matching thinning of ``target_age`` users.

  Removes the surplus of users reported at one age (a default birth year in
  the data) so that their number and value distribution match the two
  neighbouring ages. Within each (latitude band, sex) stratum, the target
  count is the mean of the counts at ``age - 1`` and ``age + 1``, and the
  target value distribution is the pooled histogram of those two ages; users
  above the target count in each 1-unit value bin are removed at random.

  Properties of the procedure (documented, not changed):

  * Removal is decided bin by bin and never adds users to under-filled bins,
    so the total removed can exceed the stratum surplus and a small fraction
    of ``target_age`` users is removed even when no artefact is present
    (shrinking as the strata grow).
  * ``age_int`` is the user's mean age rounded half-to-even (numpy), so mean
    ages of 21.5 and 22.5 both count as 22.
  * Users absent from ``user_summary`` (e.g. missing sex or latitude band,
    or no value in the summary range) are never removed and are not counted.

  Args:
    user_summary: Output of ``get_user_summary``.
    target_age: Age carrying the registration artefact.
    random_state: Seed used for the per-bin random removal.

  Returns:
    (thinned user_summary, set of removed user_ids).
  """
  df = user_summary.copy()
  df['val_rounded'] = df['val'].round().astype(int)

  user_counts = (
      df.groupby(['age_int', 'lat_band', 'gender'], observed=False)['user_id']
      .nunique()
      .unstack(level=0, fill_value=0)
  )
  users_to_drop = set()
  for stratum in user_counts.index.tolist():
    lat_band, sex = stratum
    counts = user_counts.loc[stratum]
    n_lo = counts.get(target_age - 1, 0)
    n_hi = counts.get(target_age + 1, 0)
    n_curr_total = counts.get(target_age, 0)
    n_expected = int(round(0.5 * (n_lo + n_hi)))
    if n_curr_total - n_expected <= 0:
      continue

    in_stratum = (df['lat_band'] == lat_band) & (df['gender'] == sex)
    sub_ref = df[
        in_stratum & df['age_int'].isin([target_age - 1, target_age + 1])
    ]
    sub_curr = df[in_stratum & (df['age_int'] == target_age)]
    if sub_ref.empty or sub_curr.empty:
      continue

    ref_pmf = sub_ref['val_rounded'].value_counts(normalize=True)
    curr_counts = sub_curr['val_rounded'].value_counts()
    all_bins = sorted(set(ref_pmf.index) | set(curr_counts.index))
    for r_val in all_bins:
      n_target = int(round(n_expected * ref_pmf.get(r_val, 0.0)))
      n_curr = curr_counts.get(r_val, 0)
      if n_curr > n_target:
        bin_users = sub_curr.loc[
            sub_curr['val_rounded'] == r_val, 'user_id'
        ].values
        rng = np.random.RandomState(random_state)
        users_to_drop.update(
            rng.choice(bin_users, size=n_curr - n_target, replace=False)
        )

  thinned = df[~df['user_id'].isin(users_to_drop)].drop(columns='val_rounded')
  return thinned, users_to_drop


def apply_density_based_thinning(
    df: pd.DataFrame,
    target_col: str = 'avg_resting_heart_rate_bpm',
    age_col: str = 'age',
    gender_col: str = 'gender',
    lat_col: str = 'lat_e7',
    target_age: int = 22,
    val_min: float = 30.0,
    val_max: float = 150.0,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
  """Removes the false-birth-year age artefact from a user-month DataFrame.

  Wraps ``get_user_summary`` + ``thin_histogram_matching``. Rows outside
  ``[val_min, val_max]`` are ignored when building the per-user summary but
  all rows of retained users are returned.

  Args:
    df: User-month DataFrame (clean cohort frame).
    target_col: Physiological column used for histogram matching.
    age_col: Age column (years).
    gender_col: Sex column.
    lat_col: Latitude column (degrees or E7).
    target_age: Age with the reporting artefact.
    val_min: Lower physiological bound for the summary.
    val_max: Upper physiological bound for the summary.
    random_state: Seed for the random removal.
    verbose: Print a one-line summary.

  Returns:
    (thinned_df, stats) where stats holds user counts before/after, the set of
    removed ``user_id`` values and the before/after user summaries (handy for
    the thinning-validation figures and the demographics tables). The counts
    refer to users present in the summary; users without a summary row (no
    value in range, missing sex or latitude band) are kept in ``thinned_df``
    but not counted.
  """
  valid = df[df[target_col].between(val_min, val_max)]
  summary_before = get_user_summary(
      valid, target_col, age_col, gender_col, lat_col
  )
  summary_after, dropped = thin_histogram_matching(
      summary_before, target_age=target_age, random_state=random_state
  )
  thinned_df = df[~df['user_id'].isin(dropped)]

  n_before = len(summary_before)
  stats_out = {
      'initial_users': n_before,
      'retained_users': n_before - len(dropped),
      'removed_users': len(dropped),
      'percent_removed': 100.0 * len(dropped) / n_before if n_before else 0.0,
      'dropped_user_ids': dropped,
      'user_summary_before': summary_before,
      'user_summary_after': summary_after,
  }
  if verbose:
    print(
        f'Thinning complete: removed {len(dropped):,} users'
        f' ({stats_out["percent_removed"]:.2f}%) at age {target_age};'
        f' retained {stats_out["retained_users"]:,} users.'
    )
  return thinned_df, stats_out


# ---------------------------------------------------------------------------
# Model feature engineering (shared by notebooks/02_modelling.ipynb and
# publication_figures.py) and fixed-effects prediction from saved coefficients
# ---------------------------------------------------------------------------

METRIC_TARGET_COLS: Dict[str, str] = {
    'rhr': 'avg_resting_heart_rate_bpm',
    'hrv': 'avg_hrv',
}
# Continuous predictors that are Gelman-standardised (centre, divide by 2 SD).
MODEL_CONTINUOUS_COLS: List[str] = [
    'age',
    'lat_e7',
    'abs_lat_e7',
    'avg_temp_mean',
    'avg_temp_min',
    'avg_temp_max',
    'dl_monthly',
]
TEMP_VARS_C: List[str] = ['avg_temp_mean_c', 'avg_temp_max_c', 'avg_temp_min_c']
# Temperature residualised on latitude x season (Stage 6 of the competition).
TEMP_RESIDUAL_FORMULA = (
    '{temp} ~ abs_lat_e7_c + (month_sin + month_cos + month_sin_2 +'
    ' month_cos_2) * lat_e7_c'
)
# Day of year of the 15th of each month (index 0 = January).
MID_MONTH_DAY_OF_YEAR = np.array(
    [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
)


def day_length_hours(
    latitude_deg: np.ndarray, day_of_year: np.ndarray
) -> np.ndarray:
  """Astronomical day length (hours) from latitude and day of year.

  Uses the solar declination approximation
  ``23.44 deg * sin(2 pi (284 + doy) / 365)`` and the sunrise hour angle
  ``acos(-tan(lat) tan(decl))`` clipped to the polar day (24 h) / polar night
  (0 h) limits. Refraction and the solar disc are ignored, so values differ
  from an ephemeris by a few minutes.

  Args:
    latitude_deg: Latitude in degrees (array-like, broadcast with
      ``day_of_year``).
    day_of_year: Day of the year, 1-365 (array-like).

  Returns:
    Day length in hours, same shape as the broadcast inputs.
  """
  declination = np.deg2rad(23.44) * np.sin(
      2 * np.pi * (284 + np.asarray(day_of_year, dtype=float)) / 365.0
  )
  latitude = np.deg2rad(np.asarray(latitude_deg, dtype=float))
  cos_hour_angle = np.clip(-np.tan(latitude) * np.tan(declination), -1.0, 1.0)
  return 24.0 / np.pi * np.arccos(cos_hour_angle)


def mid_month_day_length_hours(
    latitude_deg: np.ndarray, month: np.ndarray
) -> np.ndarray:
  """``day_length_hours`` on the 15th of ``month`` (1-12) at ``latitude_deg``.

  Args:
    latitude_deg: Latitude in degrees.
    month: Calendar month, 1 (January) to 12 (December).

  Returns:
    Day length in hours, same shape as the broadcast of the inputs.

  Raises:
    ValueError: a month is outside 1-12 (a 0-indexed month would otherwise
      silently select December).
  """
  month_arr = np.asarray(month, dtype=int)
  if np.any((month_arr < 1) | (month_arr > 12)):
    bad = np.unique(month_arr[(month_arr < 1) | (month_arr > 12)])
    raise ValueError(f'month must be between 1 and 12, got {bad.tolist()}')
  return day_length_hours(latitude_deg, MID_MONTH_DAY_OF_YEAR[month_arr - 1])


def add_month_harmonics(df: pd.DataFrame) -> pd.DataFrame:
  """Adds month, month_sin/cos and month_sin_2/cos_2 in place.

  Args:
    df: Frame with a datetime ``activity_period`` column.

  Returns:
    The same frame.
  """
  m = df['activity_period'].dt.month
  df['month'] = m
  df['month_sin'] = np.sin(2 * np.pi * m / 12)
  df['month_cos'] = np.cos(2 * np.pi * m / 12)
  df['month_sin_2'] = np.sin(2 * np.pi * m / 6)
  df['month_cos_2'] = np.cos(2 * np.pi * m / 6)
  return df


def add_sex_and_longitude_features(
    df: pd.DataFrame, male_share: Optional[float] = None
) -> float:
  """Adds ``gender_n``, ``abs_lat_e7``, ``sin_lng`` and ``cos_lng`` in place.

  Args:
    df: Frame with ``gender`` ('Male'/'Female'), ``lat_e7`` and ``lng_e7``.
    male_share: Mean of the Male indicator to centre ``gender_n`` on; computed
      on ``df`` when None.

  Returns:
    The male share used for centring (``gender_n = 1[Male] - male_share``).
  """
  gender_binary = df['gender'].map({'Male': 1.0, 'Female': 0.0})
  if male_share is None:
    male_share = float(gender_binary.mean())
  df['gender_n'] = gender_binary - male_share
  df['abs_lat_e7'] = df['lat_e7'].abs()
  lng_rad = np.deg2rad(df['lng_e7'] / 1e7)
  df['sin_lng'] = np.sin(lng_rad)
  df['cos_lng'] = np.cos(lng_rad)
  return male_share


def standardise_model_features(
    df: pd.DataFrame,
    cols: Sequence[str] = tuple(MODEL_CONTINUOUS_COLS),
    summary: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
  """Gelman-standardises ``cols`` and their squares in place.

  Each column is scaled as ``(x - mean) / (2 sd)`` so that coefficients are
  comparable with those of binary predictors. Adds ``{col}_c`` and
  ``{col}_sq_c``; columns absent from ``df`` are skipped. Statistics are
  computed on the rows of ``df`` (the modelling notebook calls this on the full
  cohort before subsampling) unless ``summary`` provides them.

  Args:
    df: Frame holding the raw continuous predictors; modified in place.
    cols: Column names to standardise (default: ``MODEL_CONTINUOUS_COLS``).
    summary: Statistics of an earlier call (its return value) to apply instead
      of recomputing them, so that new rows are scaled exactly like the fitted
      data. Columns without stored statistics are skipped.

  Returns:
    ``{f'{col}_mean', f'{col}_std', f'{col}_sq_mean', f'{col}_sq_std'}`` per
    column (the applied statistics when ``summary`` is given).
  """
  out: Dict[str, float] = {}
  for col in cols:
    if col not in df.columns:
      continue
    sq = df[col] ** 2
    if summary is None:
      mean_val, std_val = float(df[col].mean()), float(df[col].std())
      mean_sq, std_sq = float(sq.mean()), float(sq.std())
    elif f'{col}_std' in summary:
      mean_val, std_val = summary[f'{col}_mean'], summary[f'{col}_std']
      mean_sq, std_sq = summary[f'{col}_sq_mean'], summary[f'{col}_sq_std']
    else:
      continue
    df[f'{col}_c'] = (df[col] - mean_val) / (2 * std_val)
    df[f'{col}_sq_c'] = (sq - mean_sq) / (2 * std_sq)
    out.update({
        f'{col}_mean': mean_val,
        f'{col}_std': std_val,
        f'{col}_sq_mean': mean_sq,
        f'{col}_sq_std': std_sq,
    })
  return out


def scaling_factors_from_summary(
    summary: Mapping[str, float],
) -> Dict[str, float]:
  """``{f'{col}_c': 2 sd, f'{col}_sq_c': 2 sd(x^2)}`` as expected by ``extract_model_stats``."""
  factors: Dict[str, float] = {}
  for key, value in summary.items():
    if key.endswith('_sq_std'):
      factors[key[: -len('_sq_std')] + '_sq_c'] = 2 * value
    elif key.endswith('_std'):
      factors[key[: -len('_std')] + '_c'] = 2 * value
  return factors


def raw_scale_divisor(name: str, factors: Dict[str, float]) -> float:
  """Divisor that turns a standardised coefficient of ``name`` into raw units.

  Product over the ``:``-separated parts of the term: ``factors[part]`` for a
  standardised column, ``factors['x_c']`` for ``x_c_res``,
  ``factors['x_c'] ** 2`` for ``x_sq_c_res`` (the square of ``x_c_res``), and
  1 for anything else.

  Args:
    name: Fixed-effect term name, e.g. ``'month_sin:avg_temp_max_sq_c_res'``.
    factors: Output of :func:`scaling_factors_from_summary`.

  Returns:
    ``beta_raw = beta_std / divisor`` (same for its CI).
  """
  divisor = 1.0
  for part in name.split(':'):
    if part.endswith('_sq_c_res'):
      base = part[: -len('_sq_c_res')] + '_c'
      if base in factors:
        divisor *= factors[base] ** 2
    elif part.endswith('_res'):
      base = part[: -len('_res')]
      if base in factors:
        divisor *= factors[base]
    elif part in factors:
      divisor *= factors[part]
  return divisor


def residualise_temperature(
    df: pd.DataFrame,
    method: str = 'harmonic',
    temp_vars: Sequence[str] = tuple(TEMP_VARS_C),
    params: Optional[Mapping[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
  """Adds ``{temp}_res`` and ``avg_temp_{max,min}_sq_c_res`` in place.

  Args:
    df: Frame with the standardised temperatures, ``lat_e7_c``, ``abs_lat_e7_c``
      and the month harmonics.
    method: 'harmonic'      - OLS ``TEMP_RESIDUAL_FORMULA`` (latitude plus
      latitude x season harmonics; the modelling default); squared terms are the
      squared residuals. 'linear_lat'    - ``np.polyfit(lat_e7_c, temp, 1)``
      residual; squared terms are the squared residuals. 'linear_lat_sq' - as
      'linear_lat', but ``avg_temp_*_sq_c_res`` is the linear-latitude residual
      of ``avg_temp_*_sq_c`` itself. Ignored when ``params`` is given.
    temp_vars: Standardised temperature columns to residualise (missing
      skipped).
    params: Return value of an earlier call (``{temp: {'params': {term: coef},
      ...}}``) to apply instead of refitting, so that new rows are residualised
      exactly like the fitted data. Squared terms are the squared residuals.

  Returns:
    ``{temp: {'params': {term: coef}, 'rsquared': float}}`` per residualised
    variable (``rsquared`` is NaN when ``params`` is applied).
  """
  out: Dict[str, Dict[str, Any]] = {}
  present = [t for t in temp_vars if t in df.columns]
  if params is not None:
    for temp in present:
      if temp not in params:
        continue
      coef = pd.Series(params[temp]['params'], dtype=float)
      df[f'{temp}_res'] = df[temp] - predict_from_coefficients(df, coef)
      out[temp] = {'params': dict(params[temp]['params']), 'rsquared': np.nan}
    method = 'stored'
  elif method == 'harmonic':
    for temp in present:
      model = smf.ols(TEMP_RESIDUAL_FORMULA.format(temp=temp), data=df).fit()
      df[f'{temp}_res'] = model.resid
      out[temp] = {
          'params': {k: float(v) for k, v in model.params.items()},
          'rsquared': float(model.rsquared),
      }
  elif method in ('linear_lat', 'linear_lat_sq'):
    targets = list(present)
    if method == 'linear_lat_sq':
      targets += [
          t[:-2] + '_sq_c' for t in present if (t[:-2] + '_sq_c') in df.columns
      ]
    for temp in targets:
      slope, intercept = np.polyfit(df['lat_e7_c'], df[temp], 1)
      fitted = slope * df['lat_e7_c'] + intercept
      df[f'{temp}_res'] = df[temp] - fitted
      ss_res = float(((df[temp] - fitted) ** 2).sum())
      ss_tot = float(((df[temp] - df[temp].mean()) ** 2).sum())
      out[temp] = {
          'params': {'Intercept': float(intercept), 'lat_e7_c': float(slope)},
          'rsquared': 1.0 - ss_res / ss_tot if ss_tot else float('nan'),
      }
  else:
    raise ValueError(f'Unknown temperature residualisation method {method!r}')
  if method != 'linear_lat_sq':
    for base in ('avg_temp_max', 'avg_temp_min'):
      if f'{base}_c_res' in df.columns:
        df[f'{base}_sq_c_res'] = df[f'{base}_c_res'] ** 2
  return out


def _first_existing(candidates: Sequence[str]) -> str:
  """Returns the first candidate path that exists (else the first candidate)."""
  for path in candidates:
    if fio.file_exists(path):
      return path
  return candidates[0]


def results_csv_path(metric_results_dir: str, metric: str) -> str:
  """Path of the per-model results CSV of a run.

  Args:
    metric_results_dir: Directory holding the per-metric run outputs.
    metric: 'rhr' | 'hrv'.

  Returns:
    ``<metric_results_dir>/lme-models-results-{metric}.csv``.
  """
  candidates = [
      os.path.join(metric_results_dir, f'lme-models-results-{metric}.csv')
  ]
  return _first_existing(candidates)


def coefficients_csv_path(metric_results_dir: str, metric: str) -> str:
  """Path of the REML-refit winner coefficients CSV of a run.

  Args:
    metric_results_dir: Directory holding the per-metric run outputs.
    metric: 'rhr' | 'hrv'.

  Returns:
    ``<metric_results_dir>/model_{metric}_coefficients_interpreted.csv``.
  """
  candidates = [
      os.path.join(
          metric_results_dir, f'model_{metric}_coefficients_interpreted.csv'
      )
  ]
  return _first_existing(candidates)


FEATURE_STATS_FILE = 'feature_stats.json'


def feature_stats(
    standardisation: Dict[str, float],
    temperature_ols: Dict[str, Dict[str, Any]],
    male_share: float,
) -> Dict[str, Any]:
  """Bundles the statistics needed to rebuild the predictors of a fitted run.

  Args:
    standardisation: Return value of ``standardise_model_features``.
    temperature_ols: Return value of ``residualise_temperature``.
    male_share: Mean of the Male indicator used to centre ``gender_n``.

  Returns:
    JSON-serialisable dict understood by ``prepare_model_features``.
  """
  return {
      'standardisation': dict(standardisation),
      'temperature_ols': {
          temp: {'params': {k: float(v) for k, v in info['params'].items()}}
          for temp, info in temperature_ols.items()
      },
      'male_share': float(male_share),
  }


def write_feature_stats(
    metric_results_dir: str, run_stats: Dict[str, Any]
) -> str:
  """Writes ``FEATURE_STATS_FILE`` into ``metric_results_dir``."""
  path = os.path.join(metric_results_dir, FEATURE_STATS_FILE)
  with fio.open_file(path, 'w') as f:
    json.dump(run_stats, f, indent=2)
  return path


def read_feature_stats(metric_results_dir: str) -> Optional[Dict[str, Any]]:
  """Returns the run's ``FEATURE_STATS_FILE`` content, or None if absent."""
  path = os.path.join(metric_results_dir, FEATURE_STATS_FILE)
  if not fio.file_exists(path):
    return None
  with fio.open_file(path, 'r') as f:
    return json.load(f)


def prepare_model_features(
    df: pd.DataFrame,
    metric: str,
    temp_residual_method: str = 'harmonic',
    run_stats: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
  """Recreates, in place, every predictor used by the LME competition.

  Month harmonics, ``gender_n``/``abs_lat_e7``/``sin_lng``/``cos_lng``, 2-SD
  standardisation of ``MODEL_CONTINUOUS_COLS`` (+ squares), residualised
  temperatures and ``target_log`` (log(avg_hrv) for HRV, raw bpm for RHR).
  Standardisation and residualisation statistics are taken from ``run_stats``
  when given (the fitted run's own values) and computed on ``df`` otherwise.

  Args:
    df: Rows of the clean user-month cohort frame (see
      ``cohort_io.load_cohort``) with activity_period, gender, age, lat_e7,
      lng_e7, avg_temp_mean/min/max, [dl_monthly] and the metric target.
    metric: 'rhr' | 'hrv'.
    temp_residual_method: See ``residualise_temperature``; ignored when
      ``run_stats`` is given.
    run_stats: Output of ``feature_stats`` / ``read_feature_stats``.

  Returns:
    The same frame with the added columns.
  """
  if metric not in METRIC_TARGET_COLS:
    raise ValueError(
        f'metric must be one of {sorted(METRIC_TARGET_COLS)}, got {metric!r}'
    )
  add_month_harmonics(df)
  if run_stats is None:
    add_sex_and_longitude_features(df)
    standardise_model_features(df)
    residualise_temperature(df, temp_residual_method)
  else:
    add_sex_and_longitude_features(df, male_share=run_stats['male_share'])
    standardise_model_features(df, summary=run_stats['standardisation'])
    residualise_temperature(df, params=run_stats['temperature_ols'])
  target = METRIC_TARGET_COLS[metric]
  df['target_log'] = (
      np.log(df[target]) if metric == 'hrv' else df[target].astype(float)
  )
  return df


def predict_from_coefficients(
    df: pd.DataFrame,
    coef: pd.Series,
    formula_terms: Optional[Sequence[str]] = None,
) -> pd.Series:
  """Fixed-effects linear predictor ``sum_k beta_k * x_k`` evaluated on ``df``.

  Term semantics follow patsy/statsmodels parameter names: 'Intercept' is 1,
  'a:b:c' is the product of the columns, anything else is a column of ``df``.
  Variance-component names ending in ' Var' are ignored.

  Args:
    df: Frame holding every column referenced by the terms.
    coef: Series term -> beta (e.g. ``model_coefficients.csv['Coef.']``).
    formula_terms: Subset/order of terms to evaluate (default: all of
      ``coef.index``).

  Returns:
    Series aligned with ``df.index``.

  Raises:
    KeyError: if a referenced column is missing from ``df``.
  """
  terms = (
      list(formula_terms)
      if formula_terms is not None
      else [str(t) for t in coef.index]
  )
  pred = pd.Series(0.0, index=df.index)
  missing: List[str] = []
  for name in terms:
    if name.endswith(' Var'):
      continue
    if name == 'Intercept':
      pred += float(coef[name])
      continue
    parts = name.split(':')
    absent = [p for p in parts if p not in df.columns]
    if absent:
      missing.extend(absent)
      continue
    term = df[parts[0]].astype(float)
    for p in parts[1:]:
      term = term * df[p]
    pred += float(coef[name]) * term
  if missing:
    raise KeyError(
        'Columns required by the coefficients are missing:'
        f' {sorted(set(missing))}'
    )
  return pred
