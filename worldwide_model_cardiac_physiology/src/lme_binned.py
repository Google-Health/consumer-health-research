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

"""Fast random-intercept linear mixed models via binned sufficient statistics.

This module contains the custom LME machinery used for the worldwide
physiological seasonality models (RHR / HRV). A standard
``statsmodels.MixedLM`` fit on tens of millions of user-months from hundreds
of thousands of users is infeasible
(per-group Python loops, O(N) memory per iteration). For a *random-intercept
only* model the marginal log-likelihood depends on the data only through a
handful of sufficient statistics per group size, which makes the fit O(U) in
the number of distinct group sizes rather than O(N):

  * For every distinct number of observations per user ``n`` we accumulate
    ``X'X``, ``X'y``, ``y'y`` over all rows and ``S_X'S_X``, ``S_X'S_y``,
    ``S_y'S_y`` over per-user sums (``get_binned_stats``).
  * The marginal (ML) or restricted (REML) log-likelihood, its gradient and
    Hessian and the GLS fixed-effect solve are then closed-form sums over
    those bins (``binned_loglike_and_grad``, ``binned_hessian``,
    ``binned_get_fe_params``).

The resulting ``MixedLMResults`` object is a genuine statsmodels results
object (``params``, ``bse``, ``conf_int()``, ``bic``, ``predict`` ... all work)
so downstream reporting code is unchanged.

Entry points
------------
``BinnedMixedLM.from_formula(formula, data, groups=...)`` then ``.fit()`` –
drop-in replacement for ``smf.mixedlm``.

``inject_fast_random_effects(result)`` – vectorised BLUPs / fitted values /
residuals (statsmodels' own ``random_effects`` loops over users in Python).

``save_checkpoint`` / ``load_checkpoint`` – tiny JSON state helpers so a
multi-hour model competition can resume.

Compatibility notes
-------------------
Written against statsmodels 0.14.x and verified against statsmodels 0.15. The
module only
uses statsmodels' public API (``MixedLM`` subclassing, ``MixedLMParams``,
``MixedLMResults``, ``VCSpec``) plus the results cache (``_cache``) to inject
the pre-computed likelihood and random effects; re-check the latter when
statsmodels is upgraded.
"""

# This module re-implements the random-intercept MixedLM likelihood on binned
# sufficient statistics and fills statsmodels' results cache directly.
# pylint: disable=protected-access

# pytype: skip-file
# This module deliberately patches statsmodels' results cache / conf_int and
# sets dynamic attributes on MixedLMResults.

from __future__ import annotations

from collections.abc import Mapping
import json
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
import statsmodels.api as sm
import statsmodels.regression.mixed_linear_model as mlm
from statsmodels.regression.mixed_linear_model import MixedLMParams
from statsmodels.regression.mixed_linear_model import MixedLMResults

import file_io as fio


# -----------------------------------------------------------------------------
# conf_int() patch – statsmodels returns a bare ndarray for our wrapped results
# -----------------------------------------------------------------------------
class BinnedConfIntDescriptor:
  """Makes ``MixedLMResults.conf_int()`` return a labelled DataFrame.

  Defined at module level (not as a closure) so that results objects stay
  picklable.
  """

  def __get__(self, instance, owner):
    if instance is None:
      return self

    def _conf_int(*args, **kwargs):
      orig = getattr(instance.__class__, '_original_conf_int', None)
      ci = orig(instance, *args, **kwargs)
      if isinstance(ci, np.ndarray):
        param_names = getattr(instance, 'param_names', None)
        if param_names is not None:
          return pd.DataFrame(ci, index=param_names)
      return ci

    return _conf_int


if not hasattr(mlm.MixedLMResults, '_conf_int_patched'):
  mlm.MixedLMResults._original_conf_int = mlm.MixedLMResults.conf_int
  mlm.MixedLMResults.conf_int = BinnedConfIntDescriptor()
  mlm.MixedLMResults._conf_int_patched = True


# -----------------------------------------------------------------------------
# Sufficient statistics
# -----------------------------------------------------------------------------
def get_binned_stats(model) -> Dict[int, Dict[str, Any]]:
  """Accumulates per-group-size sufficient statistics.

  Args:
    model: A ``MixedLM``-like model exposing ``exog``, ``endog``, ``groups`` and
      ``exog_names``.

  Returns:
    ``{n: {'N_groups', 'XTX', 'XTy', 'yTy', 'SXX', 'SXy', 'Syy'}}`` keyed by the
    number of observations per group ``n``.
  """
  df = pd.DataFrame(model.exog, columns=model.exog_names)
  df['y'] = model.endog
  df['gid'] = model.groups

  grouped = df.groupby('gid')
  g_sums = grouped.sum()
  g_counts = grouped.size()

  binned = {}
  for n in sorted(g_counts.unique()):
    mask = g_counts == n
    raw_bin = df[df['gid'].isin(g_counts[mask].index)]
    x_bin = raw_bin[model.exog_names].values
    y_bin = raw_bin['y'].values
    sums_x = g_sums.loc[mask].drop('y', axis=1).values
    sums_y = g_sums.loc[mask]['y'].values
    binned[int(n)] = {
        'N_groups': int(mask.sum()),
        'XTX': x_bin.T @ x_bin,
        'XTy': x_bin.T @ y_bin,
        'yTy': float(np.sum(y_bin**2)),
        'SXX': sums_x.T @ sums_x,
        'SXy': sums_x.T @ sums_y,
        'Syy': float(np.sum(sums_y**2)),
    }
  return binned


def binned_loglike(params, binned_data, k_fe, reml=False) -> float:
  """Negative (restricted) log-likelihood without the ``log 2pi`` constant.

  Args:
    params: ``[beta (k_fe), log var_re, log var_resid]``.
    binned_data: Output of ``get_binned_stats``.
    k_fe: Number of fixed effects.
    reml: False for the marginal (ML) likelihood, True for the restricted (REML)
      likelihood, which adds ``-1/2 log|X' V^-1 X|``.

  Returns:
    Negative log-likelihood (to be minimised); ``inf`` for non-finite or
    degenerate (underflowing variance) parameters so that line searches back
    off instead of propagating NaN.
  """
  return binned_loglike_and_grad(params, binned_data, k_fe, reml)[0]


def binned_loglike_and_grad(
    params, binned_data, k_fe, reml=False
) -> Tuple[float, np.ndarray]:
  """Negative (restricted) log-likelihood and its analytic gradient.

  Used by the default fit (``analytic_gradient=True``); with
  ``analytic_gradient=False`` the fit uses ``binned_loglike`` and scipy's
  finite differences instead. The gradient is exact in
  ``beta`` and in the log-variance parameters, which keeps BFGS/L-BFGS-B stable
  when the objective is O(1e8) (tens of millions of rows).

  With ``V_n = var_resid I + var_re 11'`` for a group of size ``n`` and
  ``w_n = var_re / (var_resid + n var_re)``, the marginal log-likelihood is a
  sum over group-size bins of ``-1/2 [log|V_n| + (q_raw - w_n q_sum) /
  var_resid]``. REML adds ``-1/2 log|X' V^-1 X| = -1/2 log|A| + k_fe/2 log
  var_resid`` with ``A = sum_n (X'X_n - w_n S_X'S_X_n)``; its gradient enters
  through ``tr(A^-1 S_X'S_X_n) dw_n/dvar``.

  Args:
    params: ``[beta (k_fe), log var_re, log var_resid]``.
    binned_data: Output of ``get_binned_stats``.
    k_fe: Number of fixed effects.
    reml: False for ML, True for REML.

  Returns:
    ``(negative log-likelihood, gradient)``; ``(inf, zeros)`` for non-finite
    parameters or variances that under/overflow.
  """
  params = np.asarray(params, dtype=float)
  grad = np.zeros_like(params)
  if not np.all(np.isfinite(params)):
    return np.inf, grad
  b = params[:k_fe]
  log_var_re, log_var_resid = params[k_fe], params[k_fe + 1]
  if abs(log_var_re) > 300 or abs(log_var_resid) > 300:
    return np.inf, grad
  var_re = np.exp(log_var_re)
  var_resid = np.exp(log_var_resid)
  if not (var_re > 0 and var_resid > 0):
    return np.inf, grad

  ll = 0.0
  g_b = np.zeros(k_fe)
  g_re = 0.0  # d ll / d var_re
  g_resid = 0.0  # d ll / d var_resid
  a_mat = np.zeros((k_fe, k_fe))  # X' V^-1 X * var_resid
  for n, s in binned_data.items():
    denom = var_resid + n * var_re
    weight = var_re / denom
    r_raw = s['XTy'] - s['XTX'] @ b  # d(-q_raw/2)/db
    r_sum = s['SXy'] - s['SXX'] @ b
    q_raw = s['yTy'] - 2 * b @ s['XTy'] + b @ (s['XTX'] @ b)
    q_sum = s['Syy'] - 2 * b @ s['SXy'] + b @ (s['SXX'] @ b)
    q_eff = q_raw - weight * q_sum
    log_det = s['N_groups'] * ((n - 1) * np.log(var_resid) + np.log(denom))
    ll += -0.5 * (log_det + q_eff / var_resid)

    g_b += (r_raw - weight * r_sum) / var_resid
    # d weight/d var_re = var_resid/denom**2;
    # d weight/d var_resid = -var_re/denom**2.
    g_re += -0.5 * (s['N_groups'] * n / denom - q_sum / denom**2)
    g_resid += -0.5 * (
        s['N_groups'] * ((n - 1) / var_resid + 1.0 / denom)
        - q_eff / var_resid**2
        + var_re * q_sum / (var_resid * denom**2)
    )
    if reml:
      a_mat += s['XTX'] - weight * s['SXX']
  if reml:
    sign, log_det_a = np.linalg.slogdet(a_mat)
    if sign <= 0:
      return np.inf, grad
    ll += -0.5 * log_det_a + 0.5 * k_fe * np.log(var_resid)
    a_inv = np.linalg.inv(a_mat)
    for n, s in binned_data.items():
      denom = var_resid + n * var_re
      trace = np.sum(a_inv * s['SXX'])  # tr(A^-1 SXX), both symmetric.
      g_re += 0.5 * var_resid / denom**2 * trace
      g_resid += -0.5 * var_re / denom**2 * trace
    g_resid += 0.5 * k_fe / var_resid
  if not np.isfinite(ll):
    return np.inf, grad
  grad[:k_fe] = -g_b
  grad[k_fe] = -g_re * var_re  # chain rule for log-variance
  grad[k_fe + 1] = -g_resid * var_resid
  return -ll, grad


# -----------------------------------------------------------------------------
# Hessian / GLS solve
# -----------------------------------------------------------------------------
def binned_hessian(self, params):
  """Closed-form Hessian of the profile log-likelihood for random intercepts.

  Args:
    self: Model (``BinnedMixedLM`` or patched ``MixedLM``).
    params: Packed ``[beta, re_param]`` where ``re_param`` is sqrt(theta) if
      ``self.use_sqrt`` else theta (theta = var_re / var_resid).

  Returns:
    ``(hessian, singular_flag)``.
  """
  k_fe = self.k_fe
  beta = params[:k_fe]
  if self.use_sqrt:
    gamma = params[k_fe]
    theta = gamma**2
  else:
    theta = params[k_fe]
    gamma = None

  if not hasattr(self, 'binned_data'):
    self.binned_data = get_binned_stats(self)

  fac = self.n_totobs - (k_fe if self.reml else 0)

  xtvix = np.zeros((k_fe, k_fe))
  rvir = 0.0
  hess_fere_theta = np.zeros((1, k_fe))
  b_accum = 0.0
  d_accum = 0.0
  hess_re_trace_theta = 0.0
  grad_theta_trace = 0.0
  x_theta = np.zeros((k_fe, k_fe))  # -dA/dtheta with A = X' V^-1 X.
  f_theta = np.zeros((k_fe, k_fe))  # -d x_theta/dtheta.

  for n, s in self.binned_data.items():
    w_n = theta / (1.0 + n * theta)
    xtvix += s['XTX'] - w_n * s['SXX']
    q_raw = s['yTy'] - 2 * beta @ s['XTy'] + beta @ (s['XTX'] @ beta)
    q_sum = s['Syy'] - 2 * beta @ s['SXy'] + beta @ (s['SXX'] @ beta)
    rvir += q_raw - w_n * q_sum
    hess_fere_theta[0, :] += (s['SXy'] - s['SXX'] @ beta) / (
        1.0 + n * theta
    ) ** 2
    b_accum += q_sum / (1.0 + n * theta) ** 2
    d_accum += 2.0 * n / (1.0 + n * theta) ** 3 * q_sum
    hess_re_trace_theta += 0.5 * s['N_groups'] * n**2 / (1.0 + n * theta) ** 2
    if self.use_sqrt:
      grad_theta_trace += n * s['N_groups'] / (1.0 + n * theta)
    if self.reml:
      x_theta += s['SXX'] / (1.0 + n * theta) ** 2
      f_theta += 2.0 * n * s['SXX'] / (1.0 + n * theta) ** 3

  hess_fe = -fac * xtvix / rvir
  hess_fere_theta = -fac * hess_fere_theta / rvir
  hess_re_theta = hess_re_trace_theta - 0.5 * fac * (
      d_accum / rvir - b_accum**2 / rvir**2
  )
  grad_theta_reml = 0.0
  if self.reml:
    # Derivatives of -1/2 log|A|: 1/2 tr(A^-1 X_theta) and
    # 1/2 [tr((A^-1 X_theta)^2) - tr(A^-1 F_theta)].
    a_inv_x = np.linalg.solve(xtvix, x_theta)
    grad_theta_reml = 0.5 * np.trace(a_inv_x)
    hess_re_theta += 0.5 * (
        np.sum(a_inv_x.T * a_inv_x) - np.trace(np.linalg.solve(xtvix, f_theta))
    )

  if self.use_sqrt:
    grad_theta = (
        -0.5 * grad_theta_trace + 0.5 * fac * b_accum / rvir + grad_theta_reml
    )
    hess_fere = 2.0 * gamma * hess_fere_theta
    hess_re = 4.0 * gamma**2 * hess_re_theta + 2.0 * grad_theta
  else:
    hess_fere = hess_fere_theta
    hess_re = hess_re_theta

  m_tot = k_fe + 1
  hess = np.zeros((m_tot, m_tot))
  hess[:k_fe, :k_fe] = hess_fe
  hess[:k_fe, k_fe:] = hess_fere.T
  hess[k_fe:, :k_fe] = hess_fere
  hess[k_fe:, k_fe:] = hess_re
  return hess, bool(np.abs(hess_re) < 1e-12)


def binned_get_fe_params(self, cov_re, vcomp, tol=1e-8):
  """GLS solve for the fixed effects given the relative RE variance."""
  del vcomp, tol  # Unused for random-intercept models.
  theta = cov_re[0, 0]
  if not hasattr(self, 'binned_data'):
    self.binned_data = get_binned_stats(self)

  xtvix = np.zeros((self.k_fe, self.k_fe))
  xtviy = np.zeros(self.k_fe)
  for n, s in self.binned_data.items():
    w_n = theta / (1.0 + n * theta)
    xtvix += s['XTX'] - w_n * s['SXX']
    xtviy += s['XTy'] - w_n * s['SXy']
  try:
    return np.linalg.solve(xtvix, xtviy), False
  except np.linalg.LinAlgError:
    return np.linalg.lstsq(xtvix, xtviy, rcond=None)[0], True


# -----------------------------------------------------------------------------
# Results wrapper and fit
# -----------------------------------------------------------------------------
def wrap_binned_results(
    model, optimize_res, hessian_matrix, resid_var, hessian_skipped=False
) -> MixedLMResults:
  """Builds a statsmodels ``MixedLMResults`` from the binned optimiser output.

  ``llf``/``aic``/``bic`` are injected into the results cache so that
  statsmodels never re-evaluates the (slow, per-group) likelihood. Under REML
  (``model.reml``) ``llf`` is the restricted log-likelihood and ``aic`` /
  ``bic`` are NaN.

  Args:
    model: The fitted ``MixedLM`` model instance (provides ``k_fe``, ``nobs``,
      ``reml`` and ``exog_names``).
    optimize_res: ``scipy.optimize.OptimizeResult`` of the binned (restricted)
      likelihood; ``x`` holds the fixed effects followed by the log of the
      random-intercept variance, ``fun`` the negative log-likelihood.
    hessian_matrix: Hessian of the log-likelihood at the optimum (in
      ``sqrt(theta)`` when ``model.use_sqrt``), used to derive the parameter
      covariance.
    resid_var: Residual variance at the optimum.
    hessian_skipped: Whether the Hessian evaluation was skipped; the covariance
      and hence every standard error are then NaN (recorded on the results
      object as ``hessian_skipped``).

  Returns:
    A ``MixedLMResults`` instance carrying the fixed effects, random-intercept
    variance, covariance and cached fit statistics.
  """
  fe_len = model.k_fe
  beta = optimize_res.x[:fe_len]
  group_var = np.exp(optimize_res.x[fe_len])
  rel_var = group_var / resid_var
  result_params = np.r_[beta, rel_var]

  if hessian_skipped:
    pcov = np.full((fe_len + 1, fe_len + 1), np.nan)
  else:
    try:
      pcov = np.linalg.inv(-hessian_matrix)
    except np.linalg.LinAlgError:
      pcov = np.linalg.pinv(-hessian_matrix)
    if getattr(model, 'use_sqrt', False):
      # Delta method: the reported parameter is theta, not sqrt(theta).
      jac = np.ones(fe_len + 1)
      jac[-1] = 2.0 * np.sqrt(rel_var)
      pcov = pcov * np.outer(jac, jac)

  result = MixedLMResults(model, result_params, pcov / resid_var)
  result.hessian_skipped = hessian_skipped
  result.k_fe = fe_len
  result.k_re = 1
  result.k_re2 = 1
  result.k_vc = 0
  result.re_names = ['Group']
  result.re_names_full = ['Group Var']
  result.param_names = model.exog_names + ['user_id Var']

  result.__dict__['params_object'] = MixedLMParams.from_components(
      fe_params=beta, cov_re=np.atleast_2d(rel_var)
  )
  result.__dict__['reml'] = getattr(model, 'reml', False)
  result.__dict__['freepat'] = getattr(model, 'freepat', None)
  result.__dict__['params'] = pd.Series(result_params, index=result.param_names)

  if not hasattr(result, '_cache'):
    result._cache = {}
  reml = bool(getattr(model, 'reml', False))
  n_eff = model.nobs - (fe_len if reml else 0)
  llf = -optimize_res.fun - 0.5 * n_eff * np.log(2 * np.pi)
  k = len(result_params) + 1
  result._cache['llf'] = llf
  if reml:
    # Restricted likelihoods are not comparable across fixed-effect
    # structures; statsmodels reports NaN for the same reason.
    result._cache['aic'] = np.nan
    result._cache['bic'] = np.nan
  else:
    result._cache['aic'] = -2 * llf + 2 * k
    result._cache['bic'] = -2 * llf + k * np.log(model.nobs)
  # bse / tvalues / pvalues are deliberately NOT injected: statsmodels derives
  # them from ``cov_params() = normalized_cov * scale = pcov`` (the correct
  # fixed-effects covariance). Injecting ``sqrt(diag(pcov / resid_var))``
  # would understate every SE by a factor of sqrt(scale).

  result.method = 'Binned-BFGS'
  result.converged = optimize_res.success
  result.iterations = getattr(optimize_res, 'nit', 0)
  result.scale = resid_var
  result.fe_params = pd.Series(beta, index=model.exog_names)
  result.cov_re = np.atleast_2d(group_var)
  return result


def binned_fit(
    self,
    start_params=None,
    reml=False,
    method='BFGS',
    full_output=False,
    cov_pen=None,
    fe_pen=None,
    free=None,
    **fit_kwargs,
) -> MixedLMResults:
  """Fits a random-intercept model with the binned (restricted) likelihood.

  Steps: binned stats -> OLS warm start -> BFGS/L-BFGS-B on
  ``(beta, log var_re, log var_resid)`` -> GLS re-solve of beta -> analytic
  Hessian -> ``wrap_binned_results``.

  Args:
    self: The model.
    start_params: Ignored (kept for statsmodels signature compatibility; the OLS
      warm start converges in a handful of iterations).
    reml: False maximises the marginal likelihood (ML; use for BIC model
      comparison). True maximises the restricted likelihood (REML), which
      corrects the variance components for the fixed-effect degrees of freedom;
      ``llf`` is then the restricted log-likelihood and ``aic`` / ``bic`` are
      NaN.
    method: ``'bfgs'`` or ``'lbfgs'``; a stalled L-BFGS-B line search is retried
      with BFGS.
    full_output: Unused.
    cov_pen: Unused.
    fe_pen: Unused.
    free: Unused.
    **fit_kwargs: ``skip_hessian=True`` skips the Hessian (all standard errors
      NaN) for quick BIC screening. ``analytic_gradient=False`` lets scipy use
      finite differences instead of the exact gradient
      (``binned_loglike_and_grad``). Other keys are ignored.

  Returns:
    A ``MixedLMResults`` object.
  """
  del start_params, full_output
  skip_hessian = fit_kwargs.pop('skip_hessian', False)
  analytic_gradient = fit_kwargs.pop('analytic_gradient', True)
  self.reml = reml
  self.cov_pen = cov_pen
  self.fe_pen = fe_pen
  self._cov_sing = 0
  self._freepat = free

  method = {'lbfgs': 'L-BFGS-B', 'bfgs': 'BFGS'}.get(method.lower(), method)
  if method == 'L-BFGS-B':
    options = {'ftol': 1e-12, 'gtol': 1e-12}
  else:
    options = {'gtol': 1e-6}

  t0 = time.time()
  binned_stats = get_binned_stats(self)
  self.binned_data = binned_stats
  print(
      f'  [binned] {len(binned_stats)} group-size bins in'
      f' {time.time() - t0:.1f}s'
  )

  ols_res = sm.OLS(self.endog, self.exog).fit()
  init_params = np.r_[
      ols_res.params, [np.log(10.0), np.log(max(ols_res.mse_resid, 0.01))]
  ]

  objective_args = (binned_stats, self.k_fe, reml)
  res = optimize.minimize(
      binned_loglike_and_grad if analytic_gradient else binned_loglike,
      init_params,
      args=objective_args,
      method=method,
      jac=True if analytic_gradient else None,
      options=options,
  )
  if not np.isfinite(res.fun):
    res.fun = binned_loglike(res.x, *objective_args)
  if analytic_gradient and not res.success and method != 'BFGS':
    # L-BFGS-B line searches can stall at the precision floor; full-memory
    # BFGS stops on the gradient norm instead and converges there.
    retry = optimize.minimize(
        binned_loglike_and_grad,
        init_params,
        args=objective_args,
        method='BFGS',
        jac=True,
        options={'gtol': 1e-6},
    )
    if retry.success and np.isfinite(retry.fun):
      res = retry
  if not (res.success and np.isfinite(res.fun)):
    print(f'  [binned] WARNING: optimiser did not converge: {res.message}')

  var_re = np.exp(res.x[self.k_fe])
  var_resid = np.exp(res.x[self.k_fe + 1])
  rel_var_re = var_re / var_resid
  fe_params, _ = self.get_fe_params(
      np.atleast_2d(rel_var_re), np.zeros(self.k_vc)
  )
  res.x = np.r_[fe_params, res.x[self.k_fe :]]
  res.fun = binned_loglike(res.x, *objective_args)
  if not np.isfinite(res.fun):
    print(
        '  [binned] WARNING: log-likelihood is not finite at the GLS-updated'
        ' estimates; llf / AIC / BIC will be non-finite.'
    )

  if skip_hessian:
    h = None
  else:
    re_param = np.sqrt(rel_var_re) if self.use_sqrt else rel_var_re
    h, _ = self.hessian(np.r_[fe_params, [re_param]])

  result = wrap_binned_results(
      self, res, h, var_resid, hessian_skipped=skip_hessian
  )
  result.vcomp = np.zeros(self.k_vc)
  result.method = 'Binned-' + method
  print(
      f'  [binned] fit done in {time.time() - t0:.1f}s'
      f' (converged={res.success})'
  )
  return result


class BinnedMixedLM(sm.MixedLM):
  """Random-intercept ``MixedLM`` that never materialises per-group arrays.

  Only ``groups`` (random intercept) is supported: ``exog_re``/``exog_vc`` are
  ignored. Use ``BinnedMixedLM.from_formula(formula, data, groups='user_id_n')``
  exactly like ``smf.mixedlm``.
  """

  def __init__(
      self,
      endog,
      exog,
      groups,
      exog_re=None,
      exog_vc=None,
      use_sqrt=True,
      missing='none',
      **kwargs,
  ):
    del exog_vc
    self.use_sqrt = use_sqrt
    self.reml = True
    self.fe_pen = None
    self.re_pen = None
    self.k_vc = 0
    self.exog_vc = mlm.VCSpec([], [], [])

    # Skip MixedLM.__init__ (which splits the data per group).
    super(sm.MixedLM, self).__init__(
        endog, exog, groups=groups, exog_re=exog_re, missing=missing, **kwargs
    )

    self.k_fe = exog.shape[1]
    self.k_re = 1
    self.k_re2 = 1
    self.exog_re = np.ones((len(endog), 1), dtype=np.float64)
    self.data.exog_re = self.exog_re
    names = ['Group Var']
    self.data.param_names = self.exog_names + names
    self.data.exog_re_names = names
    self.data.exog_re_names_full = names
    self.k_params = self.k_fe + self.k_re2

    self.group_labels = np.unique(groups)
    self.n_groups = len(self.group_labels)
    self.endog_li = self.exog_li = self.exog_re_li = self.exog_re2_li = None
    self.nobs = len(self.endog)
    self.n_totobs = self.nobs
    self._aex_r = self._aex_r2 = None
    self._lin, self._quad = self._reparam()

  def fit(self, start_params=None, reml=False, method='BFGS', **fit_kwargs):
    return binned_fit(self, start_params, reml, method, **fit_kwargs)

  def hessian(self, params):
    return binned_hessian(self, params)

  def get_fe_params(self, cov_re, vcomp, tol=1e-8):
    return binned_get_fe_params(self, cov_re, vcomp, tol=tol)

  def group_list(self, array):  # pylint: disable=unused-argument
    return None


# -----------------------------------------------------------------------------
# Fast BLUPs
# -----------------------------------------------------------------------------
def inject_fast_random_effects(result: MixedLMResults) -> pd.Series:
  """Vectorised random intercepts; fills the results cache in place.

  For a random-intercept model the BLUP of user *i* is
  ``var_re * sum(resid_i) / (var_re * n_i + var_resid)`` where ``resid_i`` are
  the fixed-effects residuals. Injects ``random_effects``, ``fittedvalues`` and
  ``resid`` into ``result._cache`` so that ``sl.summary_report`` etc. do not
  trigger statsmodels' per-user Python loop. Uses the model's own ``endog`` and
  ``groups`` arrays, so rows dropped by patsy are handled consistently.

  Args:
    result: Fitted results (from ``BinnedMixedLM``).

  Returns:
    Series of BLUPs indexed by group label.
  """
  model = result.model
  cov_re = result.cov_re
  var_re = (
      cov_re.iloc[0, 0] if isinstance(cov_re, pd.DataFrame) else cov_re[0, 0]
  )
  var_resid = result.scale

  index = getattr(model.data, 'row_labels', None)
  if index is None:
    index = pd.RangeIndex(model.nobs)
  groups = pd.Series(np.asarray(model.groups), index=index)
  fixed_pred = pd.Series(np.asarray(result.predict()), index=index)
  residuals = pd.Series(np.asarray(model.endog), index=index) - fixed_pred
  grouped = residuals.groupby(groups)
  blups = (var_re * grouped.sum()) / (var_re * grouped.count() + var_resid)

  if not hasattr(result, '_cache'):
    result._cache = {}
  result._cache['random_effects'] = {k: [v] for k, v in blups.items()}
  fitted = fixed_pred + groups.map(blups)
  result._cache['fittedvalues'] = fitted
  result._cache['resid'] = residuals - groups.map(blups)
  return blups


# -----------------------------------------------------------------------------
# Checkpointing
# -----------------------------------------------------------------------------
def _json_default(obj):
  """JSON encoder for numpy scalars / arrays (keeps ints and bools typed)."""
  if isinstance(obj, np.bool_):
    return bool(obj)
  if isinstance(obj, np.integer):
    return int(obj)
  if isinstance(obj, np.floating):
    return float(obj)
  if isinstance(obj, np.ndarray):
    return obj.tolist()
  return str(obj)


def save_checkpoint(filepath: str, state: Mapping[str, Any]) -> None:
  """Writes ``state`` as JSON (creates parent directories)."""
  with fio.open_file(filepath, 'w') as f:
    json.dump(state, f, indent=2, default=_json_default)
  print(f'  [checkpoint] saved {filepath}')


def load_checkpoint(filepath: str) -> Optional[Dict[str, Any]]:
  """Reads a JSON checkpoint, or returns ``None`` if absent/unreadable."""
  if not fio.file_exists(filepath):
    return None
  try:
    with fio.open_file(filepath, 'r') as f:
      state = json.load(f)
  except (OSError, ValueError) as e:
    print(f'  [checkpoint] WARNING: could not read {filepath}: {e}')
    return None
  print(f'  [checkpoint] loaded {filepath}')
  return state
