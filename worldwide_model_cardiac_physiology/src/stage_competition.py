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

"""Progressive (stage-wise forward) LME model competition with overrides.

The model is selected in six stages. Each stage fits a small set of
candidate right-hand sides, picks the one with the lowest BIC and carries it
forward as the base of the next stage:

```
Stage1 (age form)          -> s1_rhs
Stage2 (seasonality form)  -> s2_base
Stage3 (season x day-length; diagnostic only, not carried forward)
Stage4 (latitude form +- longitude, then +- season x latitude interaction)
Stage5 (temperature form, then +- season x temperature interaction)
Stage6 (joint competition with temperature residualised on latitude/season)
```

This module holds that logic in a pure form so that

* the notebook (`notebooks/02_modelling.ipynb`) only wires in the real fitter,
* the progression can be unit-tested with a mock fitter (no real data), and
* any stage decision can be **overridden** (`STAGE_OVERRIDES`) to study a
  counterfactual path, e.g. "what if the linear age model (1.1) had progressed
  instead of the quadratic one (1.2)?".

Every fitted row is annotated with

* `stage`: decision point the row belongs to (`Stage1`, ..., `Stage4_Base`,
  `Stage4_Interaction`, ..., `Stage6`);
* `candidate`: candidate key within the stage;
* `bic_winner`: lowest BIC within its stage among converged fits (a fit that
  did not converge only wins when no candidate converged);
* `progressed`: the row that was carried forward (== `bic_winner` unless
  forced); always `False` for the diagnostic Stage 3;
* `forced`: `progressed` because of an override, not because of BIC;
* `on_path`: `False` for rows of an overridden stage that were *not* forced
  and for every Stage 3 row; the overall winner is chosen among `on_path`
  rows that converged.

With an empty override dict the progression is the default pure-BIC
selection (names, row order and winners of `02_modelling.ipynb`).
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import math
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import pandas as pd
from patsy import desc as patsy_desc

# A fitter takes (formula, model_name, **kwargs) and returns
# (fit_result_or_None, model_row_dict, vif_or_None). ``model_row_dict`` must
# contain at least 'model', 'formula' and 'BIC'.
FitFn = Callable[..., Tuple[Any, Dict[str, Any], Any]]
CheckpointFn = Callable[[Dict[str, Any]], None]
Override = Union[str, bool]
Overrides = Mapping[str, Override]

TEMP_VARS = (
    'avg_temp_mean_c',
    'avg_temp_max_c',
    'avg_temp_min_c',
    'avg_temp_max_sq_c',
    'avg_temp_min_sq_c',
)

# Decision points, in execution order. Candidate stages take a candidate key;
# interaction stages take a bool (``True`` = carry the interaction forward).
# Stage3 is diagnostic (nothing is carried forward), so it has no decision
# point.
CANDIDATE_STAGES = (
    'Stage1',
    'Stage2',
    'Stage4_Base',
    'Stage5_Base',
    'Stage6',
)
INTERACTION_STAGES = ('Stage4_Interaction', 'Stage5_Interaction')
ALL_STAGES = CANDIDATE_STAGES + INTERACTION_STAGES

# Notebook stage number of each decision point (``max_stage`` cut-off).
_STAGE_NUMBER = {
    'Stage1': 1,
    'Stage2': 2,
    'Stage3': 3,
    'Stage4_Base': 4,
    'Stage4_Interaction': 4,
    'Stage5_Base': 5,
    'Stage5_Interaction': 5,
    'Stage6': 6,
}


@dataclasses.dataclass(frozen=True)
class StageSpec:
  """Candidate right-hand-side fragments of the competition.

  Attributes:
    stage1: Baseline demographic forms (Stage 1 candidates).
    seasonal: Harmonic seasonality forms (Stage 2 ``Base_<key>`` candidates; the
      day-length alternatives are added automatically).
    lat_options: Latitude forms (Stage 4), each also tried with longitude.
    temp_options: Temperature forms (Stage 5).
  """

  stage1: Mapping[str, str]
  seasonal: Mapping[str, str]
  lat_options: Mapping[str, str]
  temp_options: Mapping[str, str]

  def candidates(self, stage: str) -> Tuple[str, ...]:
    """Returns the candidate keys a decision point can be forced to."""
    if stage == 'Stage1':
      return tuple(self.stage1)
    if stage == 'Stage2':
      return tuple(self.stage2_candidates('').keys())
    if stage == 'Stage4_Base':
      out = []
      for key in self.lat_options:
        out += [f'Lat_{key}_MainOnly', f'Lat_{key}_MainOnly_with_Lng']
      return tuple(out)
    if stage == 'Stage5_Base':
      return tuple(f'{key}_MainOnly' for key in self.temp_options)
    if stage == 'Stage6':
      return (
          '6.1_Lat_Centric_Joint',
          '6.2_Temp_Centric_Joint',
          '6.3_Joint_Competition',
      )
    raise KeyError(f'Unknown decision point {stage!r}')

  def stage2_candidates(self, s1: str) -> Dict[str, str]:
    """Stage 2 candidates (harmonics and day-length forms) on base ``s1``."""
    out = {f'Base_{k}': f'{s1} + {v}' for k, v in self.seasonal.items()}
    out['Base_DL_Linear'] = f'{s1} + dl_monthly_c'
    out['Base_DL_Quadratic'] = f'{s1} + dl_monthly_c + dl_monthly_sq_c'
    return out

  def stage2_base(self, key: str) -> str:
    """Seasonal fragment carried forward for a Stage 2 candidate key."""
    if key.startswith('Base_') and key[5:] in self.seasonal:
      return self.seasonal[key[5:]]
    return {
        'Base_DL_Linear': 'dl_monthly_c',
        'Base_DL_Quadratic': 'dl_monthly_c + dl_monthly_sq_c',
    }[key]


DEFAULT_SPEC = StageSpec(
    stage1={
        '1.1_Linear': 'age_c * gender_n',
        '1.2_Quadratic': '(age_c + age_sq_c) * gender_n',
    },
    seasonal={
        '2H': 'month_sin + month_cos',
        '4H': 'month_sin + month_cos + month_sin_2 + month_cos_2',
    },
    lat_options={'Linear': 'lat_e7_c', 'Quadratic': 'lat_e7_c + lat_e7_sq_c'},
    temp_options={
        'Mean': 'avg_temp_mean_c',
        'MaxMin_Linear': 'avg_temp_max_c + avg_temp_min_c',
        'MaxMin_Quadratic': (
            'avg_temp_max_c + avg_temp_max_sq_c + avg_temp_min_c'
            ' + avg_temp_min_sq_c'
        ),
    },
)


def residualise_temperature(formula: str) -> str:
  """Swaps every temperature term in ``formula`` for its ``_res`` version."""
  for var in TEMP_VARS:
    formula = re.sub(r'\b' + re.escape(var) + r'\b', f'{var}_res', formula)
  return formula


def _converged(row: Mapping[str, Any]) -> bool:
  """Reads a row's optional ``converged`` flag (absent counts as converged)."""
  value = row.get('converged', True)
  if isinstance(value, str):
    return value.strip().lower() != 'false'
  if isinstance(value, float) and math.isnan(value):
    return True
  return bool(value)


def _same_terms(rhs_a: str, rhs_b: str) -> bool:
  """True when two right-hand sides expand to the same set of patsy terms."""

  def terms(rhs):
    return set(patsy_desc.ModelDesc.from_formula(f'y ~ {rhs}').rhs_termlist)

  return terms(rhs_a) == terms(rhs_b)


def fingerprint(target: str, spec: StageSpec) -> str:
  """Stable digest of what a checkpoint was computed for (target + spec).

  Mapping keys are sorted, so two specs with the same candidates in a
  different insertion order share a fingerprint.

  Args:
    target: Response column of the competition.
    spec: Candidate specification.

  Returns:
    16 hex characters.
  """
  payload = json.dumps(
      {'target': target, 'spec': dataclasses.asdict(spec)}, sort_keys=True
  )
  return hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]


def _normalise_data_key(
    data_key: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
  """Copy of a data description that survives the JSON checkpoint round trip."""
  out: Dict[str, Any] = {}
  for k, v in sorted((data_key or {}).items()):
    out[str(k)] = v.item() if hasattr(v, 'item') else v  # numpy scalars
  return out


def validate_overrides(
    overrides: Optional[Overrides], spec: StageSpec = DEFAULT_SPEC
) -> Dict[str, Override]:
  """Checks override keys/values and returns them as a plain dict.

  Args:
    overrides: ``{decision_point: candidate_key | bool}``; see module doc.
    spec: Candidate specification the keys are validated against.

  Returns:
    A copy of ``overrides`` (empty dict when ``None``).

  Raises:
    ValueError: on an unknown decision point, an unknown candidate key, or a
      non-bool value for an interaction decision.
  """
  out: Dict[str, Override] = dict(overrides or {})
  for stage, value in out.items():
    if stage not in ALL_STAGES:
      raise ValueError(
          f'Unknown decision point {stage!r}; expected one of {ALL_STAGES}'
      )
    if stage in INTERACTION_STAGES:
      if not isinstance(value, bool):
        raise ValueError(f'{stage} override must be a bool, got {value!r}')
      continue
    allowed = spec.candidates(stage)
    if value not in allowed:
      raise ValueError(
          f'{stage} override {value!r} is not a candidate; expected one of'
          f' {allowed}'
      )
  return out


def select_winner(
    stage: str,
    bics: Mapping[str, float],
    overrides: Overrides,
    converged: Optional[Mapping[str, bool]] = None,
) -> Tuple[str, str]:
  """Returns ``(progressed_key, bic_winner_key)`` for a candidate stage.

  Candidates with a non-finite BIC never win. Candidates whose fit did not
  converge are excluded as well, unless no candidate converged, in which case
  the lowest finite BIC wins (callers log a warning).

  Args:
    stage: Decision point name (key into ``overrides``).
    bics: ``{candidate_key: BIC}`` of the fitted candidates.
    overrides: Validated ``STAGE_OVERRIDES``.
    converged: Optional ``{candidate_key: bool}`` convergence flags; missing
      keys count as converged.

  Returns:
    ``(progressed_key, bic_winner_key)``; equal unless ``stage`` is overridden.

  Raises:
    ValueError: if no candidate has a finite BIC, or the forced candidate is
      missing or has a non-finite BIC.
  """
  if not bics:
    raise ValueError(f'{stage}: no candidates were fitted')
  finite = {k: float(v) for k, v in bics.items() if math.isfinite(float(v))}
  bad = sorted(set(bics) - set(finite))
  if not finite:
    raise ValueError(f'{stage}: every candidate has a non-finite BIC: {bad}')
  converged = converged or {}
  eligible = {k: v for k, v in finite.items() if converged.get(k, True)}
  if not eligible:
    eligible = finite
  bic_winner = min(eligible, key=lambda k: eligible[k])
  progressed = overrides.get(stage)
  if progressed is None:
    return bic_winner, bic_winner
  if progressed not in bics:
    raise ValueError(
        f'{stage} override {progressed!r} not among fitted candidates'
        f' {tuple(bics)}'
    )
  if progressed not in finite:
    raise ValueError(
        f'{stage} override {progressed!r} has a non-finite BIC (fit did not'
        ' converge); refusing to carry it forward'
    )
  return str(progressed), bic_winner


def new_state(overrides: Optional[Overrides] = None) -> Dict[str, Any]:
  """Fresh checkpoint state for :func:`run_competition`."""
  return {
      'current_stage': 1,
      'all_stage_results': [],
      'overrides': dict(overrides or {}),
  }


def overall_winner(results: pd.DataFrame) -> pd.Series:
  """Lowest-BIC row among ``on_path`` rows that converged.

  Mirrors the per-stage rule of :func:`select_winner`: rows flagged
  ``converged == False`` are skipped unless no on-path row converged, in which
  case the lowest BIC among all on-path rows is returned. Rows of the
  diagnostic Stage 3 are written ``on_path == False`` and never win. Without
  an ``on_path`` column all rows are eligible.

  Args:
    results: Model-results rows (needs ``BIC``; uses ``on_path`` and
      ``converged`` when present).

  Returns:
    The winning row.

  Raises:
    ValueError: no on-path row to choose from.
  """
  df = results
  if 'on_path' in df.columns:
    df = df[df['on_path'].astype(bool)]
  if df.empty:
    raise ValueError('No on-path candidate rows to choose a winner from')
  if 'converged' in df.columns:
    ok = df.apply(_converged, axis=1).astype(bool)
    if ok.any():
      df = df[ok]
  return df.loc[df['BIC'].astype(float).idxmin()]


class _Competition:
  """Internal driver; see :func:`run_competition` for the public entry point."""

  def __init__(
      self,
      fit_fn: FitFn,
      target: str,
      state: Dict[str, Any],
      spec: StageSpec,
      overrides: Dict[str, Override],
      checkpoint_fn: Optional[CheckpointFn],
      fit_kwargs: Dict[str, Any],
      log: Callable[[str], None],
  ):
    self.fit_fn = fit_fn
    self.target = target
    self.state = state
    self.spec = spec
    self.overrides = overrides
    self.checkpoint_fn = checkpoint_fn
    self.fit_kwargs = fit_kwargs
    self.log = log
    self.rows: List[Dict[str, Any]] = state['all_stage_results']

  # -- helpers ---------------------------------------------------------------

  def _fit(self, stage: str, key: str, rhs: str) -> Dict[str, Any]:
    """Fits one candidate and returns its annotated (not yet ranked) row."""
    formula = f'{self.target} ~ {rhs}'
    _, row, _ = self.fit_fn(formula, f'{stage}_{key}', **self.fit_kwargs)
    row = dict(row)
    row.update(
        model=f'{stage}_{key}',
        stage=stage,
        candidate=key,
        bic_winner=False,
        progressed=False,
        forced=False,
        on_path=True,
    )
    return row

  def _discard_stage_rows(self, stage: str) -> None:
    """Drops rows of ``stage`` left behind by an interrupted earlier attempt."""
    self.rows[:] = [r for r in self.rows if r.get('stage') != stage]

  def _run_candidates(
      self, stage: str, candidates: Mapping[str, str]
  ) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Fits all candidates of a stage; returns (progressed key, rows by key)."""
    self._discard_stage_rows(stage)
    rows = {key: self._fit(stage, key, rhs) for key, rhs in candidates.items()}
    bics = {key: row['BIC'] for key, row in rows.items()}
    converged = {key: _converged(row) for key, row in rows.items()}
    progressed, bic_winner = select_winner(
        stage, bics, self.overrides, converged
    )
    forced = stage in self.overrides
    for key, row in rows.items():
      row['bic_winner'] = key == bic_winner
      row['progressed'] = key == progressed
      row['forced'] = forced and key == progressed
      row['on_path'] = (not forced) or key == progressed
      self.rows.append(row)
    bad = [k for k, v in bics.items() if not math.isfinite(float(v))]
    if bad:
      self.log(f'    [{stage}] WARNING: non-finite BIC (not converged): {bad}')
    not_converged = [
        k for k, ok in converged.items() if not ok and k not in bad
    ]
    if not_converged:
      self.log(
          f'    [{stage}] excluded from selection (did not converge):'
          f' {not_converged}'
      )
    if not converged[progressed]:
      self.log(
          f'    [{stage}] WARNING: the progressed candidate {progressed!r} did'
          ' not report convergence; inspect its fit before trusting the path.'
      )
    msg = f'\n>>> {stage} winner: {progressed} (BIC {bics[progressed]:.2f})'
    if forced and progressed != bic_winner:
      msg += f'  [FORCED by override; BIC winner was {bic_winner}]'
    self.log(msg)
    return progressed, rows

  def _interaction(
      self,
      stage: str,
      base_row: Dict[str, Any],
      inter_rhs: str,
      model_name: str,
  ) -> bool:
    """Fits the interacted model; returns whether it is carried forward."""
    self._discard_stage_rows(stage)
    row = self._fit(stage, 'Interacted', inter_rhs)
    converged = _converged(row)
    bic_better = converged and float(row['BIC']) < float(base_row['BIC'])
    if not converged:
      self.log(
          f'    [{stage}] the interacted model did not converge; it is not'
          ' carried forward on BIC.'
      )
    forced = stage in self.overrides
    keep = bool(self.overrides[stage]) if forced else bic_better
    row.update(
        model=model_name,
        bic_winner=bic_better,
        progressed=keep,
        forced=forced and keep,
        on_path=keep or not forced,
    )
    if keep:
      # Convention: the interacted row only enters the results table
      # when it progresses (its name then replaces the base winner's).
      self.rows.append(row)
    elif forced:
      # Forced *not* to interact although it was fitted: keep the audit row
      # but flag it as off-path so it cannot become the overall winner.
      row['on_path'] = False
      self.rows.append(row)
    if forced and keep != bic_better:
      self.log(
          f'    [{stage} FORCED by override: interaction={keep};'
          f' BIC preferred interaction={bic_better}]'
      )
    return keep

  def _checkpoint(self, next_stage: int, **winners: Any) -> None:
    self.state.update(
        current_stage=next_stage, all_stage_results=self.rows, **winners
    )
    if self.checkpoint_fn is not None:
      self.checkpoint_fn(self.state)
    gc.collect()

  def _due(self, stage_number: int, max_stage: int) -> bool:
    return (
        self.state['current_stage'] <= stage_number
        and max_stage >= stage_number
    )

  # -- stages ----------------------------------------------------------------

  def run(self, max_stage: int) -> None:
    """Executes the stages that are not yet checkpointed, up to ``max_stage``."""
    st, spec = self.state, self.spec

    if self._due(1, max_stage):
      w, _ = self._run_candidates('Stage1', spec.stage1)
      self._checkpoint(2, s1_rhs=spec.stage1[w])

    if self._due(2, max_stage):
      s1 = st['s1_rhs']
      w, _ = self._run_candidates('Stage2', spec.stage2_candidates(s1))
      self._checkpoint(3, s2_base=spec.stage2_base(w))

    if self._due(3, max_stage):
      s1, s2 = st['s1_rhs'], st['s2_base']
      h2, h4 = spec.seasonal.get('2H'), spec.seasonal.get('4H')
      candidates = {'Stage2_Winner': f'{s1} + {s2}'}
      if h2:
        candidates['Interacted_DL_2H'] = f'{s1} + dl_monthly_c * ({h2})'
      if h4:
        candidates['Interacted_DL_4H'] = (
            f'{s1} + (dl_monthly_c + dl_monthly_sq_c) * ({h4})'
        )
      w, rows = self._run_candidates('Stage3', candidates)
      # Diagnostic stage: the lowest-BIC candidate is recorded (``bic_winner``)
      # but nothing is carried forward, so no Stage 3 row is ``progressed``
      # or ``on_path`` and :func:`overall_winner` never picks one.
      for row in rows.values():
        row['progressed'] = False
        row['on_path'] = False
      self._checkpoint(4, s3_winner=w)

    if self._due(4, max_stage):
      s1, s2 = st['s1_rhs'], st['s2_base']
      candidates, meta = {}, {}
      for key, lat_rhs in spec.lat_options.items():
        candidates[f'Lat_{key}_MainOnly'] = f'{s1} + ({s2}) + ({lat_rhs})'
        candidates[f'Lat_{key}_MainOnly_with_Lng'] = (
            f'{s1} + ({s2}) + ({lat_rhs}) + sin_lng + cos_lng'
        )
        meta[f'Lat_{key}_MainOnly'] = (lat_rhs, False)
        meta[f'Lat_{key}_MainOnly_with_Lng'] = (lat_rhs, True)
      w, rows = self._run_candidates('Stage4_Base', candidates)
      lat_rhs, with_lng = meta[w]
      lng_part = ' + sin_lng + cos_lng' if with_lng else ''
      interact = self._interaction(
          'Stage4_Interaction',
          rows[w],
          f'{s1} + ({s2}) * ({lat_rhs}){lng_part}',
          f"Stage4_Best_{w.replace('_MainOnly', '_Interacted')}",
      )
      s4_model = (
          f"Stage4_Best_{w.replace('_MainOnly', '_Interacted')}"
          if interact
          else rows[w]['model']
      )
      self.log(f'\n>>> Stage4 winner: {s4_model} (interaction={interact})')
      self._checkpoint(
          5,
          s4_lat_rhs=lat_rhs,
          s4_with_lng=with_lng,
          s4_interact=interact,
          s4_model=s4_model,
      )

    if self._due(5, max_stage):
      s1, s2 = st['s1_rhs'], st['s2_base']
      candidates = {
          f'{key}_MainOnly': f'{s1} + ({s2}) + ({rhs})'
          for key, rhs in spec.temp_options.items()
      }
      w, rows = self._run_candidates('Stage5_Base', candidates)
      temp_rhs = spec.temp_options[w.replace('_MainOnly', '')]
      interact = self._interaction(
          'Stage5_Interaction',
          rows[w],
          f'{s1} + ({s2}) * ({temp_rhs})',
          f"Stage5_Best_{w.replace('_MainOnly', '_Interacted')}",
      )
      s5_model = (
          f"Stage5_Best_{w.replace('_MainOnly', '_Interacted')}"
          if interact
          else rows[w]['model']
      )
      self.log(f'\n>>> Stage5 winner: {s5_model} (interaction={interact})')
      self._checkpoint(
          6, s5_temp_rhs=temp_rhs, s5_interact=interact, s5_model=s5_model
      )

    if self._due(6, max_stage):
      s1, s2 = st['s1_rhs'], st['s2_base']
      lat_rhs = st['s4_lat_rhs']
      lng_part = ' + sin_lng + cos_lng' if st['s4_with_lng'] else ''
      temp_res = residualise_temperature(st['s5_temp_rhs'])
      op4 = '*' if st['s4_interact'] else '+'
      op5 = '*' if st['s5_interact'] else '+'
      candidates = {
          '6.1_Lat_Centric_Joint': (
              f'{s1} + ({s2}) {op4} ({lat_rhs}){lng_part} + {temp_res}'
          ),
          '6.2_Temp_Centric_Joint': (
              f'{s1} + ({s2}) {op5} ({temp_res}) + {lat_rhs}{lng_part}'
          ),
          '6.3_Joint_Competition': (
              f'{s1} + ({s2}) {op4} ({lat_rhs}) + ({s2}) {op5} ({temp_res})'
              f'{lng_part}'
          ),
      }
      # Candidates that expand to the same terms are fitted once. A forced
      # candidate (``overrides['Stage6']``) is the one kept, so the override
      # still names a fitted model.
      forced = self.overrides.get('Stage6')
      distinct: Dict[str, str] = {}
      for key, rhs in candidates.items():
        same = [k for k, v in distinct.items() if _same_terms(v, rhs)]
        if same and key == forced:
          del distinct[same[0]]
          self.log(f'    [Stage6] {same[0]} is identical to {key}; skipped')
          distinct[key] = rhs
        elif same:
          self.log(f'    [Stage6] {key} is identical to {same[0]}; skipped')
        else:
          distinct[key] = rhs
      w, _ = self._run_candidates('Stage6', distinct)
      self._checkpoint(7, s6_winner=w)


def run_competition(
    fit_fn: FitFn,
    target: str,
    state: Optional[Dict[str, Any]] = None,
    *,
    spec: StageSpec = DEFAULT_SPEC,
    max_stage: int = 6,
    overrides: Optional[Overrides] = None,
    checkpoint_fn: Optional[CheckpointFn] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    data_key: Optional[Mapping[str, Any]] = None,
    log: Callable[[str], None] = print,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
  """Runs (or resumes) the six-stage competition.

  Args:
    fit_fn: ``fit_fn(formula, name, **fit_kwargs) -> (result, row, vif)``;
      ``row`` is a dict with at least ``'model'``, ``'formula'``, ``'BIC'`` and,
      if available, a ``'converged'`` flag.
    target: Left-hand side of every formula (e.g.
      ``'avg_resting_heart_rate_bpm'`` or ``'target_log'``).
    state: Checkpoint state from a previous call (resumes after the last
      completed stage) or ``None`` / :func:`new_state` to start afresh.
    spec: Candidate fragments.
    max_stage: Stop after this notebook stage (1-6).
    overrides: ``{decision_point: candidate_key | bool}``; forces a candidate
      (or an interaction decision) to progress regardless of BIC. ``None`` /
      ``{}`` means pure BIC selection.
    checkpoint_fn: Called with ``state`` after every completed stage.
    fit_kwargs: Extra keyword arguments forwarded to every ``fit_fn`` call.
    data_key: Description of the fitted data (e.g. row and user counts,
      subsample fraction, thinning flag). It is stored in the checkpoint and a
      checkpoint written for different data is not resumed.
    log: Progress sink.

  Returns:
    ``(state, results)`` where ``results`` has one row per fitted candidate in
    execution order (stages up to ``max_stage`` only), with the annotation
    columns described in the module doc.

  Raises:
    ValueError: for invalid overrides, or when resuming a checkpoint that was
      written with different overrides, a different target, a different spec
      or different data (the two paths are not comparable).
  """
  overrides = validate_overrides(overrides, spec)
  if not 1 <= max_stage <= 6:
    raise ValueError(f'max_stage must be in 1..6, got {max_stage}')
  state = state if state is not None else new_state(overrides)
  state.setdefault('all_stage_results', [])
  state.setdefault('current_stage', 1)
  stored = state.setdefault('overrides', dict(overrides))
  if stored != overrides:
    raise ValueError(
        f'Checkpoint was written with overrides {stored}, current run uses'
        f' {overrides}; use FORCE_CLEAN or a different run tag.'
    )
  digest = fingerprint(target, spec)
  stored_digest = state.setdefault('fingerprint', digest)
  if stored_digest != digest:
    raise ValueError(
        'Checkpoint was written for a different target or candidate spec;'
        ' use FORCE_CLEAN or a different run tag.'
    )
  key = _normalise_data_key(data_key)
  stored_key = state.setdefault('data_key', key)
  if stored_key != key:
    raise ValueError(
        f'Checkpoint was written for different data {stored_key}, current'
        f' run uses {key}; use FORCE_CLEAN or a different run tag.'
    )
  _Competition(
      fit_fn,
      target,
      state,
      spec,
      overrides,
      checkpoint_fn,
      dict(fit_kwargs or {}),
      log,
  ).run(max_stage)
  rows = [
      r
      for r in state['all_stage_results']
      if _STAGE_NUMBER.get(r.get('stage'), 0) <= max_stage
  ]
  return state, pd.DataFrame(rows)
