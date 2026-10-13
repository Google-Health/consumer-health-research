# Worldwide Model of Cardiac Physiology

Code accompanying the paper

> **Global Geographic, Seasonal, Environmental and Demographic Drivers of
> Variations in Longitudinal Cardiac Physiology in 5 million Fitbit Users**

The repository contains the complete statistical pipeline: cohort statistics,
the six-stage BIC-selected linear mixed-effects (LME) model competition for
resting heart rate (RHR) and heart-rate variability (HRV), and the producers of
the publication tables and figures.

The Fitbit cohort itself is **not** redistributed. A step-0 notebook writes a
small synthetic dataset in the public input schema so that the whole pipeline
can be run end to end; nothing estimated from it is meaningful.

## Authors & Contacts

*   Primary Author: Katarina Vukosavljević
*   Point of Contact / Code Review: Daniel Roggen (`roggen@google.com`)

## Repository structure

```text
worldwide_model_cardiac_physiology/
├── LICENSE.txt                        # Apache License 2.0
├── README.md                          # this file
├── requirements.txt                   # Python dependencies (pip)
├── src/                               # library modules
│   ├── file_io.py                     # filesystem adapter + data-layout helpers (./data or $WPM_DATA_DIR)
│   ├── cohort_io.py                   # cohort loader (public CSV schema)
│   ├── generate_dummy_data.py         # synthetic data/dummy_data*.csv
│   ├── seasonality_lib.py             # thinning, feature engineering, model statistics, plotting
│   ├── lme_binned.py                  # fast random-intercept LME (binned likelihood) == statsmodels MixedLM
│   ├── stage_competition.py           # six-stage progressive model competition + STAGE_OVERRIDES
│   ├── paper_tables.py                # LaTeX tables (model competition, coefficients, demographics)
│   └── publication_figures.py         # publication figures
├── notebooks/
│   ├── 00_generate_dummy_data.ipynb   # writes the synthetic input CSVs into data/
│   ├── 01_cohort_stats.ipynb          # model-free: demographics tables, distributions, observed maps, thinning validation
│   ├── 02_modelling.ipynb             # six-stage BIC-selected LME competition (one run per metric)
│   └── 03_model_results.ipynb         # model-derived tables/figures from a run
└── data/
    └── README.md                      # input schema and optional reference tables
```

## Installation

Python ≥ 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

`file_io.get_data_dir()` is `./data` (or the directory in the `WPM_DATA_DIR`
environment variable). Layout:

```text
<data>/dummy_data.csv, dummy_data_hrv.csv   cohort inputs (public schema, see data/README.md)
<data>/extra_data/                          optional reference tables for the map figures
<data>/results/<run_tag>/                   one modelling run (02)
<data>/results/paper/                       optional: publication assets written by 01 + 03
                                            when SAVE_PAPER_ASSETS = True
```

Input schema (one row per user and calendar month): `id, month, age, sex,
country, latitude, longitude, temperature, temperature min, temperature max, day
light, resting heart rate | heart rate variability [, region]`. To analyse your
own data, write it in this schema and point `WPM_DATA_DIR` at it (or place the
files in `data/`). `cohort_io.load_cohort` maps the headers to the column names
and units used by the code.

The map figures (observed seasonal maps, world residual contours) additionally
need two public reference files in `<data>/extra_data/` (Natural Earth admin-0
geojson and the UNSD M49 table — sources and terms of use in `data/README.md`).
They are skipped with a message when absent. The per-capita user-representation
map of the paper is not included in this release.

## Running the pipeline

Launch Jupyter from the repository root (`jupyter lab`), open the notebooks in
`notebooks/` and run them top to bottom:

0.  **`00_generate_dummy_data.ipynb`** — writes `data/dummy_data.csv` and
    `data/dummy_data_hrv.csv` (synthetic, seeded; skipped if present). Skip this
    step when using your own data.
1.  **`01_cohort_stats.ipynb`** — everything that does not need a fitted model:
    RHR thinning of the age-22 reporting artefact
    (`seasonality_lib.apply_density_based_thinning`), demographics tables by
    latitude band, demographic distributions, observed seasonal maps,
    within-user sensitivity and the thinning-validation figures.
2.  **`02_modelling.ipynb`** — once per metric (`METRIC = 'rhr' | 'hrv'` in the
    `CONFIG` cell). Feature engineering (Gelman standardisation, annual and
    semi-annual harmonics, temperature residualised on latitude × season), then
    the six-stage competition (`stage_competition.run_competition`): baseline
    (age × sex) → seasonality (harmonics vs day length) → seasonal × day-length
    (diagnostic: fitted and reported, not carried forward) → spatial (latitude
    ± longitude) → temperature (mean vs max/min) → joint competition. Every
    candidate is a participant random-intercept LME fitted with
    `lme_binned.BinnedMixedLM` (identical estimates to `statsmodels.MixedLM`,
    O(#group sizes) instead of O(N)); each stage carries its lowest-BIC
    candidate forward, the overall winner is refitted with REML.
    `STAGE_OVERRIDES` forces a different candidate to progress (counterfactual
    paths); runs are resumable. Outputs to `results/<run_tag>/`.
3.  **`03_model_results.ipynb`** — from a run (`RUN_TAGS`) and either its
    best-performing model or a chosen model folder (`MODEL_DIRS`): all LME
    tables, empirical-Bayes random intercepts for every user, residual / QQ
    diagnostics and, with both metrics, the combined latitude-band, country,
    demographic-calibration and world-residual figures. Section 2 compares two
    runs model by model.

Notebooks 01 and 03 are **reports**: every table and figure is rendered inline
(tables as a readable preview with the LaTeX source folded underneath) and
nothing is written to disk. Set `SAVE_PAPER_ASSETS = True` in their `CONFIG`
cell to also write the publication assets — `.tex` tables and 300-dpi PNG,
SVG and PDF figures — to `results/paper/`.

The UN-region and country trend plots of notebook 02 require at least 500
users per group (`MIN_USERS_PER_GROUP` in its `CONFIG` cell; lower it for
smaller cohorts), the demographic and thinning-validation figures of notebook
01 apply minimum-count thresholds per age and latitude band, and the map
figures need the reference files described above.

## Output → code map

Output                                                   | Notebook | Producer
:------------------------------------------------------- | :------- | :-------
Demographics by latitude band (`tab:*_demographics_*`)   | 01       | `paper_tables.demographics_table`
Demographic distributions, thinning validation           | 01       | `publication_figures.generate_figure_1 / _a1 / _a2 / _a3`
fig:{rhr,hrv}_observed_map                               | 01       | `publication_figures.plot_seasonal_maps_2x2`
fig:within_user_sensitivity                              | 01       | `publication_figures.make_within_user_sensitivity`
tab:results_*, *_full_grid, supp_results_*, *_ols_params | 03       | `paper_tables.generate_all_tables`
fig:{rhr,hrv}_residuals, fig:{rhr,hrv}_qq                | 03       | `publication_figures.make_model_residuals_plot / make_random_intercepts_plot`
fig:lat_bin_fits, fig:countries_prediction_fit           | 03       | `publication_figures.make_combined_seasonal_fit_lat_bin / _country`
fig:demographic_fit                                      | 03       | `publication_figures.make_joint_demographics_calibration`
fig:global_residuals_grid                                | 03       | `publication_figures.make_world_contour_residuals`

## Third-party software

The code builds on the open-source scientific Python stack listed in
`requirements.txt` (NumPy, pandas, SciPy, statsmodels, scikit-learn, matplotlib,
seaborn, plotly, GeoPandas, Jupyter), each distributed under its own licence
(BSD-3-Clause, MIT or similar). In particular `src/lme_binned.py` subclasses
`statsmodels.regression.mixed_linear_model.MixedLM` (statsmodels, BSD-3-Clause,
Copyright © 2009–2018 statsmodels Developers) and returns genuine
`MixedLMResults` objects; no third-party source code is vendored in this
repository.

## Citation

If you use this code, please cite the paper above.

## License

Copyright 2026 Google LLC

The full license text is in [`LICENSE.txt`](LICENSE.txt).

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

## Disclaimers

This is not an officially supported Google product. This project is not eligible
for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
This project is intended for demonstration purposes only. It is not intended for
use in a production environment.

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis, prognosis, or treatment recommendations.
