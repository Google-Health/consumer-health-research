# A Consumer Cohort Yields Accurate National Step Estimates in Young and Middle-Aged Adults

This project evaluates how close national step estimates from a large-scale
consumer wearable cohort are to national health survey benchmarks in Japan
(National Health and Nutrition Survey, NHNS-J), England (English Longitudinal
Study of Ageing, ELSA), and globally across 28 countries (compared against World
Health Organization estimates of insufficient physical activity).

## Authors & Contacts

* Primary Author: Bram Schonfeldt
* Point of Contact / Code Review: Daniel Roggen (`roggen@google.com`)

## Repository Structure

```text
worldwide_step_trends/
├── LICENSE.txt                                # Apache License 2.0
├── README.md                                  # Project documentation and external setup guide
├── stratify.py                                # Core demographic stratification and imputation library
│
└── colab/
    └── analysis/                              # Interactive analysis notebooks
        ├── 0_generate_dummy_data.ipynb        # Generates synthetic datasets in ./data/ for external testing
        ├── 1_demographic_pyramids.ipynb       # Demographic pyramids and wearable sampling bias
        ├── 2_japan_england_eval.ipynb         # Validation against Japan (NHNS-J) and England (ELSA)
        ├── 3_global_eval.ipynb                # 28-country correlation with WHO inactivity prevalence
        ├── 4_filling_gaps_japan.ipynb         # Survey gap-filling and longitudinal imputation in Japan
        └── 5_uk_trends.ipynb                  # Longitudinal UK step trends by age group and sex (2015-2024)
```

## External Setup & Reproducing Results

### 1. Python Environment Requirements

Install Python 3.10+ and the required scientific computing packages:

```bash
pip install numpy pandas scipy statsmodels patsy matplotlib seaborn openpyxl pycountry pytz absl-py jupyter
```

### 2. Data Directory Layout (`./data/`)

By default, `stratify.py` reads and writes data under `./data/`. Set
`STEP_TRENDS_DATA_DIR` to use a different location:

```text
worldwide_step_trends/
└── data/
    ├── internal/
    │   ├── stratify_steps/
    │   │   ├── checkpoint_grid_2015-01-01_2026-01-01_a36731bea99396ebe673f464d73ada56.pkl
    │   │   ├── checkpoint_completed_2015-01-01_2026-01-01_a36731bea99396ebe673f464d73ada56.pkl
    │   │   ├── checkpoint_processed_2015-01-01_2026-01-01_a36731bea99396ebe673f464d73ada56.pkl
    │   │   ├── checkpoint_grid_2015-01-01_2026-01-01_9f4397029d8039e43903bb5b8b7e75ec.pkl
    │   │   ├── checkpoint_completed_2015-01-01_2026-01-01_9f4397029d8039e43903bb5b8b7e75ec.pkl
    │   │   └── checkpoint_processed_2015-01-01_2026-01-01_9f4397029d8039e43903bb5b8b7e75ec.pkl
    │   └── 2021_2023_uk_50plus_participant_steps.pkl
    │
    └── public/
        ├── WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx
        ├── WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx
        ├── who_trends.csv
        └── 2015_2024_japan_steps.pkl
```

The two precomputed `StratifySteps` checkpoint MD5 hashes correspond to the
following aggregation parameter sets:

*   `a36731bea99396ebe673f464d73ada56` (**Global 28-country grid**):
    `start_date='2015-01-01'`, `end_date='2026-01-01'`, `n_shards=100`,
    `min_days_per_user=1` (across the full study window), `min_age=20`,
    `max_age=104`, `exclude_outliers=False`, `exclude_extreme_values=False`,
    `min_steps=None`, `max_steps=172800`, `exclude_days_of_week=None`,
    `target_country=None` (all countries), `start_month=None`,
    `stop_month=None`.
*   `9f4397029d8039e43903bb5b8b7e75ec` (**Japan November Mon–Sat grid**):
    `start_date='2015-01-01'`, `end_date='2026-01-01'`, `n_shards=100`,
    `min_days_per_user=1` (across the full study window), `min_age=20`,
    `max_age=104`, `exclude_outliers=False`, `exclude_extreme_values=False`,
    `min_steps=100`, `max_steps=50000`, `exclude_days_of_week=[6]`
    (Monday–Saturday, excluding Sunday), `target_country='Japan'`,
    `start_month=11`, `stop_month=11` (November only).

*Security note: Because Python `pickle` (`.pkl`) files can execute arbitrary
code during deserialization, only load `.pkl` checkpoint files that you
generated locally or obtained from a trusted source.*

### 3. Running with Synthetic Data (`0_generate_dummy_data.ipynb`)

Because participant-level and raw cohort step tables cannot be redistributed
publicly, we provide `colab/analysis/0_generate_dummy_data.ipynb`. Running all
cells in `0_generate_dummy_data.ipynb` populates `./data/internal/` and
`./data/public/` with synthetic datasets that match the exact schema, MultiIndex
structure, and checkpoint hashes expected by notebooks `1` through `5`.

To run the full pipeline locally:

1.  Launch Jupyter from the repository root:

    ```bash
    jupyter notebook
    ```
2.  Open and run `colab/analysis/0_generate_dummy_data.ipynb` to generate
    `./data/`.
3.  Open and run notebooks `1_demographic_pyramids.ipynb` through
    `5_uk_trends.ipynb`.

## Public Reference Data Sources

To replace the synthetic public files in `./data/public/` with the official
third-party reference tables used in the study:

### UN World Population Prospects (WPP 2024)

5-year age group population Excel files from UN World Population Prospects 2024:

*   `data/public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `data/public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`

Downloaded from:
https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=Population

*   Male:
    https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx
*   Female:
    https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/EXCEL_FILES/2_Population/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx

Terms of use:
Copyright 2024 by United Nations, made available under a Creative Commons
license CC BY 3.0 IGO: http://creativecommons.org/licenses/by/3.0/igo/

### National Health and Nutrition Survey, Japan (NHNS-J)

*   `data/public/2015_2024_japan_steps.pkl` (compiled from annual
    `{year}JapanStepsEng.xlsx` reports for 2015–2019, 2022, 2023, and 2024).

Downloaded from:
https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00450171&tstat=000001041744&cycle=7&cycle_facet=cycle&metadata=1&data=1

*(Note: Selecting the English version of the e-Stat page causes no data to be
displayed; stay on the Japanese version and translate the page in your browser.
For each year, select the table "Average and standard deviation of step count -
by age group, number of people, mean, standard deviation - total, male, female,
20 years and older".)*

Terms of use: “Using the content on this website: Information made available on
this website (hereinafter referred to as “Content”) may be freely used, copied,
publicly transmitted, translated or otherwise modified on condition that the
user complies with provisions 1) to 6) below. Commercial use of Content is also
permitted. Note, however, that numerical data and data in Simple tables, graphs,
and so forth are not subject to copyright. Accordingly the terms of use does not
apply to such data, and said data may be used freely.”

### Free-Living Tracker Calibration (Nakagata et al., 2022)

The free-living step-count calibration equation (`Y_adj = (F - 4152.0) / 0.8`)
applied in the Japan NHNS-J evaluations (`2_japan_england_eval.ipynb` and
`4_filling_gaps_japan.ipynb`) is taken from:

Nakagata, T., Murakami, H., Kawakami, R., Tripette, J., Nakae, S., Yamada, Y.,
Ishikawa-Takata, K., Tanaka, S., & Miyachi, M. (2022). Step-count outcomes of 13
different activity trackers: Results from laboratory and free-living
experiments. *Gait & Posture*, 98, 24–33.
https://doi.org/10.1016/j.gaitpost.2022.08.004

### English Longitudinal Study of Ageing (ELSA)

Summary statistics taken from the study pre-print:
https://www.medrxiv.org/content/10.64898/2026.03.25.26349270v1

Terms of use: CC-BY 4.0 International license.

### Strain et al., 2024 — Global Estimates of Insufficient Physical Activity

*   `data/public/who_trends.csv`

Prevalence of insufficient physical activity taken from the supplementary table
of:
https://www.thelancet.com/journals/langlo/article/PIIS2214-109X(24)00150-5/fulltext

Terms of use: “Creative Commons: This is an open access article distributed under the terms of
the Creative Commons CC-BY license, which permits unrestricted use,
distribution, and reproduction in any medium, provided the original work is
properly cited. You are not required to obtain permission to reuse this
article.”

## Notebook Data Dependencies

### `0_generate_dummy_data.ipynb` (Synthetic Dataset Generator)

Produces in `./data/`:

*   `internal/stratify_steps/checkpoint_*_a36731bea99396ebe673f464d73ada56.pkl`
    (Global 28-country grid)
*   `internal/stratify_steps/checkpoint_*_9f4397029d8039e43903bb5b8b7e75ec.pkl`
    (Japan November grid)
*   `internal/2021_2023_uk_50plus_participant_steps.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/who_trends.csv`
*   `public/2015_2024_japan_steps.pkl`

### `1_demographic_pyramids.ipynb` (Demographic Pyramids & Sampling Bias)

Requires:

*   `internal/stratify_steps/checkpoint_grid_*_a36731be*.pkl`
*   `internal/stratify_steps/checkpoint_grid_*_9f439702*.pkl`
*   `internal/2021_2023_uk_50plus_participant_steps.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/who_trends.csv`

### `2_japan_england_eval.ipynb` (Japan & England Ground-Truth Validation)

Requires:

*   `internal/stratify_steps/checkpoint_grid_*_9f439702*.pkl`
*   `internal/2021_2023_uk_50plus_participant_steps.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/2015_2024_japan_steps.pkl`

### `3_global_eval.ipynb` (28-Country Inactivity Correlation)

Requires:

*   `internal/stratify_steps/checkpoint_grid_*_a36731be*.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/who_trends.csv`
*   `public/2015_2024_japan_steps.pkl`

### `4_filling_gaps_japan.ipynb` (Survey Gap-Filling & Imputation)

Requires:

*   `internal/stratify_steps/checkpoint_grid_*_9f439702*.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`
*   `public/2015_2024_japan_steps.pkl`

### `5_uk_trends.ipynb` (UK Longitudinal Step Trends by Age & Sex)

Requires:

*   `internal/stratify_steps/checkpoint_grid_*_a36731be*.pkl`
*   `public/WPP2024_POP_F02_2_POPULATION_5-YEAR_AGE_GROUPS_MALE.xlsx`
*   `public/WPP2024_POP_F02_3_POPULATION_5-YEAR_AGE_GROUPS_FEMALE.xlsx`

## License

Copyright 2026 Google LLC

The full license text is in [`LICENSE.txt`](LICENSE.txt).

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Disclaimers

This is not an officially supported Google product. This project is not
eligible for the
[Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
This project is intended for demonstration purposes only. It is not intended
for use in a production environment.

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind,
including but not limited to diagnosis or prognosis.
