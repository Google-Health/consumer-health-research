# `data/` — local data root

`file_io.get_data_dir()` resolves to this directory (or to `$WPM_DATA_DIR`).
Layout:

```
data/
  dummy_data.csv          synthetic RHR user-months  (public schema, below;
  dummy_data_hrv.csv      synthetic HRV user-months   written by notebook 00)
  extra_data/             optional reference tables for the maps (not shipped)
  results/<run_tag>/      written by 02_modelling.ipynb
  results/paper/          optional: publication assets (.tex, png/svg/pdf) written
                          by 01 / 03 only when SAVE_PAPER_ASSETS = True
```

## Dummy CSVs

`dummy_data*.csv` are **synthetic** (`src/generate_dummy_data.py`, seed 0): a
few hundred users across 19 countries with a crude seasonal / age / sex
structure so that the model competition does not degenerate. They exist only to
test loading and to run the notebooks end to end; nothing estimated from them is
meaningful. They are not committed: create them with
`notebooks/00_generate_dummy_data.ipynb` or

```
python src/generate_dummy_data.py --out_dir=data --n_users=300
```

## Public input schema

One row per user and calendar month. Headers are matched case-insensitively.
`cohort_io.load_cohort` renames the columns to the names in the last column,
which are the ones that appear in the model formulas, result CSVs and
coefficient tables.

| Column               | Unit / values          | Name used in the code        |
| :------------------- | :--------------------- | :--------------------------- |
| `id`                 | user identifier        | `user_id`                    |
| `month`              | calendar month         | `activity_period`            |
:                      : (`YYYY-MM`)            :                              :
| `age`                | years                  | `age`                        |
| `sex`                | `Male` / `Female`      | `gender`                     |
:                      : (also `M`/`F`)         :                              :
| `country`            | ISO-3166 alpha-2       | `country_2_alpha_cd`         |
| `latitude`,          | decimal degrees        | `lat_e7`, `lng_e7` (1e-7     |
: `longitude`          :                        : deg)                         :
| `temperature`        | monthly mean air       | `avg_temp_mean`              |
:                      : temperature, °C        :                              :
| `temperature min`    | monthly mean of the    | `avg_temp_min`               |
:                      : daily minimum, °C      :                              :
| `temperature max`    | monthly mean of the    | `avg_temp_max`               |
:                      : daily maximum, °C      :                              :
| `day light`          | mean day length, hours | `dl_monthly`                 |
| `resting heart rate` | bpm (RHR file)         | `avg_resting_heart_rate_bpm` |
| `heart rate          | ms (HRV file)          | `avg_hrv`                    |
: variability`         :                        :                              :
| `region` (optional)  | UN region label        | `un_region` (`unknown` if    |
:                      :                        : absent)                      :

`temperature min` / `temperature max` are needed because the Stage 5/6
candidates compare mean-temperature against max/min-temperature models.

The loader (`cohort_io.load_cohort`) raises a `ValueError` listing the
offending rows when `sex` is not one of the values above or when `latitude` /
`longitude` is missing or non-numeric. Only an empty `country` field counts as
missing (Namibia's code `NA` is kept).

## Reference tables (`extra_data/`, optional)

Only the map figures need them; they are not redistributed here. Download them
into `data/extra_data/` under their own terms:

*   `ne_110_admin_0_countries.geojson` — Natural Earth 1:110m Admin 0 –
    Countries (https://www.naturalearthdata.com/). Terms of use: public domain.
*   `unsd-countries.csv` — UN Statistics Division "Standard country or area
    codes for statistical use (M49)"
    (https://unstats.un.org/unsd/methodology/m49/overview/), columns `ISO-alpha2
    Code`, `ISO-alpha3 Code` and the region columns. Terms of use:
    https://www.un.org/en/about-us/terms-of-use.
