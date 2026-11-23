# Deep-Tissue Hemodynamic Sensing: Comparative Morphological Analysis and Single-Occasion Blood Pressure Status Classification using Bioimpedance and Photoplethysmography

This folder describes the [dataset](#dataset) and provides [loader code](#code) for the manuscript "Deep-Tissue Hemodynamic Sensing: Comparative Morphological Analysis and Single-Occasion Blood Pressure Status Classification using Bioimpedance and Photoplethysmography", accepted for publication in npj Health Systems. An earlier version is available as a [preprint](https://www.medrxiv.org/content/10.64898/2026.06.17.26355894v1).

## Overview

The pursuit of continuous, cuffless blood pressure (BP) monitoring is constrained by the superficial sensing depth of photoplethysmography (PPG). Impedance plethysmography (IPG) offers deeper tissue penetration, but its comparative value over PPG remains unquantified at scale. In this comparative study of 261 participants (130 single-occasion high BP status, 131 single-occasion non-high BP status), we utilized a dual-modality wearable prototype to capture simultaneous IPG and PPG signals. Over 150,000 cardiac cycles were analyzed using an unsupervised archetype discovery pipeline to quantify beat-to-beat morphological heterogeneity. IPG resolved up to three distinct morphological modes per participant, whereas co-located PPG converged into conserved, uniform profiles. Ventral forearm IPG pulse amplitude exhibited an effect for BP status (p = 0.024), a relationship absent in the co-located PPG signal. Furthermore, increasing body mass index attenuated the prevalence of steep-upstroke archetypes in IPG (p = 0.035), suggesting a potential damping effect of adipose tissue. These exploratory findings suggest that deep-tissue bioimpedance captures rich, heterogeneous hemodynamic signatures that are invisible to optical sensors. Integrating superficial optical pulse wave analysis with deep-tissue bioimpedance-based models may offer a promising multi-modal pathway for accurate wearable cardiovascular monitoring.

## Dataset

The dataset is the Impedance Plethysmography Observational Blood Pressure study dataset (the "IPG_OBP Dataset").

### Data availability

The de-identified dataset used in this study is available to approved researchers for non-commercial research only. Researchers seeking access must complete the [Dataset Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLScrHpxxaM-DXj6uFKaxel00H5a_szWPDdN_ysHTH2hlRLpYxg/viewform). Once a request is approved, the email address given in the form is added to a Google Group allowlist (`ipg-obp-dataset-external@google.com`). Membership grants read-only access to the Google Cloud Storage (GCS) bucket `gs://ipg-obp-dataset-release`, where the dataset is stored. The approval email explains how to reach the bucket.

### Terms of use

Access to and use of the dataset are governed by the [Dataset Terms of Service](term_of_service.md). Read them before you request access. In summary, and without replacing those terms:

* Use the dataset only for non-commercial academic, scientific, or clinical research.
* Do not attempt to re-identify or contact any participant, and do not link the dataset with other data sources to find out who a participant is.
* Do not share or redistribute the dataset, or any subset or modified version of it. Each collaborator must apply for access individually.
* Keep the dataset secure, with access controls and encryption at rest and in transit.
* Cite the publication and include the required notice. See [Acknowledging the dataset](#acknowledging-the-dataset).
* Report any security incident or unauthorized disclosure involving the dataset to Google promptly, at `ipg-obp-dataset-support@google.com`.
* If your access ends, delete all copies of the dataset.

For questions about the dataset or your access, write to `ipg-obp-dataset-support@google.com`.

### Contents

```text
gs://ipg-obp-dataset-release/
├── demographic_data_release_v1.csv     # one row per participant
├── train_test_split_mapping.csv        # Train or Test for each participant
└── biopac_parsed_parquet/
    └── <PIN>_collection<N>.parquet     # one waveform recording, N = 1 to 4
```

The dataset covers 261 participants. It has 1,044 files (2 CSV files and 1,042 waveform files), about 23 GB in total, holding about 97 hours of recordings sampled at 2,000 Hz.

`PIN` is the participant identifier. It links the three kinds of file. Read it as text.

### demographic_data_release_v1.csv

One row per participant (261 rows).

| Column | Description |
| :--- | :--- |
| `PIN` | Participant identifier. |
| `Age` | Age in years. |
| `Height (cm)` | Height. |
| `Weight (kg)` | Weight. |
| `BMI (kg/m2)` | Body mass index. |
| `Biological Sex` | `Male`, `Female`, or `Not-Specified`. |
| `ITA (deg)` | Individual typology angle, from colorimeter readings: arctan((L* − 50) / b*) in degrees. |
| `Monk Skin Tone` | Monk Skin Tone scale category. |
| `Skin Hydration (a.u.)` | Skin hydration from a Delfin MoistureMeterSC, in arbitrary units. See the manuscript for the measurement details. |
| `SBP (mmHg)` | Reference systolic blood pressure: the mean of two cuff measurements on the arm with the higher pressure. |
| `DBP (mmHg)` | Reference diastolic blood pressure, measured the same way. |
| `HR (bpm)` | Heart rate from the cuff. |

Some values are missing: `ITA (deg)` for 32 participants, `Skin Hydration (a.u.)` for 5, and `HR (bpm)` for 1.

The BP status used in the manuscript is not a column. Derive it from the reference pressures: a participant is `HBP_True` if SBP is 130 mmHg or more, or DBP is 80 mmHg or more, and `HBP_False` otherwise. This gives 130 `HBP_True` and 131 `HBP_False` participants.

### train_test_split_mapping.csv

One row per participant, with columns `PIN` and `Split` (`Train` or `Test`).

| Split | Participants | `HBP_False` | `HBP_True` |
| :--- | :--- | :--- | :--- |
| `Train` | 221 | 110 | 111 |
| `Test` | 40 | 21 | 19 |

The split was made with scikit-learn's `train_test_split` (`test_size=0.15`, `random_state=42`), stratified jointly by BP status, age group (18–39, 40–59, 60 and over), and obesity status (BMI above 29.9 kg/m2).

> **Note on the split sizes.** The manuscript text reports a Training Set of n = 219 and a Hold-out Test Set of n = 42. The split that was used for the analysis, and that is released here, has 221 and 40 participants. Use this file to reproduce the results.

### Waveform files

Each file `biopac_parsed_parquet/<PIN>_collection<N>.parquet` holds one recording, sampled at 2,000 Hz. The four collections are the four phases of the protocol described in the manuscript.

| Collection | Site and interface | Name in the manuscript | Nominal length | Files | Columns |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Dorsal wrist, dry electrodes, wearable prototype | Dorsal wrist (prototype) | 10 min | 261 | 7 |
| 2 | Dorsal wrist, saline, wearable prototype | Not used in the main analyses | 4 min | 261 | 7 |
| 3 | Ventral wrist, saline, wearable prototype | Ventral wrist (prototype) | 3 min | 260 | 7 |
| 4 | Ventral forearm, reference modules with gel electrodes, plus Lead I ECG | Ventral forearm (gel) | 5 min | 260 | 8 |

The channels within one file were recorded at the same time. Different collections are separate recordings, made one after another.

All columns are 64-bit floating point numbers.

| Column | Unit | Description |
| :--- | :--- | :--- |
| `time` | s | Time since the start of the recording. |
| `IPG_Z` | Ohm | Impedance channel as recorded. The values are mostly saturated, so this column is of limited use. |
| `IPG_dZdt` | Ohm/s | Time derivative of impedance, as recorded. |
| `ECG` | mV | Lead I ECG. Collection 4 only. |
| `PPG` | mV | PPG channel as recorded. |
| `IPG_integrated_dZdt` | Ohm | Cumulative trapezoidal integral of `IPG_dZdt` over `time`, starting at 0. |
| `IPG_integrated_dZdt_flattened` | Ohm | `IPG_integrated_dZdt` minus its 1 s (2,000-sample) centred moving average. |
| `PPG_flattened` | mV | `PPG` minus its 1 s (2,000-sample) centred moving average. |

The moving average pads with zeros at the ends, so the first and last 0.5 s of the two `_flattened` columns are distorted.

The manuscript's waveform analyses start from `IPG_integrated_dZdt_flattened` and `PPG_flattened`. The band-pass filtering, beat detection and later steps described in its Methods are not applied to the released files.

Recording lengths:

| Collection | Shortest | Median | Longest |
| :--- | :--- | :--- | :--- |
| 1 | 600.5 s | 601.8 s | 683.1 s |
| 2 | 240.4 s | 241.6 s | 309.6 s |
| 3 | 180.5 s | 181.7 s | 211.1 s |
| 4 | 252.5 s | 302.0 s | 335.5 s |

Known gaps:

* One participant has no collection 3 file, and one participant has no collection 4 file.
* One collection 4 recording is shorter than the nominal 5 minutes (252.5 s).

The loader notebook lists which participant and collection pairs have no file.

## Code

* `src/dataset_loader.ipynb`: signs in, loads the two CSV files, derives the BP status, checks the file inventory against the numbers above, and loads and plots the waveforms of one participant.

The analysis procedures are described in full in the manuscript's Methods. Fiducial point extraction used the publicly available pyPPG library. The remaining analysis code is not public.

## Dependencies

The loader notebook needs Python 3 with `google-cloud-storage`, `pandas`, `pyarrow`, and `matplotlib`. These are preinstalled on [colab.research.google.com](https://colab.research.google.com).

## Acknowledging the dataset

If you use or refer to the dataset in a paper, abstract, poster, or presentation:

* Cite the publication below.
* Include a notice that the data was obtained from Google under the IPG_OBP Dataset Terms of Service.

```bib
@article{thomson2026deeptissue,
  title={Deep-Tissue Hemodynamic Sensing: Comparative Morphological Analysis and Single-Occasion Blood Pressure Status Classification using Bioimpedance and Photoplethysmography},
  author={Thomson, Seamus and Jung, Seobin and Pantelopoulos, Alexandros and Deshpande, Aniket and Cai, Lawrence and Blanchard, Emily and Wasson, Jackie and Mukherjee, Debanjan and Sunden, Lindsey and Sheng, Sam and Patel, Shwetak},
  journal={npj Health Systems},
  year={2026},
  note={Accepted for publication. Preprint: https://doi.org/10.64898/2026.06.17.26355894}
}
```

## Contributing

Currently we do not accept external contributions to this repository. For more details, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[https://www.apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This license covers the code in this folder. It does not cover the dataset, which is governed by the [Dataset Terms of Service](term_of_service.md).

## Disclaimers

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security). This project is intended for demonstration purposes only. It is not intended for use in a production environment.

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.
