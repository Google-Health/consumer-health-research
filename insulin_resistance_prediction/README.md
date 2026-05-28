# Insulin Resistance Prediction From Wearables and Routine Blood Biomarkers

This repository contains [datasets](#datasets) and [reference code](#reference-code-structure) from the Insulin Resistance Prediction
manuscript "[Insulin Resistance Prediction From Wearables and Routine Blood Biomarkers](https://arxiv.org/abs/2505.03784)"

## Overview

Insulin resistance, a precursor to type 2 diabetes, is characterized by impaired insulin action in tissues. Current methods for measuring insulin resistance, while effective, are expensive, inaccessible, not widely available and hinder opportunities for early intervention. In this study, we remotely recruited the largest dataset to date across the US to study insulin resistance (N=1,165 participants, with median BMI=28 kg/m2, age=45 years, HbA1c=5.4%), incorporating wearable device time series data and blood biomarkers, including the ground-truth measure of insulin resistance, homeostatic model assessment for insulin resistance (HOMA-IR). We developed deep neural network models to predict insulin resistance based on readily available digital and blood biomarkers. Our results show that our models can predict insulin resistance by combining both wearable data and readily available blood biomarkers better than either of the two data sources separately (R2=0.5, auROC=0.80, Sensitivity=76%, and specificity 84%). The model showed 93% sensitivity and 95% adjusted specificity in obese and sedentary participants, a subpopulation most vulnerable to developing type 2 diabetes and who could benefit most from early intervention. Rigorous evaluation of model performance, including interpretability, and robustness, facilitates generalizability across larger cohorts, which is demonstrated by reproducing the prediction performance on an independent validation cohort (N=72 participants). Additionally, we demonstrated how the predicted insulin resistance can be integrated into a large language model agent to help understand and contextualize HOMA-IR values, facilitating interpretation and safe personalized recommendations. This work offers the potential for early detection of people at risk of type 2 diabetes and thereby facilitate earlier implementation of preventative strategies.

![Alt text](figures/41586_2026_10179_Fig1.webp)


## Datasets

### Data Availability

The de-identified dataset used in this study is available to approved researchers for reproducibility purposes only. Researchers seeking dataset access must complete the Insulin Resistance Dataset Access Request Form, [available here](https://docs.google.com/forms/d/e/1FAIpQLSebcfCZMQKBua7QS9MSvqr9n-fqmfJvmU4blOHXP1WZC0NFqA/viewform?usp=preview). Once approved, the email address provided in the Dataset Access Request Form will be added to a google group allowlist (`google-health-insulin-resistance-prediction-data-external@google.com`), which grants read only access to a Google Cloud Storage (GCS) Bucket where the dataset is stored. Instructions on how to access the bucket will be provided in the approval email.

Aggregated data derived from the original de-identified dataset is provided in the `/data` folder. All files are provided in [parquet format](https://parquet.apache.org/) and are loaded as [pandas dataframes](https://pandas.pydata.org/docs/index.html). 

### ir_prediction_data.pq

The file `ir_prediction_data.pq` contains a comprehensive dataset for insulin resistance prediction. Each row represents a single participant. The columns contain a wide range of features, including:

* Participant ID (`participant_id`): A unique identifier for each participant.
* Digital Biomarkers: Data from wearables, including statistics (mean, median, standard deviation) for:
  * Resting Heart Rate (`RHR_FreeLiving_mean`, `RHR_FreeLiving_median`, `RHR_FreeLiving_std`)
  * Heart Rate Variability (`HRV_FreeLiving_mean`, `HRV_FreeLiving_median`, `HRV_FreeLiving_std`)
  * Daily Steps (`STEPS_Daily_mean`, `STEPS_Daily_median`, `STEPS_Daily_std`)
  * Sleep Duration (`SLEEP_Duration_mean`, `SLEEP_Duration_median`, `SLEEP_Duration_std`)
* Lab-Measured Biomarkers: A variety of blood test results, such as:
  * Lipid Panel: `total cholesterol`, `hdl`, `triglycerides`, `ldl`
  * Metabolic Panel: `glucose`, `creatinine`, `sodium`, `potassium`
  * Other key markers like `hba1c`, `insulin`, and `crp`
* Demographics: Basic information about the participants, including `age`, `sex`, `bmi`, and `ethnicity`.
* Health Conditions and History: Self-reported health information, such as the presence of `hypertension`, `diabetes`, `CVD` (Cardiovascular Disease), and smoking status (`smoker`).
* Calculated Health Scores: Clinically relevant scores calculated from the raw data, including:
  * `homa_ir` and `homa_ir_status` (HOMA-IR is a method for quantifying insulin resistance)
  * `framingham_risk` (Framingham Risk Score for cardiovascular disease)
  * `ascvd_risk` (ASCVD Risk Score for atherosclerotic cardiovascular disease)

### data.pq

The file `data.pq` contains a comprehensive dataset for insulin resistance prediction. Each row represents a single participant. The columns have a structure similar with `ir_prediction_data.pq`.

## Reference Code Structure

- `IR_Analysis_Stats_Stratification_Visualization.ipynb`: This notebook focuses on analyzing and visualizing the cohort data. It includes code to generate descriptive statistics, stratify the data based on BMI and physical activity, and create various visualizations such as box plots, scatter plots, and heatmaps to explore the relationships between different features and insulin resistance.

- `IR_MassiveAblationStudy_DeepLearningModels_InputFeatures_SiteDataTimeWindow_CrossValidationScheme.ipynb`: This notebook conducts a large-scale ablation study using deep learning models, specifically Masked Autoencoders, to predict HOMA-IR. It systematically evaluates the impact of different input features, time windows, and cross-validation methods on model performance.

- `IR_MassiveAblationStudy_XGBoost_InputFeatures_SiteDataTimeWindow_CrossValidationScheme.ipynb`: This notebook is similar to the deep learning ablation study but focuses on XGBoost models. It explores how different input features, data time windows, and cross-validation schemes affect the performance of XGBoost in predicting HOMA-IR.

- `IR_VisualizingAblationResults.ipynb`: This notebook is dedicated to visualizing the results from the ablation studies. It includes various plots to compare the performance of different models and feature sets, such as bar charts, point plots, and scatter plots. It also includes statistical tests to compare model performance.

## Dependencies

The libraries used in the [mentioned colab notebooks](#reference-code-structure) are standard for [colab.research.google.com](https://colab.research.google.com). All python code was run using [python 3.10](https://www.python.org/downloads/release/python-3100/) on publicly available [Google Colab kernels](https://colab.research.google.com). 

In case of import failures, manually install the failing packages
by running `!pip install <package>` in a new cell.

If running the provided code using other environments, please use the following package versions:

  - altair: 5.5.0
  - bokeh: 3.7.3
  - matplotlib: 3.10.0
  - numpy: 2.0.2
  - pickle: file version 4.0
  - pandas: 2.2.2
  - pytz: 2025.2
  - pytorch: 
  - re: 2.2.1
  - scipy: 1.16.3
  - seaborn: 0.13.2
  - statsmodels: 0.14.6
  - sklearn: 1.6.1
  - torch: 2.9.0+cpu
  - tqdm: 4.67.1
  - IPython: 7.34.0
  - shap: 0.50.0
  - xgboost: 3.1.2
  - json: 2.0.9
  - tensorflow: 2.19.0
  - umap: 0.5.9.post2

## Citing this paper

```bib
@article{metwally2026insulin,
      title={Insulin Resistance Prediction From Wearables and Routine Blood Biomarkers}, 
      author={Ahmed A. Metwally and A. Ali Heydari and Daniel McDuff and Alexandru Solot and Zeinab Esmaeilpour and Anthony Z Faranesh and Menglian Zhou and David B. Savage and Conor Heneghan and Shwetak Patel and Cathy Speed and Javier L. Prieto},
      journal={Nature},
      volume={652},
      number={8109},
      pages={451--461},
      year={2026},
      publisher={Nature Publishing Group}
      url={https://doi.org/10.1038/s41586-026-10179-2}, 
}
```

## Contributing

Currently we do not accept external contributions to this repository. For more details, please see
[CONTRIBUTING.md](https://github.com/Google-Health/consumer-health-research/tree/main/insulin_resistance_prediction/CONTRIBUTING.md).

## License

Copyright 2025 Google LLC

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

This is not an officially supported Google product. This project is not eligible for the [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security). This project is intended for demonstration purposes only. It is not intended for use in a production environment.

NOTE: the content of this research code repository (i) is not intended to be a medical device; and (ii) is not intended for clinical use of any kind, including but not limited to diagnosis or prognosis.