# Dataset setup


## Overview

This document contains the detailed dataset layout reference for FHIR
Retrieval Bench. Each dataset fragment under `config/dataset/` points at a local dataset root:

```yaml
dataset:
  name: fhiragentbench
  qa_path: /path/to/fhiragentbench/qa_path
  fhir_path: /path/to/fhiragentbench/fhir_path
```

After editing a fragment, validate it by running the `check_dataset` script
(see "Sanity check" below).

## Preprocessing code

The preprocessing notebook is available: `notebooks/preprocess_data.ipynb`

It includes code for performing sanity checks (e.g., validating FHIR JSON schema, ensuring the FHIR bundle covers all the QA pairs). Whenever you add a dataset, it is recommended to run the notebook to check for any potential issues.

We obtained the following number of QA pairs for each dataset: 

* EHRQA: 5,133
* FHRAgentBench: 2,931
* MedAgentBench: 148
* FHIRPath-QA: 2,095

The preprocessed the FHIR Parquet files have the following columns:

- `Patient ID`
- `FHIR Bundle`

In case you add a new dataset, you can verify the formatting of the FHIR bundle, using `verify_with_pydantic` function implemented in `src/fhir_retrieval_bench/data/fhir_utils.py`

## `fhiragentbench`

[https://arxiv.org/abs/2509.19319](https://arxiv.org/abs/2509.19319)

### Expected fields

The QA CSV should use the following columns:

- `patient_fhir_id`
- `question`
- `true_answer`

## `ehrqa`

### Expected fields

The QA dicts should provide the fields the loader uses:

- `patient_id`
- `question`
- `correct_answer`
- `answer_choices`

## `medagentbench` 

[https://ai.nejm.org/doi/full/10.1056/AIdbp2500144](https://ai.nejm.org/doi/full/10.1056/AIdbp2500144)

### Expected fields

The QA dataframe should include the following columns:

- `eval_MRN`
- `instruction`
- `sol`

## `fhirpathqa`

[https://arxiv.org/abs/2602.23479](https://arxiv.org/abs/2602.23479)

The QA pickle is the staged FHIRPath-QA Benchmark (2,095 records) produced
by `scripts/fhirpathqa/convert_to_pickle`. The FHIR bundle parquet is built
by `scripts/fhirpathqa/build_bundles_parquet` from per-patient
`$everything` exports against MIMIC-IV on FHIR Demo loaded into HAPI
(see the upstream `setup/` pipeline at
https://github.com/mooshifrew/fhirpath-qa).

### Expected fields

The QA pickle is a list of dicts with the following keys:

- `patient_id`            (MIMIC numeric ID, used as the key into the bundle parquet)
- `question`
- `answer`                (JSON-stringified list; the loader unwraps to a string)
- `now`                   (current-datetime anchor; injected into `question_context`)

Optional metadata kept verbatim in `source_meta`:

- `query`, `perspective`, `split`, `question_template_id`,
  `question_template`, `patient_id_hash`, `s_placeholders`,
  `op_placeholders`, `t_placeholders`, `id`, `index`, `holdout`

### Perspectives

Each question template is rendered in two phrasings:

- `clinical` (1,239 records) — formal medical terminology
- `patient` (856 records) — conversational layperson language

Both perspectives are loaded together; slice on `source_meta["perspective"]`
to analyse them separately.

### Splits

The upstream stratifies records across paraphrases as
`train` (1,629), `val` (193), `test` (273) for SFT experiments. The bench
loads all splits — there is no fine-tuning here, so the partition is
informational only. Filter on `source_meta["split"] == "test"` if you want
to compare numbers against the upstream paper, which evaluates on the
test split alone.

### Patient filtering

Our MIMIC-IV on FHIR Demo v2.1.0 export covers 96 of the 100 patients
referenced by the QA records. The four missing patients
(`10002430`, `10004113`, `10010058`, `10012438`) account for 70 records;
these are dropped at preprocessing time by
`scripts/fhirpathqa/convert_to_pickle --fhir_path=...`, so the staged
pickle contains only records whose patient has a bundle in the parquet.

## Related Files

- Dataset fragments: `config/dataset/*.yaml`
- Loader implementation: `src/fhir_retrieval_bench/data/loader.py`
- Smoke test script: `scripts/check_dataset.py`