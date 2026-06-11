# FHIR Retrieval Bench

[Link to code](../src/fhir_retrieval_bench/)

## Overview

FHIR Retrieval Bench evaluates question-answering systems over patient FHIR
(Fast Healthcare Interoperability Resources) data. Each run combines a
**model**, a **dataset**, and a **strategy** fragment.

The config of each fragment is defined as YAML file under `config/` and selected by CLI flags:

```text
config/
  model/      # answer, judge, and embedding model names
  dataset/    # which benchmark to load and where it lives
  strategy/   # how FHIR data is turned into context
```

## How to run benchmarks

### 1. Choose a model and set up credentials

Model fragments live under `config/model/`. Each fragment defines the
**answer**, **judge**, and **embedding** models for a run; the answer and
judge models are independent and can come from different providers.

The simplest path is the default Google GenAI backend. Export one or more API
keys as a comma-separated list. If you specify multiple API keys, the framework rotates across keys to handle quota issues better:

```bash
export GENAI_API_KEYS="genai_key1,genai_key2,genai_key3"
```

For OpenAI-backed model fragments, export OpenAI keys the same way:

```bash
export OPENAI_API_KEYS="openai-key-1,openai-key-2"
```

To run models against Vertex AI, ensure that your machine is gcloud authorized to your GCP project and export the `GCP_PROJECT_LOCATIONS` environment variable:

```bash
export GCP_PROJECT_LOCATIONS="project_id_1:location_1,project_id_2:location_2"
```

Each model YAML fragment under `config/model/` requires an explicit `backend: "vertex"` or `backend: "public"` parameter for each model (answer, judge, embedding) to control its hosting route.

See
[docs/models.md](docs/models.md), for the full list of shipped model fragments, provider-routing rules, Vertex
Model Garden setup, and the `thinking_config` semantics across providers.

#### Verify the credentials set up

```bash
# Test Vertex API for Gemini Pro 3.1
export GCP_PROJECT_LOCATIONS="project_id_1:location_1"
python scripts/check_model_api.py --model gemini_3_flash_preview

# Test GenAI API for Gemini Pro 3.1
export GENAI_API_KEYS="genai_api_key"
python scripts/check_model_api.py --model gemini_3_flash_preview

# Test OpenAI API for GPT-5-mini
export GENAI_API_KEYS="genai_api_key" # You still need this because `config/model/gpt_5_mini` is set to use Gemini embeddings
export OPENAI_API_KEYS="oai_api_key"
python scripts/check_model_api.py --model gpt_5_mini

# Test AnthropicVertex API for Sonnet 4.6
export GENAI_API_KEYS="genai_api_key" # You still need this because `config/model/claude_sonnet_4_6` is set to use Gemini embeddings
export GCP_PROJECT_LOCATIONS="project_id_1:location_1"
python scripts/check_model_api.py --model claude_sonnet_4_6
```


### 2. Choose a dataset

Dataset fragments live under `config/dataset/`. Each fragment points at a
dataset root. By default, it is set to local `data/` directory with preprocessed datasets.

Shipped datasets:

| Dataset          | Description              |
| ---------------- | ------------------------ |
| `ehrqa`          | EHRQA benchmark          |
| `fhiragentbench` | FHIRAgentBench benchmark |
| `medagentbench`  | MedAgentBench benchmark  |
| `fhirpathqa`     | FHIRPathQA benchmark     |

For exact file expectations, see
[docs/datasets.md](docs/datasets.md).

#### Verify dataset setup

```bash
python scripts/check_dataset.py --dataset fhiragentbench --limit 3 # check only first records for FHIRAgentBench dataset
python scripts/check_dataset.py # check every fragment
```

This validates that the configured path exists, records load, and the sample
output looks reasonable.

### 3. Choose a strategy

Strategy fragments live under `config/strategy/`. A strategy controls how a
FHIR bundle is turned into LLM context.

| Strategy        | Description                                               |
| --------------- | --------------------------------------------------------- |
| `ontology_guided_retrieval` | Keyword-based scoring with medical ontology term expansion |
| `fhir2text`     | Converts FHIR resources into natural-language text blocks |
| `prefiltered`   | Status-based filtering and deduplication by concept code  |
| `embedding_csl` | Embedding-based Serialization                 |
| `ns_agent` | Graph-based representation of FHIR + Tool usage |
| `flowsheet` | Tabular views of FHIR data using FHIRPath |
| `flowsheet_agent` | Agentic representation of tabular FHIR data querying |

For more detail (including how to precompute embeddings for
`embedding_csl`), see [docs/strategies.md](docs/strategies.md).

### 4. Run an evaluation

The main CLI entry point is `scripts/run_evals.py`. It composes the
specified strategy, dataset, and model fragments into a complete config,
then runs the evaluation and writes results to the output directory.

#### Tiny run (recommended first)

Start with a small run so you can confirm everything works end to end before
spending time or quota on a full benchmark:

```bash
export OUTPUT_DIR=./outputs/current_run
python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --shuffle --max_rows 3

python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval,fhir2text,prefiltered,flowsheet,flowsheet_agent,embedding_csl,ns_agent \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --shuffle --max_rows 3
```

#### Full run

```bash
python scripts/run_evals.py \
  --dataset medagentbench \
  --strategy ontology_guided_retrieval \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR
```

#### Grid run

Any of `--strategy`, `--dataset`, and `--model` can take comma-separated
values. The CLI runs the Cartesian product:

```bash
python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval,fhir2text,prefiltered,flowsheet \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --num_workers 4
```

<!-- * 
* MedAgentBench: 148 ()
* FHRAgentBench: 2,931
* EHRQA: 5,133
* FHIRPathQA: 2,025

(148 + 2931 + 5133 + 2025) x ( 1 models ) x ( 3 strategies ) x 2 sec -> 12 hours
NS agent + embedding = 5x
-->

> Tip: This benchmark is primarily bottlenecked by API latency and quota, not local compute. You can usually run it on a gcloud instance rather than submitting a XManager job to leverage compute clusters. 
> To execute the benchmark in parallel, you only need to submit a single command for each model using the `--num_workers` flag, to maximally utilize your quota for the model. The optimal number of workers depends on your specific API quota and strategy, so you may need to just give it a few tries and monitor quota error messages printed on the terminal or check out the API usage dashboards. 

> Tip: Strategies like `ns_agent` invoke multiple LLM calls per example; for these, use a lower worker count to avoid rate-limiting. We recommend running `ns_agent` as a separate command with lower `--num_workers` than other strategies.

> Tip: You can set the `cache_dir` to save embedding caches. This is particularly useful when you run multiple experiments using the same embedding model. You can also run the `precompute_embeddings.py` script once to precompute the embeddings for the dataset.

#### All CLI options are as follows:

| Flag            | Default                             | Description                                                 |
| --------------- | ----------------------------------- | ----------------------------------------------------------- |
| `--strategy`    | required                            | Strategy fragment name(s), comma-separated                  |
| `--dataset`     | required                            | Dataset fragment name(s), comma-separated                   |
| `--model`       | required                            | Model fragment name(s), comma-separated                     |
| `--output_dir`  | `/tmp/fhir_retrieval_bench_results` | Directory for results and checkpoints                 |
| `--cache_dir`  | `/tmp/fhir_retrieval_bench_cache` | Directory for saving cache (e.g., embeddings)                 |
| `--num_workers` | `1`                                | Parallel threads per evaluation                             |
| `--shuffle`     | `false`                             | Shuffle the eval instances                                 |
| `--max_rows`    | all rows                            | Max records loaded. When `--shuffle` is set, shuffle is done first |
| `--dry_run`     | `false`                             | Print composed configs without executing                    |

#### Precomputing Embeddings

For strategies that require embedding calculations (e.g., `embedding_csl`, `ns_agent`), computing them sequentially during evaluations can become a bottleneck. You can precompute embeddings in advance for a dataset using the `precompute_embeddings.py` script:

```bash
python scripts/precompute_embeddings.py \
    --strategy=embedding_csl,ns_agent \
    --dataset=medagentbench,fhiragentbench,ehrqa \
    --embedding_model=gemini-embedding-001 \
    --cache_dir=/tmp/fhir_retrieval_bench_cache
```

* `--strategy`: The strategy config fragment names to precompute embeddings for (default: `embedding_csl,ns_agent`).
* `--dataset`: The dataset name to compute embeddings for (e.g., `ehrqa`, `fhiragentbench`).
* `--embedding_model`: The embedding model name.
* `--cache_dir`: Target directory to save cached embeddings (default: `/tmp/fhir_retrieval_bench_cache`).

#### Output directory structure

By default, results are written to `/tmp/fhir_retrieval_bench_results/`
(override with `--output-dir`). Each run writes three files, all keyed by the
same:

```text
<output-dir>/<dataset>_<strategy>_<answer-model>_<sha6>/
  results.json          # per-record results
  summary.json  # aggregate metrics
  .checkpoint.jsonl  # per-record checkpoint
```

The wall-clock timestamp lives inside the summary JSON, not the filename, so
re-running the same config overwrites prior outputs in place rather than
accumulating dated copies.

**Why the `<sha6>` digest?** The filename already carries the dataset, the
strategy, and the answer model — but not the judge model, the embedding
model, the per-call LLM knobs (temperature, `max_output_tokens`,
`thinking_config`), `dataset.path`, or `use_vertex_ai`. The 6-char SHA is a
fingerprint over all of those. Without it, two runs that differ only in
(say) judge model or temperature would silently overwrite the same file.
Encoding everything literally would make filenames unwieldy; the digest is
the compact compromise.

