# FHIR Retrieval Bench 🚀

FHIR Retrieval Bench is a clinical evaluation framework designed to benchmark question-answering systems and retrieval-augmented generation (RAG) strategies over longitudinal electronic health record (EHR) data structured as patient FHIR bundles.

This repository contains the official code implementation for the paper, packaged ready for public open-source distribution.

---

## 📦 Repository Directory Structure

```
fhir-retrieval-bench/
├── LICENSE                     # Project license
├── README.md                   # Global usage and installation guide
├── pyproject.toml              # PEP 621 package metadata and requirements
├── config/                     # Composable YAML configurations (model, dataset, strategy)
├── docs/                       # Detailed documentation
├── notebooks/                  # Example notebooks (getting started, preprocessing, analysis)
├── src/                        # Source packages
│   ├── fhir_retrieval_bench/   # Main benchmark code
│   └── ns_agent_adk/           # Neuro-Symbolic clinical reasoning agent
├── tests/                      # Pytest test suites
├── scripts/                    # Command line execution scripts
└── data/                       # Local datasets (ignored by Git)
```

---

## ⚙️ Installation & Setup

FHIR Retrieval Bench supports standard packaging and has been optimized to run natively on **Python 3.11+** and **Python 3.13**.

### 1. Create and Activate Virtual Environment
Create a local virtual environment to isolate dependencies:

```bash
python3 -m venv fhir-hopper-env
source fhir-hopper-env/bin/activate
```

### 2. Install in Editable Developer Mode
Install the repository and all required packages (including `numpy`, `pandas`, `pyarrow`, `litellm`, `google-genai`, and `google-adk`) with a single command:

```bash
pip install -e .
```

---

## 🚀 Quick Start

### 1. Verify Your Dataset Configuration
Place your dataset parquet/pickle files under the `data/` folder (ignored from Git). To check that the datasets are parsed and loaded correctly, run:

```bash
python scripts/check_dataset.py
```

### 2. Test API Model Connectivity
FHIR Retrieval Bench supports multiple LLM backends (Gemini via Google GenAI, OpenAI, Anthropic). Export your API keys and run the connectivity smoke test:

```bash
# Test Gemini API model connectivity
export GENAI_API_KEYS="your_gemini_api_key"
python scripts/check_model_api.py --model gemini_3_flash_preview
```

### 3. Launch a Cartesian Grid Evaluation Run
You can evaluate combinations of models, strategies, and datasets in parallel using the Cartesian grid expansion CLI:

```bash
export OUTPUT_DIR=./outputs/current_run

python scripts/run_evals.py \
  --dataset medagentbench,fhiragentbench,ehrqa,fhirpathqa \
  --strategy ontology_guided_retrieval,fhir2text,prefiltered,flowsheet \
  --model gemini_3_flash_preview \
  --output_dir $OUTPUT_DIR \
  --num_workers 4
```

---

## 📖 Documentation & Advanced Usage

For more comprehensive details about custom setups:
* Check out [docs/index.md](docs/index.md) for a full walkthrough.
* Read about datasets and directory structures in [docs/docs/datasets.md](docs/docs/datasets.md).
* Learn about the available LLM and embedding models in [docs/docs/models.md](docs/docs/models.md).
* Dive into the details of the retrieval/reasoning strategies (including our core Neuro-Symbolic agent) in [docs/docs/strategies.md](docs/docs/strategies.md).
