# FHIR Retrieval Bench

## Overview
FHIR Retrieval Bench evaluates question-answering systems over patient FHIR
(Fast Healthcare Interoperability Resources) data. 

Please check out the documentation in the `docs/` directory.
## Project Structure

```text
fhir_retrieval_bench/
├── config.py              # YAML config fragment loading and composition
├── data/
│   ├── base.py            # EvalInstance dataclass
│   ├── fhir_utils.py      # FHIR date parsing and bundle normalization
│   └── loader.py          # Dataset loaders
├── eval/
│   ├── judge.py           # LLM-based answer evaluation
│   └── runner.py          # Evaluation orchestration
├── strategies/
│   ├── base.py            # Strategy / RAGStrategy base classes
│   ├── ontology_guided_retrieval.py
│   ├── fhir2text.py
│   ├── prefiltered.py
│   └── embedding_csl.py
├── utils/
│   ├── api.py             # API-key rotation and retry logic
│   └── embeddings.py      # Google GenAI embedding interface
├── config/
│   ├── dataset/               # Dataset config fragments
│   ├── strategy/              # Strategy config fragments
│   └── model/                 # Model config fragments
├── docs/
│   ├── datasets.md            # Dataset layouts and field expectations
│   ├── strategies.md          # Strategy descriptions and guidance
│   └── models.md              # Model fragments, provider routing, thinking_budget
├── scripts/
│   ├── run_evals.py           # CLI entry point
│   ├── check_model_api.py           # Credential smoke test
│   ├── check_datasets.py      # Dataset path smoke test
│   └── precompute_embeddings.py # Precompute embeddings for RAG strategies
└── tests/                     # pytest test suite
```