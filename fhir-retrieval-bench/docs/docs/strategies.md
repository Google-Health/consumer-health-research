# Strategies

Each strategy fragment under `config/strategy/<name>.yaml` selects how a
FHIR bundle is turned into LLM context. Shipped fragments are tiny — they
only name the strategy — because current strategies don't expose tunable
knobs through config. To change behavior, edit the implementation under
`strategies/<name>.py`.

## Shipped strategies

| Strategy | Description | Implementation details |
|---|---|---|
| `ontology_guided_retrieval` | Keyword-based scoring with medical ontology term expansion | Implemented from scratch |
| `fhir2text` | Converts FHIR resources into natural-language text blocks | FHIR2Text serialization baseline |
| `prefiltered` | Status-based filtering and deduplication by concept code (https://arxiv.org/pdf/2402.01711) | Implemented from scratch |
| `embedding_csl` | Embedding-based Serialization | Implemented from scratch |
| `ns_agent` | Graph-based representation of FHIR + Tool usage | `src/ns_agent_adk/` |
| `flowsheet` | Tabular views of FHIR data using FHIRPath | https://www.nature.com/articles/s41746-025-01708-w |
| `flowsheet_agent` | Tabular FHIR RAG Agent with dynamic concept retrieval | ADK agent using flowsheet metadata & tabular query tools with date filtering and adaptive downsampling |

### `embedding_csl` note

`embedding_csl` strategies requires embeddings to identify relevant resources. To avoid
recomputing embeddings across runs, you can do precomputing once in
advance with `scripts/precompute_embeddings.py`.

## Adding a new strategy

1.  Implementation file under
    `src/fhir_retrieval_bench/strategies/<name>.py` subclassing `Strategy`
    or `RAGStrategy` from `strategies/base.py`.
2. Matching fragment YAML under `config/strategy/<name>.yaml`.
3. Register in `src/fhir_retrieval_bench/strategies/__init__.py` dispatch.