# Graph-Aligned Clinical Reasoning: Budget-Constrained Agent Implementation

The `ns_agent_adk` package provides a Neuro-Symbolic LLM agent designed for reasoning over longitudinal electronic health records, structured as FHIR bundles. The agent operates over a chronological hypergraph, extracting context through semantic search, taking budgeted reasoning steps, and utilizing graph-traversal tools to create comprehensive clinical summaries.

This implementation is built using the Google Agent Development Kit (ADK) and the `google-genai` library.

## Key Features

- **Chronological Hypergraphs**: Transforms standard FHIR JSON bundles into interconnected `ChronologicalHypergraph` structures to seamlessly resolve references, temporal events, and categorical data.
- **Causal Saliency Linearization**: Embeddings and temporal logic are used to construct a constrained subset of the FHIR data (a "skeleton view") that strictly guarantees fitting within the model's context window.
- **Custom Tool Execution**: The agent is equipped with a suite of custom Graph Traversal tools logic to incrementally explore the "hidden" areas of the patient graph:
    - **`inspect_node`**: Read detailed clinical descriptions of specific FHIR resources.
    - **`search_graph`**: Run semantic searches utilizing vector embeddings to find specific conditions or symptoms.
    - **`follow_links`**: Traverse structural FHIR referenced links.
    - **`filter_graph_by_time`**: Query sub-graphs based on specific date constraints.
    
## Project Structure

```text
ns_agent_adk/
├── README.md
├── __init__.py
├── engine.py       # Defines the NeuroSymbolicAgent, ADK Agent wrapper, and budgeting
├── tools.py        # Exposes ClinicalGraphTools for LLM graph reasoning and traversal
├── test_run.py     # Entrypoint script demonstrating setup, data ingestion, and querying
├── core/           # Fundamental data structures and timeline logic 
│   ├── __init__.py
│   ├── fhir_time.py
│   ├── graph.py    # Defines ChronologicalHypergraph for linking FHIR dependencies
│   └── linearizer.py # Implements CausalSaliencyLinearizer for LLM context generation
├── utils/          # General utility scripts
│   ├── __init__.py
│   ├── embeddings.py
│   ├── fhir_utils.py
│   └── temporal_parser.py
├── config/         # Configuration setup for Gemini models and embedding API
│   └── config.py
├── templates/      # Jinja system preamble templates
│   └── system_preamble.jinja2
└── tests/          # Unit tests for the agent and time-handling functionality
    ├── test_engine.py
    └── test_time_handling.py
```

## Getting Started

### Prerequisites

To test the agent with Gemini models, ensure you have the appropriate environment variables set:

```bash
export GENAI_API_KEYS="your_api_key_here"
```

You can change the backend LLM and embedding model via modifying `config/config.py`. By default, `gemini-3-flash-preview` is used for LLM and `gemini-embedding-001` is used for embeddings.

### Running the Example

Execute the python script to instantiate the graph, compute embeddings, initialize the `NeuroSymbolicAgent`, and run a general diagnostic query.

```bash
PYTHONPATH=src/ python src/ns_agent_adk/test_run.py
```

By default, the script asks the agent to _"Summarize the patient's medical history over the past 5 years."_ The script will dump both the LLM's final generated output and the full ADK reasoning step trace, including tool calls made during graph traversal.

### Evaluate on Benchmarks

```bash
python scripts/run_evals.py \
  --dataset medagentbench \
  --strategy ns_agent \
  --model gemini_3_flash_preview \
  --shuffle --max_rows 10
```


