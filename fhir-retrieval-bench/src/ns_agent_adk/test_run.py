import hashlib
import json
import os
from typing import Any

from absl import app
from absl import flags

from ns_agent_adk import engine as engine_module
from ns_agent_adk.config import config as config_module
from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.core import linearizer as linearizer_module

_INPUT_FILE = flags.DEFINE_string(
    "input_file",
    "data/sample_patient.json",
    "Path to the FHIR patient JSON bundle to test with.",
)


def transform_to_bundle(data: dict[str, Any]) -> dict[str, Any]:
  """Transforms a dictionary of resources into a FHIR bundle."""
  bundle = {"resourceType": "Bundle", "type": "collection", "entry": []}
  for resources in data.values():
    for resource in resources:
      bundle["entry"].append({"resource": resource})
  return bundle


def main(argv=None):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Setup Config
  API_KEY = os.getenv("GENAI_API_KEYS") or os.getenv("GENAI_API_KEY")

  if not API_KEY:
    print(
        "WARNING: You must set GENAI_API_KEYS in your"
        " environment to test with Gemini."
    )
    return

  config = config_module.Config(
      embedding_api_key=API_KEY,
      llm_api_key=API_KEY,
      embedding_model_name="gemini-embedding-001",
      llm_model_name="gemini-3.1-flash-lite-preview",
  )

  input_filename = _INPUT_FILE.value
  print(f"Loading FHIR graph from: {input_filename}")
  if not os.path.exists(input_filename):
    print(
        f"ERROR: Input file {input_filename} not found.\n"
        "Please run test_run.py with a valid JSON FHIR bundle path:\n"
        "  PYTHONPATH=src/ python src/ns_agent_adk/test_run.py --input_file /path/to/patient.json"
    )
    return

  try:
    with open(input_filename, "rb") as in_f:
      file_content = in_f.read()
      real_fhir_data = json.loads(file_content.decode("utf-8"))

    cache_key = hashlib.md5(file_content).hexdigest()
    fhir_bundle = transform_to_bundle(real_fhir_data)

    print("⏳ Constructing the hypergraph...")
    graph = graph_module.ChronologicalHypergraph()
    graph.build_from_bundle(fhir_bundle)
    print(f"✅ Graph built with {len(graph.spine)} hypernodes.")

    print("⏳ Precomputing node embeddings...")
    embedder = config.get_embedder()
    precomputed_node_embeddings = linearizer_module.precompute_node_embeddings(
        graph,
        embedder,
        cache_path=f"/tmp/ns_agent_adk/embedding_cache_{cache_key}.parquet",
    )

    print("⏳ Initializing ADK NeuroSymbolicAgent...")
    ns_agent = engine_module.NeuroSymbolicAgent(
        config=config,
        graph=graph,
        precomputed_node_embeddings=precomputed_node_embeddings,
    )

    query = "Summarize the patient's medical history over the past 5 years."
    print(f"⏳ Asking the agent: {query}...")
    final_response, reasoning_trace = ns_agent.execute(query)

    print("\n==================================")
    print("FINAL RESPONSE")
    print("==================================")
    print(final_response)

    print("\n==================================")
    print("ADK PARSED EVENTS")
    print("==================================")
    events_parsed = engine_module.parse_events(reasoning_trace)
    print(events_parsed["aggregate_summary"])
    for step in events_parsed["steps"]:
      print(step)

  except Exception as e:  # pylint: disable=broad-except
    print(f"Execution Error: {e}")


if __name__ == "__main__":
  app.run(main)
