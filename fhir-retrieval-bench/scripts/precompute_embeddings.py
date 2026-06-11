import os

from absl import app
from absl import flags
from absl import logging
import tqdm
import yaml

from fhir_retrieval_bench import config
from fhir_retrieval_bench.data import fhir_utils
from fhir_retrieval_bench.data import loader
from fhir_retrieval_bench.strategies import embedding_csl
from fhir_retrieval_bench.utils import api
from fhir_retrieval_bench.utils import embeddings as bench_embeddings
from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.core import linearizer as linearizer_module
from ns_agent_adk.utils import embeddings as ns_embeddings

_DATASET = flags.DEFINE_string(
    "dataset",
    None,
    "Name of the dataset (e.g. ehrqa).",
    required=True,
)
_STRATEGY = flags.DEFINE_string(
    "strategy",
    "embedding_csl,ns_agent",
    "Strategy config fragment name(s), comma-separated.",
)
_EMBEDDING_MODEL = flags.DEFINE_string(
    "embedding_model",
    "gemini-embedding-001",
    "Embedding model name.",
)
_CACHE_DIR = flags.DEFINE_string(
    "cache_dir",
    "/tmp/fhir_retrieval_bench_cache",
    "Cache directory for precomputed embeddings.",
)
_CONFIG_DIR = flags.DEFINE_string(
    "config_dir",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config",
    ),
    "Root config directory (default: <repository_root>/config).",
)


def main(argv):

  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  genai_api_keys = api.get_genai_api_keys()
  openai_api_keys = api.get_openai_api_keys()
  gcp_env = os.getenv("GCP_PROJECT_LOCATIONS", "")
  if gcp_env:
    gcp_project_and_locations = [
        tuple(project_location.split(":"))
        for project_location in gcp_env.split(",")
    ]
  else:
    gcp_project_and_locations = None

  creds = api.Credentials.from_lists(
      genai_api_keys=genai_api_keys,
      openai_api_keys=openai_api_keys,
      gcp_project_and_locations=gcp_project_and_locations,
  ).shuffled()

  embedding_backend = api.GenAIBackend(list(creds.genai_api_keys))


  bench_embedder = bench_embeddings.GenAIEmbedder(
      backend=embedding_backend,
      model_name=_EMBEDDING_MODEL.value,
  )

  ns_embedder = ns_embeddings.GenAIEmbedder(
      backend=embedding_backend,
      model_name=_EMBEDDING_MODEL.value,
  )

  if not os.path.exists(_CACHE_DIR.value):
    os.makedirs(_CACHE_DIR.value)

  strategies = [s.strip() for s in _STRATEGY.value.split(",")]
  run_embedding_csl = "embedding_csl" in strategies
  run_ns_agent = "ns_agent" in strategies

  datasets = [d.strip() for d in _DATASET.value.split(",")]

  for d in datasets:
    dataset_yaml_path = os.path.join(_CONFIG_DIR.value, "dataset", f"{d}.yaml")
    with open(dataset_yaml_path, "r") as f:
      fragment = yaml.safe_load(f)
    dataset_config = config.DatasetConfig(**fragment["dataset"])

    logging.info("Loading FHIR bundles for dataset: %s...", dataset_config.name)
    bundles = loader.load_fhir_bundles(dataset_config)
    logging.info("Loaded %d bundles.", len(bundles))

    for patient_id, fhir_bundle in tqdm.tqdm(
        bundles.items(), desc=f"Precomputing embeddings for {d}"
    ):
      logging.info("Precomputing embeddings for patient: %s...", patient_id)

      if not fhir_utils.verify_with_pydantic(fhir_bundle):
        logging.warning(
            "FHIR bundle for patient %s failed pydantic validation, skipping.",
            patient_id,
        )
        continue

      # 1. Precompute for Embedding CSL strategy
      if run_embedding_csl:
        serialized_resources, _ = embedding_csl.serialize_fhir_bundle(
            fhir_bundle
        )
        resource_texts = [text for _, text in serialized_resources]
        if resource_texts:
          csl_cache_path = os.path.join(
              _CACHE_DIR.value,
              f"embedding_csl_embeddings_{patient_id}.parquet",
          )
          embedding_csl.load_or_compute_node_embeddings(
              resource_texts, bench_embedder, cache_path=csl_cache_path
          )

      # 2. Precompute for NSAgent Strategy
      if run_ns_agent:
        graph = graph_module.ChronologicalHypergraph()
        graph.build_from_bundle(fhir_bundle)
        ns_cache_path = os.path.join(
            _CACHE_DIR.value, f"ns_agent_embeddings_{patient_id}.parquet"
        )
        linearizer_module.precompute_node_embeddings(
            graph, ns_embedder, cache_path=ns_cache_path
        )

    logging.info(
        "Successfully precomputed embeddings for dataset %s.",
        dataset_config.name,
    )


if __name__ == "__main__":
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
