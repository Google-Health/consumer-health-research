"""Tools for the FHIR Neuro-Symbolic ADK Agent."""

from absl import logging
import datetime
import json

import numpy as np

from ns_agent_adk.core import graph as graph_module
from ns_agent_adk.utils import embeddings as embeddings_module
from ns_agent_adk.utils import fhir_utils as fhir_utils_module

class ClinicalGraphTools:
  """Defines and bounds the toolset for the LLM Agent."""

  def __init__(
      self,
      graph: graph_module.ChronologicalHypergraph,
      embedder: embeddings_module.VectorCompass | None = None,
      node_embeddings: dict[str, list[float]] | None = None,
  ):
    self.graph = graph
    self.inspected_nodes = set()
    self.embedder = embedder
    self.node_embeddings = node_embeddings

  def inspect_node(self, resource_id: str) -> str:
    """Fetches the full JSON content of a specific FHIR resource.

    Usage: "Read the file." Fetches the full JSON content.
    Mandatory: You CANNOT verify values (BP, Dates, Dosage) without this.

    Args:
      resource_id: A specific ID from the skeleton (e.g., "Observation/123").

    Returns:
      A JSON string representing the missing fields for this resource ID.
    """
    logging.info(f"calling inspect_node(resource_id={resource_id})")
    node = self.graph.get_node_by_id(resource_id)
    if not node and "/" in resource_id:
      # Try stripping the resource type prefix
      node = self.graph.get_node_by_id(resource_id.split("/")[-1])
    if not node:
      return (
          f"Error: Node {resource_id} not found in graph. Did you make up"
          " the ID?"
      )

    self.inspected_nodes.add(resource_id)

    # clean_content = _strip(node.data)
    clean_content = fhir_utils_module.strip_fhir_data(node.data)
    return json.dumps(clean_content, separators=(",", ":"))

  def search_graph(self, keywords: str) -> str:
    """Searches the FULL (hidden) graph for resources containing keywords.

    Usage: "Search the dark." Scans the HIDDEN 99% of the graph.
    When to use: The Skeleton is missing a key node.

    Args:
      keywords: Clinical keywords (e.g., "Troponin", "Discharge Summary"). Also
        use this to find atemporal data like Patient demographics, Location
        details, etc.

    Returns:
      A JSON string of matched resources' ID, Type, and Timestamp.
    """
    logging.info("calling search_graph(keywords=%s)", keywords)
    keywords = keywords.lower()
    matches = []
    top_k = 5

    # Use Semantic Search if Embeddings are available
    if self.embedder and self.node_embeddings:
      try:
        keyword_emb = self.embedder.embed([keywords])[0]
        scored_nodes = []
        for node in self.graph.get_all_nodes():
          if node.id in self.node_embeddings:
            node_emb = self.node_embeddings[node.id]
            sim = self.embedder.similarity(
                np.array(keyword_emb), np.array(node_emb)
            )
            scored_nodes.append((sim, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        for sim, node in scored_nodes[:top_k]:
          matches.append({
              "id": node.id,
              "type": node.resource_type,
              "timestamp": (
                  node.timestamp.isoformat() if node.timestamp else None
              ),
              "similarity_score": round(float(sim), 3),
          })
      except Exception as e:  # pylint: disable=broad-except
        print(f"Embedding search failed, falling back to keywords: {e}")

    # Fallback to naive keyword search if semantic search found nothing
    # or is disabled
    if not matches:
      for node in self.graph.get_all_nodes():
        node_text = json.dumps(node.data).lower()
        if keywords in node_text:
          matches.append({
              "id": node.id,
              "type": node.resource_type,
              "timestamp": (
                  node.timestamp.isoformat() if node.timestamp else None
              ),
          })
          if len(matches) >= top_k:
            break

    if not matches:
      return json.dumps(
          {"message": f"No matches found for query: '{keywords}'"}
      )
    return json.dumps(matches, separators=(",", ":"))

  def follow_links(self, resource_id: str) -> str:
    """Return the IDs and Types of resources directly referenced by the node.

    Usage: "Traverse the graph based on references in resources." Returns the
    IDs of resources *referenced by* the target node.
    When to use: You have a node (e.g., MedicationRequest) and need to find its
    reason (reasonReference -> Condition) or its author (requester ->
    Practitioner), but those linked nodes are not visible in the current
    skeleton.

    Args:
      resource_id: The ID of the node you want to trace FROM.

    Returns:
      A JSON string representing references and their types.
    """
    logging.info(f"calling follow_links(resource_id=%s)", resource_id)

    node = self.graph.get_node_by_id(resource_id)
    if not node and "/" in resource_id:
      # Try stripping the resource type prefix
      node = self.graph.get_node_by_id(resource_id.split("/")[-1])
    if not node:
      return json.dumps({"error": f"Node {resource_id} not found."})

    results = []
    for ref in node.references:
      ref_id = ref.split("/")[-1] if "/" in ref else ref
      target_node = self.graph.get_node_by_id(ref_id)
      if target_node:
        results.append({
            "id": ref,
            "type": target_node.resource_type,
        })
      else:
        results.append({"id": ref, "type": "External/Missing"})

    if not results:
      return json.dumps(
          {"message": f"No references found from node {resource_id}."}
      )
    return json.dumps(results, separators=(",", ":"))

  def filter_graph_by_time(
      self,
      start_date: str | None = None,
      end_date: str | None = None,
      resource_type: str | None = None,
      keywords: str | None = None,
  ) -> str:
    """Filters graph nodes by time range, resource type, and keywords.

    Usage: "Find all observations between 2023-01-01 and 2023-12-31"
    When to use: The skeleton does not have the chronological details you need,
    and you want to specifically query a time period.

    Args:
      start_date: ISO format YYYY-MM-DD
      end_date: ISO format YYYY-MM-DD
      resource_type: The FHIR resource type to filter by (e.g., "Observation")
      keywords: Keywords to search for in the node data

    Returns:
      A JSON string of matched resources.
    """
    logging.info(
        f"calling filter_graph_by_time(%s, %s, %s, %s)",
        start_date,
        end_date,
        resource_type,
        keywords,
    )
    top_k = 10
    try:
      start_dt = (
          datetime.datetime.fromisoformat(start_date) if start_date else None
      )
      end_dt = datetime.datetime.fromisoformat(end_date) if end_date else None
      # Add time zone info if necessary, assuming naive datetimes for now
      if start_dt:
        start_dt = start_dt.replace(tzinfo=datetime.timezone.utc)
      if end_dt:
        # end_dt = end_dt.replace(tzinfo=datetime.timezone.utc)
        # Shift end_dt to the very end of the day so it is inclusive
        end_dt = end_dt.replace(
            hour=23, minute=59, second=59, tzinfo=datetime.timezone.utc
        )
    except ValueError:
      return json.dumps({"error": "Invalid date format. Use YYYY-MM-DD"})

    matches = []
    for node in self.graph.get_all_nodes():
      if not node.timestamp:
        continue

      node_ts = node.timestamp
      if node_ts.tzinfo is None:
        node_ts = node_ts.replace(tzinfo=datetime.timezone.utc)

      if start_dt and node_ts < start_dt:
        continue
      if end_dt and node_ts > end_dt:
        continue

      if resource_type and node.resource_type != resource_type:
        continue

      if keywords and keywords.lower() not in json.dumps(node.data).lower():
        continue

      matches.append({
          "id": node.id,
          "type": node.resource_type,
          "timestamp": node.timestamp.isoformat(),
      })
      if len(matches) >= top_k:
        break

    return json.dumps(matches, separators=(",", ":"))
