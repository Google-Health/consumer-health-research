"""Core graph data structures for FHIR neuro-symbolic reasoning."""

import dataclasses
import datetime
from typing import Any

from ns_agent_adk.core import fhir_time


@dataclasses.dataclass
class FHIRNode:
  """Represents a single FHIR resource within the graph."""

  id: str
  resource_type: str
  timestamp: datetime.datetime | None
  references: list[str] = dataclasses.field(default_factory=list)
  data: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class Hypernode:
  """Represents a temporal cluster of FHIR resources, often an Encounter."""

  id: str
  start_time: datetime.datetime | None
  end_time: datetime.datetime | None
  nodes: list[FHIRNode] = dataclasses.field(default_factory=list)
  is_phantom: bool = False

  def contains_timestamp(self, ts: datetime.datetime) -> bool:
    if self.start_time and self.end_time and ts:
      return self.start_time <= ts <= self.end_time
    return False


class ChronologicalHypergraph:
  """A graph structure organizing FHIR resources chronologically."""

  def __init__(self):
    """Initializes the graph."""
    self.spine: list[Hypernode] = []
    self._node_map: dict[str, FHIRNode] = {}
    self.global_nodes: list[FHIRNode] = []

  def _get_timestamp(
      self, resource: dict[str, Any]
  ) -> datetime.datetime | None:
    """Extracts a timestamp from a FHIR resource."""
    return fhir_time.get_clinical_time(resource).start_time

  def build_from_bundle(self, fhir_bundle: dict[str, Any]):
    """Builds the hypergraph from a FHIR Bundle."""
    if not fhir_bundle.get('entry'):
      return

    resources = [
        entry['resource']
        for entry in fhir_bundle['entry']
        if 'resource' in entry
    ]

    # Add all resources to the node map first
    for res in resources:
      ts = self._get_timestamp(res)
      node = FHIRNode(
          id=res.get('id', ''),
          resource_type=res.get('resourceType', ''),
          timestamp=ts,
          data=res,
          references=extract_references(res),
      )
      if node.id:
        self._node_map[node.id] = node

    encounters = sorted(
        [r for r in resources if r.get('resourceType') == 'Encounter'],
        key=lambda r: self._get_timestamp(r) or datetime.datetime.min,
    )

    # Create Hypernodes from Encounters
    for enc in encounters:
      ct = fhir_time.get_clinical_time(enc)
      start_time = ct.start_time
      end_time = ct.end_time or start_time

      hypernode = Hypernode(
          id=enc['id'], start_time=start_time, end_time=end_time
      )
      # NOTE: Encounter resource itself is NOT added to hypernode.nodes
      self.spine.append(hypernode)

    # Process other resources
    other_resources = [
        r for r in resources if r.get('resourceType') != 'Encounter'
    ]
    unmatched_timestamped_resources = []
    
    missing_timestamp_counts = {}
    for res in other_resources:
      node = self._node_map.get(res.get('id', ''))
      if not node:
        continue  # Should not happen if all resources are added above

      encounter_ref = res.get('encounter', {}).get('reference', '')
      matched = False
      if encounter_ref:
        enc_id = encounter_ref.split('/')[-1]
        for hn in self.spine:
          if hn.id == enc_id:
            hn.nodes.append(node)
            matched = True
            break

      if not matched:
        if node.timestamp:
          unmatched_timestamped_resources.append(node)
        else:
          self.global_nodes.append(node)
          count = missing_timestamp_counts.get(node.resource_type, 0)
          if count < 10:
            print(f'Info: Resource {node.id} ({node.resource_type}) has no timestamp, adding to global_nodes.')
          elif count == 10:
            print(f'Info: Resource {node.id} ({node.resource_type}) has no timestamp, adding to global_nodes. (Hiding further logs for this resource type.........)')
          missing_timestamp_counts[node.resource_type] = count + 1
    print(f'Resources with missing timestamp: {missing_timestamp_counts}')

    # Latent Episode Inference (LEI)
    for node in unmatched_timestamped_resources:

      placed = False
      for hn in self.spine:
        if hn.contains_timestamp(node.timestamp):
          hn.nodes.append(node)
          placed = True
          break

      if not placed:
        # Create Phantom Episode
        phantom_id = f'phantom-{node.id}'
        start_time = node.timestamp.replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end_time = (
            start_time
            + datetime.timedelta(days=1)
            - datetime.timedelta(microseconds=1)
        )

        phantom_hn = Hypernode(
            id=phantom_id,
            start_time=start_time,
            end_time=end_time,
            nodes=[node],
            is_phantom=True,
        )

        # Insert into sorted spine
        insert_idx = 0
        for i, hn in enumerate(self.spine):
          if hn.start_time and phantom_hn.start_time < hn.start_time:
            insert_idx = i
            break
          else:
            insert_idx = i + 1
        self.spine.insert(insert_idx, phantom_hn)

    # Sort nodes within each hypernode
    for hn in self.spine:
      hn.nodes.sort(
          key=lambda n: n.timestamp
          or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)
      )

  def get_all_nodes(self) -> list[FHIRNode]:
    """Returns all nodes in the graph."""
    return list(self._node_map.values())

  def get_node_by_id(self, node_id: str) -> FHIRNode | None:
    return self._node_map.get(node_id)

  def get_hypernode_by_id(self, hypernode_id: str) -> Hypernode | None:
    for hn in self.spine:
      if hn.id == hypernode_id:
        return hn
    return None


def extract_references(resource: dict[str, Any]) -> list[str]:
  """Extracts all FHIR references from a resource.

  A valid internal FHIR reference is expected to be a string containing '/'
  (e.g., 'Patient/123') but not starting with 'http:', 'https:', or '/'.
  This check avoids capturing URLs or absolute paths as FHIR references.
  Args:
    resource: A dictionary representing a FHIR resource.

  Returns:
    A list of strings, where each string is a FHIR reference.
  """
  references = set()

  def find_refs(data: Any):
    if isinstance(data, dict):
      for key, value in data.items():
        if key == 'reference' and isinstance(value, str):
          # Valid references are internal, relative links (e.g., "Patient/123").
          # We ignore absolute URLs or paths.
          if '/' in value and not value.startswith(('http:', 'https:', '/')):
            references.add(value)
        else:
          find_refs(value)
    elif isinstance(data, list):
      for item in data:
        find_refs(item)

  find_refs(resource)
  return sorted(list(references))
