"""FHIR utility functions."""

import json
import re


def strip_fhir_data(data, max_string_len=200):
  """Removes unnecessary boilerplate from FHIR JSON to save token context."""
  if isinstance(data, dict):
    stripped = {}
    for k, v in data.items():
      if k in [
          "text",
          "meta",
          "extension",
          "identifier",
          "contained",
          "system",
          "resourceType",
          "id",
          "subject",
          "recorder",
          "custodian",
          "eventHistory",
          "reportedBoolean",
          "context",
      ]:
        continue

      # Explicitly drop base64 encoded attachment data useless to the LLM
      if k == "data" and isinstance(v, str) and len(v) > 100:
        continue

      clean_v = strip_fhir_data(v, max_string_len)
      if clean_v in [{}, []]:
        continue

      stripped[k] = clean_v
    return stripped
  elif isinstance(data, list):
    stripped_list = [strip_fhir_data(item, max_string_len) for item in data]
    return [item for item in stripped_list if item not in [{}, []]]
  elif isinstance(data, str) and len(data) > max_string_len:
    return data[:max_string_len] + "...[TRUNC]"
  return data