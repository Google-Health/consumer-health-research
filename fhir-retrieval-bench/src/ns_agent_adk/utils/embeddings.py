"""Embedding utilities using Google Generative AI."""

import abc
import hashlib
import os
import time
from typing import Literal

from absl import logging
import numpy as np
import pandas as pd

from fhir_retrieval_bench.utils import api

DEFAULT_VECTOR_SIZE = 768
MODEL_VECTOR_SIZES = {"gemini-embedding-001": 3072}


class VectorCompass(abc.ABC):
  """Abstract base class for embedding models."""
  model_name: str
  vector_size: int
  @abc.abstractmethod
  def embed(self, texts: list[str]) -> np.ndarray:
    """Embeds a list of texts into vectors."""
    pass

  def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    norm = np.linalg.norm(vector1) * np.linalg.norm(vector2)
    return float(np.dot(vector1, vector2) / norm) if norm else 0.0


class GenAIEmbedder(VectorCompass):
  """Embedder using Google Generative AI API."""

  def __init__(self, backend: api.GenAIBackend, model_name: str = "gemini-embedding-001"):
    self.backend = backend
    self.model_name = model_name
    self.vector_size = MODEL_VECTOR_SIZES.get(model_name, DEFAULT_VECTOR_SIZE)
    logging.info(
        "GenAIEmbedder initialized | model=%s vector_size=%d",
        model_name,
        self.vector_size,
    )

  def embed_text(
      self,
      input_string: str,
      task_type: (
          Literal[
              "RETRIEVAL_QUERY",
              "RETRIEVAL_DOCUMENT",
              "SEMANTIC_SIMILARITY",
              "CLASSIFICATION",
              "CLUSTERING",
          ]
          | None
      ) = "SEMANTIC_SIMILARITY",
  ) -> np.ndarray | None:
    """Computes embeddings for a single input string."""
    embedding = self.backend.embed(self.model_name, input_string, task_type)
    if embedding.size != self.vector_size:
      raise RuntimeError(
          f"Dimension mismatch — expected {self.vector_size}, got {embedding.size} for text_len={len(input_string)}"
      )
    return embedding

  def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embeds a list of texts into vectors in parallel."""
    batch_start = time.time()
    raw_embeddings = self.backend.embed_batch(
        self.model_name,
        texts,
        batch_size=batch_size,
    )
    batch_elapsed = time.time() - batch_start

    validated: list[np.ndarray] = []
    for emb in raw_embeddings:
      if emb.size != self.vector_size:
        raise RuntimeError(
            f"Dimension mismatch — expected {self.vector_size}, got {emb.size} for batch_size={len(texts)}"
        )
      validated.append(emb)

    logging.info(
        "Batch embed done in %.2fs | total=%d",
        batch_elapsed,
        len(texts),
    )

    return np.array(validated)


def get_cache_hash(text: str, model_name: str) -> str:
  """
  Generates a deterministic SHA-256 hash for a string and its associated model.

  Args:
    text: The string content to be embedded.
    model_name: The name of the embedding model. Including this prevents cache
      poisoning if you switch models.

  Returns:
    A hex string representing the unique hash.
  """
  combined_string = f"{model_name}|||{text}"
  encoded_data = combined_string.encode("utf-8")
  hash_object = hashlib.sha256(encoded_data)
  return hash_object.hexdigest()


def update_embedding_pool(
    embedder: VectorCompass,
    embedding_pool: dict[str, np.ndarray],
    texts_to_compute: dict[str, str],
) -> None:
  """Computes embeddings for missing texts and updates the cache in-place."""
  if not texts_to_compute:
    return

  unique_hashes = list(texts_to_compute.keys())
  unique_texts = list(texts_to_compute.values())

  new_embeddings = embedder.embed(unique_texts)

  for text_hash, embedding in zip(unique_hashes, new_embeddings):
    embedding_pool[text_hash] = embedding


def load_embedding_pool(path: str) -> dict[str, np.ndarray]:
  """Loads embeddings from the given path, supporting multiple formats."""
  if not os.path.exists(path):
    return {}

  logging.info('Loading embeddings from %s', path)
  if path.endswith('.parquet'):
    try:
      with open(path, 'rb') as f:
        df_loaded = pd.read_parquet(f)
      return {index: row.values for index, row in df_loaded.iterrows()}
    except Exception as e:
      logging.error('Failed to load parquet file: %s', e)
      return {}
  else:
    raise ValueError(f'Unsupported cache format: {path}. Please use .parquet')


def save_embedding_pool(pool: dict[str, np.ndarray], path: str):
  """Saves embeddings to the given path, supporting multiple formats."""
  parent_dir = os.path.dirname(path)
  if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)

  logging.info('Saving %d embeddings to %s', len(pool), path)
  if path.endswith('.parquet'):
    df = pd.DataFrame.from_dict(pool, orient='index')
    with open(path, 'wb') as f:
      df.to_parquet(f, engine='pyarrow')
  else:
    raise ValueError(f'Unsupported cache format: {path}. Please use .parquet')