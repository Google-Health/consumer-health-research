"""Embedding utilities using Google GenAI.

Provides a ``VectorCompass`` abstract base and its concrete ``GenAIEmbedder``
implementation, which supports multi-key rotation, parallel batch embedding,
and cosine similarity scoring.
"""

import abc
import hashlib
import os
import time
from typing import Literal

from absl import logging
from immutabledict import immutabledict
import numpy as np
import pandas as pd

from fhir_retrieval_bench.utils import api


DEFAULT_VECTOR_SIZE = 768
# Known model -> dimension mappings; used to validate returned embeddings.
MODEL_VECTOR_SIZES = immutabledict({"gemini-embedding-001": 3072})


class VectorCompass(abc.ABC):
  """Abstract base class for embedding models.

  Subclasses must implement ``embed`` (batch). The ``similarity`` method
  provides cosine similarity between two vectors.

  Attributes:
    model_name: The embedding model identifier.
    vector_size: Expected dimensionality of returned embeddings.
  """

  model_name: str
  vector_size: int

  @abc.abstractmethod
  def embed(self, texts: list[str]) -> np.ndarray:
    """Embed a batch of texts.

    Args:
      texts: List of input strings.

    Returns:
      A 2-D numpy array of shape ``(len(texts), vector_size)``.
    """
    ...

  def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero norm (avoids division by zero).
    """
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / norm) if norm else 0.0

  def rank_by_similarity(
      self,
      query_emb: np.ndarray,
      texts: list[str],
      top_k: int,
  ) -> list[int]:
    """Embed texts and return indices of top-k most similar to query_emb.

    Args:
      query_emb: Pre-computed query embedding vector.
      texts: List of candidate strings to embed and rank.
      top_k: Number of top results to return.

    Returns:
      List of indices into *texts* sorted by descending similarity.
    """
    text_embs = self.embed(texts)
    similarities = np.array(
        [self.similarity(query_emb, emb) for emb in text_embs]
    )
    top_indices = np.argsort(similarities)[::-1][:top_k]

    logging.debug(
        "Similarity stats: min=%.3f max=%.3f mean=%.3f | selected top_k=%d"
        " from %d",
        similarities.min(),
        similarities.max(),
        similarities.mean(),
        top_k,
        len(texts),
    )
    return list(top_indices)


class GenAIEmbedder(VectorCompass):
  """Embedder using Google GenAI API with multi-key support.

  Embeddings only run through the public GenAI API in this build, so this
  class takes a :class:`GenAIBackend` directly rather than going through the
  :func:`get_api_client` factory.
  """

  def __init__(
      self,
      backend: api.GenAIBackend,
      model_name: str = "gemini-embedding-001",
  ):
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
      text: str,
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
    """Embed a single text string via the GenAI API.

    Uses key rotation and exponential backoff (delegated to
    ``api.embed_content``).
    Returns ``None`` if all retries fail or if the returned embedding has
    an unexpected dimension.

    Args:
      text: The input string to embed.
      task_type: Optional embedding task hint for the API.

    Returns:
      A 1-D numpy array, or ``None`` on failure.
    """
    embedding = self.backend.embed(self.model_name, text, task_type)
    if embedding.size != self.vector_size:
      raise RuntimeError(
          f"Dimension mismatch — expected {self.vector_size}, got {embedding.size} for text_len={len(text)}"
      )
    return embedding

  def embed(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Embed a list of texts using batched API calls.

    Sends up to *batch_size* texts per API call (default 100) to minimise
    the number of round-trips.  Failed batches are replaced with zero
    vectors of the correct size.

    Args:
      texts: List of input strings to embed.
      batch_size: Maximum texts per API call.

    Returns:
      A 2-D numpy array of shape ``(len(texts), vector_size)``.
    """
    logging.info("Batch embed: %d texts, batch_size=%d", len(texts), batch_size)

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
  """Generates a deterministic SHA-256 hash for a string and its associated model.

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


def save_embedding_pool(pool: dict[str, np.ndarray], path: str) -> None:
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
