# src/embedder.py
"""Embeddings + retrieval for the resume auditor.

Encodes text chunks into dense vectors using sentence-transformers, stores
them in memory, and supports top-k retrieval by cosine similarity against
a query vector.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer


# Model is fast and small (~80MB). Produces 384-dim vectors.
_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Lazy-load the model once per process. The lru_cache decorator turns
    this into a singleton: first call loads the model, subsequent calls
    return the cached instance instantly."""
    return SentenceTransformer(_MODEL_NAME)


@dataclass
class Index:
    """Holds the chunks and their pre-computed embeddings.

    `vectors` is shape (N, D) where N = number of chunks, D = embedding dim.
    Vectors are L2-normalized at build time so cosine similarity reduces
    to a single matrix-vector dot product at query time.
    """
    chunks: List[str]
    vectors: np.ndarray


def build_index(chunks: List[str]) -> Index:
    """Encode each chunk into a vector, normalize, return an Index.

    Normalization is the key trick: if all vectors are unit-length, then
    cosine(a, b) = dot(a, b). No division, no magnitude bookkeeping at
    query time.
    """
    if not chunks:
        raise ValueError("build_index called with no chunks")

    model = _get_model()
    # encode() returns a numpy array shape (N, D). normalize_embeddings=True
    # divides each row by its L2 norm, so each row is a unit vector.
    vectors = model.encode(
        chunks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return Index(chunks=chunks, vectors=vectors)


def retrieve(index: Index, query: str, k: int = 3) -> List[Tuple[str, float]]:
    """Return the top-k chunks most similar to `query`.

    Each result is (chunk_text, similarity_score). Scores are cosine
    similarities, so they live in [-1, 1]; with this model and these
    inputs you'll typically see [0.0, 0.7].
    """
    if not index.chunks:
        return []
    k = min(k, len(index.chunks))

    model = _get_model()
    # Encode the query the same way as the chunks: unit-length vector.
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]  # take the single row out of the (1, D) array

    # Cosine similarity of unit vectors == dot product.
    # index.vectors @ query_vec is a vectorized dot of every chunk against the query.
    # Result shape: (N,)
    scores = index.vectors @ query_vec

    # argsort returns indices that would sort ascending; we want descending,
    # so reverse with [::-1] and take the first k.
    top_idx = np.argsort(scores)[::-1][:k]

    return [(index.chunks[i], float(scores[i])) for i in top_idx]
