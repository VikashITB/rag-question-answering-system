"""
app/services/embedder.py — Generate dense vector embeddings for text chunks.

──────────────────────────────────────────────────────────────────────────────
MODEL CHOICE: all-MiniLM-L6-v2
──────────────────────────────────────────────────────────────────────────────
Why this model?
  - 384-dimensional embeddings — small enough for FAISS to be fast, large enough
    for high-quality semantic similarity.
  - ~22M parameters; loads in ~1–2 seconds on CPU.
  - Trained via contrastive learning on 1B+ sentence pairs; excellent semantic
    similarity benchmarks (SBERT leaderboard).
  - NOT a black box: the model card explains training data and evaluation clearly.
  - Alternative considered: text-embedding-ada-002 (OpenAI) — better quality but
    requires an API key, costs money, and introduces an external dependency.
    For a self-contained demo, all-MiniLM-L6-v2 is the right call.

Batch embedding:
  - We embed in configurable batches to avoid OOM errors on large documents.
  - encode() with show_progress_bar=False is used to keep logs clean.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
import threading
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model name — change here to switch models application-wide
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384          # must match the model's output dimension
BATCH_SIZE = 64              # chunks embedded per forward pass


class EmbeddingService:
    """
    Singleton wrapper around SentenceTransformer.
    The model is loaded once at first use (lazy load) to avoid startup latency
    when the service is imported but not yet needed.
    """

    _instance: Optional["EmbeddingService"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Lazy-load the SentenceTransformer model on first call."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
                self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
                logger.info("Embedding model loaded successfully.")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of strings.

        Args:
            texts: List of text strings to embed.

        Returns:
            2D numpy array of shape (len(texts), EMBEDDING_DIM), dtype float32.
            Float32 is required by FAISS.

        Raises:
            ValueError: If texts list is empty.
        """
        if not texts:
            raise ValueError("Cannot embed an empty list of texts.")

        self._load_model()

        logger.info("Embedding %d text(s) in batches of %d...", len(texts), BATCH_SIZE)

        embeddings = self._model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalize → cosine similarity = dot product
        )

        # Ensure float32 — FAISS requires this dtype
        embeddings = embeddings.astype(np.float32)

        logger.info(
            "Embeddings generated: shape=%s, dtype=%s",
            embeddings.shape, embeddings.dtype
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """
        Embed a single string. Returns shape (1, EMBEDDING_DIM).
        Convenience wrapper used for query embedding at search time.
        """
        return self.embed([text])

    @property
    def model_name(self) -> str:
        return EMBEDDING_MODEL_NAME

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM
