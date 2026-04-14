"""
app/services/vector_store.py — FAISS-based vector index with metadata management.

──────────────────────────────────────────────────────────────────────────────
WHY FAISS?
──────────────────────────────────────────────────────────────────────────────
- Open-source, battle-tested (Meta AI), runs fully in-process — no server to manage.
- IndexFlatIP (Inner Product on normalized vectors) = exact cosine similarity search.
  For datasets < 1M vectors, exact search is fast enough; no need for approximate
  methods (IVF, HNSW) which sacrifice recall for speed.
- Alternative: ChromaDB / Qdrant — better UX and built-in metadata filtering, but
  require running a separate process/container. FAISS is simpler for a single-node API.
- Persistence: FAISS.write_index / read_index serializes to a binary file.

DATA STRUCTURES:
  - FAISS stores only float32 vectors; it has no concept of metadata.
  - We maintain a parallel list `_metadata` where _metadata[i] corresponds to
    the vector at FAISS index position i.
  - On persist: save FAISS index as binary + metadata as JSON.
──────────────────────────────────────────────────────────────────────────────
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.services.embedder import EmbeddingService, EMBEDDING_DIM

logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = "data/vector_store/faiss.index"
METADATA_PATH = "data/vector_store/metadata.json"


@dataclass
class ChunkMetadata:
    """
    All metadata stored alongside a FAISS vector.
    The faiss_id matches the row index in the FAISS index.
    """
    faiss_id: int
    document_id: str
    filename: str
    chunk_index: int
    text: str                # original chunk text (needed at retrieval time)


class VectorStoreService:
    """
    Manages the FAISS index and its associated metadata.

    Thread-safety:
      - A single RWLock is used; we use threading.Lock() for simplicity
        (writers and readers both acquire it). For read-heavy workloads,
        use a proper reader-writer lock or separate read/write paths.
    """

    _instance: Optional["VectorStoreService"] = None
    _class_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._index = None         # FAISS index, initialized on first add
        self._metadata: List[ChunkMetadata] = []
        self._embedder = EmbeddingService.get_instance()

    @classmethod
    def get_instance(cls) -> "VectorStoreService":
        if cls._instance is None:
            with cls._class_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Index initialization ───────────────────────────────────────────────

    def _init_index(self):
        """Create a fresh FAISS IndexFlatIP index (exact cosine similarity)."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        # IndexFlatIP: exact inner product search.
        # With L2-normalized embeddings, inner product == cosine similarity.
        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        logger.info("Initialized FAISS IndexFlatIP with dim=%d", EMBEDDING_DIM)

    # ── Add vectors ────────────────────────────────────────────────────────

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: List[str],
        embeddings: np.ndarray,
    ) -> int:
        """
        Add a batch of chunks and their embeddings to the index.

        Args:
            document_id: ID of the source document.
            filename:    Original filename (for display in results).
            chunks:      List of chunk texts.
            embeddings:  np.ndarray of shape (len(chunks), EMBEDDING_DIM).

        Returns:
            Number of vectors added.

        Raises:
            ValueError: If chunks and embeddings counts don't match.
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks but {embeddings.shape[0]} embeddings."
            )

        with self._lock:
            if self._index is None:
                self._init_index()

            start_id = self._index.ntotal  # FAISS assigns IDs 0, 1, 2, ...

            self._index.add(embeddings)

            for i, chunk_text in enumerate(chunks):
                self._metadata.append(
                    ChunkMetadata(
                        faiss_id=start_id + i,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=i,
                        text=chunk_text,
                    )
                )

            added = len(chunks)
            logger.info(
                "Added %d vectors for doc '%s' (total index size: %d)",
                added, document_id, self._index.ntotal,
            )
            return added

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 4,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Retrieve the top-k most similar chunks for a query.

        Args:
            query:        Natural language query string.
            top_k:        Number of results to return.
            document_ids: Optional filter — only return chunks from these docs.

        Returns:
            List of (ChunkMetadata, similarity_score) tuples, sorted descending.

        Raises:
            ValueError: If the index is empty.
        """
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                raise ValueError(
                    "Vector store is empty. Please upload and process documents first."
                )

            # Embed the query
            query_vec = self._embedder.embed_single(query)  # shape (1, DIM)

            # Retrieve more than top_k when filtering, to compensate for post-filter loss
            fetch_k = top_k * 5 if document_ids else top_k
            fetch_k = min(fetch_k, self._index.ntotal)

            scores, indices = self._index.search(query_vec, fetch_k)

            results: List[Tuple[ChunkMetadata, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    # FAISS returns -1 for padded results when fewer vectors exist
                    continue
                meta = self._metadata[idx]

                # Apply document filter if provided
                if document_ids and meta.document_id not in document_ids:
                    continue

                results.append((meta, float(score)))
                if len(results) >= top_k:
                    break

            logger.info(
                "Search for '%s...' returned %d results (top score: %.4f)",
                query[:50], len(results), results[0][1] if results else 0.0,
            )
            return results

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self):
        """Persist FAISS index and metadata to disk."""
        if self._index is None or self._index.ntotal == 0:
            logger.info("Nothing to save — index is empty.")
            return

        import faiss
        os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

        with self._lock:
            faiss.write_index(self._index, FAISS_INDEX_PATH)
            metadata_dicts = [asdict(m) for m in self._metadata]
            with open(METADATA_PATH, "w", encoding="utf-8") as f:
                json.dump(metadata_dicts, f, indent=2)

        logger.info(
            "Saved FAISS index (%d vectors) to %s", self._index.ntotal, FAISS_INDEX_PATH
        )

    def load_if_exists(self):
        """Load FAISS index and metadata from disk if persisted files exist."""
        if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
            logger.info("No persisted vector store found — starting fresh.")
            return

        import faiss

        with self._lock:
            self._index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self._metadata = [ChunkMetadata(**d) for d in raw]

        logger.info(
            "Loaded FAISS index: %d vectors, %d metadata entries.",
            self._index.ntotal, len(self._metadata),
        )

    # ── Info ───────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def remove_document(self, document_id: str) -> int:
        """
        Remove all vectors for a given document.
        NOTE: FAISS IndexFlatIP does not support in-place deletion efficiently.
        We rebuild the index from the remaining metadata — acceptable for
        moderate document counts. For large indices, use IndexIDMap.
        """
        with self._lock:
            # Filter out the document's chunks
            remaining = [m for m in self._metadata if m.document_id != document_id]
            removed_count = len(self._metadata) - len(remaining)

            if removed_count == 0:
                return 0

            # Re-embed remaining chunks and rebuild
            import faiss
            new_index = faiss.IndexFlatIP(EMBEDDING_DIM)

            if remaining:
                # We stored texts in metadata so we can re-embed without re-uploading
                texts = [m.text for m in remaining]
                embedder = EmbeddingService.get_instance()
                embeddings = embedder.embed(texts)
                new_index.add(embeddings)
                # Update faiss_ids to match new positions
                for i, meta in enumerate(remaining):
                    meta.faiss_id = i

            self._index = new_index
            self._metadata = remaining
            logger.info(
                "Removed %d vectors for document '%s'. Index size: %d",
                removed_count, document_id, self._index.ntotal,
            )
            return removed_count
