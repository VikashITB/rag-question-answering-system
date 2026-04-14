"""
app/services/ingestion_service.py — Orchestrates the document ingestion pipeline.

Pipeline stages:
  1. Text extraction (PDF → plain text / TXT → plain text)
  2. Chunking (plain text → overlapping word-based chunks)
  3. Embedding (chunks → float32 vectors via SentenceTransformer)
  4. Indexing (vectors + metadata → FAISS index)
  5. Registry update (status, chunk count)

This service is called from a FastAPI BackgroundTask so it runs off the
request-response cycle. Status updates are written to the DocumentRegistry
so the client can poll /documents/{id}/status.
"""

import logging
import os
from typing import Optional

from app.models.document_store import DocumentRegistry
from app.models.schemas import DocumentStatus
from app.services.embedder import EmbeddingService
from app.services.vector_store import VectorStoreService
from app.utils.chunker import chunk_text
from app.utils.text_extractor import extract_text

logger = logging.getLogger(__name__)


def ingest_document(document_id: str, file_path: str, filename: str) -> None:
    """
    Full ingestion pipeline for a single document.
    Designed to be called as a background task.

    Args:
        document_id: Unique ID from DocumentRegistry.
        file_path:   Path to the saved upload file on disk.
        filename:    Original filename (for metadata / display).
    """
    registry = DocumentRegistry.get_instance()
    embedder = EmbeddingService.get_instance()
    vector_store = VectorStoreService.get_instance()

    # ── Mark as PROCESSING ────────────────────────────────────────────────
    registry.update_status(document_id, DocumentStatus.PROCESSING)
    logger.info("Starting ingestion for document '%s' (%s)", document_id, filename)

    try:
        # ── Stage 1: Text extraction ──────────────────────────────────────
        logger.info("[1/4] Extracting text from '%s'...", filename)
        raw_text = extract_text(file_path)

        if not raw_text.strip():
            raise ValueError("Extracted text is empty.")

        # ── Stage 2: Chunking ─────────────────────────────────────────────
        logger.info("[2/4] Chunking text...")
        chunks = chunk_text(raw_text)   # uses DEFAULT_CHUNK_SIZE=500, overlap=100

        if not chunks:
            raise ValueError("Chunking produced no chunks.")

        chunk_texts = [c.text for c in chunks]
        logger.info("[2/4] Produced %d chunks.", len(chunks))

        # ── Stage 3: Embedding ────────────────────────────────────────────
        logger.info("[3/4] Embedding %d chunks...", len(chunks))
        embeddings = embedder.embed(chunk_texts)

        # ── Stage 4: Index ────────────────────────────────────────────────
        logger.info("[4/4] Indexing embeddings...")
        added = vector_store.add_chunks(
            document_id=document_id,
            filename=filename,
            chunks=chunk_texts,
            embeddings=embeddings,
        )

        # ── Mark as READY ─────────────────────────────────────────────────
        registry.update_status(
            document_id, DocumentStatus.READY, chunk_count=added
        )
        logger.info(
            "Ingestion complete for '%s': %d chunks indexed.", filename, added
        )

    except Exception as exc:
        error_msg = f"{type(exc).__name__}: {str(exc)}"
        logger.error("Ingestion FAILED for '%s': %s", filename, error_msg)
        registry.update_status(
            document_id, DocumentStatus.FAILED, error=error_msg
        )
        # Don't re-raise — this runs in a background task; exceptions are silent otherwise
    finally:
        # Clean up the uploaded file after ingestion to save disk space.
        # Comment out if you need to re-process documents without re-uploading.
        _cleanup_file(file_path)


def _cleanup_file(file_path: str) -> None:
    """Remove the raw upload file after successful ingestion."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info("Cleaned up upload file: %s", file_path)
    except OSError as e:
        logger.warning("Could not delete upload file '%s': %s", file_path, e)
