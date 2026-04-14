"""
app/models/document_store.py — In-memory document registry.

Why in-memory instead of a database?
- This project is a single-instance API; persistence across restarts is handled
  by FAISS index serialization (vector_store.save/load).
- Adding SQLite/Postgres would require migrations, drivers, and connection pooling —
  unnecessary complexity for the goal of this project.
- In production at scale you'd replace this with a proper DB; the interface is
  kept thin so it's easy to swap.

Thread-safety: We use a threading.Lock because FastAPI BackgroundTasks can run
on a thread pool, so concurrent writes to shared dicts must be protected.
"""

import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from app.models.schemas import DocumentStatus


class DocumentRecord:
    """Holds metadata for a single uploaded document."""

    def __init__(self, filename: str):
        self.document_id: str = str(uuid.uuid4())
        self.filename: str = filename
        self.status: DocumentStatus = DocumentStatus.PENDING
        self.chunk_count: Optional[int] = None
        self.error: Optional[str] = None
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = datetime.utcnow()
        # Path where the raw file is stored on disk
        self.file_path: Optional[str] = None

    def touch(self):
        """Update the `updated_at` timestamp."""
        self.updated_at = datetime.utcnow()


class DocumentRegistry:
    """
    Singleton registry that maps document_id → DocumentRecord.
    All mutating methods acquire the lock.
    """

    _instance: Optional["DocumentRegistry"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self):
        self._records: Dict[str, DocumentRecord] = {}
        self._rw_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "DocumentRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── CRUD ──────────────────────────────────

    def create(self, filename: str) -> DocumentRecord:
        record = DocumentRecord(filename)
        with self._rw_lock:
            self._records[record.document_id] = record
        return record

    def get(self, document_id: str) -> Optional[DocumentRecord]:
        return self._records.get(document_id)

    def list_all(self) -> List[DocumentRecord]:
        with self._rw_lock:
            return list(self._records.values())

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: Optional[int] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Update the status of a document. Returns False if document_id not found.
        """
        with self._rw_lock:
            record = self._records.get(document_id)
            if record is None:
                return False
            record.status = status
            if chunk_count is not None:
                record.chunk_count = chunk_count
            if error is not None:
                record.error = error
            record.touch()
            return True
