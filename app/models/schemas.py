"""
app/models/schemas.py — Pydantic schemas for all API request and response bodies.

Why Pydantic?
- Automatic type validation and coercion with clear error messages.
- Self-documenting: FastAPI uses these models to generate OpenAPI docs.
- Separation of concern: API contract is defined here, not scattered in route handlers.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class DocumentStatus(str, Enum):
    PENDING = "pending"       # Uploaded, not yet processed
    PROCESSING = "processing" # Background task running
    READY = "ready"           # Embedded and indexed — queryable
    FAILED = "failed"         # Processing failed; see error field


# ──────────────────────────────────────────────
# Document Schemas
# ──────────────────────────────────────────────

class DocumentUploadResponse(BaseModel):
    """Returned immediately after a document is uploaded (before processing)."""
    document_id: str = Field(..., description="Unique ID assigned to this document")
    filename: str
    status: DocumentStatus
    message: str


class DocumentStatusResponse(BaseModel):
    """Returned when polling the status of a document."""
    document_id: str
    filename: str
    status: DocumentStatus
    chunk_count: Optional[int] = Field(
        None, description="Number of chunks generated (available after processing)"
    )
    error: Optional[str] = Field(None, description="Error message if status=failed")
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: List[DocumentStatusResponse]
    total: int


# ──────────────────────────────────────────────
# Question / Answer Schemas
# ──────────────────────────────────────────────

class QuestionRequest(BaseModel):
    """User's question payload."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer based on uploaded documents",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve (1–10)",
    )
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict retrieval to specific document IDs. If None, searches all.",
    )

    @validator("question")
    def question_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be blank or whitespace only.")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A single retrieved context chunk — returned for transparency/debugging."""
    document_id: str
    filename: str
    chunk_index: int
    text: str
    similarity_score: float = Field(
        ..., description="Cosine similarity score (0–1, higher = more relevant)"
    )


class QuestionResponse(BaseModel):
    """Full response including the generated answer and retrieved context."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")
    model_used: str


# ──────────────────────────────────────────────
# Generic / Error Schemas
# ──────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    vector_store_size: int
