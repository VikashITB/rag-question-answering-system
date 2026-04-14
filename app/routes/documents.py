"""
app/routes/documents.py — Document upload, status polling, and listing endpoints.

Endpoints:
  POST /documents/upload   — Upload a PDF or TXT file
  GET  /documents/{id}/status — Poll processing status
  GET  /documents/          — List all documents
  DELETE /documents/{id}   — Remove a document from the index
"""

import os
import shutil
import uuid

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    status,
)

from app.models.document_store import DocumentRegistry
from app.models.schemas import (
    DocumentListResponse,
    DocumentStatus,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.ingestion_service import ingest_document
from app.utils.rate_limiter import check_upload_limit

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
UPLOAD_DIR = "data/uploads"


def _extension_is_allowed(filename: str) -> bool:
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def _record_to_response(record) -> DocumentStatusResponse:
    """Convert a DocumentRecord to its Pydantic response schema."""
    return DocumentStatusResponse(
        document_id=record.document_id,
        filename=record.filename,
        status=record.status,
        chunk_count=record.chunk_count,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


# ─────────────────────────────────────────────────────────────────────────────
# POST /documents/upload
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a PDF or TXT document for indexing",
    dependencies=[Depends(check_upload_limit)],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document. Returns immediately with a document_id and 'pending' status.
    Ingestion (text extraction, chunking, embedding, indexing) runs in the background.
    Poll GET /documents/{document_id}/status to check progress.
    """
    # ── Validate filename ──────────────────────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    if not _extension_is_allowed(file.filename):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}. "
                f"Received: '{os.path.splitext(file.filename)[1]}'"
            ),
        )

    # ── Read file contents ─────────────────────────────────────────────────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // 1024 // 1024} MB.",
        )

    # ── Persist to disk ────────────────────────────────────────────────────
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    # Use a UUID-prefixed filename to avoid collisions with duplicate uploads
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f_out:
        f_out.write(contents)

    # ── Register document ──────────────────────────────────────────────────
    registry = DocumentRegistry.get_instance()
    record = registry.create(file.filename)
    record.file_path = file_path

    # ── Queue background ingestion ────────────────────────────────────────
    background_tasks.add_task(
        ingest_document,
        document_id=record.document_id,
        file_path=file_path,
        filename=file.filename,
    )

    return DocumentUploadResponse(
        document_id=record.document_id,
        filename=file.filename,
        status=DocumentStatus.PENDING,
        message=(
            "Document received. Ingestion running in background. "
            f"Poll GET /documents/{record.document_id}/status for progress."
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /documents/{document_id}/status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/{document_id}/status",
    response_model=DocumentStatusResponse,
    summary="Poll the processing status of an uploaded document",
)
def get_document_status(document_id: str):
    registry = DocumentRegistry.get_instance()
    record = registry.get(document_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    return _record_to_response(record)


# ─────────────────────────────────────────────────────────────────────────────
# GET /documents/
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents and their statuses",
)
def list_documents():
    registry = DocumentRegistry.get_instance()
    records = registry.list_all()
    responses = [_record_to_response(r) for r in records]
    return DocumentListResponse(documents=responses, total=len(responses))


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /documents/{document_id}
# ─────────────────────────────────────────────────────────────────────────────

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_200_OK,
    summary="Remove a document and its vectors from the index",
)
def delete_document(document_id: str):
    """
    Removes all vectors for this document from FAISS and deletes its registry entry.
    Note: Rebuilds the FAISS index — may be slow for large indices.
    """
    from app.services.vector_store import VectorStoreService

    registry = DocumentRegistry.get_instance()
    record = registry.get(document_id)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )

    vector_store = VectorStoreService.get_instance()
    removed = vector_store.remove_document(document_id)

    return {
        "document_id": document_id,
        "filename": record.filename,
        "vectors_removed": removed,
        "message": "Document removed from index.",
    }
