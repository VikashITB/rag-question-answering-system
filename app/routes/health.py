"""app/routes/health.py — Health check endpoint."""

from fastapi import APIRouter

from app.models.document_store import DocumentRegistry
from app.models.schemas import HealthResponse
from app.services.vector_store import VectorStoreService

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="System health check")
def health_check():
    """
    Returns basic system health info:
    - Status (always 'ok' if the server is running)
    - Number of documents indexed
    - Total vectors in the FAISS index
    """
    registry = DocumentRegistry.get_instance()
    vector_store = VectorStoreService.get_instance()

    ready_docs = [
        d for d in registry.list_all() if d.status.value == "ready"
    ]

    return HealthResponse(
        status="ok",
        documents_indexed=len(ready_docs),
        vector_store_size=vector_store.total_vectors,
    )
