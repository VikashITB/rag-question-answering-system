"""
app/routes/questions.py — Question answering endpoint.

Flow:
  1. Validate request (Pydantic)
  2. Check rate limit
  3. Embed the query
  4. Retrieve top-k chunks from FAISS
  5. Call LLM with retrieved context
  6. Return answer + retrieved chunks + latency

Latency is tracked end-to-end (steps 3–6) and returned in the response.
This gives clients and operators visibility into pipeline performance.
"""

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.document_store import DocumentRegistry
from app.models.schemas import (
    DocumentStatus,
    QuestionRequest,
    QuestionResponse,
    RetrievedChunk,
)
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.utils.rate_limiter import check_question_limit

logger = logging.getLogger(__name__)
router = APIRouter()

# Instantiate LLM service once (holds the API client)
_llm_service = LLMService()


@router.post(
    "/ask",
    response_model=QuestionResponse,
    summary="Ask a question based on uploaded documents",
    dependencies=[Depends(check_question_limit)],
)
def ask_question(request: QuestionRequest):
    """
    Ask a question. Returns an LLM-generated answer grounded in retrieved
    document chunks, along with the chunks and end-to-end latency.

    Prerequisites:
    - At least one document must have status='ready'.
    - ANTHROPIC_API_KEY must be set in the environment.
    """
    # ── Validate that we have something to query ──────────────────────────
    vector_store = VectorStoreService.get_instance()
    if vector_store.total_vectors == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "No documents are indexed yet. "
                "Upload and wait for at least one document to reach status='ready'."
            ),
        )

    # ── Validate document_ids filter if provided ──────────────────────────
    if request.document_ids:
        registry = DocumentRegistry.get_instance()
        for doc_id in request.document_ids:
            record = registry.get(doc_id)
            if record is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document '{doc_id}' not found.",
                )
            if record.status != DocumentStatus.READY:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Document '{doc_id}' is not ready for querying "
                        f"(current status: {record.status.value})."
                    ),
                )

    # ── Start latency clock ───────────────────────────────────────────────
    t_start = time.perf_counter()

    # ── Retrieve relevant chunks ──────────────────────────────────────────
    try:
        retrieved = vector_store.search(
            query=request.question,
            top_k=request.top_k,
            document_ids=request.document_ids,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    if not retrieved:
        # This can happen if document_ids filter excludes everything
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No relevant chunks found. "
                "The document filter may be too restrictive, or the documents "
                "may not contain information related to your question."
            ),
        )

    # ── Generate answer ───────────────────────────────────────────────────
    try:
        answer = _llm_service.generate_answer(
            question=request.question,
            retrieved_chunks=retrieved,
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected LLM error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LLM service error: {type(e).__name__}: {e}",
        )

    # ── Compute latency ───────────────────────────────────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        "Question answered | latency=%.1fms | chunks=%d | question='%s...'",
        latency_ms, len(retrieved), request.question[:60],
    )

    # ── Build response ────────────────────────────────────────────────────
    retrieved_chunk_responses = [
        RetrievedChunk(
            document_id=meta.document_id,
            filename=meta.filename,
            chunk_index=meta.chunk_index,
            text=meta.text,
            similarity_score=round(score, 4),
        )
        for meta, score in retrieved
    ]

    return QuestionResponse(
        question=request.question,
        answer=answer,
        retrieved_chunks=retrieved_chunk_responses,
        latency_ms=round(latency_ms, 2),
        model_used=_llm_service.model_name,
    )
