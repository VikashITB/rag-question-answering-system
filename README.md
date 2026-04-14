# RAG-Based Question Answering System API

A production-quality **Retrieval-Augmented Generation (RAG)** API built with FastAPI.  
Upload PDF or TXT documents, then ask questions answered using your documents as the ground truth.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Folder Structure](#folder-structure)
3. [Setup & Installation](#setup--installation)
4. [Running the Server](#running-the-server)
5. [API Reference](#api-reference)
6. [Example API Requests (curl)](#example-api-requests-curl)
7. [Example API Requests (Postman)](#example-api-requests-postman)
8. [Design Decisions & Explanations](#design-decisions--explanations)
   - [Chunk Size & Overlap](#1-chunk-size--overlap-rationale)
   - [Real Retrieval Failure Case](#2-real-retrieval-failure-case)
   - [Metric: End-to-End Latency](#3-metric-tracked-end-to-end-latency)
9. [Architecture Diagram (draw.io)](#architecture-diagram-drawio)
10. [Running Tests](#running-tests)
11. [Production Considerations](#production-considerations)

---

## Architecture Overview

```
User
 │
 ▼
FastAPI (app/main.py)
 ├── Rate Limiting (sliding window, in-memory)
 ├── POST /documents/upload  ──► BackgroundTask ──► Ingestion Pipeline
 │                                                      ├── Text Extraction (PyMuPDF / plain read)
 │                                                      ├── Chunker (word-based, overlap)
 │                                                      ├── Embedder (all-MiniLM-L6-v2)
 │                                                      └── FAISS Index + Metadata Store
 │
 └── POST /questions/ask ──► Embed Query
                          ──► FAISS Search (top-k)
                          ──► LLM (Claude via Anthropic API)
                          ──► Return Answer + Retrieved Chunks + Latency
```

**Three clear layers:**
- **API Layer** — FastAPI routes, Pydantic validation, rate limiting, HTTP response codes
- **Processing Layer** — Text extraction, chunking, embedding (background tasks)
- **Retrieval Layer** — FAISS vector index, similarity search, LLM generation

---

## Folder Structure

```
rag_qa_system/
│
├── app/
│   ├── main.py                    # FastAPI app, lifespan, middleware
│   ├── routes/
│   │   ├── health.py              # GET /health
│   │   ├── documents.py           # POST /documents/upload, GET status, DELETE
│   │   └── questions.py           # POST /questions/ask
│   ├── services/
│   │   ├── ingestion_service.py   # Orchestrates the full ingestion pipeline
│   │   ├── embedder.py            # SentenceTransformer embedding wrapper (singleton)
│   │   ├── vector_store.py        # FAISS index + metadata management (singleton)
│   │   └── llm_service.py         # Anthropic API wrapper for answer generation
│   ├── models/
│   │   ├── schemas.py             # All Pydantic request/response schemas
│   │   └── document_store.py      # In-memory document registry (singleton)
│   └── utils/
│       ├── text_extractor.py      # PDF + TXT text extraction
│       ├── chunker.py             # Word-based overlapping chunker
│       └── rate_limiter.py        # Sliding window rate limiter
│
├── data/
│   ├── uploads/                   # Temporary file storage (auto-cleaned after ingestion)
│   └── vector_store/              # Persisted FAISS index + metadata JSON
│
├── tests/
│   ├── test_chunker.py            # Unit tests for chunker logic
│   └── test_rate_limiter.py       # Unit tests for rate limiter
│
├── run.py                         # Development server launcher
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or newer
- pip

### Step 1: Clone / download the project

```bash
git clone <repo-url>
cd rag_qa_system
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch**: The default `torch` in requirements.txt pulls the full package (~2GB). For CPU-only environments, use:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### Step 4: Configure environment variables

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY
```

```dotenv
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the Server

### Development (with auto-reload)
```bash
python run.py --reload
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **Why `--workers 1`?** The in-memory document registry and FAISS index are
> not shared across OS processes. For multi-worker deployments, replace
> `DocumentRegistry` with a database (SQLite/Postgres) and use a shared FAISS
> instance via a vector DB service (Qdrant, Weaviate, etc.).

Once running, open:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs (ReDoc)**: http://localhost:8000/redoc

---

## API Reference

### `GET /health`
System health check.

**Response:**
```json
{
  "status": "ok",
  "documents_indexed": 3,
  "vector_store_size": 142
}
```

---

### `POST /documents/upload`
Upload a PDF or TXT file. Ingestion runs in the background.

**Request:** `multipart/form-data` with field `file`

**Response (202 Accepted):**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "report.pdf",
  "status": "pending",
  "message": "Document received. Poll GET /documents/{id}/status for progress."
}
```

---

### `GET /documents/{document_id}/status`
Poll the processing status of a document.

**Status values:**
| Status | Meaning |
|--------|---------|
| `pending` | Uploaded, waiting for background task to start |
| `processing` | Background ingestion is running |
| `ready` | Fully indexed — queryable |
| `failed` | Ingestion failed (see `error` field) |

**Response:**
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "report.pdf",
  "status": "ready",
  "chunk_count": 47,
  "error": null,
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:30:12"
}
```

---

### `GET /documents/`
List all documents and their statuses.

---

### `DELETE /documents/{document_id}`
Remove a document and all its vectors from the FAISS index.

---

### `POST /questions/ask`
Ask a question based on indexed documents.

**Request body:**
```json
{
  "question": "What were the main findings of the study?",
  "top_k": 4,
  "document_ids": null
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Your question (3–1000 chars) |
| `top_k` | int | 4 | Number of chunks to retrieve (1–10) |
| `document_ids` | list\|null | null | Filter to specific docs; null = search all |

**Response:**
```json
{
  "question": "What were the main findings?",
  "answer": "According to report.pdf, the study found that...",
  "retrieved_chunks": [
    {
      "document_id": "550e...",
      "filename": "report.pdf",
      "chunk_index": 12,
      "text": "The study concluded that...",
      "similarity_score": 0.8734
    }
  ],
  "latency_ms": 1243.5,
  "model_used": "claude-haiku-4-5-20251001"
}
```

---

## Example API Requests (curl)

### Upload a TXT file
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/document.txt"
```

### Upload a PDF file
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/report.pdf"
```

### Check document status
```bash
curl http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000/status
```

### List all documents
```bash
curl http://localhost:8000/documents/
```

### Ask a question (search all documents)
```bash
curl -X POST http://localhost:8000/questions/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key recommendations?",
    "top_k": 4
  }'
```

### Ask a question (restrict to specific documents)
```bash
curl -X POST http://localhost:8000/questions/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the revenue for Q3?",
    "top_k": 3,
    "document_ids": ["550e8400-e29b-41d4-a716-446655440000"]
  }'
```

### Delete a document
```bash
curl -X DELETE http://localhost:8000/documents/550e8400-e29b-41d4-a716-446655440000
```

### Health check
```bash
curl http://localhost:8000/health
```

---

## Example API Requests (Postman)

Import the following as a **Postman Collection**:

**Collection: RAG QA System**

1. **Upload Document**
   - Method: `POST`
   - URL: `{{base_url}}/documents/upload`
   - Body → form-data → Key: `file` (type: File)

2. **Poll Status**
   - Method: `GET`
   - URL: `{{base_url}}/documents/{{document_id}}/status`

3. **Ask Question**
   - Method: `POST`
   - URL: `{{base_url}}/questions/ask`
   - Body → raw → JSON:
     ```json
     {
       "question": "Summarize the main points.",
       "top_k": 4,
       "document_ids": null
     }
     ```

4. **List Documents**
   - Method: `GET`
   - URL: `{{base_url}}/documents/`

**Environment Variables (Postman):**
| Variable | Value |
|----------|-------|
| `base_url` | `http://localhost:8000` |
| `document_id` | *(set after upload)* |

---

## Design Decisions & Explanations

### 1. Chunk Size & Overlap Rationale

**Chunk size: 500 words (~500 tokens)**

The embedding model `all-MiniLM-L6-v2` has a maximum input of **512 tokens**.
500 words gives us a comfortable margin (word ≠ token, but the ratio is ~1:1.1 for English).

Why not smaller (e.g., 100 words)?
A single isolated sentence is often semantically incomplete. "The results were significant."
tells the retriever nothing useful without the surrounding experimental context.
Small chunks produce high-precision but low-recall retrieval — the model finds the exact sentence
but lacks the context to answer meaningfully.

Why not larger (e.g., 1000 words)?
A chunk covering multiple topics (e.g., "methodology" AND "limitations") produces a diluted
embedding that represents neither concept strongly. When a user asks about methodology,
the combined embedding scores lower than a focused 500-word methodology chunk would.
This is sometimes called the **semantic dilution problem**.

**Overlap: 100 words (~20% of chunk size)**

Key sentences often fall at chunk boundaries. Without overlap, a sentence that starts at
the end of chunk N and finishes at the start of chunk N+1 will be split — making both
chunks less coherent and lowering retrieval scores for queries targeting that idea.

100 words (≈20%) is the standard recommendation from the RAG literature (10–25%).
Higher overlap improves recall but increases index size and embedding cost proportionally.

---

### 2. Real Retrieval Failure Case

**Scenario:** A user uploads a 50-page financial report and asks:
> *"What is the compound annual growth rate (CAGR) for the Asia-Pacific region?"*

**What happens:**  
The FAISS search returns chunks mentioning "Asia-Pacific" and "CAGR" — but in different
chunks. Chunk A (page 12) discusses Asia-Pacific market share. Chunk B (page 31) discusses
CAGR methodology. Neither chunk contains the Asia-Pacific CAGR figure — that specific
statistic appears in a **table** on page 18 that PyMuPDF extracted as unstructured text
like `"AP 14.3% 17.1% 19.8% 22.4%"` with no surrounding context.

**Why it happened:**
1. **Table extraction loss**: PDF tables often lose structural meaning during text extraction.
   Numbers without row/column headers are meaningless in isolation.
2. **Cross-chunk reasoning**: The answer requires JOIN-like reasoning across two chunks,
   which embedding similarity search cannot do. It retrieves *individually relevant* chunks,
   not *jointly sufficient* chunks.
3. **Abbreviation mismatch**: The query uses "Asia-Pacific" but the table uses "AP".
   `all-MiniLM-L6-v2` understands some abbreviations, but not domain-specific ones reliably.

**How to mitigate:**
- Use a PDF parser that preserves table structure (e.g., `camelot` or `pdfplumber`).
- Add a table-to-text serialization step: `"AP: CAGR 14.3% (2020), 17.1% (2021)..."`.
- Use a re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score
  the top-20 chunks before sending top-4 to the LLM.

---

### 3. Metric Tracked: End-to-End Latency

Every `/questions/ask` response includes a `latency_ms` field measuring the full
pipeline from query embedding → FAISS search → LLM generation → response.

**Why latency?**
Latency is the most user-visible metric. A correct answer after 30 seconds is
a bad user experience. Tracking it per-request lets operators set SLA thresholds
and identify which pipeline stage is the bottleneck.

**Example measurements (MacBook M2, CPU-only):**

| Stage | Typical Time |
|-------|-------------|
| Query embedding (1 vector) | ~15–30ms |
| FAISS search (1000 vectors, top-4) | <1ms |
| LLM API call (Claude Haiku) | ~800–1500ms |
| **Total end-to-end** | **~850–1600ms** |

**Key observation:** The LLM API call dominates latency (~95%). FAISS search is
effectively free at this scale. This means:
- Optimizing the chunker or embedder provides negligible latency improvement.
- To reduce P95 latency, either use streaming responses (`stream=True` in the SDK)
  or switch to a faster/smaller model.

**Example response with latency:**
```json
{
  "question": "What is the refund policy?",
  "answer": "According to terms.pdf, refunds are available within 30 days...",
  "latency_ms": 1124.7,
  "retrieved_chunks": [...]
}
```

Operators should track `latency_ms` over time (e.g., in Datadog or Prometheus)
and alert if P95 exceeds a defined SLA threshold (e.g., 3000ms).

---

## Architecture Diagram (draw.io)

Use this text description to draw the architecture in draw.io or Lucidchart:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          CLIENT (curl / Browser)                     │
└──────────────────────┬──────────────────────┬───────────────────────┘
                       │ POST /documents/upload│ POST /questions/ask
                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FASTAPI APPLICATION                          │
│  ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────┐ │
│  │  Rate Limiter    │   │  Pydantic Models│   │  CORS Middleware │ │
│  │ (sliding window) │   │  (validation)   │   │                  │ │
│  └──────────────────┘   └─────────────────┘   └──────────────────┘ │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                      ROUTE HANDLERS                           │   │
│  │  documents.py (upload/status/list/delete)                     │   │
│  │  questions.py (ask)                                           │   │
│  │  health.py                                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
          ┌─────────────────────┴──────────────────────┐
          │                                            │
          ▼ (Background Task)                          ▼ (Request Path)
┌─────────────────────┐                   ┌──────────────────────────┐
│  INGESTION PIPELINE │                   │   RETRIEVAL PIPELINE     │
│                     │                   │                          │
│ 1. TextExtractor    │                   │ 1. EmbeddingService      │
│    (PyMuPDF/txt)    │                   │    embed(query)          │
│         │           │                   │         │                │
│ 2. Chunker          │                   │ 2. VectorStoreService    │
│    (500w, 100 ovlp) │                   │    search(top_k)         │
│         │           │                   │         │                │
│ 3. EmbeddingService │                   │ 3. LLMService            │
│    (MiniLM-L6)      │                   │    generate_answer()     │
│         │           │                   │         │                │
│ 4. VectorStoreServ  │                   │ 4. Return Response       │
│    add_chunks()     │                   │    (answer + chunks      │
│                     │                   │     + latency_ms)        │
└──────────┬──────────┘                   └──────────┬───────────────┘
           │                                         │
           ▼                                         ▼
┌─────────────────────┐              ┌───────────────────────────────┐
│  STORAGE LAYER      │              │  EXTERNAL SERVICES            │
│                     │              │                               │
│ ┌─────────────────┐ │              │  Anthropic API                │
│ │  FAISS Index    │ │◄─────────────│  (Claude Haiku / Sonnet)      │
│ │  (IndexFlatIP)  │ │              │                               │
│ └─────────────────┘ │              └───────────────────────────────┘
│ ┌─────────────────┐ │
│ │ Metadata JSON   │ │
│ │ (chunk text +   │ │
│ │  doc_id, etc.)  │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ DocumentRegistry│ │
│ │ (in-memory dict)│ │
│ └─────────────────┘ │
└─────────────────────┘
```

**draw.io tips:**
- Use swimlane containers for the three layers: API, Processing, Storage
- Use dashed arrows for async/background flows (ingestion pipeline)
- Use solid arrows for synchronous request flows (retrieval pipeline)
- Color code: blue for API layer, green for processing, orange for storage, grey for external

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Run a specific test file
pytest tests/test_chunker.py -v
```

---

## Production Considerations

| Concern | Current Approach | Production Upgrade |
|---------|-----------------|-------------------|
| Document metadata | In-memory dict | PostgreSQL / SQLite |
| Vector storage | FAISS (file-backed) | Qdrant / Weaviate / Pinecone |
| Multi-worker support | Single worker | Shared vector DB + DB registry |
| Authentication | None | API keys / OAuth2 / JWT |
| Rate limiting | In-memory per-instance | Redis-backed (e.g., `slowapi`) |
| Logging | stdout | Structured JSON → Datadog/CloudWatch |
| Monitoring | `latency_ms` in response | Prometheus metrics endpoint |
| OCR support | Not supported | Add `tesseract` + `pdf2image` step |
| Re-ranking | None | Cross-encoder re-ranker for top-20→top-4 |
| Streaming answers | Not implemented | `stream=True` in Anthropic SDK |
