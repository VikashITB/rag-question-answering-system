from contextlib import asynccontextmanager
import logging
import os
import webbrowser

from fastapi import FastAPI, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from app.routes import documents, questions, health
from app.services.vector_store import VectorStoreService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

browser_opened = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_opened

    logger.info("Starting RAG QA System...")

    if not browser_opened:
        webbrowser.open("http://127.0.0.1:8000/docs")
        browser_opened = True

    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)

    vector_service = VectorStoreService.get_instance()
    vector_service.load_if_exists()

    yield

    vector_service.save()


app = FastAPI(
    title="RAG-Based Question Answering API",
    description="Upload PDF/TXT documents and ask questions using RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(documents.router, prefix="/documents", tags=["Documents"])
app.include_router(questions.router, prefix="/questions", tags=["Questions"])


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)