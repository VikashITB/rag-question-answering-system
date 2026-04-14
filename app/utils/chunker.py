"""
app/utils/chunker.py — Split text into overlapping chunks for embedding.

──────────────────────────────────────────────────────────────────────────────
CHUNK SIZE & OVERLAP — DESIGN RATIONALE (also in README)
──────────────────────────────────────────────────────────────────────────────

Chunk size: 500 tokens (~350–400 words)
  - Why 500? Embedding models (e.g., all-MiniLM-L6-v2) have a 512-token context
    limit. 500 leaves a small safety margin for tokenization variance.
  - Too small (e.g., 100 tokens): A single sentence is often semantically
    incomplete — "The study found significant results." tells us nothing without
    the surrounding context. Retrieval quality degrades.
  - Too large (e.g., 1000 tokens): Embedding a long passage dilutes the signal —
    a chunk about both "quantum entanglement" and "cheese fermentation" produces
    an averaged embedding that matches neither query well.
  - 500 tokens balances semantic completeness vs. embedding signal clarity.

Overlap: 100 tokens (~70–80 words)
  - Why overlap at all? Key sentences frequently fall at chunk boundaries.
    Without overlap, a question about content spanning two chunks retrieves
    neither chunk confidently.
  - 100 tokens = ~20% of chunk size — the standard heuristic (10–25%) from
    the NLP literature. Enough to bridge boundaries without doubling storage.

We measure "tokens" approximately as words (split on whitespace) because:
  - It's language-agnostic and cheap to compute.
  - True tokenizer counts vary by model but word count is within ~20% for English.
──────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

# Configurable defaults — override via environment variables in production
DEFAULT_CHUNK_SIZE = 500   # approximate tokens (words)
DEFAULT_OVERLAP = 100      # approximate tokens (words)


@dataclass
class TextChunk:
    """Represents one chunk of text with its positional metadata."""
    chunk_index: int
    text: str
    word_start: int   # index of first word in original word list
    word_end: int     # index of last word (exclusive)


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> List[TextChunk]:
    """
    Split `text` into overlapping word-based chunks.

    Args:
        text:       Full document text (pre-cleaned).
        chunk_size: Target number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of TextChunk objects, ordered by position.

    Raises:
        ValueError: If chunk_size < 1 or overlap >= chunk_size.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size.")
    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    words = text.split()  # split on any whitespace — simple, consistent
    total_words = len(words)

    if total_words == 0:
        raise ValueError("Text contains no words after splitting.")

    chunks: List[TextChunk] = []
    step = chunk_size - overlap  # how far we advance each iteration
    chunk_index = 0
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words)

        chunks.append(
            TextChunk(
                chunk_index=chunk_index,
                text=chunk_text_str,
                word_start=start,
                word_end=end,
            )
        )

        chunk_index += 1

        # If we've reached the end, stop
        if end == total_words:
            break

        start += step

    logger.info(
        "Chunked text into %d chunks (chunk_size=%d, overlap=%d, total_words=%d)",
        len(chunks), chunk_size, overlap, total_words,
    )
    return chunks
