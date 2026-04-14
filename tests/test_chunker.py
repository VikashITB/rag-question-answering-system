"""
tests/test_chunker.py — Unit tests for the text chunker.
Run with: pytest tests/ -v
"""

import pytest

from app.utils.chunker import chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_basic_chunking():
    """A long text should produce multiple chunks."""
    text = " ".join([f"word{i}" for i in range(1000)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1, "Expected multiple chunks for 1000-word text."


def test_chunk_size_respected():
    """Each chunk should have at most chunk_size words."""
    text = " ".join([f"word{i}" for i in range(500)])
    chunks = chunk_text(text, chunk_size=100, overlap=10)
    for chunk in chunks:
        word_count = len(chunk.text.split())
        assert word_count <= 100, f"Chunk has {word_count} words, expected ≤ 100."


def test_overlap_present():
    """
    With overlap=50 and chunk_size=100, the last 50 words of chunk[i]
    should appear at the beginning of chunk[i+1].
    """
    words = [f"w{i}" for i in range(300)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=50)

    if len(chunks) >= 2:
        chunk0_words = chunks[0].text.split()
        chunk1_words = chunks[1].text.split()
        overlap_words = chunk0_words[-50:]
        chunk1_start = chunk1_words[:50]
        assert overlap_words == chunk1_start, "Overlap words do not match."


def test_single_chunk_for_short_text():
    """A text shorter than chunk_size should produce exactly one chunk."""
    text = "This is a short document."
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 1


def test_chunk_indices_are_sequential():
    """Chunk indices should be 0, 1, 2, ..."""
    text = " ".join([f"word{i}" for i in range(600)])
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_default_parameters_produce_chunks():
    """Default parameters should work on a realistic-sized text."""
    text = " ".join([f"sentence{i}." for i in range(2000)])
    chunks = chunk_text(text)
    assert len(chunks) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_text_raises():
    with pytest.raises(ValueError, match="empty"):
        chunk_text("")


def test_whitespace_only_raises():
    with pytest.raises(ValueError):
        chunk_text("   \n\t  ")


def test_overlap_gte_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_text("some text", chunk_size=100, overlap=100)


def test_chunk_size_zero_raises():
    with pytest.raises(ValueError, match="chunk_size must be at least 1"):
        chunk_text("some text", chunk_size=0, overlap=0)


def test_last_chunk_covers_all_words():
    """
    The last chunk should cover the remaining words even if they're fewer
    than chunk_size. No words should be silently dropped.
    """
    words = [f"w{i}" for i in range(250)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    # Collect all unique words from all chunks
    all_chunk_words = set()
    for c in chunks:
        all_chunk_words.update(c.text.split())

    for word in words:
        assert word in all_chunk_words, f"Word '{word}' was dropped during chunking."
