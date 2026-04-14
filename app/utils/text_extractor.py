"""
app/utils/text_extractor.py — Extract raw text from PDF and TXT files.

Design decisions:
- PyMuPDF (fitz) is used for PDFs over pdfminer/pypdf because it is fast,
  handles complex layouts better, and preserves reading order reliably.
- We strip excessive whitespace and normalize newlines before returning,
  so the chunker receives clean text instead of noisy PDF artifacts.
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """
    Extract text from a file.

    Args:
        file_path: Absolute or relative path to a .pdf or .txt file.

    Returns:
        Extracted plain text string.

    Raises:
        ValueError: If the file type is unsupported or extraction yields no text.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        return _extract_from_txt(path)
    elif suffix == ".pdf":
        return _extract_from_pdf(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{suffix}'. Only .pdf and .txt are supported."
        )


def _extract_from_txt(path: Path) -> str:
    """Read a plain-text file with UTF-8 encoding (fall back to latin-1)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decode failed for %s, retrying with latin-1", path.name)
        text = path.read_text(encoding="latin-1")

    cleaned = _clean_text(text)

    if not cleaned:
        raise ValueError(f"File '{path.name}' is empty or contains no readable text.")

    logger.info("Extracted %d characters from TXT: %s", len(cleaned), path.name)
    return cleaned


def _extract_from_pdf(path: Path) -> str:
    """
    Extract text page-by-page using PyMuPDF.
    Pages are joined with a double newline so chunk boundaries don't
    straddle page transitions unexpectedly.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF extraction. Install it with: pip install pymupdf"
        )

    doc = fitz.open(str(path))
    pages_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # "text" mode preserves reading order
        if text.strip():
            pages_text.append(text)

    doc.close()

    if not pages_text:
        raise ValueError(
            f"PDF '{path.name}' contains no extractable text. "
            "It may be a scanned image PDF (OCR not supported)."
        )

    full_text = "\n\n".join(pages_text)
    cleaned = _clean_text(full_text)

    logger.info(
        "Extracted %d characters from PDF (%d pages): %s",
        len(cleaned), len(pages_text), path.name
    )
    return cleaned


def _clean_text(text: str) -> str:
    """
    Normalize whitespace:
    1. Replace carriage returns with newlines.
    2. Collapse runs of 3+ newlines into two (preserve paragraph breaks).
    3. Strip leading/trailing whitespace per line.
    4. Strip overall.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces on each line (PDF artifacts)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()
