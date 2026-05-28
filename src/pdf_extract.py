# src/pdf_extract.py
"""PDF text extraction using PyMuPDF.
Handles ligatures and Unicode normalization to recover text from PDFs
with imperfect character maps."""

import unicodedata
from pathlib import Path
import fitz  # PyMuPDF's import name (legacy)


# Some PDFs encode ligatures as private-use Unicode characters instead of
# proper compatibility ligatures. NFKC normalization handles the standard
# ones (ﬁ, ﬂ, ﬀ etc); these fallbacks handle non-standard mappings we've
# observed in real-world resumes.
_LIGATURE_FALLBACK = {
    "Ɵ": "ti",   # seen mapped to the "ti" ligature glyph
    "ƞ": "tf",   # seen mapped to the "tf" ligature glyph (e.g. portfolio)
}


def extract_text(pdf_path: str | Path) -> str:
    """
    Read all pages of a PDF, return all text concatenated with double newlines
    between pages.

    Raises FileNotFoundError if the file doesn't exist.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        page_texts = []
        for page in doc:
            text = page.get_text("text") or ""
            text = _clean(text)
            if text:
                page_texts.append(text)
        return "\n\n".join(page_texts)
    finally:
        doc.close()


def _clean(text: str) -> str:
    """
    Normalize Unicode ligatures and clean up whitespace.

    1. NFKC normalization decomposes standard ligature characters
       (ﬁ → fi, ﬂ → fl, ﬀ → ff, etc.) into their base letters.
    2. Manual fallback for non-standard ligature encodings we've seen.
    3. Whitespace cleanup: collapse runs of spaces, trim lines, preserve
       paragraph breaks (so the chunker can use blank lines as anchors).
    """
    text = unicodedata.normalize("NFKC", text)
    for bad, good in _LIGATURE_FALLBACK.items():
        text = text.replace(bad, good)

    lines = []
    for line in text.splitlines():
        line = " ".join(line.split())
        lines.append(line)
    return "\n".join(lines).strip()
