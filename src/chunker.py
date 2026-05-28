# src/chunker.py
"""Text chunking for resume retrieval.

Splits a resume into chunks of roughly MIN_WORDS to MAX_WORDS words each,
with small overlap between adjacent chunks so boundary information is
preserved.
"""

import re
from typing import List


MIN_WORDS = 30
MAX_WORDS = 150
OVERLAP_WORDS = 20


def chunk_text(text: str) -> List[str]:
    """
    Split a resume's text into retrieval-friendly chunks.

    Strategy:
      1. Split on blank lines to get paragraphs.
      2. Merge paragraphs that are smaller than MIN_WORDS into the next one.
      3. Split paragraphs that are larger than MAX_WORDS at sentence boundaries.
      4. Add OVERLAP_WORDS of context from the end of each chunk to the start
         of the next.
    """
    # Step 1: paragraphs. Split on one-or-more blank lines (allowing whitespace).
    raw_paragraphs = re.split(r"\n\s*\n+", text.strip())
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Step 2: merge tiny paragraphs forward.
    merged: List[str] = []
    buffer = ""
    for para in paragraphs:
        if buffer:
            buffer = buffer + "\n" + para
        else:
            buffer = para
        if _word_count(buffer) >= MIN_WORDS:
            merged.append(buffer)
            buffer = ""
    if buffer:
        # Whatever's left over: append to the previous chunk if there is one,
        # else keep it standalone (a very short resume).
        if merged:
            merged[-1] = merged[-1] + "\n" + buffer
        else:
            merged.append(buffer)

    # Step 3: split overlong chunks at sentence boundaries.
    sized: List[str] = []
    for chunk in merged:
        if _word_count(chunk) <= MAX_WORDS:
            sized.append(chunk)
        else:
            sized.extend(_split_long(chunk))

    # Step 4: add overlap between adjacent chunks.
    if len(sized) <= 1 or OVERLAP_WORDS <= 0:
        return sized
    overlapped: List[str] = [sized[0]]
    for i in range(1, len(sized)):
        prev_words = sized[i - 1].split()
        tail = " ".join(prev_words[-OVERLAP_WORDS:])
        overlapped.append(tail + " " + sized[i])
    return overlapped


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long(text: str) -> List[str]:
    """Split a too-long block into <=MAX_WORDS pieces at sentence boundaries
    where possible, falling back to word boundaries if a single 'sentence'
    is itself too long."""
    # Naive sentence split: end-of-sentence punctuation followed by whitespace.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    pieces: List[str] = []
    buffer_words: List[str] = []
    for sent in sentences:
        sent_words = sent.split()
        if len(buffer_words) + len(sent_words) <= MAX_WORDS:
            buffer_words.extend(sent_words)
        else:
            if buffer_words:
                pieces.append(" ".join(buffer_words))
                buffer_words = []
            # If the sentence alone is longer than MAX_WORDS, hard-split it.
            if len(sent_words) > MAX_WORDS:
                for start in range(0, len(sent_words), MAX_WORDS):
                    pieces.append(" ".join(sent_words[start:start + MAX_WORDS]))
            else:
                buffer_words = sent_words
    if buffer_words:
        pieces.append(" ".join(buffer_words))
    return pieces
