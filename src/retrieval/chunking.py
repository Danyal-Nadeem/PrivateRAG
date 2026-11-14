"""Text chunking for markdown documents.

Chunk size: 500 chars, overlap: 100 chars.
Strategy: prefer splitting on blank lines (paragraph breaks);
fall back to fixed-size sliding window when no break is nearby.
"""
from __future__ import annotations

CHUNK_SIZE = 500
OVERLAP = 100


def _split_paragraphs(text: str) -> list[str]:
    """Split *text* into paragraphs on blank lines."""
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append("\n".join(current))
    return [p for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    doc_id: str,
    access_level: str,
    title: str,
) -> list[dict]:
    """Chunk *text* into overlapping windows and return metadata-rich dicts.

    Parameters
    ----------
    text:
        Raw markdown content.
    doc_id:
        Unique document identifier (e.g. ``"doc_001"``).
    access_level:
        One of ``public``, ``internal``, ``hr_restricted``, ``exec_restricted``.
    title:
        Human-readable document title.

    Returns
    -------
    list[dict]
        Each dict has keys: ``text``, ``doc_id``, ``access_level``,
        ``chunk_index``, ``title``.
    """
    paragraphs = _split_paragraphs(text)

    # Build chunks by accumulating paragraphs up to CHUNK_SIZE, then slide
    chunks: list[str] = []
    current_chars: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 1 > CHUNK_SIZE and current_chars:
            # Flush current chunk
            chunks.append("\n\n".join(current_chars))
            # Keep overlap: drop from the front until we're within OVERLAP budget
            while current_chars and current_len > OVERLAP:
                removed = current_chars.pop(0)
                current_len -= len(removed) + 2  # +2 for "\n\n"
            current_len = max(0, current_len)
        current_chars.append(para)
        current_len += para_len + 2

    if current_chars:
        chunks.append("\n\n".join(current_chars))

    # For very long paragraphs that exceed CHUNK_SIZE alone, do a fixed split
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= CHUNK_SIZE:
            final_chunks.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                final_chunks.append(chunk[start : start + CHUNK_SIZE])
                start += CHUNK_SIZE - OVERLAP

    return [
        {
            "text": c,
            "doc_id": doc_id,
            "access_level": access_level,
            "chunk_index": i,
            "title": title,
        }
        for i, c in enumerate(final_chunks)
        if c.strip()
    ]
