# domainrag/chunking.py

from dataclasses import dataclass


@dataclass
class RawDocument:
    doc_id: str
    title: str
    text: str
    source: str = "unknown"


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    source: str
    num_chars: int
    num_words: int


def chunk_by_words(
    doc: RawDocument,
    max_words: int = 160,
    overlap_words: int = 40,
) -> list[Chunk]:
    """
    Split a document into overlapping word chunks.

    Example with max_words=5, overlap_words=2:

    words:
    [0,1,2,3,4,5,6,7,8,9]

    chunk 0:
    [0,1,2,3,4]

    chunk 1:
    [3,4,5,6,7]

    chunk 2:
    [6,7,8,9]

    Why overlap?
    Because important context may sit at a boundary.
    """
    if overlap_words >= max_words:
        raise ValueError("overlap_words must be smaller than max_words")

    words = doc.text.split()

    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunk_id = f"{doc.doc_id}_chunk_{chunk_idx:04d}"

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                title=doc.title,
                text=chunk_text,
                source=doc.source,
                num_chars=len(chunk_text),
                num_words=len(chunk_words),
            )
        )

        if end == len(words):
            break

        start = end - overlap_words
        chunk_idx += 1

    return chunks