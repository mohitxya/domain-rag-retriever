# domainrag/bm25.py

import re
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Tiny tokenizer for BM25.

    Example:
        "FAISS performs vector search."
        -> ["faiss", "performs", "vector", "search"]
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 10) -> list[dict]:
        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        results = []

        for i in ranked_indices:
            chunk = self.chunks[i]
            results.append(
                {
                    "score": float(scores[i]),
                    **chunk,
                }
            )

        return results