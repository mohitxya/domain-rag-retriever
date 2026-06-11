# scripts/05_search_chunks.py

import argparse

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from domainrag.io import read_jsonl


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


class ChunkRetriever:
    def __init__(self, chunks: list[dict], model_name: str):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        embeddings = normalize(embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_embedding = normalize(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            chunk = self.chunks[int(idx)]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "text": chunk["text"],
                }
            )

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)

    retriever = ChunkRetriever(chunks, args.model)

    results = retriever.search(args.query, k=args.k)

    print(f"\nQuery: {args.query}\n")

    for rank, result in enumerate(results, start=1):
        print("=" * 80)
        print(f"Rank: {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Title: {result['title']}")
        print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()