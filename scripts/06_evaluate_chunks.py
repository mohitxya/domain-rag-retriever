# scripts/06_evaluate_chunks.py

import argparse
from collections import defaultdict

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
            show_progress_bar=False,
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
                    **chunk,
                }
            )

        return results


def recall_at_k(retrieved_titles: list[str], positive_title: str, k: int) -> float:
    return float(positive_title in retrieved_titles[:k])


def reciprocal_rank_at_k(
    retrieved_titles: list[str],
    positive_title: str,
    k: int,
) -> float:
    for rank, title in enumerate(retrieved_titles[:k], start=1):
        if title == positive_title:
            return 1.0 / rank

    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--benchmark", default="data/benchmark/queries.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)
    benchmark = read_jsonl(args.benchmark)

    retriever = ChunkRetriever(chunks, args.model)

    recall_scores = []
    rr_scores = []

    print("\nDetailed evaluation\n")

    for item in benchmark:
        query = item["query"]
        positive_title = item["positive_title"]

        results = retriever.search(query, k=args.k)
        retrieved_titles = [r["title"] for r in results]

        recall = recall_at_k(retrieved_titles, positive_title, args.k)
        rr = reciprocal_rank_at_k(retrieved_titles, positive_title, args.k)

        recall_scores.append(recall)
        rr_scores.append(rr)

        print("=" * 80)
        print(f"Query: {query}")
        print(f"Expected title: {positive_title}")
        print(f"Retrieved titles: {retrieved_titles}")
        print(f"Recall@{args.k}: {recall}")
        print(f"RR@{args.k}: {rr:.4f}")

    metrics = {
        f"recall@{args.k}": sum(recall_scores) / len(recall_scores),
        f"mrr@{args.k}": sum(rr_scores) / len(rr_scores),
    }

    print("\nFinal metrics")
    print(metrics)


if __name__ == "__main__":
    main()