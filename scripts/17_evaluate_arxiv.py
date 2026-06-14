# scripts/17_evaluate_arxiv.py

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from domainrag.io import read_jsonl


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


class ArxivRetriever:
    def __init__(
        self,
        docs: list[dict],
        model_name: str,
        batch_size: int = 64,
        doc_prefix: str = "",
    ):
        self.docs = docs
        self.model = SentenceTransformer(model_name)

        texts = [doc_prefix + doc["text"] for doc in docs]

        start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        self.embedding_seconds = time.perf_counter() - start

        embeddings = normalize(embeddings).astype("float32")

        start = time.perf_counter()
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.index_seconds = time.perf_counter() - start
        self.embedding_dim = int(embeddings.shape[1])
        self.index_size_bytes = int(embeddings.nbytes)

    def search_batch(
        self,
        queries: list[str],
        k: int,
        batch_size: int = 64,
        query_prefix: str = "",
    ):
        queries = [query_prefix + query for query in queries]

        start = time.perf_counter()
        query_embeddings = self.model.encode(
            queries,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        query_embeddings = normalize(query_embeddings).astype("float32")

        scores, indices = self.index.search(query_embeddings, k)
        search_seconds = time.perf_counter() - start

        all_results = []

        for row_scores, row_indices in zip(scores, indices):
            results = []
            for score, idx in zip(row_scores, row_indices):
                doc = self.docs[int(idx)]
                results.append(
                    {
                        "score": float(score),
                        "doc_id": doc["doc_id"],
                        "title": doc["title"],
                    }
                )
            all_results.append(results)

        return all_results, search_seconds


def recall_at_k(retrieved_ids: list[str], positive_id: str, k: int) -> float:
    return float(positive_id in retrieved_ids[:k])


def reciprocal_rank_at_k(retrieved_ids: list[str], positive_id: str, k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == positive_id:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], positive_id: str, k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == positive_id:
            return 1.0 / np.log2(rank + 1)
    return 0.0


def average_precision_at_k(retrieved_ids: list[str], positive_id: str, k: int) -> float:
    # With one relevant document, AP@k is the reciprocal rank if found.
    return reciprocal_rank_at_k(retrieved_ids, positive_id, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--benchmark", default="data/arxiv/benchmark.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    docs = read_jsonl(args.docs)
    benchmark = read_jsonl(args.benchmark)

    if args.limit_queries > 0:
        benchmark = benchmark[: args.limit_queries]

    print(f"Docs: {len(docs)}")
    print(f"Benchmark queries: {len(benchmark)}")
    print(f"Model: {args.model}")

    retriever = ArxivRetriever(
        docs=docs,
        model_name=args.model,
        batch_size=args.batch_size,
        doc_prefix=args.doc_prefix,
    )

    queries = [item["query"] for item in benchmark]
    positives = [item["positive_doc_id"] for item in benchmark]

    all_results, search_seconds = retriever.search_batch(
        queries,
        k=args.k,
        batch_size=args.batch_size,
        query_prefix=args.query_prefix,
    )

    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    mrrs_10 = []
    ndcgs_10 = []
    maps_10 = []

    for result_row, positive_id in zip(all_results, positives):
        retrieved_ids = [r["doc_id"] for r in result_row]

        recalls_1.append(recall_at_k(retrieved_ids, positive_id, 1))
        recalls_5.append(recall_at_k(retrieved_ids, positive_id, 5))
        recalls_10.append(recall_at_k(retrieved_ids, positive_id, 10))
        mrrs_10.append(reciprocal_rank_at_k(retrieved_ids, positive_id, 10))
        ndcgs_10.append(ndcg_at_k(retrieved_ids, positive_id, 10))
        maps_10.append(average_precision_at_k(retrieved_ids, positive_id, 10))

    metrics = {
        "model": args.model,
        "docs": len(docs),
        "benchmark_queries": len(benchmark),
        "query_prefix": args.query_prefix,
        "doc_prefix": args.doc_prefix,
        "recall@1": sum(recalls_1) / len(recalls_1),
        "recall@5": sum(recalls_5) / len(recalls_5),
        "recall@10": sum(recalls_10) / len(recalls_10),
        "mrr@10": sum(mrrs_10) / len(mrrs_10),
        "ndcg@10": sum(ndcgs_10) / len(ndcgs_10),
        "map@10": sum(maps_10) / len(maps_10),
        "embedding_dim": retriever.embedding_dim,
        "index_size_bytes": retriever.index_size_bytes,
        "doc_embedding_seconds": retriever.embedding_seconds,
        "index_build_seconds": retriever.index_seconds,
        "query_search_seconds": search_seconds,
        "latency_ms_per_query": (search_seconds / len(benchmark)) * 1000 if benchmark else 0.0,
    }

    print(json.dumps(metrics, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
