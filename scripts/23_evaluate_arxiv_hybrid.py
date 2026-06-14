import argparse
import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from domainrag.io import read_jsonl


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def scale_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    low = min(values)
    high = max(values)
    if high <= low:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--benchmark", default="data/arxiv/benchmark.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit-queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    docs = read_jsonl(args.docs)
    benchmark = read_jsonl(args.benchmark)
    if args.limit_queries > 0:
        benchmark = benchmark[: args.limit_queries]

    doc_ids = [doc["doc_id"] for doc in docs]
    bm25_texts = [f"{doc.get('title', '')} {doc.get('text', '')}" for doc in docs]
    bm25 = BM25Okapi([tokenize(text) for text in bm25_texts])

    model = SentenceTransformer(args.model)
    doc_texts = [args.doc_prefix + doc["text"] for doc in docs]

    start = time.perf_counter()
    doc_embeddings = model.encode(
        doc_texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    doc_embeddings = normalize(doc_embeddings).astype("float32")
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    build_seconds = time.perf_counter() - start

    recalls_1 = []
    recalls_5 = []
    recalls_10 = []
    mrrs_10 = []
    ndcgs_10 = []

    start = time.perf_counter()
    for item in benchmark:
        query = item["query"]
        positive_id = item["positive_doc_id"]

        query_embedding = model.encode(
            [args.query_prefix + query],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = normalize(query_embedding).astype("float32")
        dense_scores, dense_indices = index.search(query_embedding, args.candidate_k)

        dense_by_id = {
            doc_ids[int(idx)]: float(score)
            for score, idx in zip(dense_scores[0], dense_indices[0])
        }

        bm25_scores = bm25.get_scores(tokenize(query))
        bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )[: args.candidate_k]
        bm25_by_id = {
            doc_ids[idx]: float(bm25_scores[idx])
            for idx in bm25_indices
        }

        dense_by_id = scale_scores(dense_by_id)
        bm25_by_id = scale_scores(bm25_by_id)

        candidate_ids = set(dense_by_id) | set(bm25_by_id)
        ranked = sorted(
            candidate_ids,
            key=lambda doc_id: (
                args.dense_weight * dense_by_id.get(doc_id, 0.0)
                + (1.0 - args.dense_weight) * bm25_by_id.get(doc_id, 0.0)
            ),
            reverse=True,
        )[: args.k]

        recalls_1.append(recall_at_k(ranked, positive_id, 1))
        recalls_5.append(recall_at_k(ranked, positive_id, 5))
        recalls_10.append(recall_at_k(ranked, positive_id, 10))
        mrrs_10.append(reciprocal_rank_at_k(ranked, positive_id, 10))
        ndcgs_10.append(ndcg_at_k(ranked, positive_id, 10))

    eval_seconds = time.perf_counter() - start
    metrics = {
        "model": args.model,
        "docs": len(docs),
        "benchmark_queries": len(benchmark),
        "dense_weight": args.dense_weight,
        "candidate_k": args.candidate_k,
        "recall@1": sum(recalls_1) / len(recalls_1),
        "recall@5": sum(recalls_5) / len(recalls_5),
        "recall@10": sum(recalls_10) / len(recalls_10),
        "mrr@10": sum(mrrs_10) / len(mrrs_10),
        "ndcg@10": sum(ndcgs_10) / len(ndcgs_10),
        "map@10": sum(mrrs_10) / len(mrrs_10),
        "build_seconds": build_seconds,
        "eval_seconds": eval_seconds,
        "latency_ms_per_query": (eval_seconds / len(benchmark)) * 1000 if benchmark else 0.0,
    }

    print(json.dumps(metrics, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
