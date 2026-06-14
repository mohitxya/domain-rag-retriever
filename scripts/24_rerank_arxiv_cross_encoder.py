import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from domainrag.io import read_jsonl


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


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
    parser.add_argument("--retriever-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--limit-queries", type=int, default=200)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rerank-batch-size", type=int, default=32)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    docs = read_jsonl(args.docs)
    benchmark = read_jsonl(args.benchmark)
    if args.limit_queries > 0:
        benchmark = benchmark[: args.limit_queries]

    retriever = SentenceTransformer(args.retriever_model)
    reranker = CrossEncoder(args.reranker_model)

    doc_ids = [doc["doc_id"] for doc in docs]
    doc_texts = [args.doc_prefix + doc["text"] for doc in docs]
    docs_by_id = {doc["doc_id"]: doc for doc in docs}

    start = time.perf_counter()
    doc_embeddings = retriever.encode(
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

        query_embedding = retriever.encode(
            [args.query_prefix + query],
            batch_size=args.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = normalize(query_embedding).astype("float32")
        _, indices = index.search(query_embedding, args.candidate_k)

        candidate_ids = [doc_ids[int(idx)] for idx in indices[0]]
        pairs = [
            [query, docs_by_id[doc_id]["text"]]
            for doc_id in candidate_ids
        ]
        scores = reranker.predict(
            pairs,
            batch_size=args.rerank_batch_size,
            show_progress_bar=False,
        )

        ranked = [
            doc_id
            for doc_id, _ in sorted(
                zip(candidate_ids, scores),
                key=lambda item_score: float(item_score[1]),
                reverse=True,
            )
        ][: args.k]

        recalls_1.append(recall_at_k(ranked, positive_id, 1))
        recalls_5.append(recall_at_k(ranked, positive_id, 5))
        recalls_10.append(recall_at_k(ranked, positive_id, 10))
        mrrs_10.append(reciprocal_rank_at_k(ranked, positive_id, 10))
        ndcgs_10.append(ndcg_at_k(ranked, positive_id, 10))

    eval_seconds = time.perf_counter() - start
    metrics = {
        "retriever_model": args.retriever_model,
        "reranker_model": args.reranker_model,
        "docs": len(docs),
        "benchmark_queries": len(benchmark),
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
