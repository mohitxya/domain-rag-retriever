import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

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


def quantize_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(x)))
    scale = max_abs / 127.0 if max_abs else 1.0
    return np.clip(np.round(x / scale), -127, 127).astype("int8"), scale


def prepare_variant(x: np.ndarray, variant: str) -> tuple[np.ndarray, int, dict]:
    if variant == "float32":
        arr = x.astype("float32")
        return arr, int(arr.nbytes), {}

    if variant == "float16":
        arr = x.astype("float16")
        return arr.astype("float32"), int(arr.nbytes), {}

    if variant == "int8":
        quantized, scale = quantize_int8(x)
        return (quantized.astype("float32") * scale), int(quantized.nbytes), {"scale": scale}

    raise ValueError(f"Unknown variant: {variant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--benchmark", default="data/arxiv/benchmark.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--variants", default="float32,float16,int8")
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

    model = SentenceTransformer(args.model)
    doc_ids = [doc["doc_id"] for doc in docs]

    start = time.perf_counter()
    doc_embeddings = model.encode(
        [args.doc_prefix + doc["text"] for doc in docs],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        [args.query_prefix + item["query"] for item in benchmark],
        batch_size=args.batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    encode_seconds = time.perf_counter() - start

    doc_embeddings = normalize(doc_embeddings).astype("float32")
    query_embeddings = normalize(query_embeddings).astype("float32")

    all_metrics = []
    for variant in [item.strip() for item in args.variants.split(",") if item.strip()]:
        variant_docs, doc_storage_bytes, metadata = prepare_variant(doc_embeddings, variant)
        variant_queries, query_storage_bytes, _ = prepare_variant(query_embeddings, variant)

        variant_docs = normalize(variant_docs).astype("float32")
        variant_queries = normalize(variant_queries).astype("float32")

        recalls_1 = []
        recalls_5 = []
        recalls_10 = []
        mrrs_10 = []
        ndcgs_10 = []

        start = time.perf_counter()
        for query_embedding, item in zip(variant_queries, benchmark):
            scores = variant_docs @ query_embedding
            top_indices = np.argsort(scores)[::-1][: args.k]
            retrieved_ids = [doc_ids[int(idx)] for idx in top_indices]
            positive_id = item["positive_doc_id"]

            recalls_1.append(recall_at_k(retrieved_ids, positive_id, 1))
            recalls_5.append(recall_at_k(retrieved_ids, positive_id, 5))
            recalls_10.append(recall_at_k(retrieved_ids, positive_id, 10))
            mrrs_10.append(reciprocal_rank_at_k(retrieved_ids, positive_id, 10))
            ndcgs_10.append(ndcg_at_k(retrieved_ids, positive_id, 10))

        search_seconds = time.perf_counter() - start
        all_metrics.append(
            {
                "variant": variant,
                "model": args.model,
                "docs": len(docs),
                "benchmark_queries": len(benchmark),
                "embedding_dim": int(doc_embeddings.shape[1]),
                "doc_storage_bytes": doc_storage_bytes,
                "query_storage_bytes": query_storage_bytes,
                "recall@1": sum(recalls_1) / len(recalls_1),
                "recall@5": sum(recalls_5) / len(recalls_5),
                "recall@10": sum(recalls_10) / len(recalls_10),
                "mrr@10": sum(mrrs_10) / len(mrrs_10),
                "ndcg@10": sum(ndcgs_10) / len(ndcgs_10),
                "map@10": sum(mrrs_10) / len(mrrs_10),
                "encode_seconds": encode_seconds,
                "search_seconds": search_seconds,
                "latency_ms_per_query": (search_seconds / len(benchmark)) * 1000 if benchmark else 0.0,
                **metadata,
            }
        )

    print(json.dumps(all_metrics, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
