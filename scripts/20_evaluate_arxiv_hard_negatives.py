# scripts/20_evaluate_arxiv_hard_negatives.py

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--triples", default="data/arxiv/triples_bm25.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-prefix", default="")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    docs = read_jsonl(args.docs)
    triples = read_jsonl(args.triples)

    if args.limit > 0:
        triples = triples[: args.limit]

    docs_by_id = {
        doc["doc_id"]: doc
        for doc in docs
    }

    model = SentenceTransformer(args.model)

    total = 0
    acc1 = 0
    reciprocal_ranks = []
    ranks = []
    start = time.perf_counter()

    for triple in triples:
        query = args.query_prefix + triple["query"]
        positive_id = triple["positive_doc_id"]
        negative_ids = triple["negative_doc_ids"]

        candidate_ids = [positive_id] + negative_ids
        candidate_ids = [cid for cid in candidate_ids if cid in docs_by_id]

        if len(candidate_ids) < 2:
            continue

        candidate_texts = [
            args.doc_prefix + docs_by_id[cid]["text"]
            for cid in candidate_ids
        ]

        query_emb = model.encode(
            [query],
            convert_to_numpy=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        cand_emb = model.encode(
            candidate_texts,
            convert_to_numpy=True,
            batch_size=args.batch_size,
            show_progress_bar=False,
        )

        query_emb = normalize(query_emb)
        cand_emb = normalize(cand_emb)

        scores = (query_emb @ cand_emb.T)[0]

        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        ranked_ids = [cid for cid, _ in ranked]

        rank = ranked_ids.index(positive_id) + 1
        ranks.append(rank)

        if rank == 1:
            acc1 += 1

        reciprocal_ranks.append(1.0 / rank)
        total += 1

    metrics = {
        "model": args.model,
        "triples": args.triples,
        "query_prefix": args.query_prefix,
        "doc_prefix": args.doc_prefix,
        "num_examples": total,
        "accuracy@1": acc1 / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "avg_rank": sum(ranks) / total if total else 0.0,
        "eval_seconds": time.perf_counter() - start,
        "latency_ms_per_example": ((time.perf_counter() - start) / total) * 1000 if total else 0.0,
    }

    print(json.dumps(metrics, indent=2))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
