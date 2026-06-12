# scripts/15_evaluate_hard_negatives.py

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer

from domainrag.io import read_jsonl


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/train/triples_bm25.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    triples = read_jsonl(args.triples)
    chunks = read_jsonl(args.chunks)

    if args.limit > 0:
        triples = triples[: args.limit]

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    model = SentenceTransformer(args.model)

    total = 0
    correct_at_1 = 0
    reciprocal_ranks = []

    for triple in triples:
        query = triple["query"]
        positive_id = triple["positive_chunk_id"]
        candidate_ids = [positive_id] + triple["negative_chunk_ids"]

        # Skip examples without negatives.
        if len(candidate_ids) <= 1:
            continue

        candidate_texts = [
            chunks_by_id[cid]["text"]
            for cid in candidate_ids
        ]

        query_emb = model.encode([query], convert_to_numpy=True)
        cand_emb = model.encode(candidate_texts, convert_to_numpy=True)

        query_emb = normalize(query_emb)
        cand_emb = normalize(cand_emb)

        scores = (query_emb @ cand_emb.T)[0]

        ranked = sorted(
            zip(candidate_ids, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        ranked_ids = [cid for cid, score in ranked]

        rank = ranked_ids.index(positive_id) + 1
        rr = 1.0 / rank

        total += 1
        reciprocal_ranks.append(rr)

        if rank == 1:
            correct_at_1 += 1

    metrics = {
        "num_examples": total,
        "accuracy@1": correct_at_1 / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
    }

    print(metrics)


if __name__ == "__main__":
    main()