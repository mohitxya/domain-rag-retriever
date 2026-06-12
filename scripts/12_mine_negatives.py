# scripts/12_mine_negatives.py

import argparse
import random
from pathlib import Path

from domainrag.bm25 import BM25Retriever
from domainrag.io import read_jsonl, write_jsonl


def mine_random_negatives(
    pairs: list[dict],
    chunks: list[dict],
    num_negatives: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    triples = []

    for pair in pairs:
        positive_id = pair["positive_chunk_id"]

        candidate_ids = [
            cid for cid in chunk_ids
            if cid != positive_id
        ]

        negatives = rng.sample(
            candidate_ids,
            k=min(num_negatives, len(candidate_ids)),
        )

        triples.append(
            {
                "query_id": pair["query_id"],
                "query": pair["query"],
                "positive_chunk_id": positive_id,
                "negative_chunk_ids": negatives,
                "strategy": "random",
            }
        )

    return triples


def mine_bm25_negatives(
    pairs: list[dict],
    chunks: list[dict],
    num_negatives: int,
    candidate_k: int,
) -> list[dict]:
    retriever = BM25Retriever(chunks)

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    triples = []

    for pair in pairs:
        query = pair["query"]
        positive_id = pair["positive_chunk_id"]

        positive_text = chunks_by_id[positive_id]["text"]

        results = retriever.search(query, k=candidate_k)

        negatives = []
        seen = set()

        for result in results:
            candidate_id = result["chunk_id"]

            # Never include the known positive as negative.
            if candidate_id == positive_id:
                continue

            # Avoid duplicate candidates.
            if candidate_id in seen:
                continue

            # Avoid exact duplicate text.
            if result["text"] == positive_text:
                continue

            negatives.append(candidate_id)
            seen.add(candidate_id)

            if len(negatives) >= num_negatives:
                break

        triples.append(
            {
                "query_id": pair["query_id"],
                "query": query,
                "positive_chunk_id": positive_id,
                "negative_chunk_ids": negatives,
                "strategy": "bm25",
            }
        )

    return triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/train/pairs.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--output", default="data/train/triples.jsonl")
    parser.add_argument(
        "--strategy",
        choices=["random", "bm25"],
        default="bm25",
    )
    parser.add_argument("--num-negatives", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    chunks = read_jsonl(args.chunks)

    if not pairs:
        raise ValueError(f"No pairs found in {args.pairs}")

    if not chunks:
        raise ValueError(f"No chunks found in {args.chunks}")

    if args.strategy == "random":
        triples = mine_random_negatives(
            pairs=pairs,
            chunks=chunks,
            num_negatives=args.num_negatives,
            seed=args.seed,
        )
    else:
        triples = mine_bm25_negatives(
            pairs=pairs,
            chunks=chunks,
            num_negatives=args.num_negatives,
            candidate_k=args.candidate_k,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, triples)

    avg_negatives = sum(len(t["negative_chunk_ids"]) for t in triples) / len(triples)

    print(f"Wrote {len(triples)} triples to {args.output}")
    print(f"Strategy: {args.strategy}")
    print(f"Average negatives per query: {avg_negatives:.2f}")

    print("\nSample triples:")
    for triple in triples[:3]:
        print("=" * 80)
        print("Query:", triple["query"])
        print("Positive:", triple["positive_chunk_id"])
        print("Negatives:", triple["negative_chunk_ids"])


if __name__ == "__main__":
    main()