# scripts/13_inspect_negatives.py

import argparse

from domainrag.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/train/triples_bm25.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    triples = read_jsonl(args.triples)
    chunks = read_jsonl(args.chunks)

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    for i, triple in enumerate(triples[: args.limit], start=1):
        query = triple["query"]
        positive_id = triple["positive_chunk_id"]
        negative_ids = triple["negative_chunk_ids"]

        positive = chunks_by_id[positive_id]

        print("\n" + "=" * 100)
        print(f"Example {i}")
        print("=" * 100)

        print("\nQUERY:")
        print(query)

        print("\nPOSITIVE:")
        print(f"[{positive_id}] {positive['title']}")
        print(positive["text"])

        print("\nNEGATIVES:")

        for rank, neg_id in enumerate(negative_ids, start=1):
            neg = chunks_by_id[neg_id]
            print("-" * 100)
            print(f"Negative {rank}: [{neg_id}] {neg['title']}")
            print(neg["text"])

        print("\nQUESTION FOR YOU:")
        print("Are these negatives truly wrong, or are any false negatives?")


if __name__ == "__main__":
    main()