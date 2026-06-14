# scripts/19_inspect_arxiv_negatives.py

import argparse
import random

from domainrag.io import read_jsonl


def short(text: str, n: int = 700) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= n:
        return text
    return text[:n] + "..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--triples", default="data/arxiv/triples_bm25.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = read_jsonl(args.docs)
    triples = read_jsonl(args.triples)

    docs_by_id = {doc["doc_id"]: doc for doc in docs}

    if args.random:
        rng = random.Random(args.seed)
        triples = rng.sample(triples, k=min(args.limit, len(triples)))
    else:
        triples = triples[: args.limit]

    for i, triple in enumerate(triples, start=1):
        query = triple["query"]
        pos_id = triple["positive_doc_id"]
        neg_ids = triple["negative_doc_ids"]

        pos = docs_by_id[pos_id]

        print("\n" + "=" * 120)
        print(f"Example {i}")
        print("=" * 120)

        print("\nQUERY / TITLE:")
        print(query)

        print("\nPOSITIVE PAPER:")
        print(f"ID: {pos_id}")
        print(f"Title: {pos['title']}")
        print(short(pos["text"]))

        print("\nMINED NEGATIVES:")
        for rank, neg_id in enumerate(neg_ids, start=1):
            neg = docs_by_id.get(neg_id)
            if neg is None:
                continue

            print("\n" + "-" * 120)
            print(f"Negative {rank}")
            print(f"ID: {neg_id}")
            print(f"Title: {neg['title']}")
            print(short(neg["text"]))

        print("\nCHECK:")
        print("1. Are these actually wrong papers?")
        print("2. Are any negatives near-duplicates or same work?")
        print("3. Are they topic-similar enough to be useful?")


if __name__ == "__main__":
    main()