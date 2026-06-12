# scripts/07_make_pairs.py

import argparse
from pathlib import Path

from domainrag.io import read_jsonl, write_jsonl


def make_title_pairs(chunks: list[dict], split: str = "train") -> list[dict]:
    """
    Convert processed chunks into query-positive pairs.

    Query:
        document title

    Positive:
        chunk text

    This is weak supervision:
        title should describe the document/chunk.
    """
    pairs = []

    for chunk in chunks:
        if chunk.get("split") != split:
            continue

        title = chunk.get("title", "").strip()
        text = chunk.get("text", "").strip()

        if len(title.split()) < 2:
            continue

        if len(text.split()) < 10:
            continue

        pair = {
            "query_id": f"q_{chunk['chunk_id']}",
            "query": title,
            "positive_chunk_id": chunk["chunk_id"],
            "positive_text": text,
            "source": chunk.get("source", "unknown"),
            "pair_type": "title_to_chunk",
        }

        pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    parser.add_argument("--output", default="data/train/pairs.jsonl")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    chunks = read_jsonl(args.chunks)

    pairs = make_title_pairs(chunks, split=args.split)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, pairs)

    print(f"Wrote {len(pairs)} pairs to {args.output}")

    print("\nSample pairs:")
    for pair in pairs[:3]:
        print("=" * 80)
        print("Query:", pair["query"])
        print("Positive:", pair["positive_text"][:300])


if __name__ == "__main__":
    main()