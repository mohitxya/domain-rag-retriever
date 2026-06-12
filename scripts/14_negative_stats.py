# scripts/14_negative_stats.py

import argparse
from collections import Counter

from domainrag.io import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triples", default="data/train/triples_bm25.jsonl")
    parser.add_argument("--chunks", default="data/processed/chunks.jsonl")
    args = parser.parse_args()

    triples = read_jsonl(args.triples)
    chunks = read_jsonl(args.chunks)

    chunks_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    num_negatives = [
        len(triple["negative_chunk_ids"])
        for triple in triples
    ]

    negative_title_counts = Counter()

    for triple in triples:
        for neg_id in triple["negative_chunk_ids"]:
            title = chunks_by_id[neg_id]["title"]
            negative_title_counts[title] += 1

    print("Negative mining stats")
    print("=" * 80)
    print(f"Num triples: {len(triples)}")
    print(f"Average negatives: {sum(num_negatives) / len(num_negatives):.2f}")
    print(f"Min negatives: {min(num_negatives)}")
    print(f"Max negatives: {max(num_negatives)}")

    print("\nMost common negative titles:")
    for title, count in negative_title_counts.most_common(10):
        print(f"{title}: {count}")


if __name__ == "__main__":
    main()