# scripts/16_prepare_arxiv.py

import argparse
import hashlib
from pathlib import Path

from datasets import load_dataset

from domainrag.cleaning import clean_text
from domainrag.io import write_jsonl
from domainrag.splitting import assign_split


def stable_id(text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return h[:16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gfissore/arxiv-abstracts-2021")
    parser.add_argument("--output-docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--output-pairs", default="data/arxiv/pairs.jsonl")
    parser.add_argument("--output-benchmark", default="data/arxiv/benchmark.jsonl")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--min-abstract-words", type=int, default=40)
    parser.add_argument("--max-abstract-words", type=int, default=300)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train", streaming=True)

    documents = []
    pairs = []
    benchmark = []

    seen_titles = set()
    kept = 0

    for row in ds:
        # Dataset column names can vary. These are common for this dataset.
        title = clean_text(row.get("title", ""))
        abstract = clean_text(row.get("abstract", ""))

        if not title or not abstract:
            continue

        if title.lower() in seen_titles:
            continue

        abstract_words = abstract.split()

        if len(abstract_words) < args.min_abstract_words:
            continue

        if len(abstract_words) > args.max_abstract_words:
            abstract = " ".join(abstract_words[: args.max_abstract_words])

        doc_id = "arxiv_" + stable_id(title + abstract)
        split = assign_split(doc_id)

        doc = {
            "doc_id": doc_id,
            "title": title,
            "text": abstract,
            "source": "arxiv",
            "split": split,
            "num_words": len(abstract.split()),
        }

        documents.append(doc)

        pair = {
            "query_id": "q_" + doc_id,
            "query": title,
            "positive_doc_id": doc_id,
            "positive_text": abstract,
            "source": "arxiv",
            "split": split,
            "pair_type": "title_to_abstract",
        }

        if split == "train":
            pairs.append(pair)
        else:
            benchmark.append(
                {
                    "query_id": "q_" + doc_id,
                    "query": title,
                    "positive_doc_id": doc_id,
                    "positive_title": title,
                }
            )

        seen_titles.add(title.lower())
        kept += 1

        if kept >= args.limit:
            break

    Path(args.output_docs).parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(args.output_docs, documents)
    write_jsonl(args.output_pairs, pairs)
    write_jsonl(args.output_benchmark, benchmark)

    print("Done.")
    print(f"Documents: {len(documents)}")
    print(f"Train pairs: {len(pairs)}")
    print(f"Benchmark queries: {len(benchmark)}")
    print(f"Docs path: {args.output_docs}")
    print(f"Pairs path: {args.output_pairs}")
    print(f"Benchmark path: {args.output_benchmark}")


if __name__ == "__main__":
    main()