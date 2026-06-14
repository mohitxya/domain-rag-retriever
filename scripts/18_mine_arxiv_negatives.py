# scripts/18_mine_arxiv_negatives.py

import argparse
import random
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from domainrag.io import read_jsonl, write_jsonl


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class DocBM25:
    def __init__(self, docs: list[dict]):
        self.docs = docs

        # Use title + abstract for mining.
        # This gives BM25 enough lexical signal.
        corpus_texts = [
            f"{doc.get('title', '')} {doc.get('text', '')}"
            for doc in docs
        ]

        self.tokenized_corpus = [tokenize(text) for text in corpus_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            {
                "score": float(scores[i]),
                **self.docs[i],
            }
            for i in ranked_indices
        ]


def mine_random(
    pairs: list[dict],
    docs: list[dict],
    num_negatives: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)

    doc_ids = [doc["doc_id"] for doc in docs]

    triples = []

    for pair in pairs:
        positive_id = pair["positive_doc_id"]

        candidates = [
            doc_id for doc_id in doc_ids
            if doc_id != positive_id
        ]

        negatives = rng.sample(
            candidates,
            k=min(num_negatives, len(candidates)),
        )

        triples.append(
            {
                "query_id": pair["query_id"],
                "query": pair["query"],
                "positive_doc_id": positive_id,
                "negative_doc_ids": negatives,
                "strategy": "random",
            }
        )

    return triples


def mine_bm25(
    pairs: list[dict],
    docs: list[dict],
    num_negatives: int,
    candidate_k: int,
) -> list[dict]:
    retriever = DocBM25(docs)

    docs_by_id = {
        doc["doc_id"]: doc
        for doc in docs
    }

    triples = []

    for pair in pairs:
        query = pair["query"]
        positive_id = pair["positive_doc_id"]

        if positive_id not in docs_by_id:
            continue

        positive_text = docs_by_id[positive_id]["text"]

        results = retriever.search(query, k=candidate_k)

        negative_ids = []
        seen = set()

        for result in results:
            candidate_id = result["doc_id"]

            # Never mark the known positive as negative.
            if candidate_id == positive_id:
                continue

            if candidate_id in seen:
                continue

            # Avoid exact duplicate abstract.
            if result["text"] == positive_text:
                continue

            negative_ids.append(candidate_id)
            seen.add(candidate_id)

            if len(negative_ids) >= num_negatives:
                break

        triples.append(
            {
                "query_id": pair["query_id"],
                "query": query,
                "positive_doc_id": positive_id,
                "negative_doc_ids": negative_ids,
                "strategy": "bm25",
            }
        )

    return triples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", default="data/arxiv/documents.jsonl")
    parser.add_argument("--pairs", default="data/arxiv/pairs.jsonl")
    parser.add_argument("--output", default="data/arxiv/triples_bm25.jsonl")
    parser.add_argument("--strategy", choices=["random", "bm25"], default="bm25")
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--limit-pairs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    docs = read_jsonl(args.docs)
    pairs = read_jsonl(args.pairs)

    if args.limit_pairs > 0:
        pairs = pairs[: args.limit_pairs]

    print(f"Docs: {len(docs)}")
    print(f"Pairs: {len(pairs)}")
    print(f"Strategy: {args.strategy}")

    if args.strategy == "random":
        triples = mine_random(
            pairs=pairs,
            docs=docs,
            num_negatives=args.num_negatives,
            seed=args.seed,
        )
    else:
        triples = mine_bm25(
            pairs=pairs,
            docs=docs,
            num_negatives=args.num_negatives,
            candidate_k=args.candidate_k,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, triples)

    avg_negs = sum(len(t["negative_doc_ids"]) for t in triples) / len(triples)

    print(f"Wrote: {args.output}")
    print(f"Triples: {len(triples)}")
    print(f"Average negatives per query: {avg_negs:.2f}")


if __name__ == "__main__":
    main()