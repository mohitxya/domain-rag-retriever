#!/usr/bin/env bash
set -e

echo "Evaluating MiniLM..."
python -m scripts.17_evaluate_arxiv \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit-queries 1000

python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000


echo "Evaluating E5-small..."
python -m scripts.17_evaluate_arxiv \
  --model intfloat/e5-small-v2 \
  --limit-queries 1000

python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model intfloat/e5-small-v2 \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000


echo "Evaluating BGE-small..."
python -m scripts.17_evaluate_arxiv \
  --model BAAI/bge-small-en-v1.5 \
  --limit-queries 1000

python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model BAAI/bge-small-en-v1.5 \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000


echo "Evaluating your fine-tuned cached model..."
python -m scripts.17_evaluate_arxiv \
  --model outputs/arxiv-minilm-cached-10k \
  --limit-queries 1000

python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model outputs/arxiv-minilm-cached-10k \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000