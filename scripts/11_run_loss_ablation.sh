#!/usr/bin/env bash
set -e

echo "Evaluating base model..."
python -m scripts.06_evaluate_chunks \
  --model sentence-transformers/all-MiniLM-L6-v2

echo "Training normal MNRL..."
python -m scripts.10_train_contrastive \
  --pairs data/train/manual_pairs.jsonl \
  --loss mnrl \
  --batch-size 2 \
  --epochs 1 \
  --output outputs/ablation-mnrl

echo "Evaluating normal MNRL..."
python -m scripts.06_evaluate_chunks \
  --model outputs/ablation-mnrl

echo "Training cached MNRL..."
python -m scripts.10_train_contrastive \
  --pairs data/train/manual_pairs.jsonl \
  --loss cached_mnrl \
  --batch-size 4 \
  --mini-batch-size 2 \
  --epochs 1 \
  --output outputs/ablation-cached-mnrl

echo "Evaluating cached MNRL..."
python -m scripts.06_evaluate_chunks \
  --model outputs/ablation-cached-mnrl