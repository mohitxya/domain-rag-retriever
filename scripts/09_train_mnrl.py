# scripts/09_train_mnrl.py

import argparse
import json
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from domainrag.io import read_jsonl


def load_pairs(path: str) -> Dataset:
    """
    Convert JSONL pairs into a Hugging Face Dataset.

    Required columns for this loss:
        anchor: query text
        positive: relevant document/chunk text
    """
    rows = read_jsonl(path)

    if not rows:
        raise ValueError(f"No training rows found in {path}")

    dataset = Dataset.from_dict(
        {
            "anchor": [row["query"] for row in rows],
            "positive": [row["positive_text"] for row in rows],
        }
    )

    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/train/pairs.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="outputs/minilm-mnrl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    train_dataset = load_pairs(args.pairs)

    model = SentenceTransformer(args.model)

    train_loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="epoch",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save(args.output)

    metadata = {
        "base_model": args.model,
        "loss": "MultipleNegativesRankingLoss",
        "pairs": args.pairs,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_steps": args.max_steps,
    }

    with open(Path(args.output) / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved fine-tuned model to {args.output}")


if __name__ == "__main__":
    main()