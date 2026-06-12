# scripts/10_train_contrastive.py

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
    rows = read_jsonl(path)

    if not rows:
        raise ValueError(f"No training rows found in {path}")

    return Dataset.from_dict(
        {
            "anchor": [row["query"] for row in rows],
            "positive": [row["positive_text"] for row in rows],
        }
    )


def build_loss(
    model: SentenceTransformer,
    loss_name: str,
    mini_batch_size: int,
):
    if loss_name == "mnrl":
        return losses.MultipleNegativesRankingLoss(model)

    if loss_name == "cached_mnrl":
        return losses.CachedMultipleNegativesRankingLoss(
            model,
            mini_batch_size=mini_batch_size,
        )

    raise ValueError(f"Unknown loss: {loss_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pairs", default="data/train/pairs.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="outputs/contrastive-run")

    parser.add_argument(
        "--loss",
        choices=["mnrl", "cached_mnrl"],
        default="mnrl",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Effective contrastive batch size.",
    )

    parser.add_argument(
        "--mini-batch-size",
        type=int,
        default=8,
        help="Internal mini-batch size for cached MNRL.",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_dataset = load_pairs(args.pairs)

    model = SentenceTransformer(args.model)

    train_loss = build_loss(
        model=model,
        loss_name=args.loss,
        mini_batch_size=args.mini_batch_size,
    )

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
        seed=args.seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(output_dir))

    metadata = {
        "base_model": args.model,
        "loss": args.loss,
        "pairs": args.pairs,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }

    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()