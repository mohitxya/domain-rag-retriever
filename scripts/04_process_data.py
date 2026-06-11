# scripts/04_process_data.py

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from domainrag.cleaning import clean_text, is_good_text
from domainrag.chunking import RawDocument, chunk_by_words
from domainrag.io import read_jsonl, write_jsonl
from domainrag.splitting import assign_split


def process_with_pandas(
    input_path: str,
    output_jsonl: str,
    output_parquet: str,
    stats_path: str,
    min_chars: int,
    max_chars: int,
    max_words: int,
    overlap_words: int,
) -> None:
    rows = read_jsonl(input_path)

    cleaned_docs: list[RawDocument] = []

    for row in rows:
        doc_id = str(row["doc_id"])
        title = clean_text(row.get("title", ""))
        text = clean_text(row.get("text", ""))
        source = clean_text(row.get("source", "unknown"))

        if not is_good_text(text, min_chars=min_chars, max_chars=max_chars):
            continue

        cleaned_docs.append(
            RawDocument(
                doc_id=doc_id,
                title=title,
                text=text,
                source=source,
            )
        )

    all_chunk_rows = []

    for doc in cleaned_docs:
        split = assign_split(doc.doc_id)

        chunks = chunk_by_words(
            doc,
            max_words=max_words,
            overlap_words=overlap_words,
        )

        for chunk in chunks:
            row = asdict(chunk)
            row["split"] = split
            all_chunk_rows.append(row)

    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(output_jsonl, all_chunk_rows)

    df = pd.DataFrame(all_chunk_rows)
    df.to_parquet(output_parquet, index=False)

    stats = {
        "input_rows": len(rows),
        "kept_docs": len(cleaned_docs),
        "num_chunks": len(all_chunk_rows),
        "splits": df["split"].value_counts().to_dict() if len(df) else {},
        "avg_chunk_words": float(df["num_words"].mean()) if len(df) else 0.0,
    }

    write_jsonl(stats_path, [stats])

    print("Processing complete")
    print(stats)

def process_with_daft_preview(input_path: str) -> None:
    """
    Minimal Daft preview.

    This demonstrates that Daft can read the raw data as a dataframe.
    We keep main chunking in Python for now because chunking creates
    multiple output rows per input row, which is easier to reason about
    explicitly in v0.
    """
    try:
        import daft
    except ImportError:
        print("Daft is not installed. Install with: pip install daft")
        return

    df = daft.read_json(input_path)

    print("Daft dataframe preview:")
    print(df.limit(5).collect())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/docs.jsonl")
    parser.add_argument("--output-jsonl", default="data/processed/chunks.jsonl")
    parser.add_argument("--output-parquet", default="data/processed/chunks.parquet")
    parser.add_argument("--stats", default="data/processed/stats.jsonl")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--max-words", type=int, default=80)
    parser.add_argument("--overlap-words", type=int, default=20)
    parser.add_argument("--daft-preview", action="store_true")
    args = parser.parse_args()
    if args.daft_preview:
        process_with_daft_preview(args.input)
    process_with_pandas(
        input_path=args.input,
        output_jsonl=args.output_jsonl,
        output_parquet=args.output_parquet,
        stats_path=args.stats,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_words=args.max_words,
        overlap_words=args.overlap_words,
    )


if __name__ == "__main__":
    main()