import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from domainrag.io import read_jsonl


@dataclass
class Step:
    name: str
    command: list[str]
    kind: str = "command"
    model: str = ""
    loss: str = ""
    dataset: str = ""
    notes: str = ""


@dataclass
class Experiment:
    exp_id: str
    title: str
    steps: list[Step] = field(default_factory=list)


def py(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


EXPERIMENTS = [
    Experiment(
        "exp001",
        "bow_retrieval_from_scratch",
        [Step("Run bag-of-words retrieval", py("scripts.01_bow_retrieval"), "bow")],
    ),
    Experiment(
        "exp002",
        "dense_retrieval_minilm",
        [Step("Run dense retrieval", py("scripts.02_dense_retrieval"), "dense")],
    ),
    Experiment(
        "exp003",
        "dense_retrieval_evaluation",
        [
            Step(
                "Evaluate dense retrieval",
                py("scripts.03_evaluate_retrieval"),
                "toy_retrieval_eval",
                model="all-MiniLM-L6-v2",
                dataset="toy hardcoded docs",
            )
        ],
    ),
    Experiment(
        "exp004",
        "data_processing_chunks",
        [Step("Process default chunks", py("scripts.04_process_data"), "process_chunks")],
    ),
    Experiment(
        "exp005",
        "chunk_search",
        [
            Step(
                "Search processed chunks",
                py("scripts.05_search_chunks", "What does RAG retrieve before answering?"),
                "chunk_search",
            )
        ],
    ),
    Experiment(
        "exp006",
        "chunk_retrieval_evaluation",
        [
            Step(
                "Evaluate chunks",
                py("scripts.06_evaluate_chunks"),
                "chunk_eval",
                model="all-MiniLM-L6-v2",
                dataset="default chunks",
            )
        ],
    ),
    Experiment(
        "exp007",
        "chunk_size_ablation",
        [
            Step(
                "Process small chunks",
                py("scripts.04_process_data", "--max-words", "40", "--overlap-words", "10"),
                "process_chunks",
                dataset="40/10 chunks",
            ),
            Step(
                "Evaluate small chunks",
                py("scripts.06_evaluate_chunks"),
                "chunk_eval",
                model="all-MiniLM-L6-v2",
                dataset="40/10 chunks",
            ),
            Step(
                "Process large chunks",
                py("scripts.04_process_data", "--max-words", "120", "--overlap-words", "30"),
                "process_chunks",
                dataset="120/30 chunks",
            ),
            Step(
                "Evaluate large chunks",
                py("scripts.06_evaluate_chunks"),
                "chunk_eval",
                model="all-MiniLM-L6-v2",
                dataset="120/30 chunks",
            ),
            Step(
                "Restore default chunks",
                py("scripts.04_process_data", "--max-words", "80", "--overlap-words", "20"),
                "process_chunks",
                dataset="80/20 chunks",
            ),
            Step(
                "Evaluate default chunks",
                py("scripts.06_evaluate_chunks"),
                "chunk_eval",
                model="all-MiniLM-L6-v2",
                dataset="80/20 chunks",
            ),
        ],
    ),
    Experiment(
        "exp008",
        "make_title_pairs",
        [Step("Create title pairs", py("scripts.07_make_pairs"), "make_pairs")],
    ),
    Experiment(
        "exp009",
        "mnrl_from_scratch",
        [Step("Run MNRL from scratch", py("scripts.08_mnrl_from_scratch"), "command")],
    ),
    Experiment(
        "exp010",
        "mnrl_title_pairs_toy",
        [
            Step(
                "Train title MNRL",
                py(
                    "scripts.09_train_mnrl",
                    "--pairs",
                    "data/train/pairs.jsonl",
                    "--batch-size",
                    "2",
                    "--epochs",
                    "1",
                    "--output",
                    "outputs/title-mnrl",
                ),
                "train",
                model="outputs/title-mnrl",
                loss="MNRL",
            ),
            Step(
                "Evaluate title MNRL",
                py("scripts.06_evaluate_chunks", "--model", "outputs/title-mnrl"),
                "chunk_eval",
                model="outputs/title-mnrl",
                loss="MNRL",
                dataset="toy chunks",
            ),
        ],
    ),
    Experiment(
        "exp011",
        "mnrl_manual_pairs_toy",
        [
            Step(
                "Train manual MNRL",
                py(
                    "scripts.09_train_mnrl",
                    "--pairs",
                    "data/train/manual_pairs.jsonl",
                    "--batch-size",
                    "2",
                    "--epochs",
                    "3",
                    "--output",
                    "outputs/manual-mnrl",
                ),
                "train",
                model="outputs/manual-mnrl",
                loss="MNRL",
            ),
            Step(
                "Evaluate manual MNRL",
                py("scripts.06_evaluate_chunks", "--model", "outputs/manual-mnrl"),
                "chunk_eval",
                model="outputs/manual-mnrl",
                loss="MNRL",
                dataset="toy chunks",
            ),
        ],
    ),
    Experiment(
        "exp012",
        "cached_mnrl_toy",
        [
            Step(
                "Train normal MNRL",
                py(
                    "scripts.10_train_contrastive",
                    "--pairs",
                    "data/train/manual_pairs.jsonl",
                    "--loss",
                    "mnrl",
                    "--batch-size",
                    "2",
                    "--epochs",
                    "1",
                    "--output",
                    "outputs/test-mnrl",
                ),
                "train",
                model="outputs/test-mnrl",
                loss="MNRL",
            ),
            Step(
                "Train cached MNRL",
                py(
                    "scripts.10_train_contrastive",
                    "--pairs",
                    "data/train/manual_pairs.jsonl",
                    "--loss",
                    "cached_mnrl",
                    "--batch-size",
                    "4",
                    "--mini-batch-size",
                    "2",
                    "--epochs",
                    "1",
                    "--output",
                    "outputs/test-cached-mnrl",
                ),
                "train",
                model="outputs/test-cached-mnrl",
                loss="cached MNRL",
            ),
            Step(
                "Evaluate normal MNRL",
                py("scripts.06_evaluate_chunks", "--model", "outputs/test-mnrl"),
                "chunk_eval",
                model="outputs/test-mnrl",
                loss="MNRL",
                dataset="toy chunks",
            ),
            Step(
                "Evaluate cached MNRL",
                py("scripts.06_evaluate_chunks", "--model", "outputs/test-cached-mnrl"),
                "chunk_eval",
                model="outputs/test-cached-mnrl",
                loss="cached MNRL",
                dataset="toy chunks",
            ),
        ],
    ),
    Experiment(
        "exp013",
        "loss_ablation_toy",
        [Step("Run loss ablation shell script", ["bash", "scripts/11_run_loss_ablation.sh"], "command")],
    ),
    Experiment(
        "exp014",
        "toy_hard_negative_mining",
        [
            Step(
                "Mine random toy negatives",
                py("scripts.12_mine_negatives", "--strategy", "random", "--output", "data/train/triples_random.jsonl"),
                "mine_toy_negatives",
                dataset="random toy",
            ),
            Step(
                "Mine BM25 toy negatives",
                py("scripts.12_mine_negatives", "--strategy", "bm25", "--output", "data/train/triples_bm25.jsonl"),
                "mine_toy_negatives",
                dataset="BM25 toy",
            ),
            Step(
                "Inspect toy BM25 negatives",
                py("scripts.13_inspect_negatives", "--triples", "data/train/triples_bm25.jsonl", "--limit", "5"),
                "command",
            ),
            Step(
                "Toy negative stats",
                py("scripts.14_negative_stats", "--triples", "data/train/triples_bm25.jsonl"),
                "negative_stats",
                dataset="BM25 toy",
            ),
            Step(
                "Evaluate base on toy hard negatives",
                py("scripts.15_evaluate_hard_negatives", "--triples", "data/train/triples_bm25.jsonl", "--model", "sentence-transformers/all-MiniLM-L6-v2"),
                "toy_hard_eval",
                model="all-MiniLM-L6-v2",
                dataset="BM25 toy",
            ),
            Step(
                "Evaluate normal MNRL on toy hard negatives",
                py("scripts.15_evaluate_hard_negatives", "--triples", "data/train/triples_bm25.jsonl", "--model", "outputs/test-mnrl"),
                "toy_hard_eval",
                model="outputs/test-mnrl",
                loss="MNRL",
                dataset="BM25 toy",
            ),
            Step(
                "Evaluate cached MNRL on toy hard negatives",
                py("scripts.15_evaluate_hard_negatives", "--triples", "data/train/triples_bm25.jsonl", "--model", "outputs/test-cached-mnrl"),
                "toy_hard_eval",
                model="outputs/test-cached-mnrl",
                loss="cached MNRL",
                dataset="BM25 toy",
            ),
        ],
    ),
    Experiment(
        "exp015",
        "prepare_arxiv_1k",
        [Step("Prepare ArXiv 1k", py("scripts.16_prepare_arxiv", "--limit", "1000"), "prepare_arxiv", dataset="arxiv 1k")],
    ),
    Experiment(
        "exp016",
        "arxiv_base_eval_1k",
        [
            Step(
                "Evaluate base MiniLM on ArXiv 1k",
                py(
                    "scripts.17_evaluate_arxiv",
                    "--docs",
                    "data/arxiv/documents.jsonl",
                    "--benchmark",
                    "data/arxiv/benchmark.jsonl",
                    "--model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "--limit-queries",
                    "200",
                ),
                "arxiv_eval",
                model="all-MiniLM-L6-v2",
                dataset="arxiv 1k",
            )
        ],
    ),
    Experiment(
        "exp017",
        "arxiv_mnrl_1k",
        [
            Step(
                "Train ArXiv MNRL 1k",
                py(
                    "scripts.10_train_contrastive",
                    "--pairs",
                    "data/arxiv/pairs.jsonl",
                    "--loss",
                    "mnrl",
                    "--batch-size",
                    "16",
                    "--epochs",
                    "1",
                    "--output",
                    "outputs/arxiv-minilm-mnrl-1k",
                ),
                "train",
                model="outputs/arxiv-minilm-mnrl-1k",
                loss="MNRL",
            ),
            Step(
                "Evaluate ArXiv MNRL 1k",
                py(
                    "scripts.17_evaluate_arxiv",
                    "--docs",
                    "data/arxiv/documents.jsonl",
                    "--benchmark",
                    "data/arxiv/benchmark.jsonl",
                    "--model",
                    "outputs/arxiv-minilm-mnrl-1k",
                    "--limit-queries",
                    "200",
                ),
                "arxiv_eval",
                model="outputs/arxiv-minilm-mnrl-1k",
                loss="MNRL",
                dataset="arxiv 1k",
            ),
        ],
    ),
    Experiment(
        "exp018",
        "arxiv_cached_mnrl_1k",
        [
            Step(
                "Train ArXiv cached MNRL 1k",
                py(
                    "scripts.10_train_contrastive",
                    "--pairs",
                    "data/arxiv/pairs.jsonl",
                    "--loss",
                    "cached_mnrl",
                    "--batch-size",
                    "64",
                    "--mini-batch-size",
                    "16",
                    "--epochs",
                    "1",
                    "--output",
                    "outputs/arxiv-minilm-cached-1k",
                ),
                "train",
                model="outputs/arxiv-minilm-cached-1k",
                loss="cached MNRL",
            ),
            Step(
                "Evaluate ArXiv cached MNRL 1k",
                py(
                    "scripts.17_evaluate_arxiv",
                    "--docs",
                    "data/arxiv/documents.jsonl",
                    "--benchmark",
                    "data/arxiv/benchmark.jsonl",
                    "--model",
                    "outputs/arxiv-minilm-cached-1k",
                    "--limit-queries",
                    "200",
                ),
                "arxiv_eval",
                model="outputs/arxiv-minilm-cached-1k",
                loss="cached MNRL",
                dataset="arxiv 1k",
            ),
        ],
    ),
    Experiment(
        "exp019",
        "prepare_arxiv_10k",
        [Step("Prepare ArXiv 10k", py("scripts.16_prepare_arxiv", "--limit", "10000"), "prepare_arxiv", dataset="arxiv 10k")],
    ),
    Experiment(
        "exp020",
        "arxiv_base_eval_10k",
        [
            Step(
                "Evaluate base MiniLM on ArXiv 10k",
                py(
                    "scripts.17_evaluate_arxiv",
                    "--docs",
                    "data/arxiv/documents.jsonl",
                    "--benchmark",
                    "data/arxiv/benchmark.jsonl",
                    "--model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "--limit-queries",
                    "1000",
                ),
                "arxiv_eval",
                model="all-MiniLM-L6-v2",
                dataset="arxiv 10k",
            )
        ],
    ),
    Experiment(
        "exp021",
        "arxiv_cached_mnrl_10k",
        [
            Step(
                "Train ArXiv cached MNRL 10k",
                py(
                    "scripts.10_train_contrastive",
                    "--pairs",
                    "data/arxiv/pairs.jsonl",
                    "--loss",
                    "cached_mnrl",
                    "--batch-size",
                    "128",
                    "--mini-batch-size",
                    "16",
                    "--epochs",
                    "1",
                    "--output",
                    "outputs/arxiv-minilm-cached-10k",
                ),
                "train",
                model="outputs/arxiv-minilm-cached-10k",
                loss="cached MNRL",
            ),
            Step(
                "Evaluate ArXiv cached MNRL 10k",
                py(
                    "scripts.17_evaluate_arxiv",
                    "--docs",
                    "data/arxiv/documents.jsonl",
                    "--benchmark",
                    "data/arxiv/benchmark.jsonl",
                    "--model",
                    "outputs/arxiv-minilm-cached-10k",
                    "--limit-queries",
                    "1000",
                ),
                "arxiv_eval",
                model="outputs/arxiv-minilm-cached-10k",
                loss="cached MNRL",
                dataset="arxiv 10k",
            ),
        ],
    ),
    Experiment(
        "exp022",
        "arxiv_bm25_hard_negative_mining",
        [
            Step(
                "Mine ArXiv BM25 negatives",
                py(
                    "scripts.18_mine_arxiv_negatives",
                    "--strategy",
                    "bm25",
                    "--num-negatives",
                    "5",
                    "--candidate-k",
                    "100",
                    "--output",
                    "data/arxiv/triples_bm25.jsonl",
                ),
                "mine_arxiv_negatives",
                dataset="BM25 arxiv",
            ),
            Step(
                "Mine ArXiv random negatives",
                py(
                    "scripts.18_mine_arxiv_negatives",
                    "--strategy",
                    "random",
                    "--num-negatives",
                    "5",
                    "--output",
                    "data/arxiv/triples_random.jsonl",
                ),
                "mine_arxiv_negatives",
                dataset="random arxiv",
            ),
            Step(
                "Inspect ArXiv BM25 negatives",
                py("scripts.19_inspect_arxiv_negatives", "--triples", "data/arxiv/triples_bm25.jsonl", "--limit", "5"),
                "command",
            ),
            Step(
                "Inspect random ArXiv examples",
                py("scripts.19_inspect_arxiv_negatives", "--triples", "data/arxiv/triples_bm25.jsonl", "--limit", "5", "--random"),
                "command",
            ),
        ],
    ),
    Experiment(
        "exp023",
        "arxiv_hard_negative_evaluation",
        [
            Step(
                "Hard negatives: base MiniLM",
                py("scripts.20_evaluate_arxiv_hard_negatives", "--triples", "data/arxiv/triples_bm25.jsonl", "--model", "sentence-transformers/all-MiniLM-L6-v2", "--limit", "1000"),
                "arxiv_hard_eval",
                model="all-MiniLM-L6-v2",
                dataset="BM25 arxiv",
            ),
            Step(
                "Hard negatives: MNRL 1k",
                py("scripts.20_evaluate_arxiv_hard_negatives", "--triples", "data/arxiv/triples_bm25.jsonl", "--model", "outputs/arxiv-minilm-mnrl-1k", "--limit", "1000"),
                "arxiv_hard_eval",
                model="outputs/arxiv-minilm-mnrl-1k",
                loss="MNRL",
                dataset="BM25 arxiv",
            ),
            Step(
                "Hard negatives: cached MNRL 1k",
                py("scripts.20_evaluate_arxiv_hard_negatives", "--triples", "data/arxiv/triples_bm25.jsonl", "--model", "outputs/arxiv-minilm-cached-1k", "--limit", "1000"),
                "arxiv_hard_eval",
                model="outputs/arxiv-minilm-cached-1k",
                loss="cached MNRL",
                dataset="BM25 arxiv",
            ),
            Step(
                "Hard negatives: cached MNRL 10k",
                py("scripts.20_evaluate_arxiv_hard_negatives", "--triples", "data/arxiv/triples_bm25.jsonl", "--model", "outputs/arxiv-minilm-cached-10k", "--limit", "1000"),
                "arxiv_hard_eval",
                model="outputs/arxiv-minilm-cached-10k",
                loss="cached MNRL",
                dataset="BM25 arxiv",
            ),
        ],
    ),
    Experiment(
        "exp024",
        "encoder_baselines",
        [
            Step("MiniLM full corpus", py("scripts.17_evaluate_arxiv", "--model", "sentence-transformers/all-MiniLM-L6-v2", "--limit-queries", "1000"), "arxiv_eval", model="all-MiniLM-L6-v2", dataset="arxiv 10k"),
            Step("MiniLM hard negatives", py("scripts.20_evaluate_arxiv_hard_negatives", "--model", "sentence-transformers/all-MiniLM-L6-v2", "--triples", "data/arxiv/triples_bm25.jsonl", "--limit", "1000"), "arxiv_hard_eval", model="all-MiniLM-L6-v2", dataset="BM25 arxiv"),
            Step("E5 full corpus", py("scripts.17_evaluate_arxiv", "--model", "intfloat/e5-small-v2", "--limit-queries", "1000"), "arxiv_eval", model="e5-small-v2", dataset="arxiv 10k"),
            Step("E5 hard negatives", py("scripts.20_evaluate_arxiv_hard_negatives", "--model", "intfloat/e5-small-v2", "--triples", "data/arxiv/triples_bm25.jsonl", "--limit", "1000"), "arxiv_hard_eval", model="e5-small-v2", dataset="BM25 arxiv"),
            Step("BGE full corpus", py("scripts.17_evaluate_arxiv", "--model", "BAAI/bge-small-en-v1.5", "--limit-queries", "1000"), "arxiv_eval", model="bge-small-en-v1.5", dataset="arxiv 10k"),
            Step("BGE hard negatives", py("scripts.20_evaluate_arxiv_hard_negatives", "--model", "BAAI/bge-small-en-v1.5", "--triples", "data/arxiv/triples_bm25.jsonl", "--limit", "1000"), "arxiv_hard_eval", model="bge-small-en-v1.5", dataset="BM25 arxiv"),
            Step("Fine-tuned MiniLM full corpus", py("scripts.17_evaluate_arxiv", "--model", "outputs/arxiv-minilm-cached-10k", "--limit-queries", "1000"), "arxiv_eval", model="arxiv-minilm-cached-10k", loss="cached MNRL", dataset="arxiv 10k"),
            Step("Fine-tuned MiniLM hard negatives", py("scripts.20_evaluate_arxiv_hard_negatives", "--model", "outputs/arxiv-minilm-cached-10k", "--triples", "data/arxiv/triples_bm25.jsonl", "--limit", "1000"), "arxiv_hard_eval", model="arxiv-minilm-cached-10k", loss="cached MNRL", dataset="BM25 arxiv"),
        ],
    ),
]


def format_cmd(command: list[str]) -> str:
    return " ".join(command)


def read_count(path: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def extract_mapping(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        parsed = parse_mapping(line)
        if parsed:
            return parsed

    start = stdout.find("{")
    end = stdout.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = parse_mapping(stdout[start : end + 1])
        if parsed:
            return parsed
    return {}


def parse_mapping(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def fmt(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines) + "\n"


def summarize_step(exp: Experiment, step: Step, stdout: str, returncode: int) -> list[dict[str, Any]]:
    if returncode != 0:
        return []

    metrics = extract_mapping(stdout)
    rows: list[dict[str, Any]] = []

    if step.kind in {"toy_retrieval_eval", "chunk_eval"}:
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset or "retrieval",
                "model": step.model,
                "loss": step.loss,
                "recall@3": metrics.get("recall@3"),
                "mrr@3": metrics.get("mrr@3"),
            }
        )
    elif step.kind == "toy_hard_eval":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset,
                "model": step.model,
                "loss": step.loss,
                "accuracy@1": metrics.get("accuracy@1"),
                "mrr": metrics.get("mrr"),
                "num_examples": metrics.get("num_examples"),
            }
        )
    elif step.kind == "arxiv_eval":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset,
                "model": step.model,
                "loss": step.loss,
                "recall@1": metrics.get("recall@1"),
                "recall@5": metrics.get("recall@5"),
                "recall@10": metrics.get("recall@10"),
                "mrr@10": metrics.get("mrr@10"),
            }
        )
    elif step.kind == "arxiv_hard_eval":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset,
                "model": step.model,
                "loss": step.loss,
                "accuracy@1": metrics.get("accuracy@1"),
                "mrr": metrics.get("mrr"),
                "avg_rank": metrics.get("avg_rank"),
                "num_examples": metrics.get("num_examples"),
            }
        )
    elif step.kind == "process_chunks":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset or "default chunks",
                "input_rows": metrics.get("input_rows"),
                "kept_docs": metrics.get("kept_docs"),
                "num_chunks": metrics.get("num_chunks"),
                "avg_chunk_words": metrics.get("avg_chunk_words"),
                "splits": metrics.get("splits"),
            }
        )
    elif step.kind == "make_pairs":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": "toy title pairs",
                "pairs": read_count("data/train/pairs.jsonl"),
            }
        )
    elif step.kind == "prepare_arxiv":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset,
                "documents": read_count("data/arxiv/documents.jsonl"),
                "train_pairs": read_count("data/arxiv/pairs.jsonl"),
                "benchmark_queries": read_count("data/arxiv/benchmark.jsonl"),
            }
        )
    elif step.kind in {"mine_toy_negatives", "mine_arxiv_negatives"}:
        output = ""
        if "--output" in step.command:
            output = step.command[step.command.index("--output") + 1]
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset,
                "output": output,
                "triples": read_count(output) if output else "",
            }
        )
    elif step.kind == "negative_stats":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": step.dataset,
                "num_triples": re_search(stdout, r"Num triples: ([0-9]+)"),
                "avg_negatives": re_search(stdout, r"Average negatives: ([0-9.]+)"),
            }
        )
    elif step.kind == "bow":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": "bow retrieval",
                "query": re_search(stdout, r"Query: (.+)"),
                "top_doc": re_search(stdout, r"ID: (D[0-9]+)"),
            }
        )
    elif step.kind == "dense":
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": "dense retrieval",
                "top_docs": ", ".join(re.findall(r"Rank 1\nID: (D[0-9]+)", stdout)),
            }
        )
    elif step.kind == "chunk_search":
        title = re_search(stdout, r"Title: (.+)")
        rows.append(
            {
                "experiment": exp.exp_id,
                "task": "chunk search",
                "query": "What does RAG retrieve before answering?",
                "top_title": title,
            }
        )

    return rows


def re_search(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def run_step(step: Step, env: dict[str, str], timeout: int | None) -> tuple[int, str, str, float]:
    start = time.monotonic()
    proc = subprocess.run(
        step.command,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    elapsed = time.monotonic() - start
    return proc.returncode, proc.stdout, proc.stderr, elapsed


def write_output(
    output_path: Path,
    started_at: str,
    command_rows: list[list[Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    retrieval_rows = [
        [
            row.get("experiment"),
            row.get("task"),
            row.get("model"),
            row.get("loss"),
            row.get("recall@3"),
            row.get("mrr@3"),
        ]
        for row in summary_rows
        if "recall@3" in row
    ]
    arxiv_rows = [
        [
            row.get("experiment"),
            row.get("task"),
            row.get("model"),
            row.get("loss"),
            row.get("recall@1"),
            row.get("recall@5"),
            row.get("recall@10"),
            row.get("mrr@10"),
        ]
        for row in summary_rows
        if "recall@10" in row
    ]
    hard_rows = [
        [
            row.get("experiment"),
            row.get("task"),
            row.get("model"),
            row.get("loss"),
            row.get("accuracy@1"),
            row.get("mrr"),
            row.get("avg_rank"),
            row.get("num_examples"),
        ]
        for row in summary_rows
        if "accuracy@1" in row
    ]
    data_rows = [
        [row.get("experiment"), row.get("task"), row.get("documents"), row.get("train_pairs"), row.get("benchmark_queries")]
        for row in summary_rows
        if "documents" in row
    ]
    artifact_rows = [
        [row.get("experiment"), row.get("task"), row.get("output", ""), row.get("triples", row.get("pairs", ""))]
        for row in summary_rows
        if "triples" in row or "pairs" in row
    ]
    smoke_rows = [
        [row.get("experiment"), row.get("task"), row.get("query", ""), row.get("top_doc", row.get("top_docs", row.get("top_title", "")))]
        for row in summary_rows
        if row.get("task") in {"bow retrieval", "dense retrieval", "chunk search"}
    ]
    chunk_rows = [
        [
            row.get("experiment"),
            row.get("task"),
            row.get("input_rows"),
            row.get("kept_docs"),
            row.get("num_chunks"),
            row.get("avg_chunk_words"),
            json.dumps(row.get("splits", "")),
        ]
        for row in summary_rows
        if "num_chunks" in row
    ]

    body = [
        "# Experiment Results",
        "",
        f"Started: `{started_at}`",
        f"Finished: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Command Status",
        markdown_table(
            ["Experiment", "Step", "Status", "Seconds", "Command"],
            command_rows,
        ),
        "## Smoke Checks",
        markdown_table(["Experiment", "Task", "Query", "Observed top result"], smoke_rows),
        "## Chunk/Data Processing",
        markdown_table(["Experiment", "Task", "Input rows", "Kept docs", "Chunks", "Avg chunk words", "Splits"], chunk_rows),
        "## ArXiv Data",
        markdown_table(["Experiment", "Task", "Documents", "Train pairs", "Benchmark queries"], data_rows),
        "## Artifact Counts",
        markdown_table(["Experiment", "Task", "Output", "Count"], artifact_rows),
        "## Toy Retrieval Metrics",
        markdown_table(["Experiment", "Task", "Model", "Loss", "Recall@3", "MRR@3"], retrieval_rows),
        "## ArXiv Full-Corpus Metrics",
        markdown_table(["Experiment", "Task", "Model", "Loss", "Recall@1", "Recall@5", "Recall@10", "MRR@10"], arxiv_rows),
        "## Hard-Negative Metrics",
        markdown_table(["Experiment", "Task", "Model", "Loss", "Accuracy@1", "MRR", "Avg rank", "Examples"], hard_rows),
    ]

    output_path.write_text("\n".join(body), encoding="utf-8")


def selected_experiments(names: list[str]) -> list[Experiment]:
    if not names:
        return EXPERIMENTS
    wanted = set(names)
    selected = [
        exp
        for exp in EXPERIMENTS
        if exp.exp_id in wanted or exp.title in wanted or f"{exp.exp_id}_{exp.title}" in wanted
    ]
    missing = wanted - {exp.exp_id for exp in selected} - {exp.title for exp in selected} - {f"{exp.exp_id}_{exp.title}" for exp in selected}
    if missing:
        raise ValueError(f"Unknown experiment selector(s): {', '.join(sorted(missing))}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run experiments.md end to end and write output.md.")
    parser.add_argument("--output", default="output.md")
    parser.add_argument("--only", nargs="*", default=[], help="Optional experiment ids or titles to run.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--timeout", type=int, default=0, help="Per-step timeout in seconds. 0 means no timeout.")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = "."

    output_path = Path(args.output)
    started_at = datetime.now().isoformat(timespec="seconds")
    command_rows: list[list[Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for exp in selected_experiments(args.only):
        print(f"\n== {exp.exp_id} {exp.title} ==")
        for step in exp.steps:
            print(f"-> {step.name}")
            timeout = args.timeout if args.timeout > 0 else None
            try:
                returncode, stdout, stderr, elapsed = run_step(step, env=env, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                stdout = exc.stdout or ""
                stderr = exc.stderr or f"Timed out after {timeout} seconds"
                elapsed = float(timeout or 0)

            status = "ok" if returncode == 0 else f"failed ({returncode})"
            command_rows.append([exp.exp_id, step.name, status, f"{elapsed:.1f}", f"`{format_cmd(step.command)}`"])
            summary_rows.extend(summarize_step(exp, step, stdout, returncode))
            write_output(output_path, started_at, command_rows, summary_rows)

            if returncode != 0:
                log_dir = Path("outputs/experiment_logs")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / f"{exp.exp_id}_{slug(step.name)}.log"
                log_path.write_text(
                    f"$ {format_cmd(step.command)}\n\nSTDOUT\n{stdout}\n\nSTDERR\n{stderr}\n",
                    encoding="utf-8",
                )
                print(f"   failed; log written to {log_path}")
                if not args.continue_on_error:
                    return returncode

    write_output(output_path, started_at, command_rows, summary_rows)
    print(f"\nWrote {output_path}")
    return 0


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")


if __name__ == "__main__":
    raise SystemExit(main())
