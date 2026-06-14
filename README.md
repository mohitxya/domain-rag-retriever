# Domain RAG Retriever
[RAG Retriever](/misc/rag_retriever.png)
An end-to-end retrieval training lab for domain-specific RAG systems.

The project builds a retrieval stack from first principles, then upgrades it into a resume-ready experimental system: preprocessing, dense retrieval, evaluation, contrastive fine-tuning, cached MNRL, hard-negative mining, encoder baselines, hybrid retrieval, cross-encoder reranking, and quantization tradeoff analysis.

## Start Here

- Read the experiment plan and running log in [experiments.md](experiments.md).
- Run the full suite with [scripts/22_run_all_experiments.py](scripts/22_run_all_experiments.py).
- Fine-tune embedding models with [scripts/10_train_contrastive.py](scripts/10_train_contrastive.py).
- Evaluate full-corpus ArXiv retrieval with [scripts/17_evaluate_arxiv.py](scripts/17_evaluate_arxiv.py).
- Evaluate hard-negative ranking with [scripts/20_evaluate_arxiv_hard_negatives.py](scripts/20_evaluate_arxiv_hard_negatives.py).
- Inspect the latest generated report in [output.md](output.md).

## Why This Project Exists

Most RAG demos stop at "embed documents and search." This repo is meant to show the harder engineering work behind a credible retrieval system:

- Build repeatable datasets and benchmarks.
- Evaluate with retrieval metrics instead of vibes.
- Fine-tune embedding models and compare them against strong baselines.
- Mine hard negatives and test fine-grained ranking behavior.
- Measure quality, latency, and storage tradeoffs.
- Produce clean experiment artifacts that can support resume/project claims.

## Current Capabilities

- Toy sparse and dense retrieval from scratch.
- Chunking and preprocessing pipeline for raw documents.
- ArXiv title-to-abstract retrieval dataset preparation.
- Full-corpus retrieval evaluation with:
  - Recall@1, Recall@5, Recall@10
  - MRR@10
  - nDCG@10
  - MAP@10
  - latency per query
  - index size estimate
- Hard-negative evaluation with Accuracy@1, MRR, average rank, and latency.
- Normal MNRL and cached MNRL fine-tuning.
- BM25 and random hard-negative mining.
- Prefix-aware E5 evaluation.
- MiniLM, E5, BGE, and fine-tuned model comparisons.
- Hybrid BM25 + dense retrieval.
- Cross-encoder reranking over dense candidates.
- Float32, float16, and int8 embedding quantization experiments.
- One-command experiment runner that writes:
  - `output.md`
  - `outputs/metrics.jsonl`
  - failure logs under `outputs/experiment_logs/`

## How The Pipeline Fits Together

```text
raw docs
  -> clean/chunk
  -> make query-positive pairs
  -> train dense retriever
  -> evaluate full-corpus retrieval
  -> mine hard negatives
  -> evaluate fine-grained ranking
  -> compare baselines, hybrid retrieval, reranking, quantization
```

The canonical experiment narrative lives in [experiments.md](experiments.md). The automated version of that narrative lives in [scripts/22_run_all_experiments.py](scripts/22_run_all_experiments.py).

## Repository Layout

```text
domainrag/
  bm25.py
  chunking.py
  cleaning.py
  io.py
  splitting.py

scripts/
  01_bow_retrieval.py
  02_dense_retrieval.py
  ...
  17_evaluate_arxiv.py
  20_evaluate_arxiv_hard_negatives.py
  22_run_all_experiments.py
  23_evaluate_arxiv_hybrid.py
  24_rerank_arxiv_cross_encoder.py
  25_evaluate_arxiv_quantization.py

data/
  raw/
  processed/
  benchmark/
  arxiv/

outputs/
  metrics.jsonl
  experiment_logs/
```

## Important Files

| File | Role |
|---|---|
| [experiments.md](experiments.md) | Human-readable experiment log and project roadmap |
| [scripts/22_run_all_experiments.py](scripts/22_run_all_experiments.py) | One-command experiment runner |
| [scripts/10_train_contrastive.py](scripts/10_train_contrastive.py) | Main training script for MNRL and cached MNRL |
| [scripts/09_train_mnrl.py](scripts/09_train_mnrl.py) | Earlier/simple MNRL training script |
| [scripts/17_evaluate_arxiv.py](scripts/17_evaluate_arxiv.py) | Full-corpus ArXiv retrieval evaluation |
| [scripts/20_evaluate_arxiv_hard_negatives.py](scripts/20_evaluate_arxiv_hard_negatives.py) | Hard-negative evaluation |
| [scripts/23_evaluate_arxiv_hybrid.py](scripts/23_evaluate_arxiv_hybrid.py) | Hybrid BM25 + dense retrieval |
| [scripts/24_rerank_arxiv_cross_encoder.py](scripts/24_rerank_arxiv_cross_encoder.py) | Cross-encoder reranking |
| [scripts/25_evaluate_arxiv_quantization.py](scripts/25_evaluate_arxiv_quantization.py) | Embedding quantization tradeoff experiment |
| [requirements.txt](requirements.txt) | Minimal pip dependency list |
| [environment.yml](environment.yml) | Full Conda environment |

## Setup

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
```

Using the Conda environment:

```bash
conda env create -f environment.yml
conda activate rag
export PYTHONPATH=.
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="."
```

## Run The Full Experiment Suite

```bash
python -m scripts.22_run_all_experiments --output output.md
```

Useful variants:

```bash
python -m scripts.22_run_all_experiments --only exp001 exp004
python -m scripts.22_run_all_experiments --continue-on-error
python -m scripts.22_run_all_experiments --timeout 3600
```

The runner writes a human-readable report to `output.md` and machine-readable metrics to `outputs/metrics.jsonl`.

## Training Script

The main training entry point is [scripts/10_train_contrastive.py](scripts/10_train_contrastive.py). It supports both normal Multiple Negatives Ranking Loss and cached MNRL:

```bash
python -m scripts.10_train_contrastive \
  --pairs data/arxiv/pairs.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --loss cached_mnrl \
  --batch-size 128 \
  --mini-batch-size 16 \
  --epochs 1 \
  --output outputs/arxiv-minilm-cached-10k
```

Useful arguments:

| Argument | Meaning |
|---|---|
| `--pairs` | JSONL training pairs with `query` and `positive_text` |
| `--model` | Base Sentence Transformers model or local checkpoint |
| `--loss` | `mnrl` or `cached_mnrl` |
| `--batch-size` | Effective contrastive batch size |
| `--mini-batch-size` | Internal cached-MNRL mini-batch size |
| `--epochs` | Number of training epochs |
| `--output` | Directory where the fine-tuned model is saved |

## Key Experiments

| ID | Experiment | Purpose |
|---|---|---|
| exp001-exp003 | Toy retrieval and evaluation | Validate retrieval from first principles |
| exp004-exp008 | Data processing and pair creation | Build repeatable data artifacts |
| exp010-exp012 | MNRL and cached MNRL | Fine-tune embedding models |
| exp014 | Toy hard negatives | Validate negative mining |
| exp015-exp021 | ArXiv 1k/10k training | Move to a realistic domain dataset |
| exp022-exp023 | ArXiv hard negatives | Test fine-grained ranking |
| exp024 | Encoder baselines | Compare MiniLM, E5, BGE, and fine-tuned models |
| exp025 | Hybrid retrieval | Combine BM25 and dense scoring |
| exp026 | Cross-encoder reranking | Rerank dense candidates with a stronger relevance model |
| exp027 | Quantization | Compare quality, latency, and storage |
| exp028 | BGE fine-tuning | Fine-tune a stronger retrieval-specialized encoder |

## Important Commands

Prepare ArXiv data:

```bash
python -m scripts.16_prepare_arxiv --limit 10000
```

Evaluate a dense encoder:

```bash
python -m scripts.17_evaluate_arxiv \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit-queries 1000
```

Evaluate E5 with the required text prefixes:

```bash
python -m scripts.17_evaluate_arxiv \
  --model intfloat/e5-small-v2 \
  --query-prefix "query: " \
  --doc-prefix "passage: " \
  --limit-queries 1000
```

Train cached MNRL on ArXiv:

```bash
python -m scripts.10_train_contrastive \
  --pairs data/arxiv/pairs.jsonl \
  --loss cached_mnrl \
  --batch-size 128 \
  --mini-batch-size 16 \
  --epochs 1 \
  --output outputs/arxiv-minilm-cached-10k
```

Run hybrid BM25 + dense retrieval:

```bash
python -m scripts.23_evaluate_arxiv_hybrid \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --dense-weight 0.5 \
  --candidate-k 100 \
  --limit-queries 1000
```

Run cross-encoder reranking:

```bash
python -m scripts.24_rerank_arxiv_cross_encoder \
  --retriever-model sentence-transformers/all-MiniLM-L6-v2 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --candidate-k 50 \
  --limit-queries 200
```

Run quantization tradeoffs:

```bash
python -m scripts.25_evaluate_arxiv_quantization \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --variants float32,float16,int8 \
  --limit-queries 1000
```

## What Makes It Resume-Ready

This project demonstrates:

- Retrieval fundamentals: sparse retrieval, dense retrieval, FAISS indexing, ranking.
- ML evaluation discipline: Recall, MRR, nDCG, MAP, hard-negative evaluation.
- Fine-tuning: MNRL, cached MNRL, larger effective batch sizes.
- Data engineering: preprocessing, chunking, weak supervision, generated benchmarks.
- Systems thinking: latency, index size, quantization, reproducible result logs.
- Realistic RAG architecture: dense retrieval, BM25 fallback, hybrid scoring, reranking.

Strong resume framing:

```text
Built an end-to-end domain retrieval training lab for RAG: prepared ArXiv retrieval data, fine-tuned sentence-transformer encoders with cached contrastive learning, mined hard negatives, benchmarked MiniLM/E5/BGE baselines, added hybrid BM25+dense retrieval and cross-encoder reranking, and measured quantization tradeoffs across retrieval quality, latency, and storage.
```

## Next Robustness Improvements

- Add a small CI smoke test that runs non-network toy experiments.
- Save every experiment command and parsed metric to a timestamped run directory.
- Add model/dataset cards for the best fine-tuned checkpoint.
- Add a lightweight FastAPI or CLI demo for query-time retrieval.
- Add significance checks or bootstrap confidence intervals for metric comparisons.
- Add model-specific formatting for any future encoder families.
- Add explicit false-negative filtering before training on mined negatives.

## Notes

Large experiments may download Hugging Face models or datasets and can take a long time on CPU. Use `--only` with the experiment runner while developing, then run the full suite when you are ready to produce final tables.
