# Experiments Log — Domain RAG / Retrieval Model Training Lab

This file records the experiments from the start of the project up to encoder baselines.

The project goal is to build an end-to-end dense retrieval training pipeline:

```text
raw documents
→ preprocessing / chunking
→ dense retrieval
→ evaluation
→ query-positive pair creation
→ MNRL fine-tuning
→ cached MNRL fine-tuning
→ hard-negative mining
→ serious ArXiv dataset training
→ encoder baselines
```

All commands assume they are run from the project root.

Important setup requirement:

```bash
touch scripts/__init__.py
export PYTHONPATH=.
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH="."
```

---

## Environment

Recommended install:

```bash
pip install sentence-transformers faiss-cpu numpy pandas pyarrow tqdm rank-bm25 datasets accelerate
pip install daft
```

If `daft` install fails, try:

```bash
pip install getdaft
```

---

# exp001_bow_retrieval_from_scratch

## Goal

Build the smallest possible retrieval system using bag-of-words vectors and cosine similarity.

This validates the core retrieval loop:

```text
documents
→ vector representation
→ query vector
→ similarity scores
→ top-k ranking
```

## Command

```bash
python -m scripts.01_bow_retrieval
```

## Expected outcome

The query:

```text
What does RAG retrieve before answering?
```

should rank the RAG-related document near the top.

Expected lesson:

- Retrieval is just scoring and ranking documents for a query.
- Bag-of-words works when query and document share words.
- Bag-of-words fails on paraphrases and synonyms.

## Result

| Query | Expected top doc | Actual top doc | Pass? |
|---|---|---|---|
| What does RAG retrieve before answering? | RAG document |  |  |

## Notes

Write observations here:

```text
-
```

---

# exp002_dense_retrieval_minilm

## Goal

Replace bag-of-words vectors with neural dense embeddings using Sentence-Transformers and FAISS.

This validates the semantic retrieval loop:

```text
text
→ embedding model
→ dense vector
→ FAISS index
→ nearest-neighbor search
```

## Command

```bash
python -m scripts.02_dense_retrieval
```

## Expected outcome

Queries using paraphrases should retrieve semantically relevant documents.

Example:

```text
Query:
What does retrieval augmented generation do before answering?

Expected:
RAG document
```

Expected lesson:

- Dense retrieval can match meaning, not only word overlap.
- FAISS can rank normalized embeddings using inner product.
- If embeddings are normalized, inner product is equivalent to cosine similarity.

## Result

| Query | Expected top doc | Actual top doc | Pass? |
|---|---|---|---|
| What does retrieval augmented generation do before answering? | RAG document |  |  |
| Which library is used for vector similarity search? | FAISS document |  |  |
| How do models learn to pull similar examples together? | Contrastive learning document |  |  |

## Notes

```text
-
```

---

# exp003_dense_retrieval_evaluation

## Goal

Add proper retrieval metrics.

Metrics used:

```text
Recall@k
MRR@k
```

This validates whether the retrieval system can be measured instead of judged by vibes.

## Command

```bash
python -m scripts.03_evaluate_retrieval
```

## Expected outcome

The script should print detailed per-query retrieval results and final metrics.

Expected metrics on toy data should be high, usually close to perfect, because the corpus is very small.

## Expected output shape

```text
Detailed results

Query: ...
Expected: ...
Retrieved: [...]
Recall@3: ...
RR@3: ...

Final metrics
{'recall@3': ..., 'mrr@3': ...}
```

## Result

| Model | Corpus | Recall@3 | MRR@3 |
|---|---|---:|---:|
| all-MiniLM-L6-v2 | toy hardcoded docs |  |  |

## Notes

```text
-
```

---

# exp004_data_processing_chunks

## Goal

Move from hardcoded documents to a real preprocessing pipeline.

Pipeline:

```text
data/raw/docs.jsonl
→ clean text
→ chunk documents
→ deterministic train/dev/test split
→ data/processed/chunks.jsonl
→ data/processed/chunks.parquet
```

## Commands

```bash
python -m scripts.04_process_data
```

Optional Daft preview:

```bash
python -m scripts.04_process_data --daft-preview
```

Inspect output:

```bash
head data/processed/chunks.jsonl
```

## Expected outcome

Files should be created:

```text
data/processed/chunks.jsonl
data/processed/chunks.parquet
data/processed/stats.jsonl
```

Expected lesson:

- Retrieval units are chunks, not full documents.
- Chunk size and overlap are modeling choices.
- Splitting should happen by document ID, not chunk ID, to avoid leakage.

## Result

| Setting | Num raw docs | Num chunks | Avg chunk words | Train/dev/test distribution |
|---|---:|---:|---:|---|
| default |  |  |  |  |

## Notes

```text
-
```

---

# exp005_chunk_search

## Goal

Run semantic search over processed chunks instead of hardcoded documents.

## Command

```bash
python -m scripts.05_search_chunks "What does RAG retrieve before answering?"
```

## Expected outcome

The retrieved chunk should come from the RAG document.

Expected lesson:

- Retrieval now works on processed data files.
- This is closer to real RAG than the hardcoded toy scripts.

## Result

| Query | Expected title | Actual top title | Pass? |
|---|---|---|---|
| What does RAG retrieve before answering? | Retrieval Augmented Generation |  |  |

## Notes

```text
-
```

---

# exp006_chunk_retrieval_evaluation

## Goal

Evaluate retrieval over processed chunks using a benchmark file.

Benchmark file:

```text
data/benchmark/queries.jsonl
```

## Command

```bash
python -m scripts.06_evaluate_chunks
```

## Expected outcome

Toy benchmark should have high Recall@3 and MRR@3.

Expected lesson:

- Once data comes from files, evaluation becomes repeatable.
- The benchmark should be versioned and inspected.
- Good retrieval requires good chunking.

## Result

| Model | Chunk settings | Recall@3 | MRR@3 |
|---|---|---:|---:|
| all-MiniLM-L6-v2 | default |  |  |

## Notes

```text
-
```

---

# exp007_chunk_size_ablation

## Goal

Understand how chunk size affects retrieval.

Smaller chunks may be more precise but lose context.
Larger chunks may contain more context but become vague.

## Commands

Small chunks:

```bash
python -m scripts.04_process_data --max-words 40 --overlap-words 10
python -m scripts.06_evaluate_chunks
```

Larger chunks:

```bash
python -m scripts.04_process_data --max-words 120 --overlap-words 30
python -m scripts.06_evaluate_chunks
```

Default / medium chunks:

```bash
python -m scripts.04_process_data --max-words 80 --overlap-words 20
python -m scripts.06_evaluate_chunks
```

## Expected outcome

On the tiny toy corpus, metrics may stay perfect. The real value is inspecting retrieved chunks.

Expected lesson:

- Chunk size changes what gets retrieved.
- Tiny toy metrics may hide important behavior.
- Inspect retrieved examples, not only metrics.

## Result

| Max words | Overlap words | Recall@3 | MRR@3 | Qualitative notes |
|---:|---:|---:|---:|---|
| 40 | 10 |  |  |  |
| 80 | 20 |  |  |  |
| 120 | 30 |  |  |  |

## Notes

```text
-
```

---

# exp008_make_title_pairs

## Goal

Create weakly supervised query-positive training pairs from processed chunks.

Format:

```json
{
  "query": "document title",
  "positive_text": "chunk text"
}
```

## Command

```bash
python -m scripts.07_make_pairs
```

Inspect:

```bash
head data/train/pairs.jsonl
```

## Expected outcome

File should be created:

```text
data/train/pairs.jsonl
```

Expected lesson:

- Fine-tuning needs query-positive pairs.
- Title-to-chunk pairs are weak supervision.
- Weak supervision is noisy but useful for a first pipeline.

## Result

| Input chunks | Output pairs | Pair type |
|---:|---:|---|
|  |  | title_to_chunk |

## Notes

```text
-
```

---

# exp009_mnrl_from_scratch

## Goal

Understand Multiple Negatives Ranking Loss mathematically.

The script creates a similarity matrix:

```text
scores[i, j] = similarity(query_i, document_j)
```

The diagonal is the correct match.

## Command

```bash
python -m scripts.08_mnrl_from_scratch
```

## Expected outcome

The script should print:

```text
Scores matrix
Labels
Loss
```

Expected lesson:

- MNRL is cross-entropy over a batch similarity matrix.
- For query i, document i is the positive.
- Other documents in the batch are in-batch negatives.
- Larger batch means more negatives.

## Result

| Ran successfully? | Notes |
|---|---|
|  |  |

## Notes

```text
-
```

---

# exp010_mnrl_title_pairs_toy

## Goal

Fine-tune MiniLM on title-to-chunk toy pairs using normal MNRL.

## Command

```bash
python -m scripts.09_train_mnrl \
  --pairs data/train/pairs.jsonl \
  --batch-size 2 \
  --epochs 1 \
  --output outputs/title-mnrl
```

Evaluate:

```bash
python -m scripts.06_evaluate_chunks \
  --model outputs/title-mnrl
```

## Expected outcome

Training should run and save a model.

Metrics may improve, stay same, or worsen because the toy dataset is very small.

Expected lesson:

- The training pipeline works.
- Toy fine-tuning is only a plumbing test.
- Do not over-interpret metrics on 5 examples.

## Result

| Model | Loss | Pairs | Batch | Epochs | Recall@3 | MRR@3 |
|---|---|---:|---:|---:|---:|---:|
| outputs/title-mnrl | MNRL |  | 2 | 1 |  |  |

## Notes

```text
-
```

---

# exp011_mnrl_manual_pairs_toy

## Goal

Fine-tune on manually written query-positive pairs for the toy corpus.

Manual pairs are more realistic than title-only pairs.

## Command

```bash
python -m scripts.09_train_mnrl \
  --pairs data/train/manual_pairs.jsonl \
  --batch-size 2 \
  --epochs 3 \
  --output outputs/manual-mnrl
```

Evaluate:

```bash
python -m scripts.06_evaluate_chunks \
  --model outputs/manual-mnrl
```

## Expected outcome

Training should run and save a model.

Metrics may improve on paraphrased toy queries, but results are still not meaningful at scale.

Expected lesson:

- Better queries can improve the training signal.
- Tiny datasets can overfit.
- Evaluation is necessary after every training run.

## Result

| Model | Loss | Pairs | Batch | Epochs | Recall@3 | MRR@3 |
|---|---|---:|---:|---:|---:|---:|
| outputs/manual-mnrl | MNRL |  | 2 | 3 |  |  |

## Notes

```text
-
```

---

# exp012_cached_mnrl_toy

## Goal

Add cached MNRL / GradCache-style training.

Normal MNRL:

```text
batch size = actual contrastive batch = memory batch
```

Cached MNRL:

```text
effective batch size = contrastive batch
mini-batch size = internal memory batch
```

## Commands

Normal MNRL:

```bash
python -m scripts.10_train_contrastive \
  --pairs data/train/manual_pairs.jsonl \
  --loss mnrl \
  --batch-size 2 \
  --epochs 1 \
  --output outputs/test-mnrl
```

Cached MNRL:

```bash
python -m scripts.10_train_contrastive \
  --pairs data/train/manual_pairs.jsonl \
  --loss cached_mnrl \
  --batch-size 4 \
  --mini-batch-size 2 \
  --epochs 1 \
  --output outputs/test-cached-mnrl
```

Evaluate:

```bash
python -m scripts.06_evaluate_chunks \
  --model outputs/test-mnrl

python -m scripts.06_evaluate_chunks \
  --model outputs/test-cached-mnrl
```

## Expected outcome

Both training runs should complete and save models.

Cached MNRL may be slower.

Metrics on tiny data are not meaningful.

Expected lesson:

- Cached MNRL enables a larger effective contrastive batch.
- It trades speed for memory.
- The real value appears on larger datasets and limited GPU memory.

## Result

| Model | Loss | Batch | Mini-batch | Recall@3 | MRR@3 | Notes |
|---|---|---:|---:|---:|---:|---|
| outputs/test-mnrl | MNRL | 2 | - |  |  |  |
| outputs/test-cached-mnrl | cached MNRL | 4 | 2 |  |  |  |

## Notes

```text
-
```

---

# exp013_loss_ablation_toy

## Goal

Run a single script comparing base model, normal MNRL, and cached MNRL on the toy corpus.

## Command

```bash
bash scripts/11_run_loss_ablation.sh
```

Expected internal commands should use:

```bash
python -m scripts.06_evaluate_chunks
python -m scripts.10_train_contrastive
```

## Expected outcome

The script should:

1. Evaluate base MiniLM
2. Train normal MNRL
3. Evaluate normal MNRL
4. Train cached MNRL
5. Evaluate cached MNRL

Expected lesson:

- Automating experiments prevents missing steps.
- Even toy experiments should be reproducible.

## Result

| Model | Loss | Recall@3 | MRR@3 |
|---|---|---:|---:|
| all-MiniLM-L6-v2 | none |  |  |
| ablation-mnrl | MNRL |  |  |
| ablation-cached-mnrl | cached MNRL |  |  |

## Notes

```text
-
```

---

# exp014_toy_hard_negative_mining

## Goal

Mine random and BM25 negatives on the toy processed chunks.

This validates the hard-negative mining pipeline before using it on ArXiv.

## Commands

Random negatives:

```bash
python -m scripts.12_mine_negatives \
  --strategy random \
  --output data/train/triples_random.jsonl
```

BM25 negatives:

```bash
python -m scripts.12_mine_negatives \
  --strategy bm25 \
  --output data/train/triples_bm25.jsonl
```

Inspect:

```bash
python -m scripts.13_inspect_negatives \
  --triples data/train/triples_bm25.jsonl \
  --limit 5
```

Stats:

```bash
python -m scripts.14_negative_stats \
  --triples data/train/triples_bm25.jsonl
```

Evaluate hard negatives:

```bash
python -m scripts.15_evaluate_hard_negatives \
  --triples data/train/triples_bm25.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2

python -m scripts.15_evaluate_hard_negatives \
  --triples data/train/triples_bm25.jsonl \
  --model outputs/test-mnrl

python -m scripts.15_evaluate_hard_negatives \
  --triples data/train/triples_bm25.jsonl \
  --model outputs/test-cached-mnrl
```

## Expected outcome

Triples should be created.

BM25 negatives may not be very meaningful on a tiny corpus.

Expected lesson:

- Random negatives are easy.
- BM25 negatives are more lexically similar.
- False negatives are possible.
- Mining requires manual inspection.

## Result

| Model | Negative set | Accuracy@1 | MRR |
|---|---|---:|---:|
| all-MiniLM-L6-v2 | BM25 toy |  |  |
| outputs/test-mnrl | BM25 toy |  |  |
| outputs/test-cached-mnrl | BM25 toy |  |  |

## Manual inspection

| Example | Negative quality | False negative? | Notes |
|---:|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |

## Notes

```text
-
```

---

# exp015_prepare_arxiv_1k

## Goal

Move from toy data to a serious dataset.

Dataset:

```text
gfissore/arxiv-abstracts-2021
```

Training pair:

```text
query = paper title
positive = paper abstract
```

## Command

```bash
python -m scripts.16_prepare_arxiv \
  --limit 1000
```

Inspect:

```bash
head data/arxiv/documents.jsonl
head data/arxiv/pairs.jsonl
head data/arxiv/benchmark.jsonl
```

## Expected outcome

Files should be created:

```text
data/arxiv/documents.jsonl
data/arxiv/pairs.jsonl
data/arxiv/benchmark.jsonl
```

Expected lesson:

- Title-to-abstract is useful weak supervision.
- We now have a real domain retrieval dataset.
- This is the first serious dataset checkpoint.

## Result

| Limit | Documents | Train pairs | Benchmark queries |
|---:|---:|---:|---:|
| 1000 |  |  |  |

## Notes

```text
-
```

---

# exp016_arxiv_base_eval_1k

## Goal

Evaluate base MiniLM on ArXiv title-to-abstract retrieval.

## Command

```bash
python -m scripts.17_evaluate_arxiv \
  --docs data/arxiv/documents.jsonl \
  --benchmark data/arxiv/benchmark.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit-queries 200
```

## Expected outcome

Metrics should be non-zero and likely reasonable because titles and abstracts are semantically related.

Expected lesson:

- Establish baseline before fine-tuning.
- Never claim improvement without a base model comparison.

## Result

| Model | Docs | Query limit | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 1000 | 200 |  |  |  |  |

## Notes

```text
-
```

---

# exp017_arxiv_mnrl_1k

## Goal

Fine-tune MiniLM on 1k ArXiv title-to-abstract pairs using normal MNRL.

## Command

```bash
python -m scripts.10_train_contrastive \
  --pairs data/arxiv/pairs.jsonl \
  --loss mnrl \
  --batch-size 16 \
  --epochs 1 \
  --output outputs/arxiv-minilm-mnrl-1k
```

Evaluate:

```bash
python -m scripts.17_evaluate_arxiv \
  --docs data/arxiv/documents.jsonl \
  --benchmark data/arxiv/benchmark.jsonl \
  --model outputs/arxiv-minilm-mnrl-1k \
  --limit-queries 200
```

## Expected outcome

Training should work.

Metrics may improve, stay flat, or worsen.

Expected lesson:

- 1k is a smoke run, not a final model.
- Fine-tuning on weak supervision is not guaranteed to improve retrieval.

## Result

| Model | Loss | Train pairs | Batch | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| outputs/arxiv-minilm-mnrl-1k | MNRL |  | 16 |  |  |  |  |

## Notes

```text
-
```

---

# exp018_arxiv_cached_mnrl_1k

## Goal

Fine-tune MiniLM on 1k ArXiv pairs using cached MNRL.

## Command

```bash
python -m scripts.10_train_contrastive \
  --pairs data/arxiv/pairs.jsonl \
  --loss cached_mnrl \
  --batch-size 64 \
  --mini-batch-size 16 \
  --epochs 1 \
  --output outputs/arxiv-minilm-cached-1k
```

Evaluate:

```bash
python -m scripts.17_evaluate_arxiv \
  --docs data/arxiv/documents.jsonl \
  --benchmark data/arxiv/benchmark.jsonl \
  --model outputs/arxiv-minilm-cached-1k \
  --limit-queries 200
```

## Expected outcome

Training should work.

Cached MNRL may be slower than normal MNRL.

Expected lesson:

- Cached MNRL is useful when effective batch size matters.
- It should be evaluated against normal MNRL, not used blindly.

## Result

| Model | Loss | Train pairs | Batch | Mini-batch | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| outputs/arxiv-minilm-cached-1k | cached MNRL |  | 64 | 16 |  |  |  |  |

## Notes

```text
-
```

---

# exp019_prepare_arxiv_10k

## Goal

Scale from ArXiv 1k to ArXiv 10k.

This is the first actually useful dataset size for the application project.

## Command

```bash
python -m scripts.16_prepare_arxiv \
  --limit 10000
```

## Expected outcome

The script should create a 10k ArXiv corpus and corresponding train/benchmark files.

Expected lesson:

- The same pipeline should scale from toy to 1k to 10k.
- Scaling data should not require changing training/evaluation code.

## Result

| Limit | Documents | Train pairs | Benchmark queries |
|---:|---:|---:|---:|
| 10000 |  |  |  |

## Notes

```text
-
```

---

# exp020_arxiv_base_eval_10k

## Goal

Evaluate base MiniLM on the 10k ArXiv corpus.

## Command

```bash
python -m scripts.17_evaluate_arxiv \
  --docs data/arxiv/documents.jsonl \
  --benchmark data/arxiv/benchmark.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit-queries 1000
```

## Expected outcome

Metrics may be lower than 1k because the retrieval task is harder with more candidate documents.

Expected lesson:

- Increasing corpus size makes retrieval harder.
- Always report corpus size with metrics.

## Result

| Model | Docs | Query limit | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | 10000 | 1000 |  |  |  |  |

## Notes

```text
-
```

---

# exp021_arxiv_cached_mnrl_10k

## Goal

Train a useful ArXiv domain-adapted MiniLM model using cached MNRL on 10k pairs.

## Command

```bash
python -m scripts.10_train_contrastive \
  --pairs data/arxiv/pairs.jsonl \
  --loss cached_mnrl \
  --batch-size 128 \
  --mini-batch-size 16 \
  --epochs 1 \
  --output outputs/arxiv-minilm-cached-10k
```

If memory fails:

```bash
python -m scripts.10_train_contrastive \
  --pairs data/arxiv/pairs.jsonl \
  --loss cached_mnrl \
  --batch-size 64 \
  --mini-batch-size 8 \
  --epochs 1 \
  --output outputs/arxiv-minilm-cached-10k
```

Evaluate:

```bash
python -m scripts.17_evaluate_arxiv \
  --docs data/arxiv/documents.jsonl \
  --benchmark data/arxiv/benchmark.jsonl \
  --model outputs/arxiv-minilm-cached-10k \
  --limit-queries 1000
```

## Expected outcome

This is the first application-worthy fine-tuning run.

Expected lesson:

- Cached MNRL enables larger effective batch sizes on limited GPUs.
- The fine-tuned model must be compared to base MiniLM and other baselines.
- Improvement is not guaranteed; result interpretation matters.

## Result

| Model | Loss | Train pairs | Batch | Mini-batch | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| outputs/arxiv-minilm-cached-10k | cached MNRL |  | 128 | 16 |  |  |  |  |

## Notes

```text
-
```

---

# exp022_arxiv_bm25_hard_negative_mining

## Goal

Mine hard negatives on ArXiv using BM25.

This tests whether the model can distinguish the correct abstract from similar but wrong abstracts.

## Commands

BM25 negatives:

```bash
python -m scripts.18_mine_arxiv_negatives \
  --strategy bm25 \
  --num-negatives 5 \
  --candidate-k 100 \
  --output data/arxiv/triples_bm25.jsonl
```

Random negatives:

```bash
python -m scripts.18_mine_arxiv_negatives \
  --strategy random \
  --num-negatives 5 \
  --output data/arxiv/triples_random.jsonl
```

Inspect BM25 negatives:

```bash
python -m scripts.19_inspect_arxiv_negatives \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 5
```

Random inspection:

```bash
python -m scripts.19_inspect_arxiv_negatives \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 5 \
  --random
```

## Expected outcome

BM25 negatives should be more topically similar than random negatives.

Expected lesson:

- Hard negatives are useful because they are confusing.
- Some mined negatives may be false negatives.
- Manual inspection is part of data quality work.

## Result

| Strategy | Num triples | Avg negatives/query | Qualitative quality |
|---|---:|---:|---|
| random |  |  |  |
| BM25 |  |  |  |

## Manual inspection

| Example | BM25 negatives quality | False negative? | Notes |
|---:|---|---|---|
| 1 |  |  |  |
| 2 |  |  |  |
| 3 |  |  |  |
| 4 |  |  |  |
| 5 |  |  |  |

## Notes

```text
-
```

---

# exp023_arxiv_hard_negative_evaluation

## Goal

Evaluate models on hard-negative discrimination.

Task:

```text
candidate set = positive abstract + BM25-mined negative abstracts
```

Metric:

```text
Accuracy@1
MRR
Average rank
```

## Commands

Base MiniLM:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --triples data/arxiv/triples_bm25.jsonl \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit 1000
```

MNRL 1k:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --triples data/arxiv/triples_bm25.jsonl \
  --model outputs/arxiv-minilm-mnrl-1k \
  --limit 1000
```

Cached MNRL 1k:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --triples data/arxiv/triples_bm25.jsonl \
  --model outputs/arxiv-minilm-cached-1k \
  --limit 1000
```

Cached MNRL 10k:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --triples data/arxiv/triples_bm25.jsonl \
  --model outputs/arxiv-minilm-cached-10k \
  --limit 1000
```

## Expected outcome

This evaluation is harder than full-corpus retrieval because all candidates are topically similar.

Expected lesson:

- A model can do well on full-corpus retrieval but fail on hard negatives.
- Hard-negative eval reveals fine-grained ranking ability.
- If fine-tuning does not help here, the training data or loss may need improvement.

## Result

| Model | Training data | Loss | Accuracy@1 | MRR | Avg rank |
|---|---:|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 0 | none |  |  |  |
| outputs/arxiv-minilm-mnrl-1k | 1k | MNRL |  |  |  |
| outputs/arxiv-minilm-cached-1k | 1k | cached MNRL |  |  |  |
| outputs/arxiv-minilm-cached-10k | 10k | cached MNRL |  |  |  |

## Notes

```text
-
```

---

# exp024_encoder_baselines

## Goal

Compare strong pretrained encoder baselines against your fine-tuned MiniLM.

Models:

```text
sentence-transformers/all-MiniLM-L6-v2
intfloat/e5-small-v2
BAAI/bge-small-en-v1.5
outputs/arxiv-minilm-cached-10k
```

## Important note about E5

E5 models are usually intended to use text prefixes:

```text
query: ...
passage: ...
```

The initial baseline may be run without prefixes for simplicity. If comparing seriously, add model-specific formatting in the evaluator.

## Commands

MiniLM full-corpus:

```bash
python -m scripts.17_evaluate_arxiv \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --limit-queries 1000
```

MiniLM hard negatives:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000
```

E5 full-corpus:

```bash
python -m scripts.17_evaluate_arxiv \
  --model intfloat/e5-small-v2 \
  --limit-queries 1000
```

E5 hard negatives:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model intfloat/e5-small-v2 \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000
```

BGE full-corpus:

```bash
python -m scripts.17_evaluate_arxiv \
  --model BAAI/bge-small-en-v1.5 \
  --limit-queries 1000
```

BGE hard negatives:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model BAAI/bge-small-en-v1.5 \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000
```

Fine-tuned MiniLM full-corpus:

```bash
python -m scripts.17_evaluate_arxiv \
  --model outputs/arxiv-minilm-cached-10k \
  --limit-queries 1000
```

Fine-tuned MiniLM hard negatives:

```bash
python -m scripts.20_evaluate_arxiv_hard_negatives \
  --model outputs/arxiv-minilm-cached-10k \
  --triples data/arxiv/triples_bm25.jsonl \
  --limit 1000
```

Optional shell runner:

```bash
bash scripts/21_run_encoder_baselines.sh
```

## Expected outcome

This experiment tells whether the fine-tuned MiniLM is competitive with stronger retrieval-specialized pretrained encoders.

Expected lesson:

- A fine-tuned small model should be compared to strong pretrained baselines.
- If BGE/E5 beat your model, that is not failure; it tells you what stronger starting points to fine-tune next.
- Baselines make the project credible.

## Result: full-corpus retrieval

| Model | Fine-tuned? | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|---|---|---:|---:|---:|---:|
| all-MiniLM-L6-v2 | no |  |  |  |  |
| e5-small-v2 | no |  |  |  |  |
| bge-small-en-v1.5 | no |  |  |  |  |
| arxiv-minilm-cached-10k | yes |  |  |  |  |

## Result: hard-negative evaluation

| Model | Fine-tuned? | Accuracy@1 | MRR | Avg rank |
|---|---|---:|---:|---:|
| all-MiniLM-L6-v2 | no |  |  |  |
| e5-small-v2 | no |  |  |  |
| bge-small-en-v1.5 | no |  |  |  |
| arxiv-minilm-cached-10k | yes |  |  |  |

## Notes

```text
-
```

---

# Summary Table

Fill this after all experiments.

| Stage | Main output | Completed? |
|---|---|---|
| Toy sparse retrieval | `scripts/01_bow_retrieval.py` |  |
| Toy dense retrieval | `scripts/02_dense_retrieval.py` |  |
| Toy evaluation | `scripts/03_evaluate_retrieval.py` |  |
| Data processing | `data/processed/chunks.jsonl` |  |
| Chunk retrieval | `scripts/05_search_chunks.py` |  |
| Chunk eval | `scripts/06_evaluate_chunks.py` |  |
| Pair creation | `data/train/pairs.jsonl` |  |
| MNRL training | `outputs/title-mnrl` / `outputs/manual-mnrl` |  |
| Cached MNRL | `outputs/test-cached-mnrl` |  |
| Toy hard negatives | `data/train/triples_bm25.jsonl` |  |
| ArXiv data | `data/arxiv/documents.jsonl` |  |
| ArXiv 10k training | `outputs/arxiv-minilm-cached-10k` |  |
| ArXiv hard negatives | `data/arxiv/triples_bm25.jsonl` |  |
| Encoder baselines | result tables |  |

---

# Remaining Future Work


## 1. Model-specific formatting

Add correct query/document prefixes for E5:

```text
query: ...
passage: ...
```

Possibly add BGE-specific instruction formatting if needed.

## 2. Fine-tune stronger encoders

Run the same cached MNRL pipeline on:

```text
BAAI/bge-small-en-v1.5
intfloat/e5-small-v2
```

## 3. Quantization experiment

Compare:

```text
float32 embeddings
float16 embeddings
int8 embeddings
binary embeddings
```

Metrics:

```text
Recall@10
MRR@10
index size
latency
```

## 4. Decoder-as-encoder experiment

Try a tiny decoder model with:

```text
last-token pooling
mean pooling
weighted mean pooling
```

Compare against MiniLM/BGE/E5.

## 5. Orthogonal regularization

Only after the above is stable, explore orthogonality penalties and quantization-friendlier embedding geometry.

