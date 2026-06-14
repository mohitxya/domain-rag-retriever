# Experiment Results

Started: `2026-06-14T12:19:38`
Finished: `2026-06-14T12:58:26`

## Command Status
| Experiment | Step | Status | Seconds | Command |
| --- | --- | --- | --- | --- |
| exp001 | Run bag-of-words retrieval | ok | 0.1 | `/usr/bin/python3 -m scripts.01_bow_retrieval` |
| exp002 | Run dense retrieval | ok | 34.8 | `/usr/bin/python3 -m scripts.02_dense_retrieval` |
| exp003 | Evaluate dense retrieval | ok | 15.3 | `/usr/bin/python3 -m scripts.03_evaluate_retrieval` |
| exp004 | Process default chunks | ok | 0.6 | `/usr/bin/python3 -m scripts.04_process_data` |
| exp005 | Search processed chunks | ok | 15.9 | `/usr/bin/python3 -m scripts.05_search_chunks What does RAG retrieve before answering?` |
| exp006 | Evaluate chunks | ok | 14.7 | `/usr/bin/python3 -m scripts.06_evaluate_chunks` |
| exp007 | Process small chunks | ok | 0.5 | `/usr/bin/python3 -m scripts.04_process_data --max-words 40 --overlap-words 10` |
| exp007 | Evaluate small chunks | ok | 15.5 | `/usr/bin/python3 -m scripts.06_evaluate_chunks` |
| exp007 | Process large chunks | ok | 0.5 | `/usr/bin/python3 -m scripts.04_process_data --max-words 120 --overlap-words 30` |
| exp007 | Evaluate large chunks | ok | 15.3 | `/usr/bin/python3 -m scripts.06_evaluate_chunks` |
| exp007 | Restore default chunks | ok | 0.7 | `/usr/bin/python3 -m scripts.04_process_data --max-words 80 --overlap-words 20` |
| exp007 | Evaluate default chunks | ok | 15.4 | `/usr/bin/python3 -m scripts.06_evaluate_chunks` |
| exp008 | Create title pairs | ok | 0.1 | `/usr/bin/python3 -m scripts.07_make_pairs` |
| exp009 | Run MNRL from scratch | ok | 2.1 | `/usr/bin/python3 -m scripts.08_mnrl_from_scratch` |
| exp010 | Train title MNRL | ok | 16.8 | `/usr/bin/python3 -m scripts.09_train_mnrl --pairs data/train/pairs.jsonl --batch-size 2 --epochs 1 --output outputs/title-mnrl` |
| exp010 | Evaluate title MNRL | ok | 13.2 | `/usr/bin/python3 -m scripts.06_evaluate_chunks --model outputs/title-mnrl` |
| exp011 | Train manual MNRL | ok | 21.7 | `/usr/bin/python3 -m scripts.09_train_mnrl --pairs data/train/manual_pairs.jsonl --batch-size 2 --epochs 3 --output outputs/manual-mnrl` |
| exp011 | Evaluate manual MNRL | ok | 13.2 | `/usr/bin/python3 -m scripts.06_evaluate_chunks --model outputs/manual-mnrl` |
| exp012 | Train normal MNRL | ok | 17.2 | `/usr/bin/python3 -m scripts.10_train_contrastive --pairs data/train/manual_pairs.jsonl --loss mnrl --batch-size 2 --epochs 1 --output outputs/test-mnrl` |
| exp012 | Train cached MNRL | ok | 16.1 | `/usr/bin/python3 -m scripts.10_train_contrastive --pairs data/train/manual_pairs.jsonl --loss cached_mnrl --batch-size 4 --mini-batch-size 2 --epochs 1 --output outputs/test-cached-mnrl` |
| exp012 | Evaluate normal MNRL | ok | 13.3 | `/usr/bin/python3 -m scripts.06_evaluate_chunks --model outputs/test-mnrl` |
| exp012 | Evaluate cached MNRL | ok | 13.5 | `/usr/bin/python3 -m scripts.06_evaluate_chunks --model outputs/test-cached-mnrl` |
| exp013 | Run loss ablation shell script | ok | 77.3 | `bash scripts/11_run_loss_ablation.sh` |
| exp014 | Mine random toy negatives | ok | 0.2 | `/usr/bin/python3 -m scripts.12_mine_negatives --strategy random --output data/train/triples_random.jsonl` |
| exp014 | Mine BM25 toy negatives | ok | 0.2 | `/usr/bin/python3 -m scripts.12_mine_negatives --strategy bm25 --output data/train/triples_bm25.jsonl` |
| exp014 | Inspect toy BM25 negatives | ok | 0.1 | `/usr/bin/python3 -m scripts.13_inspect_negatives --triples data/train/triples_bm25.jsonl --limit 5` |
| exp014 | Toy negative stats | ok | 0.1 | `/usr/bin/python3 -m scripts.14_negative_stats --triples data/train/triples_bm25.jsonl` |
| exp014 | Evaluate base on toy hard negatives | ok | 16.2 | `/usr/bin/python3 -m scripts.15_evaluate_hard_negatives --triples data/train/triples_bm25.jsonl --model sentence-transformers/all-MiniLM-L6-v2` |
| exp014 | Evaluate normal MNRL on toy hard negatives | ok | 13.0 | `/usr/bin/python3 -m scripts.15_evaluate_hard_negatives --triples data/train/triples_bm25.jsonl --model outputs/test-mnrl` |
| exp014 | Evaluate cached MNRL on toy hard negatives | ok | 13.0 | `/usr/bin/python3 -m scripts.15_evaluate_hard_negatives --triples data/train/triples_bm25.jsonl --model outputs/test-cached-mnrl` |
| exp015 | Prepare ArXiv 1k | ok | 5.9 | `/usr/bin/python3 -m scripts.16_prepare_arxiv --limit 1000` |
| exp016 | Evaluate base MiniLM on ArXiv 1k | ok | 18.1 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --docs data/arxiv/documents.jsonl --benchmark data/arxiv/benchmark.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --limit-queries 200` |
| exp017 | Train ArXiv MNRL 1k | ok | 24.0 | `/usr/bin/python3 -m scripts.10_train_contrastive --pairs data/arxiv/pairs.jsonl --loss mnrl --batch-size 16 --epochs 1 --output outputs/arxiv-minilm-mnrl-1k` |
| exp017 | Evaluate ArXiv MNRL 1k | ok | 15.4 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --docs data/arxiv/documents.jsonl --benchmark data/arxiv/benchmark.jsonl --model outputs/arxiv-minilm-mnrl-1k --limit-queries 200` |
| exp018 | Train ArXiv cached MNRL 1k | ok | 28.0 | `/usr/bin/python3 -m scripts.10_train_contrastive --pairs data/arxiv/pairs.jsonl --loss cached_mnrl --batch-size 64 --mini-batch-size 16 --epochs 1 --output outputs/arxiv-minilm-cached-1k` |
| exp018 | Evaluate ArXiv cached MNRL 1k | ok | 15.8 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --docs data/arxiv/documents.jsonl --benchmark data/arxiv/benchmark.jsonl --model outputs/arxiv-minilm-cached-1k --limit-queries 200` |
| exp019 | Prepare ArXiv 10k | ok | 11.9 | `/usr/bin/python3 -m scripts.16_prepare_arxiv --limit 10000` |
| exp020 | Evaluate base MiniLM on ArXiv 10k | ok | 40.6 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --docs data/arxiv/documents.jsonl --benchmark data/arxiv/benchmark.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --limit-queries 1000` |
| exp021 | Train ArXiv cached MNRL 10k | ok | 123.1 | `/usr/bin/python3 -m scripts.10_train_contrastive --pairs data/arxiv/pairs.jsonl --loss cached_mnrl --batch-size 128 --mini-batch-size 16 --epochs 1 --output outputs/arxiv-minilm-cached-10k` |
| exp021 | Evaluate ArXiv cached MNRL 10k | ok | 39.8 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --docs data/arxiv/documents.jsonl --benchmark data/arxiv/benchmark.jsonl --model outputs/arxiv-minilm-cached-10k --limit-queries 1000` |
| exp022 | Mine ArXiv BM25 negatives | ok | 280.9 | `/usr/bin/python3 -m scripts.18_mine_arxiv_negatives --strategy bm25 --num-negatives 5 --candidate-k 100 --output data/arxiv/triples_bm25.jsonl` |
| exp022 | Mine ArXiv random negatives | ok | 2.6 | `/usr/bin/python3 -m scripts.18_mine_arxiv_negatives --strategy random --num-negatives 5 --output data/arxiv/triples_random.jsonl` |
| exp022 | Inspect ArXiv BM25 negatives | ok | 0.2 | `/usr/bin/python3 -m scripts.19_inspect_arxiv_negatives --triples data/arxiv/triples_bm25.jsonl --limit 5` |
| exp022 | Inspect random ArXiv examples | ok | 0.2 | `/usr/bin/python3 -m scripts.19_inspect_arxiv_negatives --triples data/arxiv/triples_bm25.jsonl --limit 5 --random` |
| exp023 | Hard negatives: base MiniLM | ok | 39.4 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --triples data/arxiv/triples_bm25.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --limit 1000` |
| exp023 | Hard negatives: MNRL 1k | ok | 36.9 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --triples data/arxiv/triples_bm25.jsonl --model outputs/arxiv-minilm-mnrl-1k --limit 1000` |
| exp023 | Hard negatives: cached MNRL 1k | ok | 36.9 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --triples data/arxiv/triples_bm25.jsonl --model outputs/arxiv-minilm-cached-1k --limit 1000` |
| exp023 | Hard negatives: cached MNRL 10k | ok | 37.4 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --triples data/arxiv/triples_bm25.jsonl --model outputs/arxiv-minilm-cached-10k --limit 1000` |
| exp024 | MiniLM full corpus | ok | 41.0 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --model sentence-transformers/all-MiniLM-L6-v2 --limit-queries 1000` |
| exp024 | MiniLM hard negatives | ok | 39.6 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --model sentence-transformers/all-MiniLM-L6-v2 --triples data/arxiv/triples_bm25.jsonl --limit 1000` |
| exp024 | E5 full corpus | ok | 80.9 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --model intfloat/e5-small-v2 --query-prefix query:  --doc-prefix passage:  --limit-queries 1000` |
| exp024 | E5 hard negatives | ok | 68.7 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --model intfloat/e5-small-v2 --query-prefix query:  --doc-prefix passage:  --triples data/arxiv/triples_bm25.jsonl --limit 1000` |
| exp024 | BGE full corpus | ok | 81.3 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --model BAAI/bge-small-en-v1.5 --limit-queries 1000` |
| exp024 | BGE hard negatives | ok | 67.1 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --model BAAI/bge-small-en-v1.5 --triples data/arxiv/triples_bm25.jsonl --limit 1000` |
| exp024 | Fine-tuned MiniLM full corpus | ok | 39.2 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --model outputs/arxiv-minilm-cached-10k --limit-queries 1000` |
| exp024 | Fine-tuned MiniLM hard negatives | ok | 38.1 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --model outputs/arxiv-minilm-cached-10k --triples data/arxiv/triples_bm25.jsonl --limit 1000` |
| exp025 | Hybrid BM25 + dense retrieval | ok | 94.5 | `/usr/bin/python3 -m scripts.23_evaluate_arxiv_hybrid --model sentence-transformers/all-MiniLM-L6-v2 --dense-weight 0.5 --candidate-k 100 --limit-queries 1000` |
| exp026 | Cross-encoder rerank dense candidates | ok | 91.6 | `/usr/bin/python3 -m scripts.24_rerank_arxiv_cross_encoder --retriever-model sentence-transformers/all-MiniLM-L6-v2 --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --candidate-k 50 --limit-queries 200` |
| exp027 | Evaluate float32/float16/int8 embeddings | ok | 44.3 | `/usr/bin/python3 -m scripts.25_evaluate_arxiv_quantization --model sentence-transformers/all-MiniLM-L6-v2 --variants float32,float16,int8 --limit-queries 1000` |
| exp028 | Train BGE-small cached MNRL | ok | 365.0 | `/usr/bin/python3 -m scripts.10_train_contrastive --pairs data/arxiv/pairs.jsonl --model BAAI/bge-small-en-v1.5 --loss cached_mnrl --batch-size 64 --mini-batch-size 16 --epochs 1 --output outputs/arxiv-bge-small-cached-10k` |
| exp028 | Evaluate fine-tuned BGE-small | ok | 78.2 | `/usr/bin/python3 -m scripts.17_evaluate_arxiv --model outputs/arxiv-bge-small-cached-10k --limit-queries 1000` |
| exp028 | Evaluate fine-tuned BGE-small on hard negatives | ok | 65.1 | `/usr/bin/python3 -m scripts.20_evaluate_arxiv_hard_negatives --model outputs/arxiv-bge-small-cached-10k --triples data/arxiv/triples_bm25.jsonl --limit 1000` |

## Smoke Checks
| Experiment | Task | Query | Observed top result |
| --- | --- | --- | --- |
| exp001 | bow retrieval | What does RAG retrieve before answering? | D1 |
| exp002 | dense retrieval |  | D1, D2, D3 |
| exp005 | chunk search | What does RAG retrieve before answering? | Retrieval Augmented Generation |

## Chunk/Data Processing
| Experiment | Task | Input rows | Kept docs | Chunks | Avg chunk words | Splits |
| --- | --- | --- | --- | --- | --- | --- |
| exp004 | default chunks | 5 | 5 | 5 | 41.4000 | {"train": 5} |
| exp007 | 40/10 chunks | 5 | 5 | 7 | 32.4286 | {"train": 7} |
| exp007 | 120/30 chunks | 5 | 5 | 5 | 41.4000 | {"train": 5} |
| exp007 | 80/20 chunks | 5 | 5 | 5 | 41.4000 | {"train": 5} |

## ArXiv Data
| Experiment | Task | Documents | Train pairs | Benchmark queries |
| --- | --- | --- | --- | --- |
| exp015 | arxiv 1k | 1000 | 815 | 185 |
| exp019 | arxiv 10k | 10000 | 7979 | 2021 |

## Artifact Counts
| Experiment | Task | Output | Count |
| --- | --- | --- | --- |
| exp008 | toy title pairs |  | 5 |
| exp014 | random toy | data/train/triples_random.jsonl | 5 |
| exp014 | BM25 toy | data/train/triples_bm25.jsonl | 5 |
| exp022 | BM25 arxiv | data/arxiv/triples_bm25.jsonl | 7979 |
| exp022 | random arxiv | data/arxiv/triples_random.jsonl | 7979 |

## Toy Retrieval Metrics
| Experiment | Task | Model | Loss | Recall@3 | MRR@3 |
| --- | --- | --- | --- | --- | --- |
| exp003 | toy hardcoded docs | all-MiniLM-L6-v2 |  | 1.0000 | 0.9500 |
| exp006 | default chunks | all-MiniLM-L6-v2 |  | 1.0000 | 1.0000 |
| exp007 | 40/10 chunks | all-MiniLM-L6-v2 |  | 1.0000 | 1.0000 |
| exp007 | 120/30 chunks | all-MiniLM-L6-v2 |  | 1.0000 | 1.0000 |
| exp007 | 80/20 chunks | all-MiniLM-L6-v2 |  | 1.0000 | 1.0000 |
| exp010 | toy chunks | outputs/title-mnrl | MNRL | 1.0000 | 1.0000 |
| exp011 | toy chunks | outputs/manual-mnrl | MNRL | 1.0000 | 1.0000 |
| exp012 | toy chunks | outputs/test-mnrl | MNRL | 1.0000 | 1.0000 |
| exp012 | toy chunks | outputs/test-cached-mnrl | cached MNRL | 1.0000 | 1.0000 |

## ArXiv Full-Corpus Metrics
| Experiment | Task | Model | Loss | Recall@1 | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | MAP@10 | Latency ms/query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp016 | arxiv 1k | all-MiniLM-L6-v2 |  | 0.8811 | 0.9730 | 0.9946 | 0.9232 | 0.9407 | 0.9232 | 0.5963 |
| exp017 | arxiv 1k | outputs/arxiv-minilm-mnrl-1k | MNRL | 0.9027 | 0.9838 | 0.9946 | 0.9372 | 0.9514 | 0.9372 | 0.4409 |
| exp018 | arxiv 1k | outputs/arxiv-minilm-cached-1k | cached MNRL | 0.9027 | 0.9838 | 0.9892 | 0.9358 | 0.9491 | 0.9358 | 0.5106 |
| exp020 | arxiv 10k | all-MiniLM-L6-v2 |  | 0.8180 | 0.9370 | 0.9540 | 0.8712 | 0.8918 | 0.8712 | 0.7359 |
| exp021 | arxiv 10k | outputs/arxiv-minilm-cached-10k | cached MNRL | 0.8560 | 0.9540 | 0.9710 | 0.9005 | 0.9180 | 0.9005 | 0.4675 |
| exp024 | arxiv 10k | all-MiniLM-L6-v2 |  | 0.8180 | 0.9370 | 0.9540 | 0.8712 | 0.8918 | 0.8712 | 0.4493 |
| exp024 | arxiv 10k | e5-small-v2 |  | 0.8720 | 0.9620 | 0.9730 | 0.9101 | 0.9256 | 0.9101 | 0.7464 |
| exp024 | arxiv 10k | bge-small-en-v1.5 |  | 0.8730 | 0.9550 | 0.9670 | 0.9094 | 0.9236 | 0.9094 | 0.6985 |
| exp024 | arxiv 10k | arxiv-minilm-cached-10k | cached MNRL | 0.8560 | 0.9540 | 0.9710 | 0.9005 | 0.9180 | 0.9005 | 0.4557 |
| exp025 | arxiv 10k | hybrid:minilm+bm25 |  | 0.9730 | 0.9980 | 0.9990 | 0.9845 | 0.9882 | 0.9845 | 52.2376 |
| exp026 | arxiv 10k rerank | minilm + cross-encoder/ms-marco-MiniLM-L-6-v2 |  | 0.8700 | 0.9550 | 0.9650 | 0.9088 | 0.9228 | 0.9088 | 217.1042 |
| exp028 | arxiv 10k | arxiv-bge-small-cached-10k | cached MNRL | 0.8890 | 0.9640 | 0.9740 | 0.9216 | 0.9346 | 0.9216 | 0.7199 |

## Quantization Metrics
| Experiment | Model | Variant | Recall@10 | MRR@10 | nDCG@10 | MAP@10 | Latency ms/query | Doc storage bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exp027 | sentence-transformers/all-MiniLM-L6-v2 | float32 | 0.9540 | 0.8716 | 0.8921 | 0.8716 | 0.8683 | 15360000 |
| exp027 | sentence-transformers/all-MiniLM-L6-v2 | float16 | 0.9540 | 0.8717 | 0.8921 | 0.8717 | 0.9211 | 7680000 |
| exp027 | sentence-transformers/all-MiniLM-L6-v2 | int8 | 0.9530 | 0.8709 | 0.8913 | 0.8709 | 0.9158 | 3840000 |

## Hard-Negative Metrics
| Experiment | Task | Model | Loss | Accuracy@1 | MRR | Avg rank | Examples |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exp014 | BM25 toy | all-MiniLM-L6-v2 |  | 1.0000 | 1.0000 |  | 5 |
| exp014 | BM25 toy | outputs/test-mnrl | MNRL | 1.0000 | 1.0000 |  | 5 |
| exp014 | BM25 toy | outputs/test-cached-mnrl | cached MNRL | 1.0000 | 1.0000 |  | 5 |
| exp023 | BM25 arxiv | all-MiniLM-L6-v2 |  | 0.8280 | 0.8975 | 1.3120 | 1000 |
| exp023 | BM25 arxiv | outputs/arxiv-minilm-mnrl-1k | MNRL | 0.8530 | 0.9121 | 1.2680 | 1000 |
| exp023 | BM25 arxiv | outputs/arxiv-minilm-cached-1k | cached MNRL | 0.8470 | 0.9080 | 1.2860 | 1000 |
| exp023 | BM25 arxiv | outputs/arxiv-minilm-cached-10k | cached MNRL | 0.8620 | 0.9166 | 1.2590 | 1000 |
| exp024 | BM25 arxiv | all-MiniLM-L6-v2 |  | 0.8280 | 0.8975 | 1.3120 | 1000 |
| exp024 | BM25 arxiv | e5-small-v2 |  | 0.8620 | 0.9189 | 1.2420 | 1000 |
| exp024 | BM25 arxiv | bge-small-en-v1.5 |  | 0.8740 | 0.9265 | 1.2140 | 1000 |
| exp024 | BM25 arxiv | arxiv-minilm-cached-10k | cached MNRL | 0.8620 | 0.9166 | 1.2590 | 1000 |
| exp028 | BM25 arxiv | arxiv-bge-small-cached-10k | cached MNRL | 0.8830 | 0.9335 | 1.1820 | 1000 |
