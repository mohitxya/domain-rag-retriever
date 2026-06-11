# Domain RAG

#### Goal

- Build a complete retrieval system from first principles.
- Start with a working baseline.
- Add evaluation.
- Add scalable data processing.
- Fine-tune embedding models.
- Add hard-negative mining.
- Compare encoder baselines.
- Run quantization experiments.
- Explore decoder-as-encoder retrieval.
- Keep the project educational, modular, and research-oriented.

#### Current Status

- Core retrieval pipeline is being built step by step.
- The project is currently focused on correctness, clarity, and hands-on implementation.
- The long-term goal is to turn this into a strong RAG/retrieval engineering portfolio project.

#### Built So Far

#### Project Structure

- Basic project direction defined.
- Milestone order finalized.
- Retrieval-first roadmap chosen before advanced model optimization.

#### Retrieval System

- Initial retrieval system work started.
- Focus is on understanding the full pipeline before adding complexity.

#### Learning Direction

- Sentence-transformers selected for embedding models.
- Multiple Negatives Ranking Loss selected as the main fine-tuning objective.
- Cached MNRL planned for efficient large-batch contrastive training.
- Daft selected for scalable text/data processing.
- Hard-negative mining planned after baseline evaluation.
- Quantization and regularization left for later-stage experiments.

#### Planned Milestones

#### 1. Working Retrieval System

- Build document ingestion.
- Chunk raw documents into retrievable passages.
- Embed chunks using a pretrained sentence-transformer model.
- Store embeddings in a vector index.
- Retrieve top-k relevant chunks for a query.
- Return ranked results with scores.

#### 2. Evaluation

- Create a small query-document evaluation set.
- Add retrieval metrics:
  - Recall@k
  - MRR
  - Precision@k
  - nDCG
- Compare different chunk sizes.
- Compare different embedding models.
- Track experiments in a simple table or JSON log.

#### 3. Daft Data Processing

- Use Daft for large-scale text preprocessing.
- Build streaming-friendly document pipelines.
- Add cleaning, filtering, deduplication, and chunk generation.
- Save processed datasets in a reusable format.

#### 4. MNRL Fine-Tuning

- Prepare positive query-document pairs.
- Fine-tune a sentence-transformer model using Multiple Negatives Ranking Loss.
- Compare pretrained vs fine-tuned embeddings.
- Evaluate retrieval improvement using the same benchmark.

#### 5. Cached MNRL

- Add cached Multiple Negatives Ranking Loss.
- Train with larger effective batch sizes.
- Compare:
  - normal MNRL
  - cached MNRL
  - different batch sizes
  - different learning rates

#### 6. Hard-Negative Mining

- Retrieve confusing but incorrect documents.
- Add them as hard negatives during training.
- Compare random negatives vs mined hard negatives.
- Measure whether ranking quality improves.

#### 7. Encoder Baselines

- Compare multiple encoder models.
- Evaluate tradeoffs between:
  - accuracy
  - latency
  - memory usage
  - embedding dimension
  - index size

#### 8. Quantization Experiment

- Quantize embeddings or encoder models.
- Measure retrieval quality drop.
- Measure speed and memory improvements.
- Compare full precision vs quantized retrieval.

#### 9. Decoder-as-Encoder Extension

- Explore converting decoder-only models into embedding models.
- Compare decoder-based embeddings with standard encoder models.
- Evaluate whether decoder representations are useful for retrieval.

#### 10. Orthogonal Regularization

- Study Global Orthogonal Regularization.
- Experiment with representation geometry.
- Measure whether embedding space quality improves.
- Keep this as a late-stage research extension.

#### Core Concepts Covered

#### Retrieval

- Dense retrieval
- Embeddings
- Vector search
- Similarity scoring
- Top-k ranking
- Chunking
- Query-document matching

#### Evaluation

- Ground-truth relevance labels
- Recall@k
- Mean Reciprocal Rank
- Ranking quality
- Ablation studies
- Reproducible experiments

#### Training

- Sentence-transformers
- Contrastive learning
- In-batch negatives
- Multiple Negatives Ranking Loss
- Cached loss computation
- Hard-negative mining

#### Systems

- Scalable preprocessing
- Streaming data pipelines
- Vector index storage
- Batch embedding
- Latency measurement
- Memory-aware retrieval

#### Research Extensions

- Quantization
- Representation regularization
- Decoder-to-encoder adaptation
- Embedding space analysis
- Retrieval model comparison

#### Repository Goals

- Clean implementation.
- Clear educational code.
- Strong experiment tracking.
- Reproducible results.
- Portfolio-ready README and diagrams.
- Research-paper-inspired extensions.

#### Expected Final Demo

- User enters a query.
- System retrieves relevant document chunks.
- Retrieved chunks are shown with similarity scores.
- Optional LLM answer generation can use retrieved context.
- Evaluation dashboard or script reports retrieval metrics.

#### Expected Final Outputs

- Working retrieval pipeline.
- Processed dataset pipeline.
- Evaluation benchmark.
- Fine-tuned retriever.
- Hard-negative mining pipeline.
- Baseline comparison table.
- Quantization results.
- Research notes.
- Clean README.
- Resume-worthy project summary.

#### Suggested Folder Structure

```text
domain-rag/
├── data/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── chunking/
│   ├── embedding/
│   ├── indexing/
│   ├── retrieval/
│   ├── evaluation/
│   ├── training/
│   └── utils/
├── experiments/
├── configs/
├── scripts/
├── reports/
└── README.md
```

#### Tech Stack

- Python
- PyTorch
- Sentence Transformers
- Hugging Face
- Daft
- FAISS or similar vector index
- NumPy
- Pandas or Polars
- Matplotlib
- Optional FastAPI UI/backend

#### Current Priority

- Finish the basic retrieval system.
- Add evaluation immediately after.
- Avoid advanced training until the baseline is measurable.

#### Next Steps

#### Immediate

- Implement document loading.
- Implement chunking.
- Generate embeddings.
- Build vector search.
- Run a few manual queries.

#### After That

- Create evaluation data.
- Add metrics.
- Compare baseline models.
- Only then start MNRL fine-tuning.

#### What This Project Demonstrates

- Ability to build retrieval systems from scratch.
- Understanding of modern RAG internals.
- Practical ML evaluation skills.
- Experience with embedding model fine-tuning.
- Systems thinking around data processing and inference.
- Ability to connect research papers to working code.