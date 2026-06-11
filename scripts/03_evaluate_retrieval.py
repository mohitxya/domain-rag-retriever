# scripts/03_evaluate_retrieval.py

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DOCUMENTS = [
    {
        "id": "D1",
        "text": "RAG retrieves relevant documents before generating an answer.",
    },
    {
        "id": "D2",
        "text": "FAISS performs fast vector similarity search over embeddings.",
    },
    {
        "id": "D3",
        "text": "Contrastive learning pulls related examples together and pushes unrelated examples apart.",
    },
    {
        "id": "D4",
        "text": "Judo uses kuzushi to break balance before executing a throw.",
    },
    {
        "id": "D5",
        "text": "Gradient descent updates neural network parameters using gradients.",
    },
]


BENCHMARK = [
    {
        "query":"What external knowledge does the system fetch before responding?",
        "positive_doc_id":"D1",
    },
    {
        "query": "What does RAG retrieve before answering?",
        "positive_doc_id": "D1",
    },
    {
        "query": "Which system is used for nearest-neighbor vector lookup?",
        "positive_doc_id": "D2",
    },
    {
        "query": "Which tool performs vector similarity search?",
        "positive_doc_id": "D2",
    },
    {
        "query": "What kind of learning pulls related examples together?",
        "positive_doc_id": "D3",
    },
    {
        "query": "What objective makes similar examples close in embedding space?",
        "positive_doc_id": "D3",
    },
    {
        "query": "What breaks balance in judo before a throw?",
        "positive_doc_id": "D4",
    },
    {
        "query": "What Japanese judo idea means breaking balance?",
        "positive_doc_id": "D4",
    },
    {
        "query": "What updates neural network parameters?",
        "positive_doc_id": "D5",
    },
    {
        "query": "What algorithm changes model weights using derivatives?",
        "positive_doc_id": "D5",
    },
]


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


class DenseRetriever:
    def __init__(self, documents: list[dict], model_name: str):
        self.documents = documents
        self.model = SentenceTransformer(model_name)

        texts = [doc["text"] for doc in documents]

        doc_embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        doc_embeddings = normalize(doc_embeddings).astype("float32")

        self.index = faiss.IndexFlatIP(doc_embeddings.shape[1])
        self.index.add(doc_embeddings)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        query_embedding = normalize(query_embedding).astype("float32")

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[int(idx)]

            results.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(score),
                }
            )

        return results


def recall_at_k(retrieved_ids: list[str], positive_id: str, k: int) -> float:
    return float(positive_id in retrieved_ids[:k])


def reciprocal_rank_at_k(retrieved_ids: list[str], positive_id: str, k: int) -> float:
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id == positive_id:
            return 1.0 / rank

    return 0.0


def evaluate(retriever: DenseRetriever, benchmark: list[dict], k: int = 5) -> dict:
    recall_scores = []
    rr_scores = []

    print("\nDetailed results\n")

    for item in benchmark:
        query = item["query"]
        positive_id = item["positive_doc_id"]

        results = retriever.search(query, k=k)
        retrieved_ids = [r["id"] for r in results]

        recall = recall_at_k(retrieved_ids, positive_id, k)
        rr = reciprocal_rank_at_k(retrieved_ids, positive_id, k)

        recall_scores.append(recall)
        rr_scores.append(rr)

        print("=" * 80)
        print(f"Query: {query}")
        print(f"Expected: {positive_id}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Recall@{k}: {recall}")
        print(f"RR@{k}: {rr:.4f}")

    return {
        f"recall@{k}": sum(recall_scores) / len(recall_scores),
        f"mrr@{k}": sum(rr_scores) / len(rr_scores),
    }


def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    retriever = DenseRetriever(DOCUMENTS, model_name)

    metrics = evaluate(retriever, BENCHMARK, k=3)

    print("\nFinal metrics")
    print(metrics)


if __name__ == "__main__":
    main()