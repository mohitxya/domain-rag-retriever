# scripts/02_dense_retrieval.py

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


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize each vector to unit length.

    After this:
        dot product = cosine similarity
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def build_index(model_name: str):
    model = SentenceTransformer(model_name)

    texts = [doc["text"] for doc in DOCUMENTS]

    doc_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    doc_embeddings = normalize(doc_embeddings).astype("float32")

    dim = doc_embeddings.shape[1]

    # Exact inner-product index.
    # Since vectors are normalized, inner product = cosine similarity.
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)

    return model, index


def search(query: str, model, index, k: int = 3):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    query_embedding = normalize(query_embedding).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        doc = DOCUMENTS[int(idx)]
        results.append(
            {
                "id": doc["id"],
                "text": doc["text"],
                "score": float(score),
            }
        )

    return results


def main():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("Loading model and building index...")
    model, index = build_index(model_name)

    queries = [
        "What does retrieval augmented generation do before answering?",
        "Which library is used for vector similarity search?",
        "How do models learn to pull similar examples together?",
    ]

    for query in queries:
        print("=" * 80)
        print(f"Query: {query}\n")

        results = search(query, model, index, k=3)

        for rank, result in enumerate(results, start=1):
            print(f"Rank {rank}")
            print(f"ID: {result['id']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print()


if __name__ == "__main__":
    main()