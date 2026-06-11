# scripts/01_bow_retrieval.py

import math
import re
from collections import Counter


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


def tokenize(text: str) -> list[str]:
    """
    Convert raw text into normalized tokens.

    Example:
        "RAG retrieves documents!"
        -> ["rag", "retrieves", "documents"]

    This is intentionally simple.
    """
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def bow(text: str) -> Counter:
    """
    Bag-of-words vector.

    Example:
        "rag rag documents"
        -> {"rag": 2, "documents": 1}
    """
    return Counter(tokenize(text))


def dot(a: Counter, b: Counter) -> float:
    """
    Sparse dot product.

    Only shared tokens contribute.
    """
    shared_tokens = a.keys() & b.keys()
    return sum(a[token] * b[token] for token in shared_tokens)


def norm(a: Counter) -> float:
    """
    Euclidean norm of a sparse vector.
    """
    return math.sqrt(sum(value * value for value in a.values()))


def cosine(a: Counter, b: Counter) -> float:
    """
    Cosine similarity between two sparse vectors.
    """
    denom = norm(a) * norm(b)
    if denom == 0:
        return 0.0
    return dot(a, b) / denom


def search(query: str, k: int = 3):
    query_vec = bow(query)

    scored = []

    for doc in DOCUMENTS:
        doc_vec = bow(doc["text"])
        score = cosine(query_vec, doc_vec)
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def main():
    query = "What does RAG retrieve before answering?"

    results = search(query, k=3)

    print(f"\nQuery: {query}\n")

    for rank, (score, doc) in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"ID: {doc['id']}")
        print(f"Score: {score:.4f}")
        print(f"Text: {doc['text']}")
        print()


if __name__ == "__main__":
    main()