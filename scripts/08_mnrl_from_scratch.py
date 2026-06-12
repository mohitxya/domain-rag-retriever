# scripts/08_mnrl_from_scratch.py

import torch
import torch.nn.functional as F


def main():
    torch.manual_seed(0)

    batch_size = 4
    embedding_dim = 3

    # Pretend these came from an encoder.
    query_embeddings = torch.randn(batch_size, embedding_dim)
    doc_embeddings = torch.randn(batch_size, embedding_dim)

    # Normalize so dot product becomes cosine similarity.
    query_embeddings = F.normalize(query_embeddings, dim=1)
    doc_embeddings = F.normalize(doc_embeddings, dim=1)

    # Similarity matrix:
    # scores[i, j] = similarity(query_i, doc_j)
    scores = query_embeddings @ doc_embeddings.T

    # Optional scale. Sentence-transformers often uses scale=20.
    scale = 20.0
    scores = scores * scale

    # Correct doc for query i is doc i.
    labels = torch.arange(batch_size)

    loss = F.cross_entropy(scores, labels)

    print("Scores matrix:")
    print(scores)

    print("\nLabels:")
    print(labels)

    print("\nLoss:")
    print(loss.item())

    print("\nInterpretation:")
    print("For each row q_i, cross entropy wants column i to be the largest.")


if __name__ == "__main__":
    main()