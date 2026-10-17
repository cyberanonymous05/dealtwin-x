"""Build and query a FAISS index over DealDNA embeddings for precedent transaction retrieval."""

import numpy as np
import faiss


def build_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index


def search(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 10):
    query = query_embedding.reshape(1, -1).astype("float32")
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    return scores[0], indices[0]
