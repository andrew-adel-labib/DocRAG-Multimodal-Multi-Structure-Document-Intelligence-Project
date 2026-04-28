import numpy as np
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(self, texts, embeddings):
        embeddings = np.array(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        if embeddings.size == 0:
            raise ValueError(
                "No valid embeddings provided to HybridRetriever."
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        norms[norms == 0] = 1.0

        embeddings = embeddings / norms

        self.texts = texts
        self.embeddings = embeddings

        self.vector = VectorRetriever(
            texts,
            embeddings
        )

        self.bm25 = BM25Retriever(
            texts
        )

    def retrieve(
        self,
        query,
        embedder,
        k=10
    ):
        query_emb = embedder.encode(
            [query]
        )

        query_emb = np.array(
            query_emb,
            dtype=np.float32
        )

        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        q_norm = np.linalg.norm(
            query_emb,
            axis=1,
            keepdims=True
        )

        q_norm[q_norm == 0] = 1.0

        query_emb = query_emb / q_norm

        vec_docs = self.vector.search(
            query_emb,
            k
        )

        bm_docs = self.bm25.search(
            query,
            k
        )

        combined = vec_docs + bm_docs

        seen = set()
        final = []

        for doc in combined:
            if doc not in seen:
                final.append(doc)
                seen.add(doc)

        return final[:k]