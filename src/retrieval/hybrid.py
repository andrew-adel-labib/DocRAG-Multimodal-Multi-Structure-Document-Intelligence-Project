import numpy as np
from src.retrieval.vector import VectorRetriever
from src.retrieval.bm25 import BM25Retriever


class HybridRetriever:
    def __init__(self, texts, embeddings):
        embeddings = np.array(embeddings).astype("float32")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms

        self.texts = texts
        self.embeddings = embeddings

        self.vector = VectorRetriever(texts, embeddings)
        self.bm25 = BM25Retriever(texts)

    def retrieve(self, query, embedder, k=10):
        query_emb = embedder.encode([query])

        if isinstance(query_emb, list):
            query_emb = np.array(query_emb).astype("float32")

        q_norm = np.linalg.norm(query_emb)
        if q_norm == 0:
            q_norm = 1
        query_emb = query_emb / q_norm
        
        vec_docs = self.vector.search(query_emb, k)
        bm_docs = self.bm25.search(query, k)

        combined = vec_docs + bm_docs

        seen = set()
        final = []

        for doc in combined:
            if doc not in seen:
                final.append(doc)
                seen.add(doc)

        return final[:k]