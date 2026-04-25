import numpy as np


class ContextFilter:
    def __init__(self, embedder):
        self.embedder = embedder

    def filter(self, query, docs, top_k=5):
        q_emb = self.embedder.encode([query])[0]

        doc_embs = self.embedder.encode(docs)

        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        q_emb = q_emb / np.linalg.norm(q_emb)

        scores = np.dot(doc_embs, q_emb)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [docs[i] for i in top_indices]