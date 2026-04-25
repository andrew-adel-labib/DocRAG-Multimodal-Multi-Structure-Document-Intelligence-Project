import numpy as np
import faiss


class VectorRetriever:
    def __init__(self, texts, embeddings):
        self.texts = texts
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)

    def search(self, query_emb, k=5):
        query_emb = np.array(query_emb).astype("float32")
        _, I = self.index.search(query_emb, k)
        return [self.texts[i] for i in I[0]]