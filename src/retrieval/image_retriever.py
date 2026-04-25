import numpy as np


class ImageRetriever:
    def __init__(self, docs):
        self.images = []

        for d in docs:
            for img in d.get("images", []):
                emb = img.get("embedding")

                if emb is not None and len(np.array(emb).shape) == 1:
                    self.images.append(img)

        if self.images:
            self.embeddings = np.array([img["embedding"] for img in self.images])
        else:
            self.embeddings = None

    def retrieve(self, query, clip, k=3):
        if self.embeddings is None:
            return []

        q_emb = clip.encode_text(query)

        q_emb = np.array(q_emb)

        scores = np.dot(self.embeddings, q_emb)

        idx = np.argsort(scores)[::-1][:k]

        return [self.images[i] for i in idx]