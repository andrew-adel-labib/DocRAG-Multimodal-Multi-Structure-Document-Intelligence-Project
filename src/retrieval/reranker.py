from sentence_transformers import CrossEncoder
from src.utils.logger import get_logger

logger = get_logger("Reranker")


class Reranker:
    
    # def __init__(self, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    #     self.model = CrossEncoder(model)
    
    def __init__(self, model="BAAI/bge-reranker-large"):
        self.model = CrossEncoder(model)

    def rerank(self, query, docs, top_k=5):
        try:
            pairs = [[query, d] for d in docs]
            scores = self.model.predict(pairs)

            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            return [r[0] for r in ranked[:top_k]]

        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return docs[:top_k]