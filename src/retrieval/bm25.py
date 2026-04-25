from rank_bm25 import BM25Okapi
from src.utils.logger import get_logger
from src.utils.exceptions import RetrievalError

logger = get_logger("BM25Retriever")


class BM25Retriever:
    def __init__(self, texts):
        try:
            self.texts = texts
            self.bm25 = BM25Okapi([t.split() for t in texts])
        except Exception as e:
            raise RetrievalError(f"BM25 init failed: {e}")

    def search(self, query, k=5):
        try:
            scores = self.bm25.get_scores(query.split())
            idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            return [self.texts[i] for i in idx]
        except Exception as e:
            raise RetrievalError(f"BM25 search failed: {e}")