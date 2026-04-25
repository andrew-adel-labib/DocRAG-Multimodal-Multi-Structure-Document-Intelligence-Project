from sentence_transformers import SentenceTransformer
from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger

logger = get_logger("Embedder")


class Embedder:
    # all-MiniLM-L6-v2 (fast)
    # hkunlp/instructor-xl (Instruction-tuned which is good for Q/A)
    def __init__(self, model="BAAI/bge-base-en-v1.5"):
        try:
            self.model_name = model
            self.model = SentenceTransformer(model)
            logger.info(f"Loaded embedding model: {model}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load {model}: {e}")

    def encode(self, texts):
        try:
            return self.model.encode(texts, show_progress_bar=False)
        except Exception as e:
            raise EmbeddingError(f"Encoding failed: {e}")