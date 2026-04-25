from openai import OpenAI
import os

from src.utils.logger import get_logger
from src.utils.exceptions import EmbeddingError

logger = get_logger("OpenAIEmbedder")


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        try:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model
            self.model_name = model
            logger.info(f"Loaded OpenAI embedding model: {model}")
        except Exception as e:
            raise EmbeddingError(f"OpenAI init failed: {e}")

    def encode(self, texts):
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [r.embedding for r in response.data]

        except Exception as e:
            raise EmbeddingError(f"OpenAI embedding failed: {e}")