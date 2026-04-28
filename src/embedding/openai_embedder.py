from openai import OpenAI
import os

from src.utils.logger import get_logger
from src.utils.exceptions import EmbeddingError

logger = get_logger("OpenAIEmbedder")


class OpenAIEmbedder:
    def __init__(self, model="text-embedding-3-small"):
        try:
            self.client = OpenAI(
                api_key=os.getenv(
                    "OPENAI_API_KEY"
                )
            )

            self.model = model
            self.model_name = model

            logger.info(
                f"Loaded OpenAI embedding model: {model}"
            )

        except Exception as e:
            raise EmbeddingError(
                f"OpenAI init failed: {e}"
            )

    def encode(
        self,
        texts,
        batch_size=50
    ):
        """
        Safe batched OpenAI embedding.
        Prevents:
        - API array length errors
        - Large PDF failures
        - Section-parent failures
        """

        try:
            if isinstance(texts, str):
                texts = [texts]

            texts = [
                t.strip()
                for t in texts
                if t and str(t).strip()
            ]

            if not texts:
                return []

            all_embeddings = []

            for i in range(
                0,
                len(texts),
                batch_size
            ):

                batch = texts[
                    i:i + batch_size
                ]

                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch
                )

                batch_embeddings = [
                    r.embedding
                    for r in response.data
                ]

                all_embeddings.extend(
                    batch_embeddings
                )

                logger.info(
                    f"Embedded batch {i // batch_size + 1} "
                    f"({len(batch)} texts)"
                )

            return all_embeddings

        except Exception as e:
            raise EmbeddingError(
                f"OpenAI embedding failed: {e}"
            )