from transformers import pipeline
from src.utils.logger import get_logger

logger = get_logger("QueryRewriter")


class QueryRewriter:
    def __init__(self, model="google/flan-t5-base"):
        try:
            self.model = pipeline("text-generation", model=model)
            logger.info(f"Loaded rewriting model: {model}")
        except Exception as e:
            logger.warning(f"Failed to load rewriting model: {e}")
            self.model = None

    def rewrite(self, query):
        try:
            if not self.model:
                return query

            prompt = f"Rewrite this query to be more specific: {query}"

            output = self.model(prompt, max_length=50)[0]["generated_text"]

            cleaned = output.replace(prompt, "").strip()

            return cleaned if cleaned else query

        except Exception as e:
            logger.warning(f"Rewrite failed: {e}")
            return query