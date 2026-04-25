from transformers import pipeline
from src.utils.logger import get_logger

logger = get_logger("BaselineRAG")


class BaselineRAG:
    def __init__(self, retriever, embedder):
        self.retriever = retriever
        self.embedder = embedder
        self.llm = pipeline("text-generation", model="gpt2")

    def answer(self, query):
        try:
            docs = self.retriever.retrieve(query, self.embedder)
            context = "\n".join(docs)

            prompt = f"{context}\nQ:{query}\nA:"
            return self.llm(prompt, max_length=200)[0]["generated_text"]

        except Exception as e:
            logger.error(f"Baseline RAG failed: {e}")
            return "Error"