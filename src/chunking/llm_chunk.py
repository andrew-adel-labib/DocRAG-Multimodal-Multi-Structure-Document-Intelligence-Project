from openai import OpenAI
from src.utils.logger import get_logger

logger = get_logger("LLMChunk")


class LLMChunker:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def split_text(self, text):
        prompt = f"""
Split the following text into semantically coherent sections.

RULES:
- Preserve meaning
- Split by topic shifts
- Return chunks separated ONLY by: <CHUNK>

TEXT:
{text}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        output = response.choices[0].message.content.strip()

        return [
            c.strip() for c in output.split("<CHUNK>")
            if c.strip()
        ]


def llm_chunk(docs):
    """
    Uses LLM to chunk by semantic meaning.
    """

    chunker = LLMChunker()
    chunks = []

    for doc in docs:
        try:
            text = doc.get("text", "")
            page = doc.get("page", 0)

            if not text.strip():
                continue

            sections = chunker.split_text(text)

            for sec in sections:
                chunks.append({
                    "text": sec,
                    "page": page
                })

        except Exception as e:
            logger.warning(f"LLM chunking failed: {e}")

    return chunks