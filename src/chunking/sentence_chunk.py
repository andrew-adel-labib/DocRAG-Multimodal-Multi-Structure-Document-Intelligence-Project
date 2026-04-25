import re
from src.utils.logger import get_logger

logger = get_logger("SentenceChunk")

def sentence_chunk(docs, max_sentences=5):
    chunks = []

    for d in docs:
        try:
            sentences = re.split(r'(?<=[.!?]) +', d["text"])

            for i in range(0, len(sentences), max_sentences):
                chunk = " ".join(sentences[i:i+max_sentences])
                chunks.append({"text": chunk, "page": d["page"]})

        except Exception as e:
            logger.warning(e)

    return chunks