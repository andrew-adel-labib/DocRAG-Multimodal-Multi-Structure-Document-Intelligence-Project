import numpy as np
from src.utils.logger import get_logger

logger = get_logger("LateChunk")


def late_chunk(docs, embedder, chunk_size=5):

    chunks = []

    for d in docs:
        try:
            sentences = d["text"].split(".")
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                continue

            embeddings = embedder.encode(sentences)

            for i in range(0, len(sentences), chunk_size):
                group = sentences[i:i+chunk_size]
                group_emb = embeddings[i:i+chunk_size]

                agg_emb = np.mean(group_emb, axis=0)

                chunks.append({
                    "text": ". ".join(group),
                    "embedding": agg_emb,
                    "page": d["page"]
                })

        except Exception as e:
            logger.warning(f"Late chunking failed: {e}")

    return chunks