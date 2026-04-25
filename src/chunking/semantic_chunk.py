import numpy as np
from src.utils.logger import get_logger

logger = get_logger("SemanticChunk")

def semantic_chunk(docs, embedder, threshold=0.75):
    chunks = []

    for d in docs:
        try:
            sentences = d["text"].split(".")
            embeddings = embedder.encode(sentences)

            current_chunk = [sentences[0]]

            for i in range(1, len(sentences)):
                sim = np.dot(embeddings[i-1], embeddings[i]) / (
                    np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
                )

                if sim > threshold:
                    current_chunk.append(sentences[i])
                else:
                    chunks.append({
                        "text": ".".join(current_chunk),
                        "page": d["page"]
                    })
                    current_chunk = [sentences[i]]

            if current_chunk:
                chunks.append({
                    "text": ".".join(current_chunk),
                    "page": d["page"]
                })

        except Exception as e:
            logger.warning(e)

    return chunks