import numpy as np
from sklearn.cluster import AgglomerativeClustering
from src.utils.logger import get_logger

logger = get_logger("RAPTORChunk")


def summarize_cluster(texts):
    """
    Lightweight summary placeholder.
    Replace with LLM summarization for full RAPTOR.
    """
    return " ".join(texts[:3])


def raptor_chunk(docs, embedder, max_clusters=5):
    """
    Simplified RAPTOR:
    - Sentence split
    - Embed
    - Cluster semantically
    - Summarize clusters
    """

    chunks = []

    for doc in docs:
        try:
            text = doc.get("text", "")
            page = doc.get("page", 0)

            sentences = [
                s.strip() for s in text.split(".")
                if s.strip()
            ]

            if len(sentences) < 2:
                continue

            embeddings = embedder.encode(sentences)

            n_clusters = min(max_clusters, len(sentences))

            clustering = AgglomerativeClustering(
                n_clusters=n_clusters
            )

            labels = clustering.fit_predict(embeddings)

            for cluster_id in set(labels):
                cluster_sentences = [
                    sentences[i]
                    for i in range(len(sentences))
                    if labels[i] == cluster_id
                ]

                summary = summarize_cluster(cluster_sentences)

                chunks.append({
                    "text": summary,
                    "page": page,
                    "cluster": int(cluster_id)
                })

        except Exception as e:
            logger.warning(f"RAPTOR chunking failed: {e}")

    return chunks