import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ImageRetriever:
    def __init__(self, docs):
        """
        Multimodal image retriever:
        - CLIP embedding similarity
        - OCR keyword relevance
        - Table/chart support
        """

        self.images = []

        for d in docs:
            for img in d.get("images", []):

                emb = img.get("embedding")

                if (
                    emb is not None
                    and len(np.array(emb).shape) == 1
                ):
                    self.images.append({
                        "path": img.get("path"),
                        "embedding": np.array(emb),
                        "ocr": img.get("ocr", ""),
                        "page": d.get("page", 0)
                    })

        if self.images:
            self.embeddings = np.array(
                [
                    img["embedding"]
                    for img in self.images
                ]
            )
        else:
            self.embeddings = None

    def _ocr_score(self, query, ocr_text):
        if not ocr_text:
            return 0.0

        query_terms = [
            term.lower()
            for term in query.split()
            if term.strip()
        ]

        if not query_terms:
            return 0.0

        ocr_text = ocr_text.lower()

        matches = sum(
            1
            for term in query_terms
            if term in ocr_text
        )

        return matches / len(query_terms)

    def retrieve(
        self,
        query,
        clip,
        k=5
    ):
        """
        Returns:
        top-k images ranked by:
        70% CLIP semantic similarity
        30% OCR keyword match
        """

        if (
            self.embeddings is None
            or clip is None
            or not query
        ):
            return []

        try:
            q_emb = clip.encode_text(query)

            if q_emb is None:
                return []

            q_emb = np.array(q_emb)

            clip_scores = cosine_similarity(
                [q_emb],
                self.embeddings
            )[0]

            scored = []

            for i, img in enumerate(self.images):

                clip_score = float(
                    clip_scores[i]
                )

                ocr_score = self._ocr_score(
                    query,
                    img.get("ocr", "")
                )

                total_score = (
                    0.7 * clip_score
                    + 0.3 * ocr_score
                )

                scored.append({
                    "path": img["path"],
                    "embedding": img["embedding"],
                    "ocr": img.get("ocr", ""),
                    "page": img.get("page", 0),
                    "clip_score": clip_score,
                    "ocr_score": ocr_score,
                    "score": total_score
                })

            scored = sorted(
                scored,
                key=lambda x: x["score"],
                reverse=True
            )

            return scored[:k]

        except Exception:
            return []

    def retrieve_by_ocr(
        self,
        query,
        k=5
    ):
        """
        Useful for:
        - table searches
        - exact figure labels
        - OCR-heavy docs
        """

        if not self.images:
            return []

        scored = []

        for img in self.images:
            ocr_score = self._ocr_score(
                query,
                img.get("ocr", "")
            )

            scored.append({
                **img,
                "score": ocr_score
            })

        scored = sorted(
            scored,
            key=lambda x: x["score"],
            reverse=True
        )

        return scored[:k]

    def retrieve_by_clip(
        self,
        query,
        clip,
        k=5
    ):
        if (
            self.embeddings is None
            or clip is None
        ):
            return []

        try:
            q_emb = np.array(
                clip.encode_text(query)
            )

            scores = cosine_similarity(
                [q_emb],
                self.embeddings
            )[0]

            idx = np.argsort(
                scores
            )[::-1][:k]

            return [
                {
                    **self.images[i],
                    "score": float(scores[i])
                }
                for i in idx
            ]

        except Exception:
            return []