from openai import OpenAI
from src.utils.logger import get_logger
from src.utils.text_utils import extract_exact_answer

logger = get_logger("AdvancedRAG")


class AdvancedRAG:
    def __init__(self, retriever, image_retriever, clip, reranker, rewriter, embedder):
        self.retriever = retriever
        self.image_retriever = image_retriever
        self.clip = clip
        self.reranker = reranker
        self.rewriter = rewriter
        self.embedder = embedder

        self.client = OpenAI()

    def is_visual_query(self, query):
        keywords = [
            "list", "image", "figure", "diagram", "chart",
            "table", "logo", "picture", "shown", "process", "workflow"
        ]
        return any(k in query.lower() for k in keywords)

    def is_yes_no_question(self, query):
        query = query.strip().lower()

        aux_verbs = [
            "is", "are", "was", "were",
            "do", "does", "did",
            "can", "could",
            "will", "would",
            "should", "has", "have", "had"
        ]

        return any(query.startswith(v + " ") for v in aux_verbs)

    def answer(self, query):
        try:
            q = self.rewriter.rewrite(query) if self.rewriter else query

            is_yes_no = self.is_yes_no_question(q)

            docs = self.retriever.retrieve(q, self.embedder, k=10)

            if self.reranker:
                docs = self.reranker.rerank(q, docs)

            docs = docs[:5]

            if not is_yes_no:
                exact = extract_exact_answer(q, docs)
                if exact:
                    return exact

            use_images = self.is_visual_query(q)
            image_context = ""

            if use_images:
                images = self.image_retriever.retrieve(q, self.clip)
                image_context = "\n".join([img["path"] for img in images[:3]])

            context = "\n".join(docs)

            if use_images:
                context += "\n\nImages:\n" + image_context

            if is_yes_no:
                prompt = f"""
You are a precise question answering system.

RULES:
- Answer ONLY using the provided context
- Answer with ONLY: YES or NO
- After that, give a SHORT reason (one sentence max)
- If not found → return ONLY: NOT FOUND
- Do NOT hallucinate

CONTEXT:
{context}

QUESTION:
{q}

FINAL ANSWER:
"""
            else:
                prompt = f"""
You are a precise question answering system.

RULES:
- Answer ONLY using the provided context
- Do NOT explain
- Do NOT generate extra text
- If multiple answers exist → return all
- If partial information exists → return best match
- Only return NOT FOUND if nothing relevant exists

CONTEXT:
{context}

QUESTION:
{q}

FINAL ANSWER:
"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            logger.error(f"Advanced RAG failed: {e}")
            return "Error"