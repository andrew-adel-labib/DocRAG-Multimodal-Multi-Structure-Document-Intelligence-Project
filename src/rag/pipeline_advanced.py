import os

from src.llm.vllm_client import VLLMClient
from src.llm.vision_client import VisionClient
from src.llm.openai_client import OpenAIClient
from src.llm.claude_client import ClaudeClient
# from anthropic import Anthropic

from src.utils.logger import get_logger
from src.utils.text_utils import extract_exact_answer

logger = get_logger("AdvancedRAG")


class AdvancedRAG:
    def __init__(
        self,
        retriever,
        image_retriever,
        clip,
        reranker,
        rewriter,
        embedder,
        llm_provider="openai",
        llm_model=None,
        vision_model=None
    ):
        self.retriever = retriever
        self.image_retriever = image_retriever
        self.clip = clip
        self.reranker = reranker
        self.rewriter = rewriter
        self.embedder = embedder

        provider = str(llm_provider).lower()

        if provider == "vllm":
            self.client = VLLMClient(
                model=llm_model or "meta-llama/Meta-Llama-3-8B-Instruct"
            )

        elif provider == "claude":
            self.client = ClaudeClient(
                model=llm_model or "claude-3-haiku-20240307"
            )

        else:
            self.client = OpenAIClient(
                model=llm_model or "gpt-4o-mini"
            )

        self.llm_provider = provider

        self.vision_client = VisionClient(
            model=vision_model or "Qwen/Qwen2-VL-7B-Instruct"
        )

    def is_visual_query(self, query):
        keywords = [
            "list",
            "image",
            "figure",
            "diagram",
            "chart",
            "table",
            "logo",
            "picture",
            "shown",
            "process",
            "workflow",
            "visual",
            "graph",
            "screenshot"
        ]

        query = query.lower()

        return any(
            k in query
            for k in keywords
        )

    def is_complex_visual_query(self, query):
        keywords = [
            "architecture",
            "workflow",
            "flowchart",
            "complex diagram",
            "pipeline diagram",
            "technical diagram",
            "system design",
            "network diagram",
            "block diagram"
        ]

        query = query.lower()

        return any(
            k in query
            for k in keywords
        )

    def is_yes_no_question(self, query):
        query = query.strip().lower()

        aux_verbs = [
            "is", "are", "was", "were",
            "do", "does", "did",
            "can", "could",
            "will", "would",
            "should",
            "has", "have", "had"
        ]

        return any(
            query.startswith(v + " ")
            for v in aux_verbs
        )

    def build_prompt(
        self,
        query,
        context,
        is_yes_no
    ):

        if is_yes_no:
            return f"""
You are a precise question answering system.

RULES:
- Answer ONLY using the provided context
- Answer with ONLY: YES or NO
- Then provide ONE short reason
- Use OCR, tables, and visual context if relevant
- If answer cannot be found → return ONLY: NOT FOUND
- Do NOT hallucinate

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""

        return f"""
You are a precise question answering system.

RULES:
- Answer ONLY using the provided context
- Use ALL available text, OCR, tables, and visual context
- Prefer exact values whenever possible
- Do NOT explain
- Do NOT generate unnecessary text
- If multiple answers exist → return all
- If partial info exists → return best match
- Only return NOT FOUND if absolutely nothing relevant exists

CONTEXT:
{context}

QUESTION:
{query}

FINAL ANSWER:
"""

    def build_context(
        self,
        docs,
        image_context=""
    ):
        context = "\n".join(
            [
                str(d)[:1500]
                for d in docs
                if d
            ]
        )

        if image_context:
            context += (
                "\n\nVISUAL CONTEXT:\n"
                + image_context[:3000]
            )

        return context

    def answer(self, query):
        try:
            q = (
                self.rewriter.rewrite(query)
                if self.rewriter
                else query
            )

            is_yes_no = self.is_yes_no_question(q)
            use_images = self.is_visual_query(q)

            docs = self.retriever.retrieve(
                q,
                self.embedder,
                k=10
            )

            if self.reranker:
                docs = self.reranker.rerank(
                    q,
                    docs
                )

            docs = docs[:5]

            if not is_yes_no:
                exact = extract_exact_answer(
                    q,
                    docs
                )

                if exact:
                    return exact
            image_context = ""

            if (
                use_images
                and self.image_retriever
                and self.clip
            ):
                try:
                    images = self.image_retriever.retrieve(
                        q,
                        self.clip,
                        k=5
                    )

                    if (
                        self.is_complex_visual_query(q)
                        and images
                    ):
                        return self.vision_client.analyze_image(
                            images[0]["path"],
                            q
                        )

                    ocr_texts = [
                        img.get("ocr", "")
                        for img in images
                        if img.get("ocr")
                    ]

                    if ocr_texts:
                        image_context = "\n".join(
                            ocr_texts[:3]
                        )

                    else:
                        image_context = "\n".join(
                            [
                                img["path"]
                                for img in images[:3]
                            ]
                        )

                except Exception as e:
                    logger.warning(
                        f"Image retrieval failed: {e}"
                    )

            context = self.build_context(
                docs,
                image_context
            )

            prompt = self.build_prompt(
                q,
                context,
                is_yes_no
            )

            answer = self.client.generate(
                prompt=prompt,
                temperature=0,
                max_tokens=500
            )

            if not answer:
                return "NOT FOUND"

            return answer.strip()

        except Exception as e:
            logger.error(
                f"Advanced RAG failed: {e}"
            )

            return "Error"