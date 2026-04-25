import os
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.parser.advanced_parser import parse_pdf_advanced
from src.chunking.recursive_chunk import recursive_chunk
from src.embedding.embedder import Embedder
from src.embedding.openai_embedder import OpenAIEmbedder
from src.embedding.clip_embedder import CLIPEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.image_retriever import ImageRetriever
from src.query.rewriting import QueryRewriter
from src.rag.pipeline_advanced import AdvancedRAG

load_dotenv()
logger = get_logger("MAIN")


def build_system():
    pdf_path = os.getenv("PDF_PATH")

    docs = parse_pdf_advanced(pdf_path)

    chunks = recursive_chunk(docs)
    texts = [c["text"] for c in chunks]

    use_openai = True

    if use_openai:
        embedder = OpenAIEmbedder("text-embedding-3-small")
    else:
        embedder = Embedder("BAAI/bge-base-en-v1.5")

    embeddings = embedder.encode(texts)

    retriever = HybridRetriever(texts, embeddings)

    clip = CLIPEmbedder()
    image_retriever = ImageRetriever(docs)

    reranker = Reranker()
    rewriter = QueryRewriter()

    rag = AdvancedRAG(
        retriever,
        image_retriever,
        clip,
        reranker,
        rewriter,
        embedder
    )

    return rag


def main():
    rag = build_system()

    print("\n🔥 RAG Ready (type 'exit')")

    while True:
        q = input("Question: ")

        if q.lower() == "exit":
            break

        ans = rag.answer(q)
        print("\nAnswer:\n", ans)
        print("=" * 50)


if __name__ == "__main__":
    main()