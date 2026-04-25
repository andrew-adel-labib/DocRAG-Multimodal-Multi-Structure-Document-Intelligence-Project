from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logger import get_logger

logger = get_logger("RecursiveChunk")

def recursive_chunk(docs, size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap
    )

    chunks = []

    for d in docs:
        try:
            splits = splitter.split_text(d["text"])
            for s in splits:
                chunks.append({"text": s, "page": d["page"]})
        except Exception as e:
            logger.warning(e)

    return chunks