from src.chunking.section_aware_chunk import section_aware_chunk
from src.chunking.parent_child_chunk import parent_child_chunk
from src.utils.logger import get_logger

logger = get_logger("SectionParentChunk")


def section_parent_chunk(
    docs,
    parent_size=12,
    child_size=4
):
    """
    Combined high-accuracy chunking:
    1. Section-aware preserves document structure
    2. Parent-child preserves large context + retrieval precision
    """

    try:
        logger.info(
            "Running section-aware chunking..."
        )

        section_chunks = section_aware_chunk(
            docs
        )

        if not section_chunks:
            logger.warning(
                "Section-aware returned no chunks."
            )
            return []

        section_docs = []

        for chunk in section_chunks:

            if isinstance(chunk, dict):
                text = chunk.get(
                    "text",
                    ""
                )

                page = chunk.get(
                    "page",
                    0
                )

            else:
                text = str(chunk)
                page = 0

            if text and text.strip():
                section_docs.append({
                    "text": text.strip(),
                    "page": page
                })

        if not section_docs:
            logger.warning(
                "No valid section docs after cleanup."
            )
            return []

        logger.info(
            f"Valid section docs: {len(section_docs)}"
        )

        logger.info(
            "Running parent-child chunking..."
        )

        final_chunks = parent_child_chunk(
            section_docs,
            parent_size=parent_size,
            child_size=child_size
        )

        if not final_chunks:
            logger.warning(
                "Parent-child returned no chunks."
            )
            return []

        cleaned_chunks = []

        for chunk in final_chunks:

            if isinstance(chunk, dict):
                text = chunk.get(
                    "text",
                    ""
                )

                page = chunk.get(
                    "page",
                    0
                )

            else:
                text = str(chunk)
                page = 0

            if text and text.strip():
                cleaned_chunks.append({
                    "text": text.strip(),
                    "page": page
                })

        logger.info(
            f"Section-parent generated {len(cleaned_chunks)} valid chunks."
        )

        return cleaned_chunks

    except Exception as e:
        logger.error(
            f"Combined chunking failed: {e}"
        )

        return []