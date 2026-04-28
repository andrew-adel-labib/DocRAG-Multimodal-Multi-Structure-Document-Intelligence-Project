from src.utils.logger import get_logger

logger = get_logger("ParentChildChunk")


def parent_child_chunk(
    docs,
    parent_size=1200,
    child_size=300,
    overlap=50
):
    """
    Creates:
    - Parent chunks for broad context
    - Child chunks for precise retrieval

    Returns:
    [
        {
            "id": ...,
            "text": child_text,
            "page": ...,
            "parent_id": ...,
            "parent_text": ...
        }
    ]
    """

    chunks = []
    child_id = 0

    try:
        for doc in docs:

            if isinstance(doc, dict):
                text = doc.get("text", "")
                page = doc.get("page", 0)

            else:
                text = str(doc)
                page = 0

            if not text or not text.strip():
                continue

            text = text.strip()

            if parent_size <= overlap:
                parent_step = parent_size
            else:
                parent_step = parent_size - overlap

            if child_size <= overlap:
                child_step = child_size
            else:
                child_step = child_size - overlap

            for p_start in range(
                0,
                len(text),
                parent_step
            ):

                parent_text = text[
                    p_start:p_start + parent_size
                ].strip()

                if not parent_text:
                    continue

                parent_id = (
                    f"parent_{page}_{p_start}"
                )

                for c_start in range(
                    0,
                    len(parent_text),
                    child_step
                ):

                    child_text = parent_text[
                        c_start:c_start + child_size
                    ].strip()

                    if not child_text:
                        continue

                    chunks.append({
                        "id": f"child_{child_id}",
                        "text": child_text,
                        "page": page,
                        "parent_id": parent_id,
                        "parent_text": parent_text
                    })

                    child_id += 1

        logger.info(
            f"Parent-child generated {len(chunks)} chunks."
        )

        return chunks

    except Exception as e:
        logger.error(
            f"Parent-child chunking failed: {e}"
        )

        return []