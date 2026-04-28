import re
from src.utils.logger import get_logger

logger = get_logger("SectionAwareChunk")


SECTION_PATTERNS = [
    r"\n[A-Z][A-Z\s]{3,}\n",
    r"\n\d+\.\s+.*",
    r"\nChapter\s+\d+.*",
    r"\nSection\s+\d+.*"
]


def split_by_sections(text):
    boundaries = []

    for pattern in SECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            boundaries.append(match.start())

    boundaries = sorted(list(set(boundaries)))

    if not boundaries:
        return [text]

    sections = []

    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        section = text[start:end].strip()

        if section:
            sections.append(section)

    return sections


def section_aware_chunk(docs):
    """
    Splits documents based on structural sections/headings.
    """

    chunks = []

    for doc in docs:
        try:
            text = doc.get("text", "")
            page = doc.get("page", 0)

            sections = split_by_sections(text)

            for sec in sections:
                chunks.append({
                    "text": sec,
                    "page": page
                })

        except Exception as e:
            logger.warning(f"Section-aware chunking failed: {e}")

    return chunks