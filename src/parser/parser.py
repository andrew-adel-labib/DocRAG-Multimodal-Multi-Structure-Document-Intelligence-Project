import fitz
from src.utils.logger import get_logger
from src.utils.exceptions import ParsingError

logger = get_logger("Parser")

def parse_pdf_basic(path):
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ParsingError(f"Cannot open PDF: {e}")

    results = []

    for i, page in enumerate(doc):
        try:
            text = page.get_text()
            results.append({"page": i, "text": text})
        except Exception as e:
            logger.error(f"Page {i} failed: {e}")

    return results