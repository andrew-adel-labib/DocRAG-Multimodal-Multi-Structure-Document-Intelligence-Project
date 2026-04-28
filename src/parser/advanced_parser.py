import fitz
import os

from src.utils.logger import get_logger
from src.utils.exceptions import ParsingError

from src.embedding.clip_embedder import CLIPEmbedder

from src.utils.ocr_utils import extract_text_from_image
from src.utils.table_utils import extract_tables


logger = get_logger("AdvancedParser")

clip = CLIPEmbedder()


def extract_images(
    doc,
    page,
    page_index,
    save_dir="data/images"
):
    os.makedirs(
        save_dir,
        exist_ok=True
    )

    images_info = []

    try:
        image_list = page.get_images(
            full=True
        )

        for img_index, img in enumerate(
            image_list
        ):

            try:
                xref = img[0]

                base_image = doc.extract_image(
                    xref
                )

                image_bytes = base_image["image"]
                ext = base_image["ext"]

                img_name = (
                    f"page_{page_index}_img_{img_index}.{ext}"
                )

                img_path = os.path.join(
                    save_dir,
                    img_name
                )

                with open(
                    img_path,
                    "wb"
                ) as f:
                    f.write(image_bytes)

                emb = clip.encode_image(
                    img_path
                )

                ocr_text = extract_text_from_image(
                    img_path
                )

                if (
                    ocr_text
                    and len(ocr_text.strip()) < 3
                ):
                    ocr_text = ""

                if (
                    emb is not None
                    and len(emb.shape) == 1
                ):
                    images_info.append({
                        "path": img_path,
                        "embedding": emb,
                        "ocr": ocr_text
                    })

            except Exception as img_error:
                logger.warning(
                    f"Image {img_index} extraction failed on page {page_index}: {img_error}"
                )

    except Exception as e:
        logger.warning(
            f"Image extraction failed on page {page_index}: {e}"
        )

    return images_info


def parse_pdf_advanced(path):
    try:
        doc = fitz.open(path)

    except Exception as e:
        raise ParsingError(
            f"Failed to open PDF: {e}"
        )

    results = []

    try:
        table_texts_global = extract_tables(
            path
        )

    except Exception as e:
        logger.warning(
            f"Global table extraction failed: {e}"
        )

        table_texts_global = []

    for i, page in enumerate(doc):

        try:
            blocks = page.get_text(
                "blocks"
            )

            text_blocks = []

            for b in blocks:
                block_text = str(
                    b[4]
                ).strip()

                if block_text:
                    text_blocks.append(
                        block_text
                    )

            full_text = "\n".join(
                text_blocks
            )

            heuristic_tables = []

            for line in full_text.split("\n"):
                if "|" in line or "\t" in line:
                    heuristic_tables.append(
                        line.strip()
                    )

            images = extract_images(
                doc,
                page,
                i
            )

            ocr_texts = [
                img["ocr"]
                for img in images
                if img.get("ocr")
            ]

            combined_text = full_text

            if heuristic_tables:
                combined_text += (
                    "\n\nHEURISTIC TABLES:\n"
                    + "\n".join(
                        heuristic_tables
                    )
                )

            if table_texts_global:
                combined_text += (
                    "\n\nEXTRACTED TABLES:\n"
                    + "\n".join(
                        table_texts_global
                    )
                )

            if ocr_texts:
                combined_text += (
                    "\n\nOCR:\n"
                    + "\n".join(
                        ocr_texts
                    )
                )

            combined_text = " ".join(
                combined_text.split()
            )

            results.append({
                "page": i,
                "text": combined_text,
                "tables": (
                    heuristic_tables
                    + table_texts_global
                ),
                "images": images
            })

        except Exception as e:
            logger.error(
                f"Page {i} failed: {e}"
            )

    return results