import fitz
import os
from src.utils.logger import get_logger
from src.utils.exceptions import ParsingError
from src.embedding.clip_embedder import CLIPEmbedder

logger = get_logger("AdvancedParser")

clip = CLIPEmbedder()


def extract_tables(text):
    tables, current = [], []

    for line in text.split("\n"):
        if "|" in line or "\t" in line:
            current.append(line)
        else:
            if current:
                tables.append(current)
                current = []

    if current:
        tables.append(current)

    return tables


def extract_images(doc, page, page_index, save_dir="data/images"):
    os.makedirs(save_dir, exist_ok=True)

    images_info = []

    try:
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)

            image_bytes = base_image["image"]
            ext = base_image["ext"]

            img_name = f"page_{page_index}_img_{img_index}.{ext}"
            img_path = os.path.join(save_dir, img_name)

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            emb = clip.encode_image(img_path)
            
            if emb is not None and len(emb.shape) == 1:
                images_info.append({
                    "path": img_path,
                    "embedding": emb
                })

            

    except Exception as e:
        logger.warning(f"Image extraction failed: {e}")

    return images_info


def parse_pdf_advanced(path):
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise ParsingError(f"Failed to open PDF: {e}")

    results = []

    for i, page in enumerate(doc):
        try:
            blocks = page.get_text("blocks")

            text_blocks = []
            for b in blocks:
                if b[4].strip():
                    text_blocks.append(b[4].strip())

            full_text = "\n".join(text_blocks)

            tables = extract_tables(full_text)
            table_texts = [" ".join(t) for t in tables]

            images = extract_images(doc, page, i)

            results.append({
                "page": i,
                "text": full_text + "\n" + "\n".join(table_texts),
                "tables": table_texts,
                "images": images
            })

        except Exception as e:
            logger.error(f"Page {i} failed: {e}")

    return results