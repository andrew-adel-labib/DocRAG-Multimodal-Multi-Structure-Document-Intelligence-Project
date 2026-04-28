import pytesseract
from PIL import Image


def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)

        text = pytesseract.image_to_string(
            img,
            config="--psm 6"
        )

        return clean_text(text)

    except Exception:
        return ""


def clean_text(text):
    text = " ".join(text.split())

    if len(text) < 3:
        return ""

    return text