# preprocessing/pdf/pdf_type_detector.py
import fitz

class PDFTypeDetector:
    @staticmethod
    def detect(pdf_path: str, text_threshold: int = 25):
        doc = fitz.open(pdf_path)
        text_pages = 0
        image_pages = 0

        for page in doc:
            t = page.get_text().strip()
            if len(t) >= text_threshold:
                text_pages += 1
            else:
                image_pages += 1

        if text_pages == 0:
            return "image"
        if image_pages == 0:
            return "text"
        if image_pages > text_pages:
            return "mixed-image"
        return "mixed-text"
