# preprocessing/pdf/pdf_text_extractor.py
import fitz

class PDFTextExtractor:
    @staticmethod
    def extract(pdf_path: str) -> list[str]:
        doc = fitz.open(pdf_path)
        texts = []
        for page in doc:
            try:
                txt = page.get_text().strip()
            except:
                txt = ""
            texts.append(txt)
        return texts
