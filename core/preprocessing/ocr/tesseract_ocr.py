# preprocessing/ocr/tesseract_ocr.py
import pytesseract
from PIL import Image
import re

def quality_score(t: str) -> float:
    if not t.strip():
        return 0.0
    s = t.strip()
    alpha = len(re.findall(r"[A-Za-z가-힣0-9]", s))
    noise = len(re.findall(r"[^A-Za-z가-힣0-9\s\.,!?]", s))
    return max(0, min((alpha / (len(s)+1)) - noise * 0.01, 1.0))

class TesseractOCR:

    PSM_LIST = [6, 4, 7]

    def run(self, img: Image.Image):
        best = ("", 0)

        for psm in self.PSM_LIST:
            config = f"--psm {psm} --oem 3 -c preserve_interword_spaces=1"
            text = pytesseract.image_to_string(img, lang="kor+eng", config=config)
            q = quality_score(text)

            if q > best[1]:
                best = (text.strip(), q)

        return best
