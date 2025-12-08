import pytesseract
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger("tesseract_ocr")


def run_tesseract(img, psm=6):
    """
    이미지 → Tesseract OCR 텍스트 반환
    img: OpenCV BGR 또는 PIL Image 지원
    """
    try:
        # OpenCV → PIL 변환
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        config = f"--psm {psm} --oem 1 -c preserve_interword_spaces=1"
        text = pytesseract.image_to_string(img, lang="kor+eng", config=config)
        return text.strip()

    except Exception as e:
        logger.error(f"[Tesseract OCR] 실패: {e}")
        return ""
