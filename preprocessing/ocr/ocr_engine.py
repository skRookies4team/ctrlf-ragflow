import pytesseract
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2

class OCREngine:
    @staticmethod
    def run(img: Image.Image) -> str:
        """
        OCR 품질 향상을 위한 이미지 전처리 + Tesseract OCR
        """

        # PIL -> OpenCV 변환
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 1) Grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 2) 가우시안 블러 (노이즈 제거)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # 3) 이진화(Otsu)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 4) 글자 색 반전(흰바탕 + 검은글자 형태로 맞춤)
        th = 255 - th

        # 5) 이미지 선명도 강화
        pil_img = Image.fromarray(th).filter(ImageFilter.SHARPEN)

        # 6) 명암 대비 올리기
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.0)

        # OCR 실행
        text = pytesseract.image_to_string(
            pil_img,
            lang="kor+eng",
            config="--psm 6"
        )
        return text
