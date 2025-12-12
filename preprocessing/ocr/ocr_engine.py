import pytesseract
import numpy as np
from PIL import Image
import cv2

class OCREngine:
    @staticmethod
    def run(img: Image.Image) -> str:
        """
        OCR 품질 향상을 위한 *최소한의* 전처리 + Tesseract OCR
        - PDF 슬라이드처럼 이미 깨끗한 이미지 기준
        """

        # 1) PIL -> OpenCV (혹시 컬러이면 그대로 유지)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # 2) Grayscale
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 3) 너무 작은 이미지는 리사이즈 (텍스트가 작을 때만 확대)
        h, w = gray.shape
        if w < 1200:
            scale = 1200 / w
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)

        # 4) PIL 이미지로 다시 변환 (여기서는 *강한* 이진화/반전 안 함)
        pil_img = Image.fromarray(gray)

        # 5) Tesseract 옵션 (CLI에서 잘 나왔던 것과 맞추기)
        config = (
            "--oem 1 "               # LSTM 엔진
            "--psm 6 "               # 한 페이지 내 여러 줄 텍스트
            "-c preserve_interword_spaces=1 "  # 띄어쓰기 보존
            "-c user_defined_dpi=300"          # DPI 힌트
        )

        text = pytesseract.image_to_string(
            pil_img,
            lang="kor",   # 필요하면 "kor+eng"로 바꿔도 됨
            config=config
        )
        return text.strip()
