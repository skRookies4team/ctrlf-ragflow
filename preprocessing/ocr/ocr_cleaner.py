import cv2
import numpy as np
from PIL import Image
import re


class OCRCleaner:

    # --------------------------------------------------------
    # 슬라이드 PDF / 이미지 PDF 최적화 전처리
    # --------------------------------------------------------
    def preprocess_image(self, pil_img: Image.Image) -> Image.Image:
        """
        슬라이드 PDF 최적화: grayscale → adaptive threshold →
        dilation → sharpen → resize(1.8x)
        """
        img = np.array(pil_img)

        # grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # adaptive threshold (슬라이드용 최적값)
        thr = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 10
        )

        # dilation (글자 테두리 강화)
        kernel = np.ones((2, 2), np.uint8)
        thr = cv2.dilate(thr, kernel, iterations=1)

        # sharpen
        kernel_sharp = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        thr = cv2.filter2D(thr, -1, kernel_sharp)

        # resize → OCR 인식률 증가
        thr = cv2.resize(thr, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

        return Image.fromarray(thr)

    # --------------------------------------------------------
    # 텍스트 후처리 (기본적인 노이즈 제거)
    # --------------------------------------------------------
    def postprocess_text(self, txt: str) -> str:
        if not txt:
            return ""

        t = txt

        # 제거: ASCII control, 이상한 유니코드, 잉여 공백
        t = re.sub(r"[^\S\r\n]+", " ", t)
        t = re.sub(r"[^\x09\x0A\x0D\x20-\x7E가-힣0-9.,!?():%\-/\n ]", "", t)

        # 여러 줄 개행 → 정리
        t = re.sub(r"\n{3,}", "\n\n", t)

        return t.strip()
