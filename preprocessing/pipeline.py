"""
Stable OCR Pipeline (Optimized for Education/Training Slide PDFs)
- 원본 보존 중심
- 최소한의 전처리만 적용
- 3종 OCR(Tesseract/Easy/Paddle) 비교 후 최고 품질 자동 선택
"""

import logging
import io
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import fitz
import numpy as np
from PIL import Image
import pytesseract
import cv2

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------
# EasyOCR
# ------------------------------------------------
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except:
    easyocr = None
    _EASYOCR_AVAILABLE = False

# ------------------------------------------------
# PaddleOCR
# ------------------------------------------------
try:
    from paddleocr import PaddleOCR
    _PADDLE_AVAILABLE = True
except:
    PaddleOCR = None
    _PADDLE_AVAILABLE = False


# ------------------------------------------------
# 품질 스코어
# ------------------------------------------------
def text_quality_score(t: str) -> float:
    if not t or not t.strip():
        return 0.0

    s = t.strip()
    length_score = min(len(s) / 900.0, 1.0)

    alpha_count = len(re.findall(r"[A-Za-z가-힣0-9]", s))
    noise_count = len(re.findall(r"[^0-9A-Za-z가-힣\s\.,\-\(\)\!?\":'%#]", s))

    alpha_ratio = alpha_count / max(len(s), 1)
    noise_ratio = noise_count / max(len(s), 1)

    penalty = min(noise_ratio * 2.0, 0.45)
    repeat_penalty = len(re.findall(r"(.)\1{4,}", s)) * 0.05

    score = (0.7 * length_score + 0.3 * alpha_ratio)
    score *= (1 - penalty) * (1 - min(repeat_penalty, 0.3))

    return float(max(0, min(score, 1)))


# ------------------------------------------------
# Pipeline
# ------------------------------------------------
class PreprocessPipeline:
    def __init__(self):
        self.run_id = "RUN"

        # EasyOCR
        self.easy_reader = None
        if _EASYOCR_AVAILABLE:
            try:
                self.easy_reader = easyocr.Reader(["ko", "en"], gpu=False)
            except:
                self.easy_reader = None

        # PaddleOCR
        self.paddle_reader = None
        if _PADDLE_AVAILABLE:
            try:
                self.paddle_reader = PaddleOCR(lang="korean", use_gpu=False)
            except:
                self.paddle_reader = None

    # ------------------------------------------------
    # Pixmap → PIL
    # ------------------------------------------------
    def _pixmap_to_pil(self, pix: fitz.Pixmap):
        return Image.open(io.BytesIO(pix.tobytes("png")))

    # ------------------------------------------------
    # 소프트 전처리 (너 PDF에 최적화)
    # ------------------------------------------------
    def soft_preprocess(self, img: Image.Image) -> Image.Image:
        """
        - threshold / CLAHE / 강한 sharpen 제거
        - 글자를 최대한 유지하면서 가볍게 선명화
        """
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # light sharpen (약한 화질 향상)
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0]
        ])
        sharp = cv2.filter2D(arr, -1, kernel)

        gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray)

    # ------------------------------------------------
    # Tesseract
    # ------------------------------------------------
    def _ocr_tesseract(self, img):
        psm_modes = [4, 6, 7]

        best_q = 0
        best_txt = ""

        for psm in psm_modes:
            config = f"--psm {psm} --oem 3 -c preserve_interword_spaces=1"
            try:
                txt = pytesseract.image_to_string(img, lang="kor+eng", config=config)
                q = text_quality_score(txt)
                if q > best_q:
                    best_q = q
                    best_txt = txt.strip()
            except:
                continue

        return best_txt, best_q

    # ------------------------------------------------
    # EasyOCR
    # ------------------------------------------------
    def _ocr_easy(self, img):
        if not self.easy_reader:
            return "", 0.0
        try:
            arr = np.array(img.convert("RGB"))
            result = self.easy_reader.readtext(arr, detail=0, paragraph=True)
            txt = "\n".join(result)
            return txt, text_quality_score(txt)
        except:
            return "", 0.0

    # ------------------------------------------------
    # PaddleOCR
    # ------------------------------------------------
    def _ocr_paddle(self, img):
        if not self.paddle_reader:
            return "", 0.0
        try:
            arr = np.array(img.convert("RGB"))
            result = self.paddle_reader.ocr(arr)
            lines = [line[1][0] for line in result[0]]
            txt = "\n".join(lines)
            return txt, text_quality_score(txt)
        except:
            return "", 0.0

    # ------------------------------------------------
    # 페이지 OCR
    # ------------------------------------------------
    def _run_page(self, img: Image.Image):
        processed = self.soft_preprocess(img)

        tess_txt, tess_q = self._ocr_tesseract(processed)
        easy_txt, easy_q = self._ocr_easy(processed)
        paddle_txt, paddle_q = self._ocr_paddle(processed)

        results = [
            ("tesseract", tess_txt, tess_q),
            ("easyocr", easy_txt, easy_q),
            ("paddle", paddle_txt, paddle_q),
        ]
        best_engine, best_txt, best_q = max(results, key=lambda x: x[2])

        return best_txt, {
            "best_engine": best_engine,
            "best_quality": best_q,
            "ocr_len": len(best_txt),
            "scores": {
                "tesseract": tess_q,
                "easy": easy_q,
                "paddle": paddle_q,
            }
        }

    # ------------------------------------------------
    # 전체 PDF 처리
    # ------------------------------------------------
    def run(self, input_pdf: str):
        pdf_path = Path(input_pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))

        pages = []
        qualities = []
        chunks = []

        for idx, page in enumerate(doc):
            pix = page.get_pixmap(dpi=350, alpha=False)
            img = self._pixmap_to_pil(pix)

            text, meta = self._run_page(img)
            meta["page"] = idx

            pages.append({
                "index": idx,
                "text": text,
                "meta": meta
            })
            qualities.append(meta["best_quality"])

            chunks.append({
                "chunk_index": idx,
                "text": text,
                "quality": meta["best_quality"]
            })

            logger.info(f"[Page {idx}] best={meta['best_engine']} q={meta['best_quality']:.3f}")

        avg_q = sum(qualities) / max(1, len(qualities))
        logger.info(f"OCR 완료: avg_quality={avg_q:.3f}")

        return {
            "run_id": self.run_id,
            "page_count": len(pages),
            "avg_quality": avg_q,
            "pages": pages,
            "chunks": chunks,
        }
