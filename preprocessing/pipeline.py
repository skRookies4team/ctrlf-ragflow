"""
Stable OCR Pipeline + LLM Correction
- Soft preprocess (light sharpen)
- 3 OCR(Tesseract/Easy/Paddle) 비교 후 최고 품질 선택
- Qwen LLM 자동 교정(오타/띄어쓰기/단어복원)
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

# --------------------------------------------
# LLM 자동 교정 가져오기
# --------------------------------------------
try:
    from preprocessing.llm.llm_correction import llm_correct_text
except Exception as e:
    print("[WARNING] llm_correct_text import 실패:", e)
    def llm_correct_text(x): 
        return x  # fallback


logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO)


# ------------------------------------------------
# 품질 스코어 계산 함수
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
# OCR Pipeline
# ------------------------------------------------
class PreprocessPipeline:
    def __init__(self):
        self.run_id = "RUN"

        # EasyOCR
        try:
            import easyocr
            self.easy_reader = easyocr.Reader(["ko", "en"], gpu=False)
            self.easy_available = True
        except:
            self.easy_reader = None
            self.easy_available = False

        # PaddleOCR
        try:
            from paddleocr import PaddleOCR
            self.paddle_reader = PaddleOCR(lang="korean", use_gpu=False)
            self.paddle_available = True
        except:
            self.paddle_reader = None
            self.paddle_available = False

        logger.info("[INIT] OCR Pipeline 준비 완료")

    # ------------------------------------------------
    # Pixmap → PIL 변환
    # ------------------------------------------------
    def _pixmap_to_pil(self, pix: fitz.Pixmap):
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    # ------------------------------------------------
    # Soft Preprocess (가벼운 sharpen)
    # ------------------------------------------------
    def soft_preprocess(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # 약한 sharpen 필터
        kernel = np.array([
            [0, -0.5, 0],
            [-0.5, 3, -0.5],
            [0, -0.5, 0],
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
        if not self.easy_available:
            return "", 0.0

        try:
            arr = np.array(img)
            result = self.easy_reader.readtext(arr, detail=0, paragraph=True)
            txt = "\n".join(result)
            return txt, text_quality_score(txt)
        except:
            return "", 0.0

    # ------------------------------------------------
    # PaddleOCR
    # ------------------------------------------------
    def _ocr_paddle(self, img):
        if not self.paddle_available:
            return "", 0.0

        try:
            arr = np.array(img)
            result = self.paddle_reader.ocr(arr)
            if not result or not result[0]:
                return "", 0.0

            lines = [line[1][0] for line in result[0]]
            txt = "\n".join(lines)
            return txt, text_quality_score(txt)
        except:
            return "", 0.0

    # ------------------------------------------------
    # 한 페이지 OCR + LLM 교정
    # ------------------------------------------------
    def _run_page(self, img: Image.Image):
        processed = self.soft_preprocess(img)

        tess_txt, tess_q = self._ocr_tesseract(processed)
        easy_txt, easy_q = self._ocr_easy(processed)
        paddle_txt, paddle_q = self._ocr_paddle(processed)

        results = [
            ("tesseract", tess_txt, tess_q),
            ("easyocr", easy_txt, easy_q),
            ("paddleocr", paddle_txt, paddle_q),
        ]

        best_engine, best_txt, best_q = max(results, key=lambda x: x[2])

        # ----------------------------------------
        # 🔥 LLM 자동 교정 (빈 텍스트 보호)
        # ----------------------------------------
        if best_txt.strip():
            try:
                corrected = llm_correct_text(best_txt)
            except Exception as e:
                print("[LLM ERROR] 교정 실패:", e)
                corrected = best_txt
        else:
            corrected = ""

        corrected_q = text_quality_score(corrected)

        return corrected, {
            "best_engine": best_engine,
            "ocr_quality": best_q,
            "corrected_quality": corrected_q,
            "ocr_len": len(best_txt),
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
        chunks = []
        qualities = []

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

            chunks.append({
                "chunk_index": idx,
                "text": text,
                "quality": meta["corrected_quality"]
            })

            qualities.append(meta["corrected_quality"])

            logger.info(f"[Page {idx}] engine={meta['best_engine']} corrected_q={meta['corrected_quality']:.3f}")

        avg_q = sum(qualities) / max(1, len(qualities))

        logger.info(f"전체 완료: 평균 품질 score={avg_q:.3f}")

        return {
            "run_id": self.run_id,
            "page_count": len(pages),
            "avg_quality": avg_q,
            "pages": pages,
            "chunks": chunks,
        }
