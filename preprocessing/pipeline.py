# preprocessing/pipeline.py
"""
이미지/비정형형 PDF → OCR + LLM 교정 파이프라인 (PyMuPDF 제거 버전)

- PDF → 이미지: pdf2image 사용
- 텍스트 영역 감지: PaddleOCR(det)
- OCR: Tesseract (crop 기반)
- LLM 교정: safe_llm_correct
"""

import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

# LLM 안전 교정 래퍼
from preprocessing.llm.llm_correction import safe_llm_correct

logger = logging.getLogger("pipeline")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

# ------------------------------------------------------
# PaddleOCR Detector
# ------------------------------------------------------
try:
    from paddleocr import PaddleOCR
    OCR_DETECTOR = PaddleOCR(lang="korean", use_angle_cls=False)
    DET_AVAILABLE = True
    logger.info("[INIT] PaddleOCR detector ready")
except Exception as e:
    OCR_DETECTOR = None
    DET_AVAILABLE = False
    logger.warning(f"[INIT] PaddleOCR detector unavailable → {e}")

# ------------------------------------------------------
# 텍스트 품질 평가
# ------------------------------------------------------
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
    return max(0.0, min(score, 1.0))

# ------------------------------------------------------
# OCR Pipeline
# ------------------------------------------------------
class PreprocessPipeline:

    def __init__(self):
        self.run_id = "RUN"
        self.detector = OCR_DETECTOR
        self.det_available = DET_AVAILABLE
        logger.info(f"[INIT] Pipeline (det={self.det_available}, tesseract=on)")

    # soft preprocess
    def soft_preprocess(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
        sharp = cv2.filter2D(arr, -1, kernel)
        gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray)

    # tesseract
    def _ocr_tesseract(self, img: Image.Image) -> Tuple[str, float]:
        psm_modes = [4, 6, 7]
        best_q = 0.0
        best_txt = ""

        for psm in psm_modes:
            config = f"--psm {psm} --oem 3 -c preserve_interword_spaces=1"
            try:
                txt = pytesseract.image_to_string(img, lang="kor+eng", config=config)
                q = text_quality_score(txt)
                if q > best_q:
                    best_q = q
                    best_txt = txt.strip()
            except Exception:
                continue

        return best_txt, best_q

    # 영역 감지
    def _detect_text_regions(self, page_bgr):
        if not self.det_available:
            return []

        try:
            result = self.detector.ocr(page_bgr, rec=False, cls=False)
        except Exception:
            return []

        if not result:
            return []

        lines = result[0] if (isinstance(result, list) and len(result) == 1) else result

        boxes = []
        for line in lines:
            try:
                pts = np.array(line[0]).astype(np.int32)
                x, y, w, h = cv2.boundingRect(pts)
            except:
                continue

            if w < 60 or h < 25:
                continue
            boxes.append((x, y, w, h))

        boxes.sort(key=lambda b: (b[1], b[0]))
        return boxes

    # 한 페이지 처리
    def _run_page(self, img_pil: Image.Image, page_idx: int):
        arr = np.array(img_pil)
        page_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        boxes = self._detect_text_regions(page_bgr)
        region_texts = []

        # 영역별 OCR
        for x, y, w, h in boxes:
            crop = page_bgr[y:y+h, x:x+w]
            crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_proc = self.soft_preprocess(crop_pil)
            txt, q = self._ocr_tesseract(crop_proc)

            if q < 0.15 or len(txt.strip()) < 15:
                continue

            region_texts.append(txt)

        # fallback: 전체 페이지 OCR
        if not region_texts:
            proc = self.soft_preprocess(img_pil)
            best_txt, best_q = self._ocr_tesseract(proc)
            raw_text = best_txt
        else:
            raw_text = "\n\n".join(region_texts)
            best_q = text_quality_score(raw_text)

        corrected = safe_llm_correct(raw_text, page_idx)
        corrected_q = text_quality_score(corrected)

        meta = {
            "page": page_idx,
            "ocr_quality": float(best_q),
            "corrected_quality": float(corrected_q),
            "raw_len": len(raw_text),
            "corrected_len": len(corrected),
        }

        return corrected, meta

    # 전체 실행
    def run(self, input_pdf: str):
        pdf_path = Path(input_pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(pdf_path)

        logger.info(f"[START] PDF → image 변환: {pdf_path.name}")

        # PDF → 이미지 변환 (dpi=300~350 권장)
        pages = convert_from_path(str(pdf_path), dpi=300)

        all_pages = []
        chunks = []
        qualities = []

        for idx, img_pil in enumerate(pages):
            txt, meta = self._run_page(img_pil, idx)

            all_pages.append({"index": idx, "text": txt, "meta": meta})
            chunks.append({"chunk_index": idx, "text": txt, "quality": meta["corrected_quality"]})
            qualities.append(meta["corrected_quality"])

        avg_q = sum(qualities) / max(1, len(qualities))

        return {
            "run_id": self.run_id,
            "page_count": len(all_pages),
            "avg_quality": avg_q,
            "pages": all_pages,
            "chunks": chunks,
        }
