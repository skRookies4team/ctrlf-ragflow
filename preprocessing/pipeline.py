import logging
import json
import base64
import io
import re
import os
from pathlib import Path
from uuid import uuid4
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageOps
import pytesseract
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class PreprocessPipeline:
    def __init__(self):
        self.run_id = str(uuid4())
        logger.info(f"[Pipeline Init] run_id={self.run_id}")
        self.always_run_ocr = True  # 🔥 슬라이드 PDF 등 전 페이지 OCR 강제 실행

    # ---------------- 이미지 텍스트 포함 판단 ----------------
    def _contains_text(self, image: Image.Image) -> bool:
        """이미지에 텍스트 구조가 있는지 대략 판단"""
        try:
            gray = np.array(image.convert("L"), dtype=np.int32)
            diff = np.abs(np.diff(gray, axis=1))
            return np.mean(diff) > 4  # 5 → 4로 살짝 완화
        except Exception:
            return False

    # ---------------- 이미지 전처리 필요 판단 ----------------
    def _needs_preprocess(self, image: Image.Image) -> bool:
        """이미지 그래디언트 분산 기반으로 흐림 정도 판단"""
        try:
            gray = np.array(image.convert("L"), dtype=float)
            grads = np.gradient(gray)
            return np.var(grads) < 45  # 40 → 45로 완화
        except Exception:
            return True

    # ---------------- 이미지 전처리 적용 ----------------
    def _apply_preprocess(self, image: Image.Image) -> Image.Image:
        """글자/슬라이드 OCR 정확도 향상 최소 전처리"""
        try:
            img = ImageOps.exif_transpose(image)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = ImageEnhance.Contrast(img).enhance(1.6)
            img = ImageEnhance.Sharpness(img).enhance(1.3)
            return img
        except Exception as e:
            logger.warning(f"[Image Preprocess fail]: {e}")
            return image

    # ---------------- Adaptive 이진화 ----------------
    def _adaptive_binarize(self, img: Image.Image) -> Image.Image:
        """과도하게 뭉개지는 이진화 방지 + 글자 보존 강화"""
        try:
            im = img.convert("L")
            hist = im.histogram()
            total = sum(hist)
            if total == 0:
                return im
            mean = sum(i * hist[i] for i in range(256)) / total
            var = sum(((i - mean) ** 2) * hist[i] for i in range(256)) / total
            std = var ** 0.5
            thresh = int(max(120, min(200, mean + 0.3 * std)))
            return im.point(lambda p: 255 if p > thresh else 0)
        except Exception:
            return img

    # ---------------- 이미지 OCR용 전처리 ----------------
    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """OCR 이미지 전처리 + 이진화"""
        img = self._apply_preprocess(image)
        img = self._adaptive_binarize(img)
        return img


    # ---------------- 텍스트 영역 탐지(옵션인데 당장 안써도됨) ----------------
    def _detect_text_regions(self, image_gray_np: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """텍스트가 있을만한 영역을 분할 탐지 (필요시 활용가능)"""
        h, w = image_gray_np.shape
        boxes = []
        step_h, step_w = h // 2, w // 2
        for y in range(0, h, step_h):
            for x in range(0, w, step_w):
                x2, y2 = min(x + step_w, w), min(y + step_h, h)
                crop = image_gray_np[y:y2, x:x2]
                if crop.size == 0:
                    continue
                if np.var(crop) > 16:
                    boxes.append((x, y, x2, y2))
        return boxes

    # ---------------- 영역 기반 OCR(옵션) ----------------
    def _ocr_from_image_regions(self, img: Image.Image, boxes: List[Tuple[int, int, int, int]]) -> str:
        """영역 단위 OCR 실행"""
        text = ""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            try:
                region = img.crop((x1, y1, x2, y2))
                ocr = pytesseract.image_to_string(region, lang="kor+eng").strip()
                if ocr:
                    text += f"\n--- OCR region {i} ---\n" + ocr
            except Exception:
                logger.warning(f"[OCR region fail] {i}")
        return text.strip()

    # ---------------- PDF 텍스트 추출 ----------------
    def extract_text_from_pdf(self, pdf_path: Path) -> List[str]:
        """PyMuPDF로 페이지별 TEXT 추출"""
        texts = []
        try:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                t = page.get_text("text") or ""
                texts.append(t.strip())
                logger.info(f"[PDF TEXT] page={i}, len={len(t.strip())}")
            doc.close()
        except Exception as e:
            logger.error("[PDF open fail]", e)
        return texts

    # ---------------- PDF OCR 추출 ----------------
    def extract_ocrs_from_pdf(self, pdf_path: Path, min_len_skip: int = 35) -> List[str]:
        """PDF->이미지->OCR 실행"""
        ocrs = []
        try:
            doc = fitz.open(str(pdf_path))
            for i, page in enumerate(doc):
                txt = page.get_text("text").strip()
                if len(txt) < min_len_skip or self.always_run_ocr:
                    pix = page.get_pixmap(dpi=450, alpha=False)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    img = self.preprocess_image_for_ocr(img)
                    ocr_text = pytesseract.image_to_string(img, lang="kor+eng",
                                config=r"--psm 4 --oem 1 -c preserve_interword_spaces=1").strip()
                    ocrs.append(ocr_text)
                else:
                    ocrs.append("")
                logger.info(f"[OCR TEXT] page={i}, len={len(ocr_text)}")
            doc.close()
        except Exception as e:
            logger.error("OCR extract fail %s", e)
        return ocrs

    # ---------------- 텍스트 품질 점수 ----------------
    def _text_quality_score(self, t: str) -> float:
        if not t:
            return 0.0
        s = t.strip()
        length_score = min(len(s) / 900.0, 1.0)
        alpha_count = len(re.findall(r"[A-Za-z가-힣0-9]", s))
        noise_count = len(re.findall(r"[^0-9A-Za-z가-힣\s\.,\-\(\)\!?\":'%#]", s))
        alpha_ratio = alpha_count / max(1, len(s))
        penalty = min(noise_count / max(1, len(s)), 0.35)
        return length_score * (0.7 + 0.3 * alpha_ratio) * (1.0 - penalty)

    # ---------------- PDF Text + OCR 병합 ----------------
    def merge_page_texts(self, py_text: str, ocr_text: str) -> str:
        py = py_text.strip()
        ocr = ocr_text.strip()
        if not py and not ocr:
            return ""
        py_score = self._text_quality_score(py)
        ocr_score = self._text_quality_score(ocr)
        if py_score >= ocr_score * 1.1:
            return py
        if ocr_score >= py_score * 1.1:
            return ocr
        merged = []
        seen = set()
        for line in (py.splitlines() + ocr.splitlines()):
            l = line.strip()
            if l and l not in seen:
                merged.append(l)
                seen.add(l)
        return "\n".join(merged)

    def merge_pdf_texts(self, py_texts: List[str], ocr_texts: List[str]) -> str:
        pages = []
        for py, ocr in zip(py_texts, ocr_texts):
            merged_page = self.merge_page_texts(py, ocr)
            if merged_page.strip():
                pages.append(merged_page)
        return "\n\n===PAGE_BREAK===\n\n".join(pages)

    # ---------------- 노이즈 제거 ----------------
    def remove_noise_patterns(self, text: str) -> str:
        if not text:
            return ""

        # 슬라이드용 relaxed noise 패턴 리스트
        relaxed_patterns = [r"={5,}", r"-{5,}", r"_{5,}", r"\|{5,}", r"\n{4,}", r"[ \t]{3,}"]

        for p in relaxed_patterns:
            text = re.sub(p, " ", text)

        text = re.sub(r"[ ]{2,}", " ", text)
        return text.strip()

    # ---------------- Word 단위 청킹 (슬라이드 friendly) ----------------
    def safe_smart_chunk(self, text: str, max_len: int = 1200) -> List[str]:
        if not text:
            return []
        chunks = []
        buf = ""
        for line in text.splitlines():
            if not line.strip():
                continue
            if not buf:
                buf = line
            elif len(buf) + len(line) + 1 <= max_len:
                buf += " " + line
            else:
                chunks.append(buf.strip())
                buf = line
        if buf:
            chunks.append(buf.strip())
        return [c for c in chunks if len(c) > 15]

    # ---------------- 실행 메인 ----------------
    def run(self, input_pdf: str, chunk_size: int = 1200) -> Dict[str, Any]:
        pdf_path = Path(input_pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"❌ PDF not found: {input_pdf}")

        py_texts = self.extract_text_from_pdf(pdf_path)
        ocr_texts = self.extract_ocrs_from_pdf(pdf_path)

        merged = self.merge_pdf_texts(py_texts, ocr_texts)
        clean = self.remove_noise_patterns(merged)
        chunks = self.safe_smart_chunk(clean, max_len=chunk_size)

        payload = {
            "meta": {"run_id": self.run_id},
            "num_chunks": len(chunks),
            "chunks": [{"index": i, "text": c} for i, c in enumerate(chunks)],
        }

        out_path = Path("pipeline_result.json")
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[저장 완료] {out_path}")

        return {
            "doc_type": "slide-friendly",
            "run_id": self.run_id,
            "num_chunks": len(chunks),
            "result_json": payload,
        }
