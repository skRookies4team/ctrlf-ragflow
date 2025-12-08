import logging
import io
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import pytesseract
import cv2 

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO)

# ------------------------------------------------
# EasyOCR (선택)
# ------------------------------------------------
try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    easyocr = None
    _EASYOCR_AVAILABLE = False
    logger.info("[OCR] easyocr 미설치 – Tesseract만 사용합니다.")


# ------------------------------------------------
# 텍스트 품질 점수 업그레이드 (정확도 ↑)
# ------------------------------------------------
def text_quality_score(t: str) -> float:
    """OCR 결과 텍스트 품질 점수 (0~1)"""
    if not t:
        return 0.0
    s = t.strip()
    if not s:
        return 0.0

    length_score = min(len(s) / 900.0, 1.0)

    alpha_count = len(re.findall(r"[A-Za-z가-힣0-9]", s))
    noise_count = len(re.findall(r"[^0-9A-Za-z가-힣\s\.,\-\(\)\!?\":'%#]", s))

    alpha_ratio = alpha_count / max(len(s), 1)
    noise_ratio = noise_count / max(len(s), 1)

    # 노이즈 penalty 강화
    penalty = min(noise_ratio * 1.8, 0.45)

    # 반복문자 패널티 강화
    repeated = len(re.findall(r"(.)\1{4,}", s))
    rep_penalty = min(repeated * 0.05, 0.25)

    score = (0.7 * length_score + 0.3 * alpha_ratio) * (1.0 - penalty) * (1.0 - rep_penalty)
    return float(max(0.0, min(score, 1.0)))


# ------------------------------------------------
# Main Pipeline
# ------------------------------------------------
class PreprocessPipeline:
    def __init__(self):
        self.run_id = "RUN"
        self.easyocr_reader: Optional[Any] = None

        if _EASYOCR_AVAILABLE:
            try:
                self.easyocr_reader = easyocr.Reader(["ko", "en"], gpu=False)
            except:
                self.easyocr_reader = None

        logger.info(f"[Pipeline Init] run_id={self.run_id}")

    # ---------------- 이미지 변환 ----------------
    def _pixmap_to_pil(self, pix: fitz.Pixmap) -> Image.Image:
        img_bytes = pix.tobytes("png")
        return Image.open(io.BytesIO(img_bytes))

    # ---------------- 이미지 품질 진단 ----------------
    def _is_blurry(self, img: Image.Image) -> bool:
        try:
            gray = np.array(img.convert("L"), dtype=float)
            g = np.gradient(gray)
            v = float(np.var(g))
            return v < 55.0
        except:
            return False

    def _is_low_contrast(self, img: Image.Image) -> bool:
        try:
            gray = np.array(img.convert("L"), dtype=float)
            return float(np.std(gray)) < 35.0
        except:
            return False

    # ---------------- Light Preprocess 강화 ----------------
    def _light_preprocess(self, img: Image.Image) -> Image.Image:
        try:
            im = ImageOps.exif_transpose(img)

            im = im.filter(ImageFilter.MedianFilter(size=3))
            im = ImageEnhance.Sharpness(im).enhance(1.6)
            im = ImageEnhance.Contrast(im).enhance(1.35)
            im = ImageOps.autocontrast(im)

            return im
        except:
            return img

    # ---------------- Heavy Preprocess 강화 ----------------
    def _heavy_preprocess(self, img: Image.Image) -> Image.Image:
        try:
            im = ImageOps.exif_transpose(img).convert("L")
            w, h = im.size
            im = im.resize((int(w * 1.8), int(h * 1.8)), Image.LANCZOS)

            # Bilateral blur (엣지 유지하면서 노이즈 제거)
            arr = np.array(im)
            arr = cv2.bilateralFilter(arr, d=7, sigmaColor=50, sigmaSpace=50)
            im = Image.fromarray(arr)

            im = ImageEnhance.Sharpness(im).enhance(2.2)

            # Adaptive threshold
            hist = im.histogram()
            total = sum(hist)
            mean = sum(i * hist[i] for i in range(256)) / (total + 1)
            var = sum(((i - mean) ** 2) * hist[i] for i in range(256)) / (total + 1)
            std = var ** 0.5

            thresh = int(max(90, min(200, mean + 0.25 * std)))
            im = im.point(lambda p: 255 if p > thresh else 0)

            return im
        except:
            return img

    # ---------------- PSM 자동 선택 ----------------
    def _auto_psm(self, img: Image.Image) -> int:
        """문서 형태 기반 PSM 자동 선택"""
        w, h = img.size
        if h > w * 1.2:
            return 11  # block text detection
        return 6

    # ---------------- OCR: Tesseract ----------------
    def _ocr_tesseract(self, img: Image.Image, psm: Optional[int] = None) -> str:
        try:
            if psm is None:
                psm = self._auto_psm(img)

            cfg = f"--psm {psm} --oem 1 -c preserve_interword_spaces=1"
            return pytesseract.image_to_string(img, lang="kor+eng", config=cfg).strip()
        except:
            return ""

    # ---------------- OCR: EasyOCR ----------------
    def _ocr_easyocr(self, img: Image.Image) -> str:
        try:
            if self.easyocr_reader is None:
                return ""
            arr = np.array(img.convert("RGB"))
            results = self.easyocr_reader.readtext(arr, detail=0, paragraph=True)
            return "\n".join(r.strip() for r in results if r.strip())
        except:
            return ""

    # ---------------- 단일 페이지 처리 ----------------
    def _run_page_ocr(self, img: Image.Image) -> Tuple[str, Dict[str, Any]]:
        blur = self._is_blurry(img)
        low_contrast = self._is_low_contrast(img)

        # Light + Tess
        light = self._light_preprocess(img)
        txt_light = self._ocr_tesseract(light)
        q_light = text_quality_score(txt_light)

        best_text, best_q = txt_light, q_light
        best_engine = "tesseract_light"
        used_heavy = False
        used_easy = False

        # Heavy 필요 조건
        if best_q < 0.65 or blur or low_contrast:
            heavy = self._heavy_preprocess(img)
            txt_heavy = self._ocr_tesseract(heavy)
            q_heavy = text_quality_score(txt_heavy)

            if q_heavy > best_q:
                best_q = q_heavy
                best_text = txt_heavy
                best_engine = "tesseract_heavy"
                used_heavy = True
        else:
            q_heavy = 0.0

        # EasyOCR
        if self.easyocr_reader is not None and best_q < 0.80:
            txt_easy = self._ocr_easyocr(img)
            q_easy = text_quality_score(txt_easy)

            if q_easy > best_q:
                best_q = q_easy
                best_text = txt_easy
                best_engine = "easyocr"
                used_easy = True
        else:
            txt_easy = ""
            q_easy = 0.0

        info = {
            "blur": blur,
            "low_contrast": low_contrast,
            "q_light": q_light,
            "q_heavy": q_heavy,
            "q_easy": q_easy,
            "best_quality": best_q,
            "best_engine": best_engine,
            "used_heavy": used_heavy,
            "used_easyocr": used_easy,
            "ocr_len": len(best_text),
        }

        return best_text, info

    # ---------------- 전체 PDF 처리 ----------------
    def run(self, input_pdf: str) -> Dict[str, Any]:
        pdf_path = Path(input_pdf)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))

        pages = []
        qualities = []

        for i, page in enumerate(doc):
            try:
                pix = page.get_pixmap(dpi=260, alpha=False)
                img = self._pixmap_to_pil(pix)

                text, info = self._run_page_ocr(img)
                info["page_index"] = i

                pages.append({
                    "index": i,
                    "text": text,
                    "langeffect": info,
                })
                qualities.append(info["best_quality"])

                logger.info(
                    f"[Page {i}] engine={info['best_engine']} "
                    f"q={info['best_quality']:.3f} len={info['ocr_len']}"
                )
            except Exception as e:
                logger.error(f"[Page {i}] 처리 실패: {e}")
                pages.append({
                    "index": i,
                    "text": "",
                    "langeffect": {"page_index": i, "error": str(e)},
                })
                qualities.append(0.0)

        doc.close()

        avg_q = sum(qualities) / max(1, len(qualities))
        result = {
            "run_id": self.run_id,
            "page_count": len(pages),
            "avg_quality": avg_q,
            "pages": pages,
        }

        logger.info(f"[Pipeline 완료] avg_quality={avg_q:.3f}")
        return result
 