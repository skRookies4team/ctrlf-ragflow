# pipeline.py

import io
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

from preprocessing.ocr.ocr_cleaner import OCRCleaner
from preprocessing.llm.llm_correction import LLMCorrector

import concurrent.futures
import os

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception as e:
    EASY_AVAILABLE = False
    easyocr = None

logger = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

# ================================================================
# Super-Resolution (옵션)
# ================================================================
SR_AVAILABLE = False
SuperResolutionClass = None

try:
    from preprocessing.sr.super_resolution import SuperResolution as _SR
    SuperResolutionClass = _SR
    SR_AVAILABLE = True
    logger.info("[INIT] SuperResolution(RealESRGAN) 사용 가능")
except Exception as e:
    logger.warning(f"[INIT] SuperResolution 사용 불가 → 기본 업샘플 ({e})")
    SR_AVAILABLE = False
    SuperResolutionClass = None


# ================================================================
# 텍스트 품질 점수 / 한글 비율
# ================================================================
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
    return float(max(0.0, min(score, 1.0)))


def hangul_ratio(t: str) -> float:
    if not t:
        return 0.0
    total = len(t)
    if total == 0:
        return 0.0
    hangul = len(re.findall(r"[가-힣]", t))
    return hangul / total


# ================================================================
# 메인 파이프라인 클래스 (Strong Mode)
# ================================================================
class PreprocessPipeline:

    def __init__(self, dpi: int = 500,
                 save_debug_png: bool = False,
                 debug_dir: str = "debug_pages",
                 mode: str = "strong") -> None:
        """
        Strong Mode: Tesseract + EasyOCR 병합 + LLM 최소 교정
        """
        self.run_id = "RUN"
        self.dpi = dpi
        self.save_debug_png = save_debug_png
        self.debug_dir = Path(debug_dir)
        self.mode = mode

        if self.save_debug_png:
            self.debug_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = OCRCleaner()
        self.llm = LLMCorrector()

        # Super-Resolution
        if SR_AVAILABLE and SuperResolutionClass:
            try:
                self.sr = SuperResolutionClass(scale=2)
            except Exception as e:
                logger.warning(f"[INIT] SuperResolution 초기화 실패 → 기본 업샘플 사용 ({e})")
                self.sr = None
        else:
            self.sr = None

        # EasyOCR Reader
        if EASY_AVAILABLE:
            try:
                # GPU=False 로 고정 (환경에 따라 오류 방지)
                self.easy_reader = easyocr.Reader(["ko", "en"], gpu=False)
                logger.info("[INIT] EasyOCR Reader 초기화 완료")
            except Exception as e:
                logger.warning(f"[INIT] EasyOCR 초기화 실패 → Tesseract만 사용 ({e})")
                self.easy_reader = None
        else:
            self.easy_reader = None

        logger.info(
            f"[INIT] Strong Pipeline Ready (SR={'on' if self.sr else 'off'}, "
            f"DPI={self.dpi}, EasyOCR={'on' if self.easy_reader else 'off'})"
        )

    # ------------------------------------------------------------
    def _pixmap_to_pil(self, pix: fitz.Pixmap) -> Image.Image:
        # pixmap을 PNG 바이트로 뽑아서 PIL로 변환 → 사실상 PDF→PNG
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

    # ------------------------------------------------------------
    def _apply_super_resolution(self, pil_img: Image.Image) -> Image.Image:
        try:
            if self.sr:
                return self.sr.upscale(pil_img)
        except Exception as e:
            logger.warning(f"[SR] upscale 실패 → 기본 업샘플 사용 ({e})")

        w, h = pil_img.size
        # SR이 없으면 기본 2배 업샘플
        return pil_img.resize((w * 2, h * 2), Image.LANCZOS)

    # ------------------------------------------------------------
    def _ocr_tesseract_psm_cycle(self, img: Image.Image, lang: str) -> Tuple[str, float]:
        """
        여러 PSM을 돌려서 best 결과 선택 (한글 슬라이드 최적화)
        """
        best_q = 0.0
        best_txt = ""

        psm_orders_first = [6, 4]
        psm_orders_extra = [7, 11, 12]

        # 1차: 자주 잘 먹는 PSM
        for psm in psm_orders_first:
            config = f"--psm {psm} --oem 3 -c preserve_interword_spaces=1"
            try:
                txt = pytesseract.image_to_string(img, lang=lang, config=config)
                q = text_quality_score(txt)
                if q > best_q:
                    best_q = q
                    best_txt = txt.strip()
            except Exception as e:
                logger.warning(f"[OCR] Tesseract 실패 (lang={lang}, psm={psm}) → {e}")
                continue

        # 품질이 어느 정도 나오면 여기서 종료
        if best_q >= 0.75:
            return best_txt, best_q

        # 2차: 나머지 PSM까지 시도
        for psm in psm_orders_extra:
            config = f"--psm {psm} --oem 3 -c preserve_interword_spaces=1"
            try:
                txt = pytesseract.image_to_string(img, lang=lang, config=config)
                q = text_quality_score(txt)
                if q > best_q:
                    best_q = q
                    best_txt = txt.strip()
            except Exception as e:
                logger.warning(f"[OCR] Tesseract 실패 (lang={lang}, psm={psm}) → {e}")
                continue

        return best_txt, best_q

    # ------------------------------------------------------------
    def _ocr_tesseract_strong(self, img: Image.Image) -> Tuple[str, float]:
        """
        Strong 모드용 Tesseract:
        1) kor+eng
        2) 품질이 낮으면 kor-only fallback
        """
        txt_koeng, q_koeng = self._ocr_tesseract_psm_cycle(img, "kor+eng")

        # 한국어-only fallback: kor+eng 결과가 별로면 kor만
        txt_ko, q_ko = self._ocr_tesseract_psm_cycle(img, "kor")

        # 둘 다 완전 쓰레기면 일단 q 높은 쪽
        if q_koeng < 0.01 and q_ko < 0.01:
            return (txt_koeng if q_koeng >= q_ko else txt_ko), max(q_koeng, q_ko)

        # kor-only가 확실히 더 좋은 경우
        if q_ko >= q_koeng + 0.03:
            return txt_ko, q_ko

        # 그 외에는 ko+eng를 기본으로 사용
        return txt_koeng, q_koeng

    # ------------------------------------------------------------
    def _ocr_easyocr_strong(self, img: Image.Image) -> Tuple[str, float]:
        """
        Strong 모드용 EasyOCR:
        - 한글/영문 혼합 슬라이드에서 잘 먹는 편
        """
        if not self.easy_reader:
            return "", 0.0

        try:
            np_img = np.array(img)  # RGB
            # EasyOCR에서 detail=0, paragraph=True로 문단 단위 추출
            results = self.easy_reader.readtext(np_img, detail=0, paragraph=True)
            txt = "\n".join([r.strip() for r in results if isinstance(r, str) and r.strip()])
            q = text_quality_score(txt)
            return txt, q
        except Exception as e:
            logger.warning(f"[OCR] EasyOCR 실패 → {e}")
            return "", 0.0

    # ------------------------------------------------------------
    def _hybrid_merge_text(self,
                           tess_txt: str,
                           tess_q: float,
                           easy_txt: str,
                           easy_q: float) -> Tuple[str, float, str]:
        """
        Tesseract / EasyOCR 둘 다 돌린 뒤:
        - 품질 점수 + 한글 비율로 최종 선택
        """
        # 둘 다 텍스트 없으면 빈 문자열
        if not tess_txt and not easy_txt:
            return "", 0.0, "none"

        # 한글 비율
        t_hr = hangul_ratio(tess_txt)
        e_hr = hangul_ratio(easy_txt)

        # 기본 규칙:
        # 1) 품질 점수가 0.05 이상 차이 나면 점수 높은 쪽
        if tess_q >= easy_q + 0.05:
            return tess_txt, tess_q, "tesseract"
        if easy_q >= tess_q + 0.05:
            return easy_txt, easy_q, "easyocr"

        # 2) 점수가 비슷하면 한글 비율이 높은 쪽을 우선
        if e_hr >= t_hr + 0.10:
            # EasyOCR가 한글 비율이 확실히 높을 때
            chosen_txt = easy_txt
            chosen_q = easy_q
            engine = "easyocr"
        else:
            chosen_txt = tess_txt
            chosen_q = tess_q
            engine = "tesseract"

        # 3) 둘 다 품질이 너무 낮으면 한글 비율만 보고 선택
        if max(tess_q, easy_q) < 0.35:
            if e_hr > t_hr:
                return easy_txt, max(easy_q, tess_q), "easyocr"
            else:
                return tess_txt, max(tess_q, easy_q), "tesseract"

        return chosen_txt, chosen_q, engine

    # ------------------------------------------------------------
    def _run_page_strong(self, page, idx: int):
        """
        Strong Mode: 텍스트 레이어 → (SR) → Tesseract + EasyOCR → Hybrid → 클리닝 → LLM
        """
        logger.info(f"[LOOP] page {idx} start")

        # 0) 텍스트 레이어 먼저 시도 (있으면 그게 제일 깨끗함)
        extracted = ""
        try:
            extracted = page.get_text("text") or ""
        except Exception as e:
            logger.warning(f"[PAGE {idx}] text layer 추출 실패 → OCR로 진행 ({e})")

        if extracted.strip():
            # 텍스트 레이어에도 약간의 공백/줄바꿈 잡음이 있으니 클리닝 + LLM만
            cleaned = self.cleaner.postprocess_text(extracted)
            cleaned_q = text_quality_score(cleaned)

            # LLM 최소 교정
            corrected = self.llm.correct(cleaned, page_idx=idx, mode="strong")
            final = (corrected or cleaned).strip()
            final_q = text_quality_score(final)

            logger.info(
                f"[PAGE {idx}] TEXT_LAYER 사용, cleaned_q={cleaned_q:.3f}, final_q={final_q:.3f}, len={len(final)}"
            )

            meta = {
                "page": idx,
                "engine": "text_layer",
                "t_q": 1.0,
                "e_q": 0.0,
                "cleaned_q": float(cleaned_q),
                "final_q": float(final_q),
                "raw_len": len(extracted),
                "cleaned_len": len(cleaned),
                "final_len": len(final),
            }
            return final, meta

        # 1) PDF 페이지를 고해상도 PNG로 렌더링
        pix = page.get_pixmap(dpi=self.dpi, alpha=False)
        pil_image = self._pixmap_to_pil(pix)

        # 디버그용 PNG 저장 옵션
        if self.save_debug_png:
            debug_path = self.debug_dir / f"page_{idx:03d}.png"
            pil_image.save(debug_path)
            logger.info(f"[PAGE {idx}] debug PNG 저장 → {debug_path}")

        # 2) Super-Resolution 또는 기본 업샘플
        if self.sr:
            pil_for_ocr = self._apply_super_resolution(pil_image)
        else:
            # Strong 모드에서는 SR이 없어도 기본 1.5배 업샘플 적용 (가독성↑)
            w, h = pil_image.size
            pil_for_ocr = pil_image.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS)

        # 3) 이미지 전처리 (흑백+이진화 등)
        proc_img = self.cleaner.preprocess_image(pil_for_ocr)

        # 4) Tesseract / EasyOCR 모두 수행
        tess_txt, tess_q = self._ocr_tesseract_strong(proc_img)
        easy_txt, easy_q = self._ocr_easyocr_strong(proc_img)

        merged_txt, merged_q, engine = self._hybrid_merge_text(
            tess_txt, tess_q, easy_txt, easy_q
        )

        # 5) 클리닝
        cleaned = self.cleaner.postprocess_text(merged_txt or "")
        cleaned_q = text_quality_score(cleaned)

        # 6) LLM 최소 교정
        corrected = self.llm.correct(cleaned, page_idx=idx, mode="strong")
        final = (corrected or cleaned).strip()
        final_q = text_quality_score(final)

        logger.info(
            f"[PAGE {idx}] OCR done, engine={engine}, "
            f"t_q={tess_q:.3f}, e_q={easy_q:.3f}, "
            f"cleaned_q={cleaned_q:.3f}, final_q={final_q:.3f}, len={len(final)}"
        )

        meta = {
            "page": idx,
            "engine": engine,
            "t_q": float(tess_q),
            "e_q": float(easy_q),
            "cleaned_q": float(cleaned_q),
            "final_q": float(final_q),
            "raw_len": len(merged_txt or ""),
            "cleaned_len": len(cleaned),
            "final_len": len(final),
        }

        return final, meta

    # ------------------------------------------------------------
    def run(self, input_pdf: str) -> Dict[str, Any]:
        logger.info(f"[START] PDF → Strong OCR 처리 시작: {input_pdf}")

        pdf_path = Path(input_pdf)
        if not pdf_path.exists():
            logger.error(f"[ERROR] PDF 파일 없음: {pdf_path}")
            raise FileNotFoundError(pdf_path)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"[ERROR] PDF open 실패 ({pdf_path}) → {e}")
            raise

        page_count = len(doc)
        logger.info(f"[START] PDF 페이지 수={page_count}")

        pages: List[Dict[str, Any]] = []
        final_qualities: List[float] = []

        # Strong 모드는 페이지 수가 엄청 많지 않으니까 우선 순차 처리
        for i, page in enumerate(doc):
            try:
                txt, meta = self._run_page_strong(page, i)
            except Exception as e:
                logger.error(f"[ERR] page {i} OCR 처리 실패 → {e}")
                txt = ""
                meta = {
                    "page": i,
                    "engine": "error",
                    "t_q": 0.0,
                    "e_q": 0.0,
                    "cleaned_q": 0.0,
                    "final_q": 0.0,
                    "raw_len": 0,
                    "cleaned_len": 0,
                    "final_len": 0,
                }

            pages.append({"index": i, "text": txt, "meta": meta})
            final_qualities.append(meta.get("final_q", 0.0))

        # 평균 품질
        avg_q = sum(final_qualities) / max(1, len(final_qualities))
        logger.info(f"[DONE] Strong Mode 평균 품질 = {avg_q:.3f}")

        # index 기준 정렬
        pages.sort(key=lambda x: x["index"])

        return {
            "run_id": self.run_id,
            "page_count": page_count,
            "avg_quality": avg_q,
            "pages": pages,
            "chunks": [
                {
                    "chunk_index": p["index"],
                    "text": p["text"],
                    "quality": p["meta"]["final_q"],
                }
                for p in pages
            ],
        }
