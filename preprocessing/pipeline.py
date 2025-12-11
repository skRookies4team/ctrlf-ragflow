# pipeline.py
"""
PreprocessPipeline - PDF → (페이지 단위) 전처리 + OCR + LLM 교정 + 페이지 청킹

- 이미지/슬라이드형 PDF에 최적화
- 페이지별 OCR → 클리닝 → LLM 교정까지 처리
- 최종 결과는 pages + chunks(page-level) 구조로 반환
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF
from PIL import Image

from preprocessing.ocr.engine_smart import SmartOCREngine, text_quality_score
from preprocessing.llm.llm_correction import LLMCorrector

# Cleaner optional
try:
    from ocr.ocr_cleaner import OCRCleaner  # type: ignore
    _HAS_OCR_CLEANER = True
except Exception:
    OCRCleaner = None  # type: ignore
    _HAS_OCR_CLEANER = False

logger = logging.getLogger("pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:pipeline:%(message)s",
    )


# ============================================================
# 간단한 텍스트 클리너 (fallback)
# ============================================================
def _simple_clean_text(text: str) -> str:
    if not text:
        return ""

    s = text.replace("\t", " ")
    s = s.replace("~ ~", "~").replace("— —", "—")

    while "  " in s:
        s = s.replace("  ", " ")

    lines = [ln.rstrip() for ln in s.splitlines()]
    cleaned_lines = []

    for ln in lines:
        if len(ln.strip()) == 0:
            # 빈 줄은 하나만 허용
            if cleaned_lines and cleaned_lines[-1] == "":
                continue
            cleaned_lines.append("")
        else:
            cleaned_lines.append(ln)

    return "\n".join(cleaned_lines).strip()


# ============================================================
# PreprocessPipeline
# ============================================================
class PreprocessPipeline:
    def __init__(
        self,
        use_llm: bool = True,
        strong_mode: bool = True,
    ) -> None:
        self.strong_mode = strong_mode
        self.ocr = SmartOCREngine(use_easyocr=True, easyocr_gpu=False)
        self.llm = LLMCorrector()
        self.use_llm = use_llm

        if _HAS_OCR_CLEANER and OCRCleaner is not None:
            self.cleaner = OCRCleaner()
        else:
            self.cleaner = None

    # --------------------------------------------------------
    @staticmethod
    def _page_has_text_layer(page: "fitz.Page") -> Optional[str]:
        try:
            txt = page.get_text("text")
        except Exception:
            return None

        if not txt or not txt.strip():
            return None

        if len(txt.strip()) < 40:
            return None

        return txt

    # --------------------------------------------------------
    @staticmethod
    def _page_to_pil(page: "fitz.Page", zoom: float = 2.0) -> Image.Image:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img

    # --------------------------------------------------------
    def _clean_text(self, text: str) -> str:
        if self.cleaner is not None:
            if hasattr(self.cleaner, "postprocess_text"):
                return self.cleaner.postprocess_text(text)
            elif hasattr(self.cleaner, "clean"):
                return self.cleaner.clean(text)

        return _simple_clean_text(text)

    # --------------------------------------------------------
    def run(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = str(pdf_path)
        logger.info("======= PDF 처리 시작: %s =======", pdf_path)

        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        logger.info("[START] PDF 페이지 수=%d", page_count)

        pages_result: List[Dict[str, Any]] = []

        for page_idx in range(page_count):
            logger.info("[LOOP] page %d start", page_idx)
            page = doc.load_page(page_idx)

            # ---------------- 텍스트 레이어 ----------------
            text_layer = self._page_has_text_layer(page)
            if text_layer is not None:
                raw_text = text_layer
                engine = "text_layer"
                t_q = text_quality_score(raw_text)
                e_q = 0.0

                logger.info(
                    "[PAGE %d] text layer 사용, t_q=%.3f, len=%d",
                    page_idx,
                    t_q,
                    len(raw_text),
                )
            else:
                # ---------------- Strong OCR ----------------
                img = self._page_to_pil(page, zoom=2.0)
                raw_text, ocr_meta = self.ocr.strong_ocr(
                    img_for_ocr=img, page_idx=page_idx
                )

                engine = ocr_meta.get("engine", "tesseract")
                t_q = float(ocr_meta.get("t_q", 0.0))
                e_q = float(ocr_meta.get("e_q", 0.0))

                logger.info(
                    "[PAGE %d] OCR done, engine=%s, t_q=%.3f, e_q=%.3f, len=%d",
                    page_idx,
                    engine,
                    t_q,
                    e_q,
                    len(raw_text),
                )

            # ---------------- 1차 클리닝 --------------------
            clean_text = self._clean_text(raw_text)
            cleaned_q = text_quality_score(clean_text)

            # ---------------- 2차 LLM 교정 ------------------
            final_text = clean_text
            final_q = cleaned_q

            if self.use_llm:
                final_text, llm_meta = self.llm.correct_page(
                    clean_text,
                    page_idx=page_idx,
                    quality=cleaned_q,
                )
                if llm_meta.get("used_llm") and not llm_meta.get("rollback", False):
                    final_q = text_quality_score(final_text)

                if llm_meta.get("used_llm"):
                    logger.info(
                        "[PAGE %d] LLM 교정 사용, rollback=%s, len=%d",
                        page_idx,
                        llm_meta.get("rollback", False),
                        len(final_text),
                    )

            pages_result.append(
                {
                    "page": page_idx,
                    "text": final_text,
                    "raw_text": raw_text,
                    "clean_text": clean_text,
                    "engine": engine,
                    "t_q": float(t_q),
                    "e_q": float(e_q),
                    "cleaned_q": float(cleaned_q),
                    "final_q": float(final_q),
                }
            )

        doc.close()

        if pages_result:
            avg_quality = sum(p["final_q"] for p in pages_result) / len(pages_result)
        else:
            avg_quality = 0.0

        logger.info("[DONE] Strong Mode 평균 품질 = %.3f", float(avg_quality))
        logger.info("→ PreprocessPipeline 완료")

        # ============================================================
        # ⭐⭐ 페이지 단위 청킹 생성 (중요 ⭐⭐)
        # ============================================================
        chunks = []
        for p in pages_result:
            txt = p["text"].strip()
            if txt:
                chunks.append(
                    {
                        "text": txt,
                        "page_index": p["page"],
                    }
                )

        return {
            "pdf_path": pdf_path,
            "page_count": page_count,
            "pages": pages_result,
            "avg_quality": float(avg_quality),
            "chunks": chunks,     # ← 반드시 필요!!
        }


# ------------------------------------------------------------------
# 단독 실행 (테스트 용)
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("사용법: python pipeline.py <pdf_path>")
        sys.exit(1)

    pdf = sys.argv[1]
    pipeline = PreprocessPipeline()
    result = pipeline.run(pdf)

    print("\n===== PIPELINE RESULT =====")
    print("페이지 수:", result["page_count"])
    print("청크 수:", len(result["chunks"]))

    for c in result["chunks"][:2]:
        print("\n--- 미리보기 ---")
        print(c["text"][:500])
