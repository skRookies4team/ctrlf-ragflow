# preprocessing/pipeline.py
import logging
from pathlib import Path

from ocr.engine_smart import SmartOCREngine
from preprocessing.llm.llm_correction import LLMCorrector
from preprocessing.chunking.chunker_page import chunk_page_level

logger = logging.getLogger("pipeline")


class PreprocessPipeline:

    def __init__(self):
        self.ocr = SmartOCREngine()
        self.llm = LLMCorrector()

    # -------------------------------------------------------
    def run(self, file_path: str):
        logger.info("======= PDF 처리 시작: %s =======", file_path)

        pages = self.ocr.run(file_path)
        total_pages = len(pages)
        logger.info("[START] PDF 페이지 수=%d", total_pages)

        llm_cleaned_pages = []

        # -------------------------------
        # PAGE LOOP (OCR → LLM cleaning)
        # -------------------------------
        for idx, p in enumerate(pages):
            logger.info("[LOOP] page %d start", idx)

            text = p["text"]
            quality = p["quality"]

            # LLM 교정
            cleaned_text, meta = self.llm.correct_page(
                text=text,
                page_idx=idx,
                quality=quality
            )

            logger.info(
                "[PAGE %d] LLM 교정 사용, rollback=%s, len=%d",
                idx, meta["rollback"], len(cleaned_text)
            )

            llm_cleaned_pages.append(cleaned_text)

        # -------------------------------
        # PAGE-LEVEL CHUNKING (각 페이지 1청크)
        # -------------------------------

        chunks = []
        for idx, txt in enumerate(llm_cleaned_pages):
            page_chunks = chunk_page_level(txt, idx)
            chunks.extend(page_chunks)

        logger.info("[DONE] Strong Mode 평균 품질 = %.3f",
                    sum(p["quality"] for p in pages) / total_pages)

        logger.info("→ PreprocessPipeline 완료")
        logger.info("→ 파이프라인 청크 %d개 반환", len(chunks))

        return {
            "pages": llm_cleaned_pages,
            "chunks": chunks
        }
