import logging
from typing import Dict, Any
import numpy as np

from .tesseract_ocr import run_tesseract
from .paddle_ocr import run_paddle_ocr

logger = logging.getLogger("ensemble_ocr")


class EnsembleOCREngine:
    """
    여러 OCR 모델(Tesseract + PaddleOCR)을 실행하고
    품질 점수가 가장 높은 텍스트를 선택하는 Ensemble OCR 엔진
    """

    def __init__(self):
        self.engines = {
            "tesseract": run_tesseract,
            "paddle": run_paddle_ocr,
        }

    def score_text(self, text: str) -> float:
        """텍스트 품질 자동 점수"""
        if not text:
            return 0.0

        cleaned = text.strip()
        if not cleaned:
            return 0.0

        # 알파벳/한글/숫자 비율
        import re
        alpha = len(re.findall(r"[A-Za-z가-힣0-9]", cleaned))
        noise = len(re.findall(r"[^0-9A-Za-z가-힣\s\.,\-\(\)]", cleaned))

        alpha_ratio = alpha / max(len(cleaned), 1)
        noise_ratio = noise / max(len(cleaned), 1)

        score = alpha_ratio * (1 - min(noise_ratio, 0.4))
        return float(max(0.0, min(score, 1.0)))

    def run(self, img) -> Dict[str, Any]:
        """
        모든 OCR 엔진을 실행 → 최고 품질 결과 선택
        """

        results = {}
        for name, engine in self.engines.items():
            try:
                text = engine(img)
                score = self.score_text(text)

                results[name] = {
                    "text": text,
                    "score": score,
                }

                logger.info(f"[OCR] {name} score={score:.3f}")

            except Exception as e:
                logger.error(f"[OCR] {name} 실행 실패: {e}")

                results[name] = {
                    "text": "",
                    "score": 0.0,
                }

        # 최고 점수 선택
        best_engine = max(results, key=lambda k: results[k]["score"])
        best_text = results[best_engine]["text"]

        return {
            "best_engine": best_engine,
            "best_text": best_text,
            "engine_scores": results,
        }
