# ocr/engine_smart.py
import logging
import re
from typing import Tuple, Dict, Any, Optional

import numpy as np
import pytesseract
from PIL import Image
import cv2

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

logger = logging.getLogger("smart_ocr")


# ============================================================
# 텍스트 품질 점수
# ============================================================
def text_quality_score(t: str) -> float:
    """
    OCR 결과 텍스트 품질을 0.0 ~ 1.0 사이로 스코어링.
    - 길이
    - 한글/영문/숫자 비율
    - 노이즈 문자 비율
    - 같은 글자 반복 패턴
    """
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


# ============================================================
# Smart OCR Engine (Tesseract + EasyOCR Fallback)
# ============================================================
class SmartOCREngine:
    """
    CPU 환경 기준 최적:
    - 기본: Tesseract + 이미지 전처리
    - 보조: 품질이 낮은 페이지에서만 EasyOCR Fallback
    """

    def __init__(
        self,
        use_easyocr: bool = True,
        easyocr_gpu: bool = False,
        easyocr_langs=None,
    ) -> None:
        self.use_easyocr = use_easyocr and EASY_AVAILABLE
        self.easyocr_gpu = easyocr_gpu
        self.easy_reader: Optional["easyocr.Reader"] = None

        if easyocr_langs is None:
            easyocr_langs = ["ko", "en"]
        self.easyocr_langs = easyocr_langs

        if not EASY_AVAILABLE and use_easyocr:
            logger.warning("[SmartOCR] easyocr 미설치 → Tesseract only 모드로 동작")

    # --------------------------------------------------------
    # 이미지 전처리 (슬라이드 PDF 기준)
    # --------------------------------------------------------
    @staticmethod
    def _prepare_image_for_ocr(img: Image.Image) -> Image.Image:
        """
        - Grayscale
        - 해상도 보정 (너무 작으면 확대)
        - CLAHE 대비 향상
        - 노이즈 제거
        - 큰 텍스트 블록만 크롭 (여백 제거)
        """
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 해상도 보정
        h, w = gray.shape
        target_min_width = 1500
        if w < target_min_width:
            scale = target_min_width / float(w)
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        # 약간의 블러로 노이즈 완화
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # CLAHE 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 적당한 이진화 (너무 aggressive 하지 않게)
        _, th = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 텍스트 블록 추출 (여백 제거)
        # 너무 공격적이면 안 되므로 "최대 컨투어"만 사용
        try:
            contours, _ = cv2.findContours(
                255 - th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                # 가장 큰 컨투어
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 0.25 * th.shape[0] * th.shape[1]:
                    th = th[y : y + h, x : x + w]
        except Exception:
            # 컨투어 실패해도 그냥 전체 사용
            pass

        return Image.fromarray(th)

    # --------------------------------------------------------
    def _init_easyocr(self):
        if not self.use_easyocr:
            return
        if self.easy_reader is not None:
            return
        try:
            logger.info(
                "[SmartOCR] EasyOCR Reader 초기화 중 (langs=%s, gpu=%s)",
                self.easyocr_langs,
                self.easyocr_gpu,
            )
            self.easy_reader = easyocr.Reader(
                self.easyocr_langs, gpu=self.easyocr_gpu
            )
            logger.info("[SmartOCR] EasyOCR Reader 초기화 완료")
        except Exception as e:
            logger.warning(
                "[SmartOCR] EasyOCR 초기화 실패 → Tesseract only (%s)", e
            )
            self.easy_reader = None
            self.use_easyocr = False

    # --------------------------------------------------------
    def _ocr_tesseract(self, img: Image.Image) -> Tuple[str, float]:
        """
        Tesseract 강화 버전:
        - PSM 4, 6
        - preserve_interword_spaces=1
        - 이미지 전처리 포함
        """
        pre_img = self._prepare_image_for_ocr(img)

        best_txt = ""
        best_q = 0.0

        for psm in [4, 6]:
            config = (
                f"--psm {psm} "
                "--oem 3 "
                "-c preserve_interword_spaces=1 "
                "-c user_defined_dpi=300"
            )
            try:
                txt = pytesseract.image_to_string(
                    pre_img, lang="kor+eng", config=config
                )
                q = text_quality_score(txt)
                if q > best_q:
                    best_q = q
                    best_txt = txt
            except Exception as e:
                logger.warning(
                    "[SmartOCR] Tesseract 실패 (psm=%d) → %s", psm, e
                )

        return best_txt.strip(), float(best_q)

    # --------------------------------------------------------
    def _ocr_easyocr(self, img: Image.Image) -> Tuple[str, float]:
        """
        EasyOCR는 fallback/보조 채널로 사용.
        """
        if not self.use_easyocr:
            return "", 0.0

        self._init_easyocr()
        if self.easy_reader is None:
            return "", 0.0

        try:
            np_img = np.array(img)
            result = self.easy_reader.readtext(
                np_img, detail=0, paragraph=True
            )
            text = "\n".join([r for r in result if isinstance(r, str)])
            q = text_quality_score(text)
            return text.strip(), float(q)
        except Exception as e:
            logger.warning("[SmartOCR] EasyOCR 실패 → %s", e)
            return "", 0.0

    # --------------------------------------------------------
    def strong_ocr(
        self,
        img_for_ocr: Image.Image,
        page_idx: int = -1,
        easy_fallback_threshold: float = 0.45,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Strong Mode:
        - 1차: 무조건 Tesseract 실행
        - 2차: Tesseract 품질이 threshold 미만이면 EasyOCR Fallback
        - 3차: 중간 품질(0.45~0.65)에서도 EasyOCR 비교 실행
        - 최종: 더 품질 좋은 쪽 채택
        """
        # 1) Tesseract
        t_text, t_q = self._ocr_tesseract(img_for_ocr)

        e_text = ""
        e_q = 0.0
        used_engine = "tesseract"

        # 2) 품질이 낮으면 EasyOCR fallback
        if t_q < easy_fallback_threshold and self.use_easyocr:
            logger.info(
                "[SmartOCR][PAGE %d] Tesseract 품질 %.3f < %.3f → EasyOCR Fallback 시도",
                page_idx,
                t_q,
                easy_fallback_threshold,
            )
            e_text, e_q = self._ocr_easyocr(img_for_ocr)
        elif self.use_easyocr and 0.45 <= t_q <= 0.65:
            # 중간 품질이면 비교용으로 EasyOCR도 실행
            logger.info(
                "[SmartOCR][PAGE %d] Tesseract 중간 품질 %.3f → EasyOCR 비교용 실행",
                page_idx,
                t_q,
            )
            e_text, e_q = self._ocr_easyocr(img_for_ocr)

        # 3) 최종 선택
        final_text = t_text
        if e_q > t_q + 0.05 and e_q >= 0.4:
            final_text = e_text
            used_engine = "easyocr"
        elif e_q > 0.0:
            used_engine = "tesseract+easyocr"

        meta = {
            "page": page_idx,
            "engine": used_engine,
            "t_q": float(t_q),
            "e_q": float(e_q),
            "raw_len": len(final_text or ""),
        }

        return final_text.strip(), meta
