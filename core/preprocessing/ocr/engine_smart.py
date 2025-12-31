# ocr/engine_smart.py
import logging
import re
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image

try:
    import easyocr
    EASY_AVAILABLE = True
except Exception:
    EASY_AVAILABLE = False

# torch는 easyocr 내부에서 쓰지만, 여기서 쓰레드 제한용으로 optional
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

logger = logging.getLogger("smart_ocr")


# ============================================================
# OCR 텍스트 품질 점수
# ============================================================
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


class SmartOCREngine:
    """
    안정화 최종:
    - 기본: Tesseract
    - EasyOCR: "구제 가능" + "이미지 크기 안전" 조건에서만 실행
    - EasyOCR 입력은 강제 다운스케일 적용
    - EasyOCR readtext 전후로 시간 로그 / OOM 즉시 폴백
    """

    def __init__(
        self,
        use_easyocr: bool = True,
        easyocr_gpu: bool = False,
        easyocr_langs=None,
        #여기부터 '멈춤 방지' 튜닝 파라미터
        hard_skip_tq_below: float = 0.05,     # tesseract q가 이보다 낮으면 easyocr 아예 스킵
        easy_min_tq_to_try: float = 0.10,     # 너무 낮은 경우는 구제 불가 → easyocr 비추(스킵)
        easy_max_pixels: int = 2_500_000,     # easyocr에 넣는 최대 픽셀(예: 2.5MP)
        easy_max_side: int = 1600,            # 다운스케일 최대 한 변
        torch_num_threads: int = 1,           # CPU 폭주 방지(1~2 추천)
    ) -> None:
        self.use_easyocr = use_easyocr and EASY_AVAILABLE
        self.easyocr_gpu = easyocr_gpu
        self.easy_reader: Optional["easyocr.Reader"] = None

        if easyocr_langs is None:
            easyocr_langs = ["ko", "en"]
        self.easyocr_langs = easyocr_langs

        self.hard_skip_tq_below = hard_skip_tq_below
        self.easy_min_tq_to_try = easy_min_tq_to_try
        self.easy_max_pixels = easy_max_pixels
        self.easy_max_side = easy_max_side
        self.torch_num_threads = torch_num_threads

        if not EASY_AVAILABLE and use_easyocr:
            logger.warning("[SmartOCR] easyocr 미설치 → Tesseract only")

        # torch thread 제한(있으면 적용)
        if TORCH_AVAILABLE and self.use_easyocr:
            try:
                torch.set_num_threads(max(1, int(torch_num_threads)))
            except Exception:
                pass

    # --------------------------------------------------------
    @staticmethod
    def _prepare_image_for_ocr(img: Image.Image) -> Image.Image:
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        if w < 1500:
            scale = 1500 / float(w)
            gray = cv2.resize(
                gray,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(th)

    # --------------------------------------------------------
    @staticmethod
    def _downscale_for_easyocr(img: Image.Image, max_pixels: int, max_side: int) -> Image.Image:
        """
        EasyOCR은 큰 이미지에서 CPU/메모리 폭주로 '멈춘 것처럼' 보이는 경우가 잦음.
        그래서 입력 이미지를 안전하게 다운스케일.
        """
        if img.mode != "RGB":
            img = img.convert("RGB")

        w, h = img.size
        pixels = w * h

        # 1) 한 변 제한
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            nw, nh = int(w * scale), int(h * scale)
            img = img.resize((nw, nh), Image.BILINEAR)
            w, h = img.size
            pixels = w * h

        # 2) 픽셀 수 제한(추가로 한 번 더)
        if pixels > max_pixels:
            scale = (max_pixels / float(pixels)) ** 0.5
            nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR)

        return img

    # --------------------------------------------------------
    def _init_easyocr(self):
        if not self.use_easyocr or self.easy_reader is not None:
            return
        try:
            logger.info(
                "[SmartOCR] EasyOCR Reader 초기화 (langs=%s, gpu=%s)",
                self.easyocr_langs,
                self.easyocr_gpu,
            )
            self.easy_reader = easyocr.Reader(self.easyocr_langs, gpu=self.easyocr_gpu)
            logger.info("[SmartOCR] EasyOCR Reader 초기화 완료")
        except Exception as e:
            logger.warning("[SmartOCR] EasyOCR 초기화 실패 → %s", e)
            self.use_easyocr = False
            self.easy_reader = None

    # --------------------------------------------------------
    def _ocr_tesseract(self, img: Image.Image) -> Tuple[str, float]:
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
                txt = pytesseract.image_to_string(pre_img, lang="kor+eng", config=config)
                q = text_quality_score(txt)
                if q > best_q:
                    best_q = q
                    best_txt = txt
            except Exception as e:
                logger.warning("[SmartOCR] Tesseract 실패 → %s", e)

        return best_txt.strip(), float(best_q)

    # --------------------------------------------------------
    def _ocr_easyocr(self, img: Image.Image, page_idx: int) -> Tuple[str, float]:
        if not self.use_easyocr:
            return "", 0.0

        self._init_easyocr()
        if self.easy_reader is None:
            return "", 0.0

        # ✅ 핵심: easyocr 입력 다운스케일
        safe_img = self._downscale_for_easyocr(
            img, max_pixels=self.easy_max_pixels, max_side=self.easy_max_side
        )
        np_img = np.array(safe_img)

        try:
            t0 = time.time()
            logger.info(
                "[SmartOCR][PAGE %d] EasyOCR readtext start (w=%d,h=%d,pixels=%d)",
                page_idx,
                safe_img.size[0],
                safe_img.size[1],
                safe_img.size[0] * safe_img.size[1],
            )
            result = self.easy_reader.readtext(np_img, detail=0, paragraph=True)
            text = "\n".join([r for r in result if isinstance(r, str)])
            q = text_quality_score(text)
            logger.info(
                "[SmartOCR][PAGE %d] EasyOCR readtext done (sec=%.2f, q=%.3f, len=%d)",
                page_idx,
                time.time() - t0,
                q,
                len(text or ""),
            )
            return text.strip(), float(q)

        except (MemoryError, RuntimeError) as e:
            # PyTorch OOM / not enough memory / alloc 실패 등
            logger.warning("[SmartOCR][PAGE %d] EasyOCR OOM/RuntimeError → %s", page_idx, e)
            return "", 0.0
        except Exception as e:
            logger.warning("[SmartOCR][PAGE %d] EasyOCR 실패 → %s", page_idx, e)
            return "", 0.0

    # --------------------------------------------------------
    def strong_ocr(
        self,
        img_for_ocr: Image.Image,
        page_idx: int = -1,
        easy_fallback_threshold: float = 0.45,
    ) -> Tuple[str, Dict[str, Any]]:

        # 1) Tesseract
        t_text, t_q = self._ocr_tesseract(img_for_ocr)

        # 🔒 HARD CUTOFF 1: 구제 불가급이면 easyocr 자체를 스킵
        if t_q < self.hard_skip_tq_below:
            logger.info(
                "[SmartOCR][PAGE %d] t_q=%.3f < %.3f → EasyOCR 스킵(구제 불가)",
                page_idx, t_q, self.hard_skip_tq_below
            )
            return t_text.strip(), {
                "page": page_idx,
                "engine": "tesseract",
                "t_q": float(t_q),
                "e_q": 0.0,
                "skipped_easyocr": True,
                "raw_len": len(t_text or ""),
            }

        # 🔒 HARD CUTOFF 2: 너무 낮으면(의미 없는 이미지/QR/도형) easyocr로도 구제 어려움
        if t_q < self.easy_min_tq_to_try:
            logger.info(
                "[SmartOCR][PAGE %d] t_q=%.3f < %.3f → EasyOCR 비추(스킵)",
                page_idx, t_q, self.easy_min_tq_to_try
            )
            return t_text.strip(), {
                "page": page_idx,
                "engine": "tesseract",
                "t_q": float(t_q),
                "e_q": 0.0,
                "skipped_easyocr": True,
                "raw_len": len(t_text or ""),
            }

        e_text = ""
        e_q = 0.0
        used_engine = "tesseract"
        tried_easyocr = False

        # 2) EasyOCR 시도 조건
        if self.use_easyocr and t_q < easy_fallback_threshold:
            tried_easyocr = True
            logger.info(
                "[SmartOCR][PAGE %d] Tesseract 품질 %.3f → EasyOCR 시도",
                page_idx, t_q
            )
            e_text, e_q = self._ocr_easyocr(img_for_ocr, page_idx)

        elif self.use_easyocr and 0.45 <= t_q <= 0.65:
            tried_easyocr = True
            logger.info(
                "[SmartOCR][PAGE %d] 중간 품질 %.3f → EasyOCR 비교",
                page_idx, t_q
            )
            e_text, e_q = self._ocr_easyocr(img_for_ocr, page_idx)

        # 3) 최종 선택
        final_text = t_text
        if e_q > t_q + 0.05 and e_q >= 0.40:
            final_text = e_text
            used_engine = "easyocr"
        elif tried_easyocr:
            used_engine = "tesseract+easyocr"

        return final_text.strip(), {
            "page": page_idx,
            "engine": used_engine,
            "t_q": float(t_q),
            "e_q": float(e_q),
            "tried_easyocr": tried_easyocr,
            "raw_len": len(final_text or ""),
        }
