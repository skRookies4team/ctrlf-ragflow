# preprocessing/ocr/preprocess.py
import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("ocr_preprocess")


def _to_gray(img: Image.Image) -> np.ndarray:
    """PIL 이미지를 그레이스케일 OpenCV 배열로 변환."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return gray


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Otsu 이진화 + 약한 블러
    너무 강한 전처리는 한글을 뭉치게 하니 최소로만.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thr = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thr


def _crop_margin(bin_img: np.ndarray, border: int = 10) -> np.ndarray:
    """
    텍스트가 있는 영역만 남기고 바깥 여백 제거.
    (강한 Smart Crop이 아니라, 안전한 Margin Crop 수준)
    """
    # 텍스트(검은색)에 해당하는 영역 찾기
    # 이진 이미지에서 0(검정)에 가까운 부분을 텍스트로 간주
    ys, xs = np.where(bin_img < 250)
    if len(xs) == 0 or len(ys) == 0:
        # 텍스트가 없다고 판단되면 원본 유지
        return bin_img

    x_min = max(int(xs.min()) - border, 0)
    x_max = min(int(xs.max()) + border, bin_img.shape[1])
    y_min = max(int(ys.min()) - border, 0)
    y_max = min(int(ys.max()) + border, bin_img.shape[0])

    cropped = bin_img[y_min:y_max, x_min:x_max]
    return cropped


def preprocess_for_ocr(
    pil_image: Image.Image,
    strong_crop: bool = True
) -> Image.Image:
    """
    OCR 전용 이미지 전처리:
    1. Grayscale
    2. Otsu 이진화
    3. (선택) 여백 Crop
    4. 약한 dilation (연결)
    """
    gray = _to_gray(pil_image)
    bin_img = _binarize(gray)

    if strong_crop:
        bin_img = _crop_margin(bin_img, border=12)

    # 너무 강하지 않은 dilation (한글이 서로 붙지 않을 정도)
    kernel = np.ones((2, 2), np.uint8)
    bin_img = cv2.dilate(bin_img, kernel, iterations=1)

    # 다시 PIL Image로 변환
    out = Image.fromarray(bin_img)
    return out
