# preprocessing/pdf/page_image_extractor_final.py

import hashlib
from pathlib import Path
from typing import Dict, Any, List

import fitz
import numpy as np


# ============================================================
# utils
# ============================================================

def pix_hash(pix: fitz.Pixmap) -> str:
    return hashlib.sha1(pix.samples).hexdigest()


def is_flat_or_mask_image(pix: fitz.Pixmap) -> bool:
    """
    ❌ 제거 대상
    - 거의 단색 (검정/흰색)
    - 마스크 레이어만 있는 이미지
    """
    try:
        arr = np.frombuffer(pix.samples, dtype=np.uint8)
        if arr.size == 0:
            return True

        # 색 분산 거의 없음 → 배경 / 마스크
        if arr.std() < 6.0:
            return True

        return False
    except Exception:
        return True


def is_photo_like_image(pix: fitz.Pixmap) -> bool:
    """
    ✅ 통과 대상
    - 색 분산 충분
    - RGB 채널 다양
    """
    try:
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(-1, pix.n)
        if arr.shape[0] < 1000:
            return False

        std = arr.std()
        unique_colors = len(np.unique(arr[:, :3], axis=0))

        return std > 12.0 and unique_colors > 300
    except Exception:
        return False


# ============================================================
# extractor (FINAL)
# ============================================================

def extract_visual_blocks(
    page: fitz.Page,
    page_idx: int,
    save_dir: Path,
    global_hashes: Dict[str, str],
    min_size: int = 220,
) -> List[Dict[str, Any]]:
    """
    🔥 최종 전략 extractor

    - page.get_images() 사용
    - "사진처럼 나온 이미지"만 사용
    - 배경 / 마스크 / 벡터 결과 전부 폐기
    - 페이지마다 성공/실패가 갈리는 구조를 그대로 인정
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    images = page.get_images(full=True) or []
    if not images:
        return results

    for idx, img in enumerate(images):
        xref = img[0]

        try:
            pix = fitz.Pixmap(page.parent, xref)

            # CMYK → RGB
            if pix.n >= 5:
                pix = fitz.Pixmap(fitz.csRGB, pix)

        except Exception:
            continue

        # ------------------------------
        # 1. 너무 작은 이미지 컷
        # ------------------------------
        if pix.width < min_size or pix.height < min_size:
            continue

        # ------------------------------
        # 2. 배경 / 마스크 컷
        # ------------------------------
        if is_flat_or_mask_image(pix):
            continue

        # ------------------------------
        # 3. 사진 같은 이미지만 통과
        # ------------------------------
        if not is_photo_like_image(pix):
            continue

        # ------------------------------
        # 4. 중복 제거
        # ------------------------------
        h = pix_hash(pix)
        if h in global_hashes:
            continue
        global_hashes[h] = f"{page_idx}_{idx}"

        # ------------------------------
        # 5. 저장
        # ------------------------------
        out = save_dir / f"page_{page_idx:03d}_img_{idx:02d}.png"
        if pix.colorspace is None:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        elif pix.colorspace.n != 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)

        pix.save(str(out))


        results.append({
            "page_index": page_idx,
            "image_path": str(out),
            "width": pix.width,
            "height": pix.height,
        })

    return results
