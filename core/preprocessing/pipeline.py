import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz
from PIL import Image

from core.preprocessing.llm.llm_correction import LLMCorrector
from core.preprocessing.ocr.engine_smart import (SmartOCREngine,
                                                 text_quality_score)
from core.preprocessing.pdf.page_image_extractor_relaxed import \
    extract_visual_blocks

# ============================================================
# logger
# ============================================================
logger = logging.getLogger("pipeline")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:pipeline:%(message)s",
    )


# ============================================================
# LLM image caption (STRICT & SAFE)
# ============================================================
def generate_image_caption(
    image_path: str,
    near_text: str,
    llm: LLMCorrector,
) -> Optional[str]:
    if near_text and len(near_text.strip()) > 200:
        return None

    prompt = f"""
다음은 교육자료에서 추출된 이미지에 대한 설명입니다.

규칙:
- "이미지는"으로 시작하지 말 것
- 요약/추측 금지
- 사실 기반 설명만 작성
- 2~3문장 이내

이미지 주변 문맥:
{near_text}
""".strip()

    caption, meta = llm.correct_page(prompt, page_idx=-1, quality=0.0)
    return caption.strip() if meta.get("used_llm") else None


# ============================================================
# OCR postprocess (완화)
# ============================================================
def deduplicate_similar_lines(text: str, th: float = 0.92) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    kept: List[str] = []
    for ln in lines:
        if any(SequenceMatcher(None, ln, k).ratio() >= th for k in kept):
            continue
        kept.append(ln)
    return "\n".join(kept)


def drop_noise_lines(text: str) -> str:
    lines: List[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if len(s) < 8:
            continue
        if re.search(r"[|~_]{2,}", s):
            continue
        lines.append(s)
    return "\n".join(lines)


def restore_spacing_soft(text: str) -> str:
    text = re.sub(r"([가-힣])\s+([을를은는이가])", r"\1\2", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def final_ocr_postprocess(text: str) -> str:
    text = deduplicate_similar_lines(text)
    text = drop_noise_lines(text)
    text = restore_spacing_soft(text)
    return text.strip()


# ============================================================
# split → merge 핵심 로직
# ============================================================
def split_paragraphs(text: str) -> List[str]:
    """
    리플릿/홍보물 대응:
    - 줄 단위 분리
    - 길이 필터 완화
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines


def merge_short_chunks(
    texts: List[str],
    min_len: int = 120,
    max_len: int = 700,
) -> List[str]:
    """
    🔥 핵심 병합 단계

    - 너무 짧은 문장/제목/캡션을 앞뒤와 자동 병합
    - RAG에 적합한 문단 길이로 정규화
    """
    merged: List[str] = []
    buffer = ""

    for t in texts:
        if not buffer:
            buffer = t
            continue

        # buffer + t 가 너무 길어지면 buffer 확정
        if len(buffer) + len(t) > max_len:
            if len(buffer) >= min_len:
                merged.append(buffer)
                buffer = t
            else:
                buffer = buffer + " " + t
        else:
            buffer = buffer + " " + t

    if buffer and len(buffer) >= min_len:
        merged.append(buffer)

    return merged


def is_text_unreliable(q: float, text: str) -> bool:
    return q < 0.20 or len(text.strip()) < 25


def is_duplicate(a: str, b: str, th: float = 0.88) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= th


# ============================================================
# Pipeline
# ============================================================
class PreprocessPipeline:
    def __init__(
        self,
        use_llm: bool = True,
        image_storage_root: str = "sample/storage/pdf_images",
    ):
        self.ocr = SmartOCREngine(use_easyocr=True, easyocr_gpu=False)
        self.llm = LLMCorrector()
        self.use_llm = use_llm
        self.image_storage_root = Path(image_storage_root)
        self.global_image_hashes: Dict[str, str] = {}

    # -------------------------
    @staticmethod
    def _page_has_text_layer(page: fitz.Page) -> Optional[str]:
        txt = page.get_text("text")
        return txt if txt and len(txt.strip()) >= 40 else None

    @staticmethod
    def _page_to_pil(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    # -------------------------
    def run(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)

        image_dir = self.image_storage_root / pdf_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        chunks: List[Dict[str, Any]] = []
        prev_text: Optional[str] = None

        for page_idx in range(doc.page_count):
            page = doc.load_page(page_idx)

            # ================= TEXT =================
            text_layer = self._page_has_text_layer(page)
            if text_layer:
                raw_text = text_layer
                logger.info(f"[PAGE {page_idx}] text-layer 사용")
            else:
                img = self._page_to_pil(page)
                raw_text, _ = self.ocr.strong_ocr(img, page_idx)

            clean = final_ocr_postprocess(raw_text)

            # 1️⃣ split
            splitted = split_paragraphs(clean)

            # 2️⃣ merge (🔥 과분리 해결 핵심)
            merged = merge_short_chunks(splitted)

            for para in merged:
                q = text_quality_score(para)

                if self.use_llm and q < 0.26:
                    para, _ = self.llm.correct_page(para, page_idx, q)

                if is_text_unreliable(q, para):
                    continue
                if prev_text and is_duplicate(prev_text, para):
                    continue

                prev_text = para
                chunks.append({
                    "type": "text",
                    "text": para,
                    "page_index": page_idx,
                })
                
            # 🔒 fallback: 이 페이지에서 text chunk가 하나도 없으면
            if not any(c["page_index"] == page_idx for c in chunks):
                fallback_text = clean.strip()
                if len(fallback_text) >= 60:
                    chunks.append({
                        "type": "page_fallback",
                        "text": fallback_text,
                        "page_index": page_idx,
                    })
                    logger.warning(
                        f"[PAGE {page_idx}] text chunk 0개 → page fallback 생성"
                    )


            # ================= IMAGE =================
            visuals = extract_visual_blocks(
                page=page,
                page_idx=page_idx,
                save_dir=image_dir,
                global_hashes=self.global_image_hashes,
            )
            

            for v in visuals:
                caption = generate_image_caption(
                    image_path=v["image_path"],
                    near_text=clean,
                    llm=self.llm,
                )
                if not caption:
                    continue

                chunks.append({
                    "type": "image_caption",
                    "text": caption,
                    "image_path": v["image_path"],
                    "page_index": page_idx,
                })

        doc.close()
        logger.info("→ PreprocessPipeline 완료 (chunks=%d)", len(chunks))

        return {
            "pdf_path": str(pdf_path),
            "chunks": chunks,
        }
