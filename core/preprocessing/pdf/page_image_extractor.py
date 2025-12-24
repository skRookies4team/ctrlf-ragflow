import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from difflib import SequenceMatcher

import fitz
from PIL import Image

from preprocessing.ocr.engine_smart import SmartOCREngine, text_quality_score
from preprocessing.llm.llm_correction import LLMCorrector
from preprocessing.pdf.page_image_extractor_relaxed import extract_visual_blocks


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
# image caption (LLM)
# ============================================================
def generate_image_caption(
    image_path: str,
    near_text: str,
    llm: LLMCorrector,
) -> Optional[str]:
    prompt = f"""
다음은 직장 내 성희롱·괴롭힘 예방 교육 자료에서 추출된 이미지에 대한 설명을 생성해야 합니다.

규칙:
- 요약 금지 / 추측 금지
- 정보 중심 설명
- 2~5문장
- 불릿, 마크다운 사용 금지

이 이미지가 사용된 문맥:
{near_text}
"""
    caption, meta = llm.correct_page(prompt, page_idx=-1, quality=0.0)
    return caption.strip() if meta.get("used_llm") else None


# ============================================================
# OCR 후처리
# ============================================================
def deduplicate_similar_lines(text: str, th: float = 0.92) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    kept = []
    for ln in lines:
        if any(SequenceMatcher(None, ln, k).ratio() >= th for k in kept):
            continue
        kept.append(ln)
    return "\n".join(kept)


def restore_korean_spacing(text: str) -> str:
    text = re.sub(r"(은|는|이|가|을|를|에|에서|으로|로|와|과)", r" \1", text)
    return re.sub(r"\s+", " ", text).strip()


def drop_garbled_tokens(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        if re.search(r"[A-Z]{3,}[^가-힣]{2,}", ln):
            continue
        if re.search(r"[|~_]{2,}", ln):
            continue
        lines.append(ln)
    return "\n".join(lines)


IMAGE_META_KEYWORDS = [
    "이미지는", "이 이미지는", "위 이미지는",
    "사진에는", "그림에는", "도표에는", "그래프에는"
]


def remove_image_meta_sentences(text: str) -> str:
    return "\n".join(
        ln for ln in text.splitlines()
        if not any(k in ln for k in IMAGE_META_KEYWORDS)
    )


def final_ocr_postprocess(text: str) -> str:
    text = deduplicate_similar_lines(text)
    text = drop_garbled_tokens(text)
    text = restore_korean_spacing(text)
    text = remove_image_meta_sentences(text)
    return text.strip()


# ============================================================
# text utils
# ============================================================
def simple_clean_text(text: str) -> str:
    if not text:
        return ""
    s = text.replace("\t", " ")
    while "  " in s:
        s = s.replace("  ", " ")
    return "\n".join(ln.rstrip() for ln in s.splitlines()).strip()


def split_paragraphs(text: str) -> List[str]:
    return [
        p.strip()
        for p in text.split("\n\n")
        if len(p.strip()) >= 45
    ]


def is_text_unreliable(q: float, text: str) -> bool:
    return q < 0.25 or len(text.strip()) < 25


_TOKEN_RE = re.compile(r"[가-힣]{2,}")


def compute_anomaly_score(text: str, vocab: Set[str]) -> float:
    tokens = _TOKEN_RE.findall(text)
    if not tokens:
        return 0.0
    unknown = [t for t in tokens if t not in vocab]
    return min(len(unknown) / max(len(tokens), 1), 1.0)


def is_duplicate(a: str, b: str, th: float = 0.85) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= th


# ============================================================
# question type tagging
# ============================================================
QUESTION_PATTERNS = {
    "definition": ["정의", "이란", "의미"],
    "procedure": ["절차", "방법", "단계", "처리"],
    "sanction": ["제재", "처벌", "징계", "과태료"],
    "case": ["사례", "판결", "예시"],
}


def compute_question_type_scores(text: str) -> Dict[str, float]:
    scores = {}
    for k, words in QUESTION_PATTERNS.items():
        scores[k] = sum(text.count(w) for w in words) / max(len(text) / 80, 1)
    return scores


# ============================================================
# PreprocessPipeline
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

        self.vocab: Set[str] = set()
        self.global_image_hashes: Set[str] = set()

    @staticmethod
    def _page_has_text_layer(page: fitz.Page) -> Optional[str]:
        txt = page.get_text("text")
        return txt if txt and len(txt.strip()) >= 40 else None

    @staticmethod
    def _page_to_pil(page: fitz.Page, zoom: float = 2.0) -> Image.Image:
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    def _update_vocab(self, text: str):
        self.vocab.update(_TOKEN_RE.findall(text))

    # --------------------------------------------------------
    # main
    # --------------------------------------------------------
    def run(self, pdf_path: str) -> Dict[str, Any]:
        pdf_path = Path(pdf_path)
        doc = fitz.open(pdf_path)
        page_count = doc.page_count

        image_dir = self.image_storage_root / pdf_path.stem
        image_dir.mkdir(parents=True, exist_ok=True)

        chunks = []
        prev_text = None

        for page_idx in range(page_count):
            page = doc.load_page(page_idx)

            # ---------- OCR ----------
            text_layer = self._page_has_text_layer(page)
            if text_layer:
                raw = text_layer
            else:
                img = self._page_to_pil(page)
                raw, _ = self.ocr.strong_ocr(img, page_idx)

            # ---------- text clean ----------
            clean = simple_clean_text(raw)
            clean = final_ocr_postprocess(clean)
            self._update_vocab(clean)

            # ---------- text chunks ----------
            for para in split_paragraphs(clean):
                q = text_quality_score(para)
                anomaly = compute_anomaly_score(para, self.vocab)

                if self.use_llm and q < 0.45 and anomaly > 0.6:
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
                    "question_type_scores": compute_question_type_scores(para),
                })

            # ---------- image chunks ----------
            visuals = extract_visual_blocks(
                page=page,
                page_idx=page_idx,
                save_dir=image_dir,
                global_hashes=self.global_image_hashes,
            )

            for v in visuals:
                caption = generate_image_caption(
                    v["image_path"], clean, self.llm
                )
                if not caption:
                    continue

                chunks.append({
                    "type": "image",
                    "text": caption,
                    "image_path": v["image_path"],
                    "page_index": page_idx,
                })

        doc.close()

        logger.info("→ PreprocessPipeline 완료 (chunks=%d)", len(chunks))

        return {
            "pdf_path": str(pdf_path),
            "page_count": page_count,
            "chunks": chunks,
        }
