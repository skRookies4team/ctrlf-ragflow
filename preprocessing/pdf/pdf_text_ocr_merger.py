# preprocessing/pdf/pdf_text_ocr_merger.py
import re
from typing import List

def _score(text: str) -> float:
    if not text:
        return 0.0
    s = text.strip()
    length_score = min(len(s) / 1000.0, 1.0)
    alpha_count = len(re.findall(r"[A-Za-z가-힣]", s))
    nonword_count = len(re.findall(r"[^0-9A-Za-z가-힣\s\.,\-()]", s))
    alpha_ratio = alpha_count / max(1, len(s))
    penalty = min(nonword_count / max(1, len(s)), 0.5)
    return length_score * (0.5 + 0.5 * alpha_ratio) * (1 - penalty)

def merge_page(py: str, ocr: str) -> str:
    py_score = _score(py)
    ocr_score = _score(ocr)

    if py_score >= ocr_score * 1.2:
        return py
    if ocr_score >= py_score * 1.2:
        return ocr

    lines = []
    seen = set()
    for l in (py.splitlines() + ocr.splitlines()):
        ls = l.strip()
        if ls and ls not in seen:
            lines.append(ls)
            seen.add(ls)
    return "\n".join(lines)

def merge_all(py_texts: List[str], ocr_texts: List[str]) -> str:
    pages = [merge_page(p, o) for p, o in zip(py_texts, ocr_texts)]
    pages = [p for p in pages if p.strip()]
    return "\n\n===PAGE_BREAK===\n\n".join(pages)
