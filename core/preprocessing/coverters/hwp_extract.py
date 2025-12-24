"""
HWP 텍스트 변환 어댑터

- HWP → DOCX (LibreOffice)
- DOCX → text blocks / table blocks (구조화)
"""

import logging
import os
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any

from docx import Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.text.paragraph import Paragraph
from docx.table import Table

import re
from collections import Counter

logger = logging.getLogger(__name__)

# ============================================================
# LibreOffice (soffice) 경로
# ============================================================

def _get_soffice_cmd() -> str:
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
        return "soffice"
    return "soffice"


def _check_libreoffice_available() -> bool:
    try:
        subprocess.run([_get_soffice_cmd(), "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False
    
def _stable_table_id(
    doc_name: str,
    idx: int,
    headers: list[str | None],
    rows: list[list[str | None]],
) -> str:
    """
    문서명 + 테이블 순번 + 구조 기반으로
    실행할 때마다 동일한 table_id 생성
    """
    raw = f"{doc_name}|{idx}|{headers}|{rows}".encode(
        "utf-8", errors="ignore"
    )
    h = hashlib.sha1(raw).hexdigest()[:8]
    return f"tbl_{doc_name}_{idx:03d}_{h}"

def _is_toc_paragraph(text: str) -> bool:
    t = text.replace(" ", "").strip()

    # 1️⃣ 딱 "목차"
    if t == "목차":
        return True

    # 2️⃣ "제1장", "제 1 장" 단독
    if re.fullmatch(r"제\d+장", t):
        return True

    # 3️⃣ 목차에서 흔한 패턴
    if re.match(r"제\d+장.*", text) and len(text) < 30:
        return True

    return False

# ============================================================
# DOCX → text / table blocks (개선 버전)
# ============================================================



def _normalize_headers(headers: List[str | None]) -> List[str]:
    """
    병합 셀로 인해 반복된 헤더 제거
    """
    normalized = []
    prev = None
    for h in headers:
        h = (h or "").strip()
        if not h:
            continue
        if h != prev:
            normalized.append(h)
        prev = h
    return normalized


def _flatten_cells(rows: List[List[str | None]]) -> List[str]:
    return [
        c.strip()
        for r in rows
        for c in r
        if c and c.strip()
    ]


def _looks_like_toc(headers: List[str], rows: List[List[str | None]]) -> bool:
    """
    목차/차례 표 감지
    """
    joined = " ".join(headers + _flatten_cells(rows))

    toc_patterns = [
        r"목\s*차",
        r"차\s*례",
        r"제\s*\d+\s*장",
        r"\d+\.\s*[가-힣]",
    ]

    for p in toc_patterns:
        if re.search(p, joined):
            return True

    return False

def _is_meaningful_table(headers, rows):
    def _looks_like_toc(headers, rows) -> bool:
        joined = " ".join(headers + _flatten_cells(rows))
        return bool(re.search(r"(목\s*차|차\s*례|제\s*\d+\s*장)", joined))

    # 최소 구조
    if len(headers) < 2 or len(rows) < 1:
        return False

    header_text = " ".join(headers)
    cell_text = " ".join(_flatten_cells(rows))

    field_keywords = [
        "구분", "항목", "사업", "연도", "분기",
        "금액", "합계", "비고", "내용",
        "계획", "실적", "예산",
    ]

    # ✅ CASE 1: 헤더/셀 중 하나라도 키워드 있으면 OK
    if any(k in header_text for k in field_keywords) \
       or any(k in cell_text for k in field_keywords):
        if not _looks_like_toc(headers, rows):
            return True

    # ✅ CASE 2: 양식표 구제 조건 (중요)
    # 값은 비어 있어도 "열 구조 + 행 반복"이 있으면 허용
    if len(headers) >= 3 and len(rows) >= 2:
        # 분기/합계/연도 같은 패턴
        structure_hint = ["1/4", "2/4", "3/4", "4/4", "합계", "연도"]
        if any(h in header_text for h in structure_hint):
            return True

    return False

def extract_docx_blocks(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    doc = Document(str(path))

    blocks: List[Dict[str, Any]] = []
    doc_name = path.stem

    body = doc.element.body
    table_idx = 0

    in_toc = False  # ✅ 목차 영역 스킵 모드

    for element in body.iterchildren():
        # ----------------------------
        # Paragraph
        # ----------------------------
        if isinstance(element, CT_P):
            p = Paragraph(element, doc)
            text = (p.text or "").strip()
            if not text:
                continue

            # ✅ 목차 시작 감지 (딱 "목차")
            if text.replace(" ", "").strip() == "목차":
                in_toc = True
                continue

            # ✅ 목차 영역이면: "제 n 조" 시작 전까지 전부 스킵
            if in_toc:
                if re.match(r"^\s*제\s*\d+\s*조", text):
                    in_toc = False  # 본문 시작
                else:
                    continue

            # ✅ 기존 단일 목차 패턴도 스킵(제1장 같은 라인)
            if _is_toc_paragraph(text):
                continue

            blocks.append({"type": "text", "text": text})
            continue

        # ----------------------------
        # Table
        # ----------------------------
        if isinstance(element, CT_Tbl):
            t = Table(element, doc)

            grid: List[List[str]] = []
            for row in t.rows:
                grid.append([(cell.text or "").strip() for cell in row.cells])

            if not grid or all(all(not c for c in r) for r in grid):
                continue

            headers = grid[0]
            rows = grid[1:] if len(grid) > 1 else []

            # ✅ 목차 표는 통째로 스킵
            if _looks_like_toc(headers, rows):
                continue

            table_id = _stable_table_id(doc_name, table_idx, headers, rows)
            table_idx += 1

            blocks.append({
                "type": "table",
                "table": {
                    "table_id": table_id,
                    "doc": doc_name,
                    "page": None,
                    "headers": headers,
                    "rows": rows,
                }
            })
            continue

    return blocks


def _is_title_row(cells: List[str | None]) -> bool:
    texts = [c for c in cells if c]
    if not texts:
        return False
    # 전부 비슷하거나 너무 긴 문장 → 제목
    if len(set(texts)) == 1:
        return True
    if sum(len(t) for t in texts) > 40:
        return True
    return False

# ============================================================
# HWP → DOCX → blocks
# ============================================================

def convert_hwp_to_blocks(hwp_path: str) -> Dict[str, Any]:
    """
    HWP → DOCX → text_blocks / table_blocks
    """
    hwp_path = Path(hwp_path).resolve()

    if not hwp_path.exists():
        raise FileNotFoundError(hwp_path)

    if not _check_libreoffice_available():
        raise RuntimeError("LibreOffice not available")

    cmd = _get_soffice_cmd()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        subprocess.run(
            [
                cmd,
                "--headless",
                "--infilter=Hwp2002_File",
                "--convert-to", "docx",
                str(hwp_path),
                "--outdir", str(tmpdir),
            ],
            capture_output=True,
            timeout=120,
        )

        docx_path = tmpdir / f"{hwp_path.stem}.docx"
        if not docx_path.exists():
            raise RuntimeError("HWP → DOCX 변환 실패")

        blocks = extract_docx_blocks(docx_path)

        text_blocks = [b for b in blocks if b["type"] == "text"]
        table_blocks = [b["table"] for b in blocks if b["type"] == "table"]

        return {
            "docx_path": str(docx_path),
            "blocks": blocks,   # ✅ 그대로 전달
        }

