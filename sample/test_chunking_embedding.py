#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
PDF / DOCX / TXT 자동 처리 + 문서 타입 판별 + 자동 패턴 감지 완전판
"""

import os
import sys
import time
import requests
import re
from pathlib import Path
from typing import List, Sequence
from dotenv import load_dotenv
from ragflow_sdk.modules.dataset import DataSet

# =======================
# 0. RAGFlow SDK import
# =======================
try:
    from ragflow_sdk import RAGFlow
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "sdk" / "python"))
    from ragflow_sdk import RAGFlow

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

HOST_ADDRESS = os.getenv("RAGFLOW_HOST", "http://localhost")
API_KEY = os.getenv("RAGFLOW_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "RAGFLOW_EMBEDDING_MODEL",
    "text-embedding-004@Gemini"
)

if not API_KEY:
    print("❌ RAGFLOW_API_KEY 환경 변수를 설정하세요.")
    sys.exit(1)


# ===========================================================
# 출력 유틸
# ===========================================================
def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_step(n, text):
    print(f"\n[단계 {n}] {text}")
    print("-" * 60)


# ===========================================================
# 1. 문서 타입 자동 판단
# ===========================================================
def detect_document_type(raw_text: str) -> str:
    """
    문서 형태 자동 판단
    - regulation: '제 n조'가 많이 등장하는 규정류
    - structured: ◇/◊/○/숫자 헤더가 많은 보안/방침 문서
    - general: 일반 문서 (보고서, 회의록 등)
    """

    text = raw_text.replace("\x01", " ").replace("\u00a0", " ")
    lines = text.splitlines()

    article = sum(1 for l in lines if re.match(r"^\s*제\s*\d+\s*조", l))
    diamond = sum(1 for l in lines if re.match(r"^\s*[◊◆◇]", l))
    circle = sum(1 for l in lines if re.match(r"^\s*[○●◯O]", l))
    number = sum(1 for l in lines if re.match(r"^\s*\d+\.", l))

    paragraph_count = len([p for p in text.split("\n\n") if p.strip()])
    length = len(text)

    if article >= 3:
        return "regulation"

    if article == 0 and (diamond >= 2 or circle >= 2):
        return "structured"

    if number >= 5:
        return "structured"

    if length > 2000 and paragraph_count >= 3:
        return "general"

    return "general"


# ===========================================================
# 2. heading 패턴 자동 감지
# ===========================================================
def detect_heading_patterns_from_text(raw_text: str) -> list[str]:
    text = raw_text.replace("\x01", " ").replace("\u00a0", " ")
    lines = text.splitlines()

    candidate = {
        "article": r"^\s*제\s*\d+\s*조",
        "diamond": r"^\s*[◊◆◇]\s*",
        "circle": r"^\s*[○●◯O]\s*",
        "special": r"^\s*[令※]\s*",
        "number_top": r"^\s*\d+\.\s+",
        "number_sub": r"^\s*\d+\.\d+\s+",
    }

    counts = {k: 0 for k in candidate.keys()}
    compiled = {k: re.compile(v) for k, v in candidate.items()}

    for l in lines:
        s = l.rstrip()
        for k, pat in compiled.items():
            if pat.match(s):
                counts[k] += 1

    article = counts["article"]
    diamond = counts["diamond"]
    circle = counts["circle"]
    special = counts["special"]
    num = counts["number_top"] + counts["number_sub"]

    # 규정류
    if article >= 3:
        active = ["article"]

    # 보안관리규정
    elif article == 0 and diamond >= 1:
        active = ["diamond"]
        if circle + special >= 1:
            active += ["circle", "special"]

    # 경영방침, 숫자 구조 문서
    elif num >= 3:
        active = ["number_top", "number_sub"]

    else:
        nonzero = [k for k, v in counts.items() if v > 0]
        if nonzero:
            max_cnt = max(counts[k] for k in nonzero)
            active = [k for k in nonzero if counts[k] == max_cnt]
        else:
            active = ["article"]

    return [candidate[k] for k in active]


# ===========================================================
# 3. 길이 기반 스플릿 함수
# ===========================================================
def split_long_chunk_with_heading(chunk_text: str, max_chars: int) -> list[str]:
    if len(chunk_text) <= max_chars:
        return [chunk_text]

    lines = chunk_text.splitlines()
    heading = lines[0]
    body = "\n".join(lines[1:]).strip()
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]

    max_body = max_chars - len(heading) - 10
    chunks = []
    buf = []

    def flush():
        nonlocal buf
        if not buf:
            return
        text = heading + "\n" + "\n\n".join(buf)
        chunks.append(text)
        buf = []

    for p in paras:
        if not buf:
            cand = p
        else:
            cand = "\n\n".join(buf + [p])

        if len(cand) <= max_body:
            buf.append(p)
        else:
            flush()
            if len(p) > max_body:
                # 강제 자르기
                s = 0
                while s < len(p):
                    part = p[s:s + max_body]
                    chunks.append(heading + "\n" + part)
                    s += max_body
            else:
                buf = [p]

    flush()
    return chunks


# ===========================================================
# 4. 규정형 청킹
# ===========================================================
def split_text_by_rules(raw_text: str,
                        heading_patterns: Sequence[str],
                        max_chars: int,
                        strict_heading_only=False) -> List[str]:
    """
    strict_heading_only=True → 길이 기준 분할 OFF (조 단위 유지)
    """
    lines = raw_text.splitlines()
    compiled = [re.compile(p) for p in heading_patterns]
    coarse = []
    buf = []

    def is_heading(line):
        s = line.strip()
        for pat in compiled:
            if pat.match(s):
                return True
        return False

    def flush():
        nonlocal buf
        if not buf:
            return None
        t = "\n".join(buf).strip()
        buf = []
        return t

    for line in lines:
        if is_heading(line):
            f = flush()
            if f:
                coarse.append(f)
            buf = [line]
        else:
            buf.append(line)

    last = flush()
    if last:
        coarse.append(last)

    # 길이 기반 분할 해제 (규정류)
    if strict_heading_only:
        return [c for c in coarse if len(c.strip()) > 20]

    # structured 문서 → 필요시 길이 분할
    final = []
    for ch in coarse:
        if len(ch) <= max_chars:
            final.append(ch)
        else:
            final.extend(split_long_chunk_with_heading(ch, max_chars))

    return [c for c in final if len(c.strip()) > 20]


# ===========================================================
# 5. 파일 타입별 chunk 함수
# ===========================================================
def extract_text_pdf(path: Path) -> str:
    import pdfplumber
    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


def extract_text_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def extract_text_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_document(path: Path) -> list[str]:
    ext = path.suffix.lower()

    # 1) 텍스트 추출
    if ext == ".pdf":
        raw = extract_text_pdf(path)
    elif ext == ".docx":
        raw = extract_text_docx(path)
    else:
        raw = extract_text_txt(path)

    raw = raw.replace("\x01", " ").replace("\u00a0", " ")

    # 2) 문서 타입 판단
    doc_type = detect_document_type(raw)
    print(f"   → 문서 타입: {doc_type}")

    # 3) 헤더 패턴 자동 감지
    patterns = detect_heading_patterns_from_text(raw)
    print(f"   → 사용 heading_patterns: {patterns}")

    # 4) 타입별 청킹 전략
    if doc_type == "regulation":
        # 조 단위 완전 보존
        return split_text_by_rules(raw, patterns, max_chars=999999, strict_heading_only=True)

    elif doc_type == "structured":
        # 적용범위/책임과권한/정보보안 같은 문서
        return split_text_by_rules(raw, patterns, max_chars=2000)

    else:
        # 일반 문서: 문단 + 길이 기준
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        chunks = []
        buf = []
        max_chars = 800

        for p in paras:
            candidate = "\n\n".join(buf + [p]) if buf else p
            if len(candidate) <= max_chars:
                buf.append(p)
            else:
                chunks.append("\n\n".join(buf))
                buf = [p]

        if buf:
            chunks.append("\n\n".join(buf))

        return chunks


# ===========================================================
# 메인
# ===========================================================
def main():
    print_section("RAGFlow 커스텀 청킹 + add_chunk (PDF/DOCX/TXT 포함)")

    # ------------------------------------
    # 1) 서버 연결
    # ------------------------------------
    print_step(1, "서버 연결")
    try:
        r = requests.get(f"{HOST_ADDRESS}/api/v1/datasets",
                         headers={"Authorization": f"Bearer {API_KEY}"})
        rag = RAGFlow(API_KEY, HOST_ADDRESS)
        print("✅ RAGFlow 연결 성공")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return

    # ------------------------------------
    # 2) dataset 폴더 검색 (pdf + docx + txt)
    # ------------------------------------
    print_step(2, "dataset 폴더 스캔")
    dataset_dir = Path(__file__).parent / "dataset"

    pdfs = list(dataset_dir.glob("*.pdf"))
    docxs = list(dataset_dir.glob("*.docx"))
    txts = list(dataset_dir.glob("*.txt"))
    files = sorted(pdfs + docxs + txts)

    if not files:
        print("❌ 처리할 파일이 없습니다.")
        return

    print("📂 처리 파일:")
    for f in files:
        print("   -", f.name)

    # ------------------------------------
    # 3) Dataset 생성
    # ------------------------------------
    print_step(3, "데이터셋 생성")
    dataset_name = f"auto_chunk_{int(time.time())}"

    parser_config = DataSet.ParserConfig(rag, {"raptor": {"use_raptor": False}})

    dataset = rag.create_dataset(
        name=dataset_name,
        description="자동 청킹 규정/지침 DOCX/PDF/TXT 포함",
        chunk_method="manual",
        embedding_model=EMBEDDING_MODEL,
        parser_config=parser_config,
    )

    print(f"✅ Dataset 생성 완료: {dataset.id}")

    # ------------------------------------
    # 4) 파일별 업로드 + 청킹 + add_chunk
    # ------------------------------------
    print_step(4, "파일 업로드 + 청킹")

    for fpath in files:
        print(f"\n======= {fpath.name} 처리 =======")

        with open(fpath, "rb") as f:
            blob = f.read()

        doc = dataset.upload_documents(
            [{"display_name": fpath.name, "blob": blob}]
        )[0]

        print(f"→ 업로드 완료 (doc.id={doc.id})")

        chunks = chunk_document(fpath)
        print(f"→ 생성된 청크 수: {len(chunks)}")

        for idx, c in enumerate(chunks, 1):
            doc.add_chunk(content=c)
            if idx <= 2:
                print(f"\n  [미리보기 청크 {idx}]")
                print(c[:200] + "...")

        print(f"→ 총 {len(chunks)}개 청크 추가 완료")

    # ------------------------------------
    # 5) 검색 테스트
    # ------------------------------------
    print_step(5, "검색 테스트")

    query = "이 문서의 목적은 무엇인가?"
    results = rag.retrieve(
        dataset_ids=[dataset.id],
        question=query,
        top_k=5,
    )

    for i, r in enumerate(results, 1):
        print(f"\n[검색 {i}]")
        print(r.content[:200] + "...")


if __name__ == "__main__":
    main()
