"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
HWP / PDF / PPT / DOCX / TXT / CSV 자동 처리 + 문서 타입 판별 + 자동 패턴 감지까지 완성

✅ 추가:
- --input / --domain 인자 지원
  - --input이 있으면: 업로드된 "단일 파일"만 처리하고 종료
  - 없으면: 기존 DOMAIN_DIRS 배치 처리
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher

# =======================
# 0. 경로/환경 설정
# =======================
from pathlib import Path
from typing import List, Sequence

import pdfplumber
import requests
from docx import Document
from dotenv import load_dotenv

# ✅ OpenAI SDK (openai 선택일 때만 실제로 사용)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent

# ragflow 루트를 파이썬 모듈 경로에 추가 → preprocessing 패키지 import 가능
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

# milvus 환경설정
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")

# =======================
# ✅ OpenAI 환경변수
# =======================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
# text-embedding-3-large -> 3072, text-embedding-3-small -> 1536
OPENAI_EMBED_DIM = int(os.getenv("OPENAI_EMBED_DIM", "3072"))

# OpenAI client는 "openai embedding"을 실제로 쓸 때만 생성
openai_client = None

# =======================
# ✅ 임베딩 함수 (OpenAI) - openai 선택일 때만 사용
# =======================
def embed_text(text: str) -> List[float]:
    """
    OpenAI Embedding 생성.
    - 반환 벡터 차원은 모델에 의해 고정됨 (large=3072, small=1536)
    """
    global openai_client

    if openai_client is None:
        raise RuntimeError(
            "❌ OpenAI client not initialized. "
            "Set EMBEDDING_MODEL_SELECTED=openai and provide OPENAI_API_KEY."
        )

    text = (text or "").strip()
    if not text:
        return [0.0] * OPENAI_EMBED_DIM

    resp = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=text
    )
    vec = resp.data[0].embedding

    # 방어: dim mismatch 감지
    if len(vec) != OPENAI_EMBED_DIM:
        raise ValueError(
            f"❌ Embedding dim mismatch: got {len(vec)}, expected {OPENAI_EMBED_DIM}. "
            f"Check OPENAI_EMBED_MODEL/OPENAI_EMBED_DIM and Milvus collection dim."
        )
    return vec


# =======================
# 1. RAGFlow SDK import
# =======================
try:
    from ragflow_sdk import RAGFlow
except ImportError:
    # ✅ 레포 구조: /workspace/libs/sdk/python/ragflow_sdk
    sys.path.insert(0, str(BASE_DIR / "libs" / "sdk" / "python"))
    # (혹시 예전 구조도 같이 커버)
    sys.path.insert(0, str(BASE_DIR / "sdk" / "python"))
    from ragflow_sdk import RAGFlow


from embedding_provider import EmbeddingProvider
from milvus_proxy import MilvusProxy
from ragflow_sdk.modules.dataset import DataSet

from core.preprocessing.classifier.document_classifier import DocumentClassifier
from core.preprocessing.coverters.hwp_extract import extract_docx_blocks  # (폴더명이 coverters인 구조 유지)

# =======================
# 2. 커스텀 전처리 모듈 import
# =======================
from core.preprocessing.coverters.hwp_to_docx import HwpAdapter
from core.preprocessing.pipeline import PreprocessPipeline
from core.storage.table_store import TableStore

TABLE_STORAGE_DIR = SCRIPT_DIR / "storage"
table_store = TableStore(TABLE_STORAGE_DIR)
embedder = EmbeddingProvider()

# =======================
# 3. 환경 변수 (RAGFlow)
# =======================
# ✅ worker 컨테이너에서 localhost가 아니라 ragflow 서비스로 접근해야 함
HOST_ADDRESS = os.getenv("RAGFLOW_HOST") or os.getenv("HOST_ADDRESS") or "http://ragflow-cpu:9380"
API_KEY = os.getenv("RAGFLOW_API_KEY")

# ✅ RAGFlow 내부 설정용(표기용) 모델명은 네 환경에 맞게 유지해도 되고,
# 실제 임베딩은 EmbeddingProvider()가 선택된 embedding_model로 수행
EMBEDDING_MODEL = os.getenv("RAGFLOW_EMBEDDING_MODEL", OPENAI_EMBED_MODEL)

if not API_KEY:
    print("❌ RAGFLOW_API_KEY 환경 변수를 설정하세요.")
    sys.exit(1)

# =======================
# 4. 공용 객체
# =======================
hwp_adapter = HwpAdapter()
classifier = DocumentClassifier()
preprocess_pipeline = PreprocessPipeline()


# ===========================================================
# 출력 유틸
# ===========================================================
def chunk_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def print_step(n: int, text: str):
    print(f"\n[단계 {n}] {text}")
    print("-" * 60)


# ===========================================================
# 1. 문서 타입 자동 판단 (텍스트 기반 규정/방침 판별용)
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
def detect_heading_patterns_from_text(raw_text: str) -> List[str]:
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
def split_long_chunk_with_heading(chunk_text: str, max_chars: int) -> List[str]:
    if len(chunk_text) <= max_chars:
        return [chunk_text]

    lines = chunk_text.splitlines()
    heading = lines[0]
    body = "\n".join(lines[1:]).strip()
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]

    max_body = max_chars - len(heading) - 10
    chunks: List[str] = []
    buf: List[str] = []

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
def split_text_by_rules(
    raw_text: str,
    heading_patterns: Sequence[str],
    max_chars: int,
    strict_heading_only: bool = False
) -> List[str]:
    """
    strict_heading_only=True → 길이 기준 분할 OFF (조 단위 유지)
    """
    lines = raw_text.splitlines()
    compiled = [re.compile(p) for p in heading_patterns]
    coarse: List[str] = []
    buf: List[str] = []

    def is_heading(line: str) -> bool:
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
    final: List[str] = []
    for ch in coarse:
        if len(ch) <= max_chars:
            final.append(ch)
        else:
            final.extend(split_long_chunk_with_heading(ch, max_chars))

    return [c for c in final if len(c.strip()) > 20]


def chunk_docx_blocks_with_rules(
    blocks: list[dict],
    max_chars_structured: int = 2000
) -> list[str]:
    """
    extract_docx_blocks 결과(blocks)를 받아서
    - 표는 JSON 저장 + TABLE 마커 유지
    - 전체 텍스트를 조(제 n 조) 기준으로 청킹
    - structured/general은 기존 로직 준용
    반환: list[str] (최종 청크)
    """

    ordered_chunks: list[str] = []

    # 1) 블록을 문서 흐름대로 하나의 raw 텍스트로 재구성
    for blk in blocks:
        if blk.get("type") == "text":
            t = (blk.get("text") or "").strip()
            if t:
                ordered_chunks.append(t)

        elif blk.get("type") == "table":
            table = blk.get("table") or {}

            # 표 JSON 저장
            table_store.save_table(
                table_id=table.get("table_id"),
                doc=table.get("doc"),
                page=table.get("page"),
                headers=table.get("headers"),
                rows=table.get("rows"),
            )

            # 직전 문단에 TABLE 참조 결합
            table_id = table.get("table_id")
            if ordered_chunks:
                ordered_chunks[-1] += (
                    f"\n\n[TABLE:{table_id}] "
                    "표 데이터는 별도 JSON으로 저장되어 있습니다."
                )
            else:
                ordered_chunks.append(
                    f"[TABLE:{table_id}] 표 데이터는 별도 JSON으로 저장되어 있습니다."
                )

    raw = "\n".join(ordered_chunks).strip()
    if not raw:
        return []

    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "regulation":
        return split_text_by_rules(raw, patterns, max_chars=999999, strict_heading_only=True)

    if doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=max_chars_structured)

    return ordered_chunks


# ==========================================
# HWP / 슬라이드형 PDF 전처리 파이프라인 래퍼
# ==========================================
def preprocess_to_chunks(path: Path, chunk_size: int = 1200) -> list[str]:
    """
    PreprocessPipeline을 실행해서 청크 리스트(list[str])로 변환.
    """
    result = preprocess_pipeline.run(str(path), chunk_size=chunk_size)

    # 1) result가 문자열이면 → JSON 파일 경로라고 가정하고 로드
    if isinstance(result, str):
        try:
            with open(result, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return [result.strip()] if result.strip() else []
    else:
        data = result

    items = []

    if isinstance(data, dict):
        if "result_json" in data:
            rj = data["result_json"]
            if isinstance(rj, dict):
                if "chunks" in rj and isinstance(rj["chunks"], list):
                    items = rj["chunks"]
            elif isinstance(rj, list):
                items = rj
        elif "chunks" in data and isinstance(data["chunks"], list):
            items = data["chunks"]
    elif isinstance(data, list):
        items = data

    if not items:
        return []

    chunks: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("text") or item.get("content") or ""
        else:
            text = str(item)

        text = text.strip()
        if text:
            chunks.append(text)

    return chunks


# =========================
# CER(문자 오류율) 계산 유틸 (선택적)
# =========================
def cer(pred: str, truth: str) -> float:
    import numpy as np

    p = list(pred)
    t = list(truth)

    dp = np.zeros((len(t) + 1, len(p) + 1), dtype=int)

    for i in range(len(t) + 1):
        dp[i][0] = i
    for j in range(len(p) + 1):
        dp[0][j] = j

    for i in range(1, len(t) + 1):
        for j in range(1, len(p) + 1):
            cost = 0 if t[i - 1] == p[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[len(t)][len(p)] / max(1, len(t))


def eval_cer_for_pdf_text(pdf_path: Path, extracted_text: str) -> None:
    gt_root = pdf_path.parent / "solution"
    gt_path = gt_root / f"{pdf_path.stem}.txt"

    if not gt_path.exists():
        print(f"   ⚠ CER 스킵: 정답 파일 없음 → {gt_path}")
        return

    truth = gt_path.read_text(encoding="utf-8", errors="ignore")
    pred = extracted_text

    truth_norm = truth.replace("\r\n", "\n").strip()
    pred_norm = pred.replace("\r\n", "\n").strip()

    score = cer(pred_norm, truth_norm)
    print(f"   ✅ CER 평가 결과: {score * 100:.2f}% (문자 오류율)")
    print(f"      → 문자 정확도(대략): {(1 - score) * 100:.2f}%")


# ===========================================================
# 5. DOCX / TXT / CSV 전용 chunk 함수
# ===========================================================
def extract_text_docx(path: Path) -> str:
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def extract_text_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_csv(path: Path) -> str:
    lines: list[str] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return ""

    header = rows[0]
    for row in rows[1:]:
        for col, val in zip(header, row):
            col_s = str(col).strip()
            val_s = str(val).strip()
            if col_s or val_s:
                lines.append(f"{col_s}: {val_s}")
        lines.append("")

    return "\n".join(lines).strip()


def chunk_document(path: Path) -> List[str]:
    ext = path.suffix.lower()

    if ext == ".docx":
        raise RuntimeError("DOCX는 extract_docx_blocks 경로만 사용해야 합니다")
    elif ext == ".csv":
        raw = extract_text_csv(path)
    else:
        raw = extract_text_txt(path)

    if not raw.strip():
        return []

    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "regulation":
        return split_text_by_rules(raw, patterns, max_chars=999999, strict_heading_only=True)
    elif doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=2000)
    else:
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        chunks: List[str] = []
        buf: List[str] = []

        for p in paras:
            candidate = "\n\n".join(buf + [p]) if buf else p
            if len(candidate) <= 800:
                buf.append(p)
            else:
                if buf:
                    chunks.append("\n\n".join(buf))
                buf = [p]
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks


def chunk_text_pdf(path: Path) -> list[str]:
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")

    raw = "\n".join(pages)

    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "regulation":
        return split_text_by_rules(raw, patterns, max_chars=999999, strict_heading_only=True)
    elif doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=2000)
    else:
        paras = [p.strip() for p in raw.split("\n\n") if p.strip()]
        chunks: list[str] = []
        buf: list[str] = []

        for p in paras:
            candidate = "\n\n".join(buf + [p]) if buf else p
            if len(candidate) <= 800:
                buf.append(p)
            else:
                if buf:
                    chunks.append("\n\n".join(buf))
                buf = [p]
        if buf:
            chunks.append("\n\n".join(buf))

        return chunks


# ===========================================================
# 메인 설정값 & 도메인 디렉터리 정의
# ===========================================================
MAX_CHUNK_LEN = 8000

SCRIPT_DIR = Path(__file__).parent  # sample 폴더 기준

DOMAIN_DIRS = {
    "직무교육":        SCRIPT_DIR / "dataset_직무교육",
    "장애인인식개선교육": SCRIPT_DIR / "dataset_장애인인식개선",
    "직장내괴롭힘교육": SCRIPT_DIR / "dataset_괴롭힘교육",
    "직장내성희롱교육": SCRIPT_DIR / "dataset_성희롱교육",
    "정보보안교육":    SCRIPT_DIR / "dataset_정보보안교육",
    "사내규정":        SCRIPT_DIR / "dataset_사내규정",
}


def compare_with_solution(dataset_dir: Path, fpath: Path, chunks: list[str]):
    solution_dir = dataset_dir / "solution"
    solution_txt_path = solution_dir / f"{fpath.stem}.txt"

    if not solution_txt_path.exists():
        print(f"  [유사도] solution txt 없음 → {solution_txt_path.name} (스킵)")
        return

    try:
        solution_text = solution_txt_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        solution_text = solution_txt_path.read_text(encoding="cp949")

    chunk_text = "\n".join(c for c in chunks if c and c.strip())
    sim = SequenceMatcher(None, chunk_text, solution_text).ratio()
    print(f"  [유사도] 전체 청킹 결과 vs solution/{solution_txt_path.name}: {sim*100:.2f}%")


# ===========================================================
# 청크 추가 유틸 (RAGFlow + (옵션) Milvus 동시 저장 버전) - 정리본
# ===========================================================
def add_chunks_safe(
    doc,
    chunks,
    milvus=None,
    dataset_id=None,
    doc_id=None,
    embedding_model="openai",
    experiment_tag=None,
):
    """
    RAGFlow: 무조건 chunk 추가
    Milvus: chunk_hash 기준 중복 차단 후 일괄 insert
    """

    print(f"→ 생성된 청크 수: {len(chunks)}")

    if not chunks:
        print("⚠ 청크 0개 → 저장 스킵")
        return

    milvus_payload = []
    added = 0

    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = (chunk.get("text") or "").strip()
            chunk_type = chunk.get("type", "text")
            metadata = {
                "source": chunk_type,
                "page_index": chunk.get("page_index"),
                "image_path": chunk.get("image_path"),
            }
        else:
            text = str(chunk).strip()
            chunk_type = "text"
            metadata = {"source": "text"}

        if not text:
            continue

        chash = chunk_hash(text)

        if chunk_type == "image_caption" and metadata.get("image_path"):
            content = text + f"\n\n[IMAGE_PATH]{metadata['image_path']}[/IMAGE_PATH]"
        else:
            content = text

        doc.add_chunk(content=text)
        added += 1

        if milvus and dataset_id and doc_id:
            if milvus.exists_chunk_hash(dataset_id, doc_id, chash):
                continue

            try:
                embedding = embedder.embed(text, embedding_model)
                milvus_payload.append({
                    "dataset_id": dataset_id,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "chunk_hash": chash,
                    "text": text,
                    "embedding": embedding,
                    "metadata": {
                        **metadata,
                        "embedding_model": embedding_model,
                        "experiment": experiment_tag,
                    },
                })
            except Exception as e:
                print(f"⚠ embedding 실패 (chunk {idx}): {e}")

        if idx <= 2:
            print(f"\n[미리보기 {idx}] ({chunk_type})")
            print(text[:200] + ("..." if len(text) > 200 else ""))

    print(f"→ RAGFlow 청크 추가 완료: {added}개")

    if milvus and milvus_payload:
        milvus.insert_chunks(
            dataset_id=dataset_id,
            chunks=milvus_payload,
        )
        print(f"→ Milvus 적재 완료: {len(milvus_payload)}개")
    else:
        print("→ Milvus 적재 없음 (중복 또는 embedding 실패)")


# ===========================================================
# ✅ 추가: 단일 파일 모드 처리 함수
# ===========================================================
def process_single_file_mode(
    rag,
    milvus,
    embedding_model_selected: str,
    experiment_tag: str,
    input_path: Path,
    domain: str,
    doc_id: str | None,
    version: int | None,
    replace: bool,
):
    """
    --input 으로 들어온 파일 1개만 처리
    - dataset은 "그 도메인 1개"만 생성
    - solution 비교는 단일모드에서는 기본 스킵(필요하면 나중에 경로 인자 추가 가능)
    """

    print("\n" + "#" * 60)
    print("### 단일 파일 모드")
    print(f"### 도메인: {domain}")
    print(f"### 파일: {input_path}")
    print("#" * 60)

    # 도메인 1개용 dataset 생성
    dataset_name = f"upload_{domain}_{int(time.time())}"
    parser_config = DataSet.ParserConfig(rag, {"raptor": {"use_raptor": False}})
    dataset = rag.create_dataset(
        name=dataset_name,
        description=f"{domain} 업로드 단일 파일 처리",
        chunk_method="manual",
        embedding_model=EMBEDDING_MODEL,
        parser_config=parser_config,
    )
    print(f"✅ [{domain}] Dataset 생성 완료: {dataset.id} (name={dataset_name})")

    fpath = input_path.resolve()
    ext = fpath.suffix.lower().lstrip(".")
    print(f"\n======= [{domain}] {fpath.name} 처리 =======")

    effective_doc_id = (doc_id or fpath.name)

    if replace and milvus:
        try:
            milvus.delete_file(dataset_id=domain, doc_id=effective_doc_id)
            print(f"✅ replace=true → Milvus 기존 doc 삭제 완료 (dataset_id={domain}, doc_id={effective_doc_id})")
        except Exception as e:
            print(f"⚠ replace 삭제 실패 (Milvus): {e}")

    if version is not None:
        print(f"ℹ version={version} (추적용)")

    # HWP/HWPX → DOCX로 변환
    if ext in ("hwp", "hwpx"):
        print(f"[HWP] {fpath.name} → DOCX로 변환")
        docx_path = hwp_adapter.to_docx(str(fpath))
        fpath = Path(docx_path)
        ext = "docx"

    with open(fpath, "rb") as fb:
        blob = fb.read()

    doc = dataset.upload_documents(
        [{"display_name": fpath.name, "blob": blob}]
    )[0]
    print(f"→ 업로드 완료 (doc.id={doc.id})")

    # PDF / PPT / PPTX
    if ext in ("pdf", "ppt", "pptx"):
        if ext == "pdf":
            doc_type = classifier.classify(str(fpath))
        else:
            doc_type = "ppt"

        print(f"→ 문서 타입: {doc_type}")

        if doc_type == "text_pdf":
            print("→ [텍스트 PDF] 로컬 규정형 청킹 사용")
            chunks = chunk_text_pdf(fpath)
            add_chunks_safe(
                doc,
                chunks,
                milvus=milvus,
                dataset_id=domain,
                doc_id=effective_doc_id,
                embedding_model=embedding_model_selected,
                experiment_tag=experiment_tag,
            )
            return

        print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")
        pipeline_result = preprocess_pipeline.run(str(fpath))
        chunks = pipeline_result.get("chunks", [])
        add_chunks_safe(
            doc,
            chunks,
            milvus=milvus,
            dataset_id=domain,
            doc_id=effective_doc_id,
            embedding_model=embedding_model_selected,
            experiment_tag=experiment_tag,
        )
        print(f"→ 파이프라인 청크 {len(chunks)}개 반환")
        return

    # CSV / DOCX / TXT
    if ext in ("csv", "docx", "txt"):
        print("→ [CSV/DOCX/TXT] 기존 규정형 청킹 사용")

        if ext == "docx":
            print("→ DOCX blocks 기반 처리 + 조단위(제 n 조) 청킹 적용")
            blocks = extract_docx_blocks(fpath)
            docx_chunks = chunk_docx_blocks_with_rules(blocks)
            add_chunks_safe(
                doc,
                docx_chunks,
                milvus=milvus,
                dataset_id=domain,
                doc_id=effective_doc_id,
                embedding_model=embedding_model_selected,
                experiment_tag=experiment_tag,
            )
            return

        chunks = chunk_document(fpath)
        add_chunks_safe(
            doc,
            chunks,
            milvus=milvus,
            dataset_id=domain,
            doc_id=effective_doc_id,
            embedding_model=embedding_model_selected,
            experiment_tag=experiment_tag,
        )
        return

    print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")


# ===========================================================
# 메인
# ===========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="업로드 단일 파일 처리 경로")
    parser.add_argument("--domain", default="default", help="단일 파일 모드 도메인명")

    # ✅ 기존 유지 + alias 추가
    parser.add_argument("--doc_id", default=None, help="문서 식별자(docId)")
    parser.add_argument("--doc-id", dest="doc_id", default=None, help="문서 식별자(docId) (alias)")

    parser.add_argument("--version", type=int, default=None, help="문서 버전")
    parser.add_argument("--replace", default="false", help="true면 기존 docId 교체")

    args = parser.parse_args()
    replace_flag = str(args.replace).lower() in ("1", "true", "yes", "y", "on")

    print_section("RAGFlow 커스텀 청킹 + add_chunk (HWP/PDF/PPT/DOCX/TXT/CSV 포함)")

    # =========================
    # 🔥 실험 / 임베딩 설정
    # =========================
    EMBEDDING_MODEL_SELECTED = os.getenv("EMBEDDING_MODEL_SELECTED", "openai")
    EXPERIMENT_TAG = os.getenv("EXPERIMENT_TAG", "A")

    MODEL_DIM_MAP = {
        "openai": 3072,
        "sroberta": 768,
    }

    COLLECTION_NAME_MAP = {
        "openai": "ragflow_chunks",
        "sroberta": "ragflow_chunks_sroberta",
    }

    if EMBEDDING_MODEL_SELECTED not in MODEL_DIM_MAP:
        raise ValueError(f"지원하지 않는 embedding model: {EMBEDDING_MODEL_SELECTED}")

    # ✅ openai 선택일 때만 OPENAI 키/클라이언트 요구
    global openai_client
    if EMBEDDING_MODEL_SELECTED == "openai":
        if OpenAI is None:
            raise RuntimeError("❌ openai 패키지가 설치되어 있지 않습니다.")
        if not OPENAI_API_KEY:
            raise RuntimeError("❌ EMBEDDING_MODEL_SELECTED=openai 인데 OPENAI_API_KEY가 없습니다.")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    # ------------------------------------
    # 1) 서버 연결
    # ------------------------------------
    print_step(1, "서버 연결")
    try:
        _ = requests.get(
            f"{HOST_ADDRESS}/api/v1/datasets",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=10,
        )
        rag = RAGFlow(API_KEY, HOST_ADDRESS)
        print("✅ RAGFlow 연결 성공")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return

    # ------------------------------------
    # Milvus 연결
    # ------------------------------------
    print_step(1, "Milvus 연결")
    print(f"[DEBUG] EMBEDDING_MODEL_SELECTED={EMBEDDING_MODEL_SELECTED}")

    try:
        embedding_model = EMBEDDING_MODEL_SELECTED
        collection_name = COLLECTION_NAME_MAP[embedding_model]
        print(f"[DEBUG] Milvus collection={collection_name}")

        milvus = MilvusProxy(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=collection_name,
            dim=MODEL_DIM_MAP[embedding_model],
        )
        print("✅ Milvus 연결/컬렉션 준비 완료")
    except Exception as e:
        print(f"❌ Milvus 연결 실패 (일단 RAGFlow만 진행): {e}")
        milvus = None

    # ===========================================================
    # ✅ 단일 파일 모드 (--input)
    # ===========================================================
    if args.input:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            print(f"❌ 단일 파일 경로가 존재하지 않습니다: {input_path}")
            return

        process_single_file_mode(
            rag=rag,
            milvus=milvus,
            embedding_model_selected=EMBEDDING_MODEL_SELECTED,
            experiment_tag=EXPERIMENT_TAG,
            input_path=input_path,
            domain=args.domain,
            doc_id=args.doc_id,
            version=args.version,
            replace=replace_flag,
        )
        return

    # ==============================================
    # ★ 도메인별로 Dataset을 순차적으로 구성 ★ (기존 배치)
    # ==============================================
    for domain, dataset_dir in DOMAIN_DIRS.items():
        print("\n" + "#" * 60)
        print(f"### 도메인: {domain}")
        print(f"### 로컬 폴더: {dataset_dir}")
        print("#" * 60)

        if not dataset_dir.exists():
            print(f"⚠️  폴더가 없습니다. 스킵: {dataset_dir}")
            continue

        print_step(2, f"[{domain}] dataset 폴더 스캔")

        pdfs = list(dataset_dir.glob("*.pdf"))
        ppts = list(dataset_dir.glob("*.ppt")) + list(dataset_dir.glob("*.pptx"))
        hwps = list(dataset_dir.glob("*.hwp")) + list(dataset_dir.glob("*.hwpx"))
        docxs = list(dataset_dir.glob("*.docx"))
        txts = list(dataset_dir.glob("*.txt"))
        csvs = list(dataset_dir.glob("*.csv"))

        files = sorted(pdfs + ppts + hwps + docxs + txts + csvs)

        if not files:
            print(f"❌ [{domain}] 처리할 파일이 없습니다.")
            continue

        print("📂 처리 파일:")
        for f in files:
            print("   -", f.name)

        print_step(3, f"[{domain}] 데이터셋 생성")
        dataset_name = f"auto_{domain}_{int(time.time())}"

        parser_config = DataSet.ParserConfig(rag, {"raptor": {"use_raptor": False}})

        dataset = rag.create_dataset(
            name=dataset_name,
            description=f"{domain} 전용 자동 청킹 데이터셋",
            chunk_method="manual",
            embedding_model=EMBEDDING_MODEL,
            parser_config=parser_config,
        )

        print(f"✅ [{domain}] Dataset 생성 완료: {dataset.id} (name={dataset_name})")

        print_step(4, f"[{domain}] 파일 업로드 + 청킹")

        for fpath in files:
            fpath = fpath.resolve()
            ext = fpath.suffix.lower().lstrip(".")
            print(f"\n======= [{domain}] {fpath.name} 처리 =======")

            if ext in ("hwp", "hwpx"):
                print(f"[HWP] {fpath.name} → DOCX로 변환")
                docx_path = hwp_adapter.to_docx(str(fpath))
                fpath = Path(docx_path)
                ext = "docx"

            if ext in ("pdf", "ppt", "pptx"):
                if ext == "pdf":
                    doc_type = classifier.classify(str(fpath))
                else:
                    doc_type = "ppt"

                print(f"→ 문서 타입: {doc_type}")

                if doc_type == "text_pdf":
                    print("→ [텍스트 PDF] 로컬 규정형 청킹 사용")

                    with open(fpath, "rb") as fb:
                        blob = fb.read()

                    doc = dataset.upload_documents(
                        [{"display_name": fpath.name, "blob": blob}]
                    )[0]
                    print(f"→ 업로드 완료 (doc.id={doc.id})")

                    chunks = chunk_text_pdf(fpath)
                    compare_with_solution(dataset_dir, fpath, chunks)

                    add_chunks_safe(
                        doc,
                        chunks,
                        milvus=milvus,
                        dataset_id=domain,
                        doc_id=fpath.name,
                        embedding_model=EMBEDDING_MODEL_SELECTED,
                        experiment_tag=EXPERIMENT_TAG,
                    )
                    continue

                print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")

                with open(fpath, "rb") as fb:
                    blob = fb.read()

                doc = dataset.upload_documents(
                    [{"display_name": fpath.name, "blob": blob}]
                )[0]
                print(f"→ 업로드 완료 (doc.id={doc.id})")

                pipeline_result = preprocess_pipeline.run(str(fpath))
                print("→ PreprocessPipeline 완료")

                chunks = pipeline_result.get("chunks", [])
                add_chunks_safe(
                    doc,
                    chunks,
                    milvus=milvus,
                    dataset_id=domain,
                    doc_id=fpath.name,
                    embedding_model=EMBEDDING_MODEL_SELECTED,
                    experiment_tag=EXPERIMENT_TAG,
                )
                print(f"→ 파이프라인 청크 {len(chunks)}개 반환")
                continue

            if ext in ("csv", "docx", "txt"):
                print("→ [CSV/DOCX/TXT] 기존 규정형 청킹 사용")

                with open(fpath, "rb") as fb:
                    blob = fb.read()

                doc = dataset.upload_documents(
                    [{"display_name": fpath.name, "blob": blob}]
                )[0]
                print(f"→ 업로드 완료 (doc.id={doc.id})")

                if ext == "docx":
                    print("→ DOCX blocks 기반 처리 + 조단위(제 n 조) 청킹 적용")
                    blocks = extract_docx_blocks(fpath)
                    docx_chunks = chunk_docx_blocks_with_rules(blocks)

                    add_chunks_safe(
                        doc,
                        docx_chunks,
                        milvus=milvus,
                        dataset_id=domain,
                        doc_id=fpath.name,
                        embedding_model=EMBEDDING_MODEL_SELECTED,
                        experiment_tag=EXPERIMENT_TAG,
                    )
                    continue

                chunks = chunk_document(fpath)
                compare_with_solution(dataset_dir, fpath, chunks)

                add_chunks_safe(
                    doc,
                    chunks,
                    milvus=milvus,
                    dataset_id=domain,
                    doc_id=fpath.name,
                    embedding_model=EMBEDDING_MODEL_SELECTED,
                    experiment_tag=EXPERIMENT_TAG,
                )
                continue

            print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")

        print_step(5, f"[{domain}] 검색 테스트")
        query = f"{domain} 관련 문서의 목적은 무엇인가?"
        results = rag.retrieve(
            dataset_ids=[dataset.id],
            question=query,
            top_k=3,
        )

        for i, r in enumerate(results, 1):
            print(f"\n[검색 {i}]")
            print(r.content[:200] + "...")

if __name__ == "__main__":
    main()
