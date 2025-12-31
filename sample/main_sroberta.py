"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
HWP / PDF / PPT / DOCX / TXT / CSV 자동 처리 + 문서 타입 판별 + 자동 패턴 감지까지 완성
- ✅ sRoBERTa(512) 대응: SAFE_MAX_TOKENS=480 + overlap=80
- ✅ (요청 반영) "원래대로" 청킹 유지 + 앞에 ①②③(또는 (1)(2)...) 항 번호만 prefix로 붙여 저장
  -> regulation 전용 구조청킹은 DOCX에서 사용하지 않음 (UI에서 호가 쪼개지는 문제 방지)
"""

import csv
import hashlib
import json
import os
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Sequence

import pdfplumber
import requests
from dotenv import load_dotenv

# =======================
# 0. 경로/환경 설정
# =======================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent.parent

# ragflow 루트를 파이썬 모듈 경로에 추가 → preprocessing 패키지 import 가능
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

# milvus 환경설정
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

# =======================
# ✅ vLLM + sroberta 환경변수
# =======================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
if not VLLM_BASE_URL:
    raise RuntimeError("VLLM_BASE_URL is not set. Check your .env file.")

VLLM_BASE_URL = VLLM_BASE_URL.rstrip("/")
if not VLLM_BASE_URL.endswith("/v1"):
    VLLM_BASE_URL += "/v1"


VLLM_EMBED_MODEL_ID = os.getenv("VLLM_EMBED_MODEL_ID")  # 필수
EMBEDDING_ENDPOINT = f"{VLLM_BASE_URL}/embeddings"

# sroberta dim
SROBERTA_EMBED_DIM = int(os.getenv("SROBERTA_EMBED_DIM", "768"))

if not VLLM_EMBED_MODEL_ID:
    raise RuntimeError("❌ VLLM_EMBED_MODEL_ID 가 .env에 없습니다.")

MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ragflow_chunks_sroberta")

# =======================
# ✅ 토큰/청킹 설정 (sRoBERTa 512 대응)
# =======================
SAFE_MAX_TOKENS = 480     # ✅ target chunk
OVERLAP_TOKENS = 80       # ✅ overlap


def embed_text(text: str) -> List[float]:
    """
    vLLM OpenAI-compatible embeddings endpoint로 sroberta 임베딩 생성
    """
    text = (text or "").strip()
    if not text:
        return [0.0] * SROBERTA_EMBED_DIM

    r = requests.post(
        EMBEDDING_ENDPOINT,
        json={"input": [text], "model": VLLM_EMBED_MODEL_ID},
        timeout=60
    )
    r.raise_for_status()
    vec = r.json()["data"][0]["embedding"]

    if len(vec) != SROBERTA_EMBED_DIM:
        raise ValueError(
            f"❌ Embedding dim mismatch: got {len(vec)}, expected {SROBERTA_EMBED_DIM}. "
            "Check VLLM_EMBED_MODEL_ID / SROBERTA_EMBED_DIM / Milvus collection dim."
        )
    return vec


def approx_tokens_ko(text: str) -> int:
    """
    토큰 근사치:
    - 한국어/혼합 텍스트에서 보수적으로 잡기 위해
      '글자수/2.2' 정도로 근사 (너무 빡세면 2.0~2.8 조절)
    """
    text = text or ""
    if not text.strip():
        return 0
    return max(1, int(len(text) / 2.2))


def hard_split_by_chars(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def split_long_text_naturally(text: str, target_tokens: int = SAFE_MAX_TOKENS) -> list[str]:
    """
    1) 문장 단위(마침표/다/요/?!/개행)로 최대한 유지
    2) 그래도 길면 공백 기준
    3) 최후엔 글자수 기준 hard split
    """
    text = (text or "").strip()
    if not text:
        return []

    if approx_tokens_ko(text) <= target_tokens:
        return [text]

    # 문장 경계 후보로 split
    parts = re.split(r'(?:[\.!\?\n]\s+|다\.\s+|\n\s*)', text)
    parts = [p.strip() for p in parts if p and p.strip()]

    chunks = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for p in parts:
        cand = (buf + " " + p).strip() if buf else p
        if approx_tokens_ko(cand) <= target_tokens:
            buf = cand
        else:
            flush()
            if approx_tokens_ko(p) <= target_tokens:
                buf = p
            else:
                # 문장 자체가 너무 김 → 공백/구두점 기준 더 쪼갬
                sub = re.split(r'(\s+|,|;|:)', p)
                sub = [s for s in sub if s and s.strip()]
                sub_buf = ""
                for s in sub:
                    cand2 = (sub_buf + " " + s).strip() if sub_buf else s
                    if approx_tokens_ko(cand2) <= target_tokens:
                        sub_buf = cand2
                    else:
                        if sub_buf:
                            chunks.append(sub_buf.strip())
                        sub_buf = s.strip()

                if sub_buf:
                    if approx_tokens_ko(sub_buf) <= target_tokens:
                        chunks.append(sub_buf.strip())
                    else:
                        # 최후: hard split
                        max_chars = int(target_tokens * 2.2)
                        chunks.extend([c.strip() for c in hard_split_by_chars(sub_buf, max_chars) if c.strip()])

    flush()
    return [c for c in chunks if c.strip()]


def apply_overlap(chunks: list[str], overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """
    chunk 사이에 overlap 적용 (토큰 근사 → 글자수로 환산)
    """
    if not chunks or overlap_tokens <= 0:
        return chunks

    overlap_chars = int(overlap_tokens * 2.2)  # approx
    out = [chunks[0].strip()]

    for i in range(1, len(chunks)):
        prev = out[-1]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        merged = (tail.strip() + "\n" + chunks[i].strip()).strip()
        out.append(merged)

    return out


def split_with_overlap(text: str, target_tokens: int = SAFE_MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[str]:
    """
    자연 분할 + overlap
    """
    parts = split_long_text_naturally(text, target_tokens=target_tokens)
    return apply_overlap(parts, overlap_tokens=overlap_tokens)


def mean_pool(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return [0.0] * SROBERTA_EMBED_DIM
    n = len(vectors)
    out = [0.0] * len(vectors[0])
    for v in vectors:
        for i, x in enumerate(v):
            out[i] += float(x)
    for i in range(len(out)):
        out[i] /= n
    return out


def embed_text_once(text: str) -> list[float]:
    text = (text or "").strip()
    if not text:
        return [0.0] * SROBERTA_EMBED_DIM

    r = requests.post(
        EMBEDDING_ENDPOINT,
        json={"input": [text], "model": VLLM_EMBED_MODEL_ID},
        timeout=60
    )
    r.raise_for_status()
    vec = r.json()["data"][0]["embedding"]
    if len(vec) != SROBERTA_EMBED_DIM:
        raise ValueError(f"dim mismatch: {len(vec)} vs {SROBERTA_EMBED_DIM}")
    return vec


def embed_text_safe(text: str) -> list[float]:
    """
    - 1차: 480 기준 자연 분할 후 각 조각 임베딩
    - 2차: 400류 에러 발생 시 더 잘게(반으로) 쪼개서 재시도
    - 최종: mean pooling 해서 1개 벡터
    """
    text = (text or "").strip()
    if not text:
        return [0.0] * SROBERTA_EMBED_DIM

    pieces = split_long_text_naturally(text, SAFE_MAX_TOKENS)
    vectors = []

    for piece in pieces:
        try:
            vectors.append(embed_text_once(piece))
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 400:
                smaller = split_long_text_naturally(piece, target_tokens=max(120, SAFE_MAX_TOKENS // 2))
                for sp in smaller:
                    vectors.append(embed_text_once(sp))
            else:
                raise

    return mean_pool(vectors)


# =======================
# 1. RAGFlow SDK import
# =======================
try:
    from ragflow_sdk import RAGFlow
except ImportError:
    sys.path.insert(0, str(BASE_DIR / "sdk" / "python"))
    from ragflow_sdk import RAGFlow

from milvus_proxy import MilvusProxy
from ragflow_sdk.modules.dataset import DataSet
from storage.table_store import TableStore

from core.preprocessing.classifier.document_classifier import \
    DocumentClassifier
from core.preprocessing.coverters.hwp_extract import extract_docx_blocks
# =======================
# 2. 커스텀 전처리 모듈 import
# =======================
from core.preprocessing.coverters.hwp_to_docx import HwpAdapter
from core.preprocessing.pipeline import PreprocessPipeline

TABLE_STORAGE_DIR = SCRIPT_DIR / "storage"
table_store = TableStore(TABLE_STORAGE_DIR)

# =======================
# 3. 환경 변수 (RAGFlow)
# =======================
HOST_ADDRESS = os.getenv("RAGFLOW_HOST", "http://localhost")
API_KEY = os.getenv("RAGFLOW_API_KEY")

EMBEDDING_MODEL = os.getenv("RAGFLOW_EMBEDDING_MODEL", "sroberta@local")

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
# ✅ (NEW) chunk 앞에 ①②③...(또는 (1)(2)...) 항 번호만 prefix로 붙이기
# - 기존 청킹 결과는 그대로 두고, "표시용/검색용 prefix"만 최소 개입
# ===========================================================
CIRCLED_SIMPLE = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


def prefix_paragraph_marker(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t

    # 이미 맨 앞에 붙어 있으면 그대로
    if re.match(r"^[%s]" % re.escape(CIRCLED_SIMPLE), t):
        return t
    if re.match(r"^\(\d+\)", t):
        return t

    # 본문 첫 부분(너무 멀리 가면 엉뚱한 곳 잡음)에서만 항 번호 탐색
    head = t[:400]

    m = re.search(r"([%s])" % re.escape(CIRCLED_SIMPLE), head)
    if m:
        return f"{m.group(1)} {t}"

    m = re.search(r"(\(\s*\d+\s*\))", head)
    if m:
        mark = re.sub(r"\s+", "", m.group(1))  # "( 1 )" -> "(1)"
        return f"{mark} {t}"

    return t


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
# 2. heading 패턴 자동 감지 (structured/general 용)
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

    if article >= 3:
        active = ["article"]
    elif article == 0 and diamond >= 1:
        active = ["diamond"]
        if circle + special >= 1:
            active += ["circle", "special"]
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
# 3. 길이 기반 스플릿 함수 (structured/general용)
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
        cand = p if not buf else "\n\n".join(buf + [p])

        if len(cand) <= max_body:
            buf.append(p)
        else:
            flush()
            if len(p) > max_body:
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
# 4. structured/general 청킹
# ===========================================================
def split_text_by_rules(
    raw_text: str,
    heading_patterns: Sequence[str],
    max_chars: int,
    strict_heading_only: bool = False
) -> List[str]:
    """
    structured/general 전용:
    strict_heading_only=True → 길이 기준 분할 OFF (헤딩 단위 유지)
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

    if strict_heading_only:
        return [c for c in coarse if len(c.strip()) > 20]

    final: List[str] = []
    for ch in coarse:
        if len(ch) <= max_chars:
            final.append(ch)
        else:
            final.extend(split_long_chunk_with_heading(ch, max_chars))

    return [c for c in final if len(c.strip()) > 20]


# ===========================================================
# 5. DOCX blocks → chunk (표 분리 포함)
#    ✅ 요청 반영: DOCX에서는 regulation 구조청킹 사용하지 않고 "원래대로"
# ===========================================================
def chunk_docx_blocks_with_rules(
    blocks: list[dict],
    max_chars_structured: int = 2000
) -> list[str]:
    """
    extract_docx_blocks 결과(blocks)를 받아서
    - 표는 JSON 저장 + TABLE 마커 유지
    - structured는 헤딩 기반 청킹
    - regulation/general은 ✅ blocks 순서 그대로(원래대로) 반환
    """

    # blocks가 dict 형태면(구버전) 내부에서 꺼내기
    if isinstance(blocks, dict):
        blocks = blocks.get("blocks") or blocks.get("text_blocks") or blocks.get("items") or []

    ordered_chunks: list[str] = []

    for blk in blocks:
        if blk.get("type") == "text":
            t = (blk.get("text") or "").strip()
            if t:
                ordered_chunks.append(t)

        elif blk.get("type") == "table":
            table = blk.get("table") or {}

            table_store.save_table(
                table_id=table.get("table_id"),
                doc=table.get("doc"),
                page=table.get("page"),
                headers=table.get("headers"),
                rows=table.get("rows"),
            )

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

    if doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=max_chars_structured)

    # ✅ regulation/general 모두 blocks 기반 그대로
    return ordered_chunks


# ==========================================
# HWP / 슬라이드형 PDF 전처리 파이프라인 래퍼
# ==========================================
def preprocess_to_chunks(path: Path, chunk_size: int = 1200) -> list[str]:
    result = preprocess_pipeline.run(str(path), chunk_size=chunk_size)

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


# ===========================================================
# DOCX / TXT / CSV 전용 chunk 함수 (CSV/TXT만)
# ===========================================================
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

    if ext == ".csv":
        raw = extract_text_csv(path)
    else:
        raw = extract_text_txt(path)

    if not raw.strip():
        return []

    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=2000)

    # ✅ 원래대로: 문단 뭉치
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

    raw = "\n".join(pages).strip()
    if not raw:
        return []

    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=2000)

    # ✅ 원래대로: 문단 뭉치
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
MAX_CHUNK_LEN = 8000  # 너무 긴 청크 방지용 (필요하면 4000~6000 정도로 줄여도 됨)

DOMAIN_DIRS = {
    "직무교육":            SCRIPT_DIR / "dataset_직무교육",
    "장애인인식개선교육":   SCRIPT_DIR / "dataset_장애인인식개선",
    "직장내괴롭힘교육":     SCRIPT_DIR / "dataset_괴롭힘교육",
    "직장내성희롱교육":     SCRIPT_DIR / "dataset_성희롱교육",
    "정보보안교육":        SCRIPT_DIR / "dataset_정보보안교육",
    "사내규정":            SCRIPT_DIR / "dataset_사내규정",
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
# 청크 추가 유틸 (RAGFlow + (옵션) Milvus 동시 저장 버전)
# ===========================================================
def add_chunks_safe(
    doc,
    chunks,
    milvus=None,
    dataset_id: str | None = None,
    doc_id: str | None = None,
):
    print(f"→ 생성된 청크 수: {len(chunks)}")

    if not chunks:
        print("⚠ 청크 0개 → 이번 실행에서는 아무 것도 저장하지 않음 (재시도 가능)")
        return

    milvus_payload = []
    added = 0

    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            text = (chunk.get("text") or chunk.get("content") or "").strip()
            chunk_type = chunk.get("type", "text")
        else:
            text = str(chunk).strip()
            chunk_type = "text"

        if not text:
            continue

        # 너무 긴 청크 보호(혹시 모르는 케이스)
        if len(text) > MAX_CHUNK_LEN:
            text = text[:MAX_CHUNK_LEN]

        # ✅ 요청 반영: 앞에 ①② / (1)(2) 만 붙여준다 (청킹은 그대로)
        text = prefix_paragraph_marker(text)

        chash = chunk_hash(text)

        # 1) RAGFlow 저장 (항상)
        doc.add_chunk(content=text)
        added += 1

        # 2) Milvus 저장 (옵션)
        if milvus and dataset_id and doc_id:
            if milvus.exists_chunk_hash(dataset_id, doc_id, chash):
                continue

            try:
                embedding = embed_text_safe(text)
                milvus_payload.append({
                    "dataset_id": dataset_id,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "chunk_hash": chash,
                    "text": text,
                    "embedding": embedding,
                })
            except Exception as e:
                print(f"⚠ embedding 실패 (chunk {idx}): {e}")

        # 미리보기
        if idx <= 2:
            print(f"\n[미리보기 {idx}] ({chunk_type})")
            print(text[:200] + ("..." if len(text) > 200 else ""))

    print(f"→ RAGFlow 청크 추가 완료: {added}개")

    # 3) Milvus 일괄 insert
    if milvus and milvus_payload:
        milvus.insert_chunks(
            dataset_id=dataset_id,
            chunks=milvus_payload,
        )
        print(f"→ Milvus 적재 완료: {len(milvus_payload)}개")
    else:
        print("→ Milvus 적재 없음 (중복 또는 embedding 실패)")


# ===========================================================
# 메인
# ===========================================================
def main():
    print_section("RAGFlow 커스텀 청킹 + add_chunk (HWP/PDF/PPT/DOCX/TXT/CSV 포함)")

    # ------------------------------------
    # 1) 서버 연결
    # ------------------------------------
    print_step(1, "서버 연결")
    try:
        _ = requests.get(
            f"{HOST_ADDRESS}/api/v1/datasets",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=5,
        )
        rag = RAGFlow(API_KEY, HOST_ADDRESS)
        print("✅ RAGFlow 연결 성공")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return

    # ------------------------------------
    # 1-1) Milvus 연결
    # ------------------------------------
    print_step(1, "Milvus 연결")
    try:
        milvus = MilvusProxy(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name="ragflow_chunks_sroberta",
            dim=SROBERTA_EMBED_DIM,
        )
        print("✅ Milvus 연결/컬렉션 준비 완료")
    except Exception as e:
        print(f"❌ Milvus 연결 실패 (일단 RAGFlow만 진행): {e}")
        milvus = None

    # ==============================================
    # 도메인별 Dataset 구성
    # ==============================================
    for domain, dataset_dir in DOMAIN_DIRS.items():
        print("\n" + "#" * 60)
        print(f"### 도메인: {domain}")
        print(f"### 로컬 폴더: {dataset_dir}")
        print("#" * 60)

        if not dataset_dir.exists():
            print(f"⚠️  폴더가 없습니다. 스킵: {dataset_dir}")
            continue

        # ------------------------------------
        # 2) 도메인 폴더 안 파일 스캔
        # ------------------------------------
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

        # ------------------------------------
        # 3) 도메인별 Dataset 생성
        # ------------------------------------
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

        # ------------------------------------
        # 4) 파일별 업로드 + 청킹
        # ------------------------------------
        print_step(4, f"[{domain}] 파일 업로드 + 청킹")

        for fpath in files:
            fpath = fpath.resolve()
            ext = fpath.suffix.lower().lstrip(".")
            print(f"\n======= [{domain}] {fpath.name} 처리 =======")

            # 4-1. HWP/HWPX → DOCX 변환
            if ext in ("hwp", "hwpx"):
                print(f"[HWP] {fpath.name} → DOCX로 변환")
                docx_path = hwp_adapter.to_docx(str(fpath))
                fpath = Path(docx_path)
                ext = "docx"

            # 4-2. PDF / PPT / PPTX 처리
            if ext in ("pdf", "ppt", "pptx"):
                if ext == "pdf":
                    doc_type = classifier.classify(str(fpath))
                else:
                    doc_type = "ppt"

                print(f"→ 문서 타입: {doc_type}")

                # 텍스트 기반 PDF
                if doc_type == "text_pdf":
                    print("→ [텍스트 PDF] 로컬 청킹 사용")

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
                    )
                    continue

                # 이미지 기반 PDF / PPT
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
                add_chunks_safe(doc, chunks, milvus=milvus, dataset_id=domain, doc_id=fpath.name)
                print(f"→ 파이프라인 청크 {len(chunks)}개 반환")
                continue

            # 4-3. CSV / DOCX / TXT
            if ext in ("csv", "docx", "txt"):
                print("→ [CSV/DOCX/TXT] 로컬 청킹 사용")

                with open(fpath, "rb") as fb:
                    blob = fb.read()

                doc = dataset.upload_documents(
                    [{"display_name": fpath.name, "blob": blob}]
                )[0]
                print(f"→ 업로드 완료 (doc.id={doc.id})")

                # DOCX: blocks 기반
                if ext == "docx":
                    print("→ DOCX blocks 기반 처리 (원래대로) + prefix(①②...)만 부여")

                    blocks = extract_docx_blocks(fpath)
                    docx_chunks = chunk_docx_blocks_with_rules(blocks)

                    add_chunks_safe(
                        doc,
                        docx_chunks,
                        milvus=milvus,
                        dataset_id=domain,
                        doc_id=fpath.name,
                    )
                    continue

                # CSV/TXT
                chunks = chunk_document(fpath)
                compare_with_solution(dataset_dir, fpath, chunks)

                add_chunks_safe(
                    doc,
                    chunks,
                    milvus=milvus,
                    dataset_id=domain,
                    doc_id=fpath.name,
                )
                continue

            print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")

        # ------------------------------------
        # 5) 도메인별 검색 테스트
        # ------------------------------------
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

    print("\n✅ 전체 완료")


if __name__ == "__main__":
    main()
