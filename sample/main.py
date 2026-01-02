"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
HWP / PDF / PPT / DOCX / TXT / CSV 자동 처리 + 문서 타입 판별 + 자동 패턴 감지까지 완성

✅ 추가:
- --input / --domain 인자 지원
  - --input이 있으면: 업로드된 "단일 파일"만 처리하고 종료
  - 없으면: 기존 DOMAIN_DIRS 배치 처리

✅ 운영형 수정(핵심):
- dataset을 매번 새로 만들지 않고 "도메인당 1개 고정 dataset"을 재사용(get_or_create)
- replace=true면 Milvus 뿐 아니라 RAGFlow 문서도 삭제 후 재업로드
- add_chunk에서 content 변수 사용 버그 수정 (content=text -> content=content)

✅ 신규 요구사항 반영(핵심):
- --ingest-id / --meta-json 인자 지원 (ingest-worker가 넘김)
- 처리 결과는 main.py가 직접 콜백하지 않고, ingest-worker가 콜백함
- main.py는 worker가 파싱할 수 있도록 stdout에 다음 라인을 남김:
  INGEST_STATS_JSON={"chunks":123, ...}
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from uuid import uuid4
from difflib import SequenceMatcher

# =======================
# 0. 경로/환경 설정
# =======================
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

    resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=text)
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


def safe_ascii(s: str) -> str:
    if not s:
        return "default"
    # ascii로 slug 시도
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", s).strip("_")
    if slug:
        return slug[:40]
    # 전부 한글/특수면 해시로
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


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
# ✅ (신규) worker가 파싱할 ingest stats 출력
# ===========================================================
def print_ingest_stats(chunks: int, extra: Optional[Dict[str, Any]] = None) -> None:
    """
    ingest-worker(app.py)가 stdout에서 파싱하는 라인:
      INGEST_STATS_JSON={...}
    """
    payload: Dict[str, Any] = {"chunks": int(chunks or 0)}
    if extra and isinstance(extra, dict):
        payload.update(extra)
    try:
        print("INGEST_STATS_JSON=" + json.dumps(payload, ensure_ascii=False))
    except Exception:
        # 어떤 상황에서도 최소 형태는 남기기
        print("INGEST_STATS_JSON=" + '{"chunks":%d}' % int(chunks or 0))


# ===========================================================
# (운영형 핵심) RAGFlow REST 헬퍼 (SDK가 없는 기능도 안정적으로 처리)
# ===========================================================
def _auth_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}"}


def _list_datasets_rest() -> List[Dict[str, Any]]:
    url = f"{HOST_ADDRESS}/api/v1/datasets"
    resp = requests.get(url, headers=_auth_headers(), timeout=20)
    resp.raise_for_status()
    data = resp.json()
    # 다양한 응답 형태 방어
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "datasets" in data and isinstance(data["datasets"], list):
            return data["datasets"]
    if isinstance(data, list):
        return data
    return []


def _find_dataset_by_name(dataset_name: str) -> Optional[Dict[str, Any]]:
    # 1) SDK에 목록 기능이 있으면 우선 사용 (형태가 버전에 따라 달라서 여기선 패스)
    try:
        if hasattr(RAGFlow, "list_datasets") and callable(getattr(RAGFlow, "list_datasets")):
            pass
    except Exception:
        pass

    # 2) REST로 찾기
    try:
        datasets = _list_datasets_rest()
        for d in datasets:
            if (d.get("name") or d.get("dataset_name")) == dataset_name:
                return d
    except Exception as e:
        print(f"⚠ dataset 목록 조회 실패(REST): {e}")
    return None


def get_or_create_dataset(rag: Any, domain: str) -> Any:
    """
    도메인당 1개 고정 dataset 재사용.
    - name: domain_{safe_ascii(domain)}
    """
    dataset_name = f"domain_{safe_ascii(domain)}"

    found = _find_dataset_by_name(dataset_name)
    if found and found.get("id"):
        ds_id = found["id"]
        print(f"✅ [{domain}] 기존 Dataset 재사용: {ds_id} (name={dataset_name})")

        # SDK로 dataset 객체를 얻는 다양한 케이스 방어
        try:
            if hasattr(rag, "get_dataset") and callable(getattr(rag, "get_dataset")):
                return rag.get_dataset(ds_id)
            if hasattr(rag, "dataset") and callable(getattr(rag, "dataset")):
                return rag.dataset(ds_id)
        except Exception:
            pass

        # 최후: DataSet 객체를 직접 만들어서 id만 주입 시도
        try:
            ds = DataSet(rag, {"id": ds_id})
            ds.id = ds_id  # type: ignore
            return ds
        except Exception:
            print("⚠ SDK로 기존 dataset 핸들 생성 실패 → 새로 생성 시도(이름 중복 가능)")

    # 없으면 생성
    parser_config = DataSet.ParserConfig(rag, {"raptor": {"use_raptor": False}})
    dataset = rag.create_dataset(
        name=dataset_name,
        description=f"{domain} 운영형 고정 데이터셋",
        chunk_method="manual",
        embedding_model=EMBEDDING_MODEL,
        parser_config=parser_config,
    )
    print(f"✅ [{domain}] Dataset 생성 완료: {dataset.id} (name={dataset_name})")
    return dataset


def _list_documents_rest(dataset_id: str) -> List[Dict[str, Any]]:
    url = f"{HOST_ADDRESS}/api/v1/datasets/{dataset_id}/documents"
    resp = requests.get(url, headers=_auth_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            return data["data"]
        if "documents" in data and isinstance(data["documents"], list):
            return data["documents"]
    if isinstance(data, list):
        return data
    return []


def _delete_document_rest(dataset_id: str, document_id: str) -> bool:
    candidates = [
        f"{HOST_ADDRESS}/api/v1/datasets/{dataset_id}/documents/{document_id}",
        f"{HOST_ADDRESS}/api/v1/datasets/{dataset_id}/document/{document_id}",
    ]
    for url in candidates:
        try:
            resp = requests.delete(url, headers=_auth_headers(), timeout=30)
            if resp.status_code in (200, 204):
                return True
        except Exception:
            continue
    return False


def delete_ragflow_document_by_name(dataset: Any, effective_doc_name: str) -> bool:
    """
    replace=true일 때 RAGFlow쪽 문서도 지우기.
    - display_name / name / filename 중 하나가 effective_doc_name과 동일한 문서를 삭제 시도
    """
    ds_id = getattr(dataset, "id", None)
    if not ds_id:
        return False

    try:
        docs = _list_documents_rest(str(ds_id))
        target_ids = []
        for d in docs:
            name = d.get("display_name") or d.get("name") or d.get("filename") or d.get("doc_name")
            if name == effective_doc_name and d.get("id"):
                target_ids.append(d["id"])
        deleted_any = False
        for did in target_ids:
            ok = _delete_document_rest(str(ds_id), str(did))
            deleted_any = deleted_any or ok
        return deleted_any
    except Exception as e:
        print(f"⚠ RAGFlow 문서 삭제 실패(REST): {e}")
        return False


# ===========================================================
# 1. 문서 타입 자동 판단 (텍스트 기반 규정/방침 판별용)
# ===========================================================
def detect_document_type(raw_text: str) -> str:
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
                    part = p[s : s + max_body]
                    chunks.append(heading + "\n" + part)
                    s += max_body
            else:
                buf = [p]

    flush()
    return chunks


def split_text_by_rules(
    raw_text: str,
    heading_patterns: Sequence[str],
    max_chars: int,
    strict_heading_only: bool = False,
) -> List[str]:
    lines = raw_text.splitlines()
    compiled = [re.compile(p) for p in heading_patterns]
    coarse: List[str] = []
    buf: List[str] = []

    def is_heading(line: str) -> bool:
        s = line.strip()
        return any(pat.match(s) for pat in compiled)

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


def chunk_docx_blocks_with_rules(blocks: list[dict], max_chars_structured: int = 2000) -> list[str]:
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
                ordered_chunks[-1] += f"\n\n[TABLE:{table_id}] 표 데이터는 별도 JSON으로 저장되어 있습니다."
            else:
                ordered_chunks.append(f"[TABLE:{table_id}] 표 데이터는 별도 JSON으로 저장되어 있습니다.")

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
            if isinstance(rj, dict) and "chunks" in rj and isinstance(rj["chunks"], list):
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
    "직무교육": SCRIPT_DIR / "dataset_직무교육",
    "장애인인식개선교육": SCRIPT_DIR / "dataset_장애인인식개선",
    "직장내괴롭힘교육": SCRIPT_DIR / "dataset_괴롭힘교육",
    "직장내성희롱교육": SCRIPT_DIR / "dataset_성희롱교육",
    "정보보안교육": SCRIPT_DIR / "dataset_정보보안교육",
    "사내규정": SCRIPT_DIR / "dataset_사내규정",
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
    반환: 실제로 RAGFlow에 add_chunk된 개수
    """
    print(f"→ 생성된 청크 수: {len(chunks)}")

    if not chunks:
        print("⚠ 청크 0개 → 저장 스킵")
        return 0

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

        # ✅ FIX: content 변수 사용 (원래 버그: content=text)
        doc.add_chunk(content=content)
        added += 1

        if milvus and dataset_id and doc_id:
            if milvus.exists_chunk_hash(dataset_id, doc_id, chash):
                continue

            try:
                embedding = embedder.embed(text, embedding_model)
                milvus_payload.append(
                    {
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
                    }
                )
            except Exception as e:
                print(f"⚠ embedding 실패 (chunk {idx}): {e}")

        if idx <= 2:
            print(f"\n[미리보기 {idx}] ({chunk_type})")
            print(text[:200] + ("..." if len(text) > 200 else ""))

    print(f"→ RAGFlow 청크 추가 완료: {added}개")

    if milvus and milvus_payload:
        milvus.insert_chunks(dataset_id=dataset_id, chunks=milvus_payload)
        print(f"→ Milvus 적재 완료: {len(milvus_payload)}개")
    else:
        print("→ Milvus 적재 없음 (중복 또는 embedding 실패)")

    return added


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
) -> int:
    """
    --input 으로 들어온 파일 1개만 처리
    - dataset은 "도메인당 1개 고정"으로 재사용
    - replace=true면 (1) RAGFlow 문서 삭제 (2) Milvus 삭제 후 재업로드
    반환: 실제 add_chunk된 개수
    """
    print("\n" + "#" * 60)
    print("### 단일 파일 모드")
    print(f"### 도메인: {domain}")
    print(f"### 파일: {input_path}")
    print("#" * 60)

    dataset = get_or_create_dataset(rag, domain)

    fpath = input_path.resolve()
    ext = fpath.suffix.lower().lstrip(".")
    print(f"\n======= [{domain}] {fpath.name} 처리 =======")

    effective_doc_id = (doc_id or fpath.name)

    # ✅ 운영형: replace=true면 RAGFlow 문서도 삭제
    if replace:
        try:
            deleted = delete_ragflow_document_by_name(dataset, effective_doc_id)
            if deleted:
                print(f"✅ replace=true → RAGFlow 기존 문서 삭제 완료 (name={effective_doc_id})")
            else:
                print(f"ℹ replace=true → RAGFlow 기존 문서 없음/삭제불가 (name={effective_doc_id})")
        except Exception as e:
            print(f"⚠ replace RAGFlow 삭제 실패: {e}")

        if milvus:
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

    # ✅ display_name을 effective_doc_id로 고정 → replace 시 찾기 쉬움
    doc = dataset.upload_documents([{"display_name": effective_doc_id, "blob": blob}])[0]
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
            added = add_chunks_safe(
                doc,
                chunks,
                milvus=milvus,
                dataset_id=domain,
                doc_id=effective_doc_id,
                embedding_model=embedding_model_selected,
                experiment_tag=experiment_tag,
            )
            return int(added)

        print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")
        pipeline_result = preprocess_pipeline.run(str(fpath))
        chunks = pipeline_result.get("chunks", []) if isinstance(pipeline_result, dict) else []
        added = add_chunks_safe(
            doc,
            chunks,
            milvus=milvus,
            dataset_id=domain,
            doc_id=effective_doc_id,
            embedding_model=embedding_model_selected,
            experiment_tag=experiment_tag,
        )
        print(f"→ 파이프라인 청크 {len(chunks)}개 반환")
        return int(added)

    # CSV / DOCX / TXT
    if ext in ("csv", "docx", "txt"):
        print("→ [CSV/DOCX/TXT] 기존 규정형 청킹 사용")

        if ext == "docx":
            print("→ DOCX blocks 기반 처리 + 조단위(제 n 조) 청킹 적용")
            blocks = extract_docx_blocks(fpath)
            docx_chunks = chunk_docx_blocks_with_rules(blocks)
            added = add_chunks_safe(
                doc,
                docx_chunks,
                milvus=milvus,
                dataset_id=domain,
                doc_id=effective_doc_id,
                embedding_model=embedding_model_selected,
                experiment_tag=experiment_tag,
            )
            return int(added)

        chunks = chunk_document(fpath)
        added = add_chunks_safe(
            doc,
            chunks,
            milvus=milvus,
            dataset_id=domain,
            doc_id=effective_doc_id,
            embedding_model=embedding_model_selected,
            experiment_tag=experiment_tag,
        )
        return int(added)

    print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")
    return 0


def _parse_meta_json(meta_json: str | None) -> Dict[str, Any]:
    if not meta_json:
        return {}
    try:
        obj = json.loads(meta_json)
        if isinstance(obj, dict):
            return obj
        return {"_meta_raw": obj}
    except Exception:
        return {"_meta_raw": meta_json}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="업로드 단일 파일 처리 경로")
    parser.add_argument("--domain", default="default", help="단일 파일 모드 도메인명")

    # ✅ 기존 유지 + alias 추가
    parser.add_argument("--doc_id", default=None, help="문서 식별자(docId)")
    parser.add_argument("--doc-id", dest="doc_id", default=None, help="문서 식별자(docId) (alias)")

    parser.add_argument("--version", type=int, default=None, help="문서 버전")
    parser.add_argument("--replace", default="false", help="true면 기존 docId 교체")

    # ✅ 신규: worker가 넘기는 추적 인자
    parser.add_argument("--ingest-id", dest="ingest_id", default=None, help="ingest 작업 식별자(추적용)")
    parser.add_argument("--meta-json", dest="meta_json", default=None, help="meta JSON 문자열")

    args = parser.parse_args()
    replace_flag = str(args.replace).lower() in ("1", "true", "yes", "y", "on")

    ingest_id = args.ingest_id or str(uuid4())
    meta = _parse_meta_json(args.meta_json)

    # docId는 단일 파일 모드에서 doc_id 우선, 없으면 파일명
    if args.input:
        effective_doc_id = args.doc_id or Path(args.input).name
    else:
        effective_doc_id = args.doc_id or "batch"

    print_section("RAGFlow 커스텀 청킹 + add_chunk (운영형: dataset 재사용/replace 지원 + worker-callback)")

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

    global openai_client
    if EMBEDDING_MODEL_SELECTED == "openai":
        if OpenAI is None:
            raise RuntimeError("❌ openai 패키지가 설치되어 있지 않습니다.")
        if not OPENAI_API_KEY:
            raise RuntimeError("❌ EMBEDDING_MODEL_SELECTED=openai 인데 OPENAI_API_KEY가 없습니다.")
        openai_client = OpenAI(api_key=OPENAI_API_KEY)

    chunks_added = 0
    milvus = None

    # stats에 남길 부가 정보(원하면 더 추가 가능)
    stats_extra = {
        "ingestId": ingest_id,
        "docId": effective_doc_id,
        "version": args.version,
        "domain": args.domain if args.input else None,
        "replace": replace_flag,
        "status": "UNKNOWN",
    }

    try:
        # ------------------------------------
        # 1) 서버 연결
        # ------------------------------------
        print_step(1, "서버 연결")
        _ = requests.get(
            f"{HOST_ADDRESS}/api/v1/datasets",
            headers=_auth_headers(),
            timeout=10,
        )
        rag = RAGFlow(API_KEY, HOST_ADDRESS)
        print("✅ RAGFlow 연결 성공")

        # ------------------------------------
        # 2) Milvus 연결
        # ------------------------------------
        print_step(2, "Milvus 연결")
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
                raise RuntimeError(f"단일 파일 경로가 존재하지 않습니다: {input_path}")

            chunks_added = process_single_file_mode(
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

            stats_extra["status"] = "COMPLETED"
            print_ingest_stats(chunks_added, extra=stats_extra)
            return

        # ==============================================
        # ★ 도메인별 배치 처리 (기존 유지)
        # ==============================================
        # 배치 모드는 ingest-worker 콜백 스펙(단일 doc 단위)과 매칭이 애매해서,
        # 여기서는 stats 출력만 하되 status는 BATCH로 남김.
        for domain, dataset_dir in DOMAIN_DIRS.items():
            print("\n" + "#" * 60)
            print(f"### 도메인: {domain}")
            print(f"### 로컬 폴더: {dataset_dir}")
            print("#" * 60)

            if not dataset_dir.exists():
                print(f"⚠️  폴더가 없습니다. 스킵: {dataset_dir}")
                continue

            print_step(3, f"[{domain}] dataset 폴더 스캔")

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

            print_step(4, f"[{domain}] 데이터셋 확보(get_or_create)")
            dataset = get_or_create_dataset(rag, domain)

            print_step(5, f"[{domain}] 파일 업로드 + 청킹")

            for fpath in files:
                fpath = fpath.resolve()
                ext = fpath.suffix.lower().lstrip(".")
                print(f"\n======= [{domain}] {fpath.name} 처리 =======")

                effective_doc_id_batch = fpath.name

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

                    with open(fpath, "rb") as fb:
                        blob = fb.read()

                    doc = dataset.upload_documents([{"display_name": effective_doc_id_batch, "blob": blob}])[0]
                    print(f"→ 업로드 완료 (doc.id={doc.id})")

                    if doc_type == "text_pdf":
                        print("→ [텍스트 PDF] 로컬 규정형 청킹 사용")
                        chunks = chunk_text_pdf(fpath)
                        compare_with_solution(dataset_dir, fpath, chunks)

                        chunks_added += int(
                            add_chunks_safe(
                                doc,
                                chunks,
                                milvus=milvus,
                                dataset_id=domain,
                                doc_id=effective_doc_id_batch,
                                embedding_model=EMBEDDING_MODEL_SELECTED,
                                experiment_tag=EXPERIMENT_TAG,
                            )
                        )
                        continue

                    print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")
                    pipeline_result = preprocess_pipeline.run(str(fpath))
                    chunks = pipeline_result.get("chunks", []) if isinstance(pipeline_result, dict) else []
                    chunks_added += int(
                        add_chunks_safe(
                            doc,
                            chunks,
                            milvus=milvus,
                            dataset_id=domain,
                            doc_id=effective_doc_id_batch,
                            embedding_model=EMBEDDING_MODEL_SELECTED,
                            experiment_tag=EXPERIMENT_TAG,
                        )
                    )
                    print(f"→ 파이프라인 청크 {len(chunks)}개 반환")
                    continue

                if ext in ("csv", "docx", "txt"):
                    print("→ [CSV/DOCX/TXT] 기존 규정형 청킹 사용")

                    with open(fpath, "rb") as fb:
                        blob = fb.read()

                    doc = dataset.upload_documents([{"display_name": effective_doc_id_batch, "blob": blob}])[0]
                    print(f"→ 업로드 완료 (doc.id={doc.id})")

                    if ext == "docx":
                        print("→ DOCX blocks 기반 처리 + 조단위(제 n 조) 청킹 적용")
                        blocks = extract_docx_blocks(fpath)
                        docx_chunks = chunk_docx_blocks_with_rules(blocks)

                        chunks_added += int(
                            add_chunks_safe(
                                doc,
                                docx_chunks,
                                milvus=milvus,
                                dataset_id=domain,
                                doc_id=effective_doc_id_batch,
                                embedding_model=EMBEDDING_MODEL_SELECTED,
                                experiment_tag=EXPERIMENT_TAG,
                            )
                        )
                        continue

                    chunks = chunk_document(fpath)
                    compare_with_solution(dataset_dir, fpath, chunks)

                    chunks_added += int(
                        add_chunks_safe(
                            doc,
                            chunks,
                            milvus=milvus,
                            dataset_id=domain,
                            doc_id=effective_doc_id_batch,
                            embedding_model=EMBEDDING_MODEL_SELECTED,
                            experiment_tag=EXPERIMENT_TAG,
                        )
                    )
                    continue

                print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")

            print_step(6, f"[{domain}] 검색 테스트")
            query = f"{domain} 관련 문서의 목적은 무엇인가?"
            results = rag.retrieve(dataset_ids=[dataset.id], question=query, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"\n[검색 {i}]")
                print(r.content[:200] + "...")

        stats_extra["status"] = "BATCH_COMPLETED"
        print_ingest_stats(chunks_added, extra=stats_extra)
        return

    except Exception as e:
        # ✅ 실패여도 worker가 stdout에서 stats 뽑을 수 있게 남김
        stats_extra["status"] = "FAILED"
        stats_extra["error"] = str(e)[:500]
        print_ingest_stats(0, extra=stats_extra)
        # worker가 stderr로 잡을 수 있게 re-raise
        raise


if __name__ == "__main__":
    main()
