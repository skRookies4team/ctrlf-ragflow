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

import milvus_proxy
from milvus_proxy import MilvusProxy

print("🔥 milvus_proxy loaded from:", milvus_proxy.__file__)

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
HOST_ADDRESS = (
    os.getenv("RAGFLOW_BASE_URL")
    or os.getenv("RAGFLOW_HOST")
    or os.getenv("HOST_ADDRESS")
    or "http://ragflow-cpu:80"
)
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


def safe_dataset_name(domain: str) -> str:
    # 한글/영문/숫자/공백/하이픈/언더스코어 허용
    s = (domain or "default").strip()
    s = re.sub(r"[^\w가-힣\s-]+", "", s)      # 특수문자 제거
    s = re.sub(r"\s+", "_", s)               # 공백 -> _
    s = s.strip("_")
    if not s:
        return "default"
    return s[:50]


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
    base = extra or {}
    meta_obj = base.get("meta") or {}
    meta_safe = dict(meta_obj) if isinstance(meta_obj, dict) else {"_meta_raw": meta_obj}

    payload = {
        "ingestId": base.get("ingestId"),
        "docId": base.get("docId"),
        "version": base.get("version"),
        "status": base.get("status"),
        "meta": meta_safe,
        "stats": {"chunks": int(chunks or 0)},
    }

    for k in ("ragDatasetPk", "ragDocumentPk", "uploadName", "domain", "replace", "error", "systemDocId"):
        v = base.get(k)
        if v is not None:
            payload["meta"][k] = v

    # ✅ 1줄 + 직렬화 안전
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    print("INGEST_STATS_JSON=" + line)


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
    dataset_name = f"domain_{safe_dataset_name(domain)}"

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

def _get_dataset_id(dataset: Any) -> Optional[str]:
    """
    RAGFlow SDK dataset 객체는 버전에 따라 id 필드가 다를 수 있어서
    여러 후보를 순차적으로 확인.
    """
    if dataset is None:
        return None

    for key in ("id", "dataset_id", "datasetId"):
        v = getattr(dataset, key, None)
        if v:
            return str(v)

    # dataset.data / dataset._data 형태 방어
    data = getattr(dataset, "data", None) or getattr(dataset, "_data", None)
    if isinstance(data, dict) and data.get("id"):
        return str(data["id"])

    # dict로 넘어오는 경우
    if isinstance(dataset, dict) and dataset.get("id"):
        return str(dataset["id"])

    return None


def delete_ragflow_document_by_name(dataset: Any, effective_doc_name: str) -> bool:
    """
    replace=true일 때 RAGFlow쪽 문서도 지우기.
    - display_name / name / filename 중 하나가 effective_doc_name과 동일한 문서를 삭제 시도
    """
    ds_id = _get_dataset_id(dataset)
    if not ds_id:
        print("⚠ delete_ragflow_document_by_name: dataset id not found -> skip")
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

DEPT_KEYWORDS = [
    "전체 부서",
    "총무팀",
    "기획팀",
    "마케팅팀",
    "인사팀",
    "재무팀",
    "개발팀",
    "영업팀",
    "법무팀",
]

def infer_department_from_filename(filename: str) -> str:
    """
    파일명에 부서명이 포함되면 해당 부서를 반환.
    없으면 '전체 부서' 반환.
    """
    name = (filename or "").replace(" ", "").strip()

    # 우선순위: '전체'가 섞여있으면 전체 부서로 처리하고 싶다면 여기서 먼저 체크
    if "전체" in name:
        return "전체 부서"

    for dept in DEPT_KEYWORDS:
        if dept == "전체 부서":
            continue
        if dept.replace(" ", "") in name:
            return dept

    return "전체 부서"


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
    dataset_id: str | None = None,
    doc_id: str | None = None,
    embedding_model="openai",
    experiment_tag=None,
    department: str | None = None,   # ✅ 추가
    document_title: str | None = None,
):
    """
    RAGFlow: 무조건 chunk 추가
    Milvus: chunk_hash 기준 중복 차단 후 일괄 insert
    반환: 실제로 RAGFlow에 add_chunk된 개수
    """
    print(f"[TRACE][ADD] start: chunks={len(chunks)} dataset_id={dataset_id} doc_id={doc_id} embedding_model={embedding_model} exp={experiment_tag}")

    if not chunks:
        print("[TRACE][ADD] ⚠ chunks=0 -> skip (no saving)")
        return 0

    milvus_payload = []
    added = 0

    # 통계용 카운터
    addchunk_ok = 0
    addchunk_fail = 0
    exists_check_ok = 0
    exists_check_fail = 0
    dup_skip = 0
    embed_ok = 0
    embed_fail = 0

    for idx, chunk in enumerate(chunks, start=1):
        # ---- normalize ----
        try:
            if isinstance(chunk, dict):
                text = (chunk.get("text") or "").strip()
                chunk_type = chunk.get("type", "text")
                metadata = {
                    "source": chunk_type,
                    "page_index": chunk.get("page_index"),
                    "page_num": chunk.get("page_num"),              # ✅ 추가
                    "section": chunk.get("section"),                # ✅ 추가
                    "section_path": chunk.get("section_path"),      # ✅ 추가
                    "document_title": chunk.get("document_title"),  # ✅ 추가
                    "image_path": chunk.get("image_path"),
                }
            else:
                text = str(chunk).strip()
                chunk_type = "text"
                metadata = {"source": "text"}
        except Exception as e:
            print(f"[TRACE][ADD][ERROR] normalize failed idx={idx}: {repr(e)}")
            continue

        if not text:
            if idx <= 3:
                print(f"[TRACE][ADD] empty text skip idx={idx} type={chunk_type}")
            continue

        chash = chunk_hash(text)

        if chunk_type == "image_caption" and metadata.get("image_path"):
            content = text + f"\n\n[IMAGE_PATH]{metadata['image_path']}[/IMAGE_PATH]"
        else:
            content = text

        # ✅ ChatSource 메타 (없으면 안전 기본값)
        doc_title = (metadata.get("document_title") or document_title or "").strip()[:256]

        # page_num: chunk에 page_num이 있으면 그대로, 없으면 page_index(+1)로 유추
        page_num_val = metadata.get("page_num", None)
        if page_num_val is None and metadata.get("page_index") is not None:
            try:
                page_num_val = int(metadata["page_index"]) + 1
            except Exception:
                page_num_val = None

        page_num = None
        if page_num_val not in (None, ""):
            try:
                page_num = int(page_num_val)
            except Exception:
                page_num = None

        section = (metadata.get("section") or "").strip()[:128]
        section_path = (metadata.get("section_path") or "").strip()[:256]


        # ---- RAGFlow add_chunk ----
        try:
            doc.add_chunk(content=content)
            added += 1
            addchunk_ok += 1
        except Exception as e:
            addchunk_fail += 1
            print(f"[TRACE][ADD][ERROR] doc.add_chunk failed idx={idx} len={len(content)}: {repr(e)}")
            # add_chunk 실패하면 milvus도 의미 없으니 다음 chunk로
            continue

        # ---- Milvus payload build ----
        if milvus and dataset_id and doc_id:
            # 1) 중복 체크
            try:
                exists_check_ok += 1
                if milvus.exists_chunk_hash(dataset_id, doc_id, chash, department=department):
                    dup_skip += 1
                    if dup_skip <= 3:
                        print(f"[TRACE][ADD] dup_skip idx={idx} hash={chash[:10]} dept={department}...")
                    continue
            except Exception as e:
                exists_check_fail += 1
                # 중복 체크가 실패해도 insert 자체는 시도할 수 있게 계속 진행
                print(f"[TRACE][ADD][WARN] exists_chunk_hash failed idx={idx}: {repr(e)} (continue to embed/insert)")

            # 2) 임베딩 생성
            try:
                embedding = embedder.embed(text, embedding_model)
                embed_ok += 1

                # 임베딩 차원/형태 확인 (초반 1~2개만)
                if embed_ok <= 2:
                    if isinstance(embedding, list):
                        print(f"[TRACE][ADD] embedding_ok idx={idx} dim={len(embedding)}")
                    else:
                        print(f"[TRACE][ADD][WARN] embedding type idx={idx}: {type(embedding)}")

                # section/section_path 기본 채움(텍스트 chunk에서도 최소 UX 확보)
                if not section:
                    first_line = (text.splitlines()[0].strip() if text else "")
                    section = first_line[:128]
                if not section_path and section:
                    section_path = section[:256]

                milvus_payload.append(
                    {
                        "dataset_id": dataset_id,
                        "doc_id": doc_id,
                        "chunk_id": idx,
                        "chunk_hash": chash,
                        "text": text,
                        "embedding": embedding,
                        "department": (department or "ALL").strip()[:32],  # ✅ 이걸로 끝

                        # ✅ ChatSource 메타 4종
                        "document_title": doc_title,
                        "page_num": page_num,
                        "section": section,
                        "section_path": section_path,
                    }
                )
            except Exception as e:
                embed_fail += 1
                print(f"[TRACE][ADD][ERROR] ⚠ embedding failed idx={idx} len(text)={len(text)}: {repr(e)}")

        # ---- preview ----
        if idx <= 2:
            print(f"\n[TRACE][ADD][PREVIEW {idx}] ({chunk_type})")
            print(text[:200] + ("..." if len(text) > 200 else ""))

    print(f"[TRACE][ADD] ragflow_added={added} addchunk_ok={addchunk_ok} addchunk_fail={addchunk_fail}")
    print(f"[TRACE][ADD] exists_check_ok={exists_check_ok} exists_check_fail={exists_check_fail} dup_skip={dup_skip}")
    print(f"[TRACE][ADD] embed_ok={embed_ok} embed_fail={embed_fail} milvus_payload={len(milvus_payload)}")

    # ---- Milvus insert ----
    if milvus and dataset_id and doc_id and not milvus_payload:
        # ✅ 운영 정책: Milvus 필수인데 넣을 게 없으면 실패 처리
        raise RuntimeError(
            f"Milvus payload is empty -> FAIL by policy "
            f"(dataset_id={dataset_id}, doc_id={doc_id}, chunks={len(chunks)})"
        )

    if milvus and milvus_payload:
        print(f"[TRACE][ADD] calling milvus.insert_chunks payload={len(milvus_payload)} dataset_id={dataset_id} doc_id={doc_id}")
        try:
            milvus.insert_chunks(dataset_id=dataset_id, chunks=milvus_payload)
            print(f"[TRACE][ADD] ✅ milvus.insert_chunks OK inserted={len(milvus_payload)}")
        except Exception as e:
            print(f"[TRACE][ADD][ERROR] ❌ milvus.insert_chunks failed: {repr(e)}")
            raise RuntimeError(f"Milvus insert failed: {repr(e)}") from e
    else:
        reason = []
        if not milvus:
            reason.append("milvus=None")
        if not milvus_payload:
            reason.append("payload=0")
        print(f"[TRACE][ADD] milvus skipped ({', '.join(reason)})")

    return added

def process_single_file_mode(
    rag,
    milvus,
    embedding_model_selected: str,
    experiment_tag: str,
    input_path: Path,
    domain: str,
    doc_id: str | None,          # ✅ 이제 "원본 문서키"로 쓸 값 (없으면 파일명 그대로)
    version: int | None,
    replace: bool,
    milvus_dataset_id: str,
    milvus_doc_id: str | None,   # ✅ 이제 "원본 문서키"로 쓸 값 (없으면 doc_id로 확정)
    department: str,
) -> tuple[int, str, dict]:
    """
    단일 파일 모드 정책(권장):
    - ✅ Milvus/docId(시스템 키): 원본 파일명 유지 (.hwp 포함)
    - ✅ RAGFlow 업로드명: 변환 후 확장자(.docx 등)로 업로드
    - ✅ replace 삭제도 분리:
        - RAGFlow: 업로드명(uploadName) 기준 삭제
        - Milvus: 원본 docId 기준 삭제
    """
    print("\n" + "#" * 60)
    print("### 단일 파일 모드")
    print(f"### domain(arg) = {domain}")
    print(f"### input_path = {input_path}")
    print(f"### embedding_model_selected = {embedding_model_selected}")
    print("#" * 60)

    # ---------------------------
    # [1] Dataset 결정
    # ---------------------------
    dataset = get_or_create_dataset(rag, domain)
    print(
        "[TRACE-1] dataset resolved -> "
        f"id={getattr(dataset, 'id', None)}, "
        f"name={getattr(dataset, 'name', None)}"
    )

    orig_path = input_path.expanduser().resolve()
    if not orig_path.exists():
        raise RuntimeError(f"단일 파일 경로가 존재하지 않습니다: {orig_path}")

    orig_ext = orig_path.suffix.lower()  # ".hwp" / ".pdf" ...
    orig_name = orig_path.name          # "파일.hwp"

    # ✅ Milvus/시스템 docId: "원본 파일명(.hwp 포함)"을 기본으로
    # - 외부에서 doc_id/milvus_doc_id를 줬으면 그걸 우선
    # - 안 줬으면 input 파일명 그대로
    system_doc_id = (milvus_doc_id or doc_id or orig_name).strip()
    if not system_doc_id:
        system_doc_id = orig_name

    print(f"[TRACE-2] system_doc_id (Milvus key) = {system_doc_id}")

    print(f"[DEPT] single-file department={department}")

    # ---------------------------
    # [2] HWP/HWPX -> DOCX 변환 (업로드용)
    # ---------------------------
    upload_path = orig_path
    if orig_ext in (".hwp", ".hwpx"):
        print(f"[TRACE-3] HWP/HWPX detected → DOCX convert")
        docx_path = hwp_adapter.to_docx(str(orig_path))
        upload_path = Path(docx_path).resolve()
        if not upload_path.exists():
            raise RuntimeError(f"HWP 변환 실패: {upload_path}")

    upload_ext = upload_path.suffix.lower()  # ".docx" 등
    upload_name = f"{orig_path.stem}{upload_ext}"   # 원본 stem 기반
    # 예: (교재)...예방교육 + .docx
    # 위 줄은 system_doc_id가 "파일.hwp"면 stem="파일" -> "파일.docx"
    # (권장) 업로드명은 원본 stem 기반: 사람이 보기 쉬움
    print(
        "[TRACE-4] upload file -> "
        f"upload_path={upload_path}, upload_name={upload_name}"
    )

    # ---------------------------
    # [3] replace 처리 (삭제 분리)
    # ---------------------------
    if replace:
        print("[TRACE-5] replace=true → 기존 문서 삭제 시도")

        # ✅ RAGFlow는 업로드명(확장자 포함) 기준으로 삭제
        try:
            deleted = delete_ragflow_document_by_name(dataset, upload_name)
            print(f"[TRACE-5] RAGFlow delete(upload_name) result = {deleted}")
        except Exception as e:
            print(f"[TRACE-5][WARN] RAGFlow delete failed: {e}")

        # ✅ Milvus는 원본 docId(system_doc_id) 기준으로 삭제
        if milvus and milvus_dataset_id and system_doc_id:
            try:
                milvus.delete_file(dataset_id=milvus_dataset_id, doc_id=system_doc_id)
                print("[TRACE-5] Milvus delete(system_doc_id) OK")
            except Exception as e:
                print(f"[TRACE-5][WARN] Milvus delete failed: {e}")

    # ---------------------------
    # [4] 업로드
    # ---------------------------
    with open(upload_path, "rb") as fb:
        blob = fb.read()
    print(f"[TRACE-6] upload blob size = {len(blob)} bytes")

    doc = dataset.upload_documents(
        [{"display_name": upload_name, "name": upload_name, "blob": blob}]
    )[0]

    print(
        "[TRACE-6] uploaded -> "
        f"ragflow_doc.id={doc.id}, display_name={upload_name}"
    )

    stats_extra2 = {
        "ragDatasetPk": str(getattr(dataset, "id", "")),
        "ragDocumentPk": str(getattr(doc, "id", "")),
        "uploadName": upload_name,

        "systemDocId": system_doc_id,
    }

    # ---------------------------
    # [5] 청킹/추출
    # ---------------------------
    ext = upload_ext.lstrip(".")  # "docx" / "pdf" / ...

    if ext in ("pdf", "ppt", "pptx"):
        if ext == "pdf":
            doc_type = classifier.classify(str(upload_path))
        else:
            doc_type = "ppt"
        print(f"[TRACE-7] doc_type = {doc_type}")

        if doc_type == "text_pdf":
            chunks = chunk_text_pdf(upload_path)
            added = add_chunks_safe(
                doc,
                chunks,
                milvus=milvus,
                dataset_id=milvus_dataset_id,
                doc_id=system_doc_id,                # ✅ Milvus docId는 원본키
                embedding_model=embedding_model_selected,
                experiment_tag=experiment_tag,
                department=department,
                document_title=orig_path.stem,
            )
            return int(added), upload_name, stats_extra2

        pipeline_result = preprocess_pipeline.run(str(upload_path))
        chunks = pipeline_result.get("chunks", []) if isinstance(pipeline_result, dict) else []
        added = add_chunks_safe(
            doc,
            chunks,
            milvus=milvus,
            dataset_id=milvus_dataset_id,
            doc_id=system_doc_id,                    # ✅ Milvus docId는 원본키
            embedding_model=embedding_model_selected,
            experiment_tag=experiment_tag,
            department=department,
            document_title=orig_path.stem,
        )
        return int(added), upload_name, stats_extra2

    if ext in ("csv", "docx", "txt"):
        print(f"[TRACE-7] structured file type = {ext}")

        if ext == "docx":
            blocks = extract_docx_blocks(upload_path)
            docx_chunks = chunk_docx_blocks_with_rules(blocks)
            added = add_chunks_safe(
                doc,
                docx_chunks,
                milvus=milvus,
                dataset_id=milvus_dataset_id,
                doc_id=system_doc_id,                # ✅ Milvus docId는 원본키
                embedding_model=embedding_model_selected,
                experiment_tag=experiment_tag,
                department=department,
                document_title=orig_path.stem,
            )
            return int(added), upload_name, stats_extra2

        chunks = chunk_document(upload_path)
        added = add_chunks_safe(
            doc,
            chunks,
            milvus=milvus,
            dataset_id=milvus_dataset_id,
            doc_id=system_doc_id,                    # ✅ Milvus docId는 원본키
            embedding_model=embedding_model_selected,
            experiment_tag=experiment_tag,
            department=department,
            document_title=orig_path.stem,
        )
        return int(added), upload_name, stats_extra2

    print(f"[TRACE] unsupported extension .{ext}")
    return 0, upload_name, stats_extra2


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

    # ✅ 서버 운영: 단일 문서 모드만 허용 (배치 금지)
    if not args.input:
        # stdout에 stats 남기고, worker가 실패로 처리하게 함
        ingest_id = args.ingest_id or str(uuid4())
        meta = _parse_meta_json(args.meta_json)
        stats_extra = {
            "ingestId": ingest_id,
            "docId": args.doc_id or "batch",
            "version": args.version,
            "domain": None,
            "replace": replace_flag,
            "status": "FAILED",
            "meta": meta,
            "error": "Batch mode disabled on server. Use --input.",
        }
        print_ingest_stats(0, extra=stats_extra)
        raise RuntimeError("Batch mode disabled on server. Use --input.")

    ingest_id = args.ingest_id or str(uuid4())
    meta = _parse_meta_json(args.meta_json)


    DEPARTMENT = (meta.get("department") if isinstance(meta, dict) else None) or "ALL"
    DEPARTMENT = str(DEPARTMENT).strip()[:32] if DEPARTMENT else "ALL"

    if args.input:
        _in_path = Path(args.input)
        doc_key_for_stats = args.doc_id or _in_path.stem
        upload_name_for_stats = f"{doc_key_for_stats}{_in_path.suffix.lower()}"
    else:
        doc_key_for_stats = args.doc_id or "batch"
        upload_name_for_stats = None

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
        "openai": "ragflow_chunks_openai",
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
    # milvus = None

    MILVUS_DATASET_ID = os.getenv("INGEST_DATASET_ID")
    if not MILVUS_DATASET_ID:
        raise RuntimeError("INGEST_DATASET_ID is required in single-file mode")

    SYSTEM_DOC_ID = (args.doc_id or os.getenv("INGEST_DOC_ID") or "").strip()
    if not SYSTEM_DOC_ID:
        SYSTEM_DOC_ID = Path(args.input).name  # 최후 fallback은 실제 파일명.확장자

    print(f"[TRACE]: AI에서 받은 [DOCID] args.doc_id={args.doc_id!r}")
    print(f"[TRACE][DOCID] env.INGEST_DOC_ID={os.getenv('INGEST_DOC_ID')!r}")
    print(f"[TRACE]: 최종 [DOCID] SYSTEM_DOC_ID={SYSTEM_DOC_ID!r}")

    # stats에 남길 부가 정보(원하면 더 추가 가능)
    stats_extra = {
        "ingestId": ingest_id,
        "docId": SYSTEM_DOC_ID,
        "version": args.version,
        "domain": args.domain if args.input else None,
        "replace": replace_flag,
        "status": "UNKNOWN",
        "meta": meta,
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

        # (필수 env 체크)
        if not MILVUS_HOST or not MILVUS_PORT:
            raise RuntimeError("MILVUS_HOST/MILVUS_PORT env missing")

        embedding_model = EMBEDDING_MODEL_SELECTED
        collection_name = COLLECTION_NAME_MAP[embedding_model]
        print(f"[DEBUG] Milvus collection={collection_name}")

        try:
            milvus = MilvusProxy(
                host=MILVUS_HOST,
                port=MILVUS_PORT,
                collection_name=collection_name,
                dim=MODEL_DIM_MAP[embedding_model],
            )
            print("✅ Milvus 연결/컬렉션 준비 완료")
            stats_extra["milvusCollection"] = collection_name
            stats_extra["milvusDim"] = MODEL_DIM_MAP[embedding_model]

        except Exception as e:
            # ✅ 운영 정책: Milvus 필수 → 즉시 실패
            raise RuntimeError(f"Milvus connect failed: {e}") from e


        # ===========================================================
        # ✅ 단일 파일 모드 (--input)
        # ===========================================================
        if args.input:
            input_path = Path(args.input).expanduser().resolve()
            if not input_path.exists():
                raise RuntimeError(f"단일 파일 경로가 존재하지 않습니다: {input_path}")

            chunks_added, final_upload_name, extra2 = process_single_file_mode(
                rag=rag,
                milvus=milvus,
                embedding_model_selected=EMBEDDING_MODEL_SELECTED,
                experiment_tag=EXPERIMENT_TAG,
                input_path=input_path,
                domain=args.domain,
                doc_id=SYSTEM_DOC_ID,

                version=args.version,
                replace=replace_flag,
                milvus_dataset_id=MILVUS_DATASET_ID,
                milvus_doc_id=SYSTEM_DOC_ID,
                department=DEPARTMENT,   # ✅ 추가
            )

            if isinstance(extra2, dict):
                stats_extra.update(extra2)

            stats_extra["uploadName"] = final_upload_name
            stats_extra["docId"] = SYSTEM_DOC_ID  # ✅ stats도 원본명으로

            stats_extra["status"] = "COMPLETED"
            print_ingest_stats(chunks_added, extra=stats_extra)
            return

        # ===========================================================
        # ✅ 배치 모드는 Milvus 적재 OFF (권장)
        # - 배치는 AI 요청 스펙(datasetId/docId 매칭)이 없어서 Milvus 키를 안정적으로 못 맞춤
        # - 따라서 RAGFlow 업로드 + add_chunk만 수행
        # ===========================================================
        # milvus = None

        # ==============================================
        # ★ 도메인별 배치 처리 (기존 유지)
        # ==============================================
        # 배치 모드는 ingest-worker 콜백 스펙(단일 doc 단위)과 매칭이 애매해서,
        # 여기서는 stats 출력만 하되 status는 BATCH로 남김.
        # for domain, dataset_dir in DOMAIN_DIRS.items():
        #     print("\n" + "#" * 60)
        #     print(f"### 도메인: {domain}")
        #     print(f"### 로컬 폴더: {dataset_dir}")
        #     print("#" * 60)

        #     if not dataset_dir.exists():
        #         print(f"⚠️  폴더가 없습니다. 스킵: {dataset_dir}")
        #         continue

        #     print_step(3, f"[{domain}] dataset 폴더 스캔")

        #     pdfs = list(dataset_dir.glob("*.pdf"))
        #     ppts = list(dataset_dir.glob("*.ppt")) + list(dataset_dir.glob("*.pptx"))
        #     hwps = list(dataset_dir.glob("*.hwp")) + list(dataset_dir.glob("*.hwpx"))
        #     docxs = list(dataset_dir.glob("*.docx"))
        #     txts = list(dataset_dir.glob("*.txt"))
        #     csvs = list(dataset_dir.glob("*.csv"))

        #     files = sorted(pdfs + ppts + hwps + docxs + txts + csvs)

        #     if not files:
        #         print(f"❌ [{domain}] 처리할 파일이 없습니다.")
        #         continue

        #     print("📂 처리 파일:")
        #     for f in files:
        #         print("   -", f.name)

        #     print_step(4, f"[{domain}] 데이터셋 확보(get_or_create)")
        #     dataset = get_or_create_dataset(rag, domain)

        #     print_step(5, f"[{domain}] 파일 업로드 + 청킹")

        #     for fpath in files:
        #         orig_path = fpath.resolve()
        #         orig_ext = orig_path.suffix.lower().lstrip(".")
        #         system_doc_id = orig_path.name              # ✅ 원본키: 규정01.hwp (Milvus 기준)

        #         # ✅ 업로드용 파일/이름(=RAGFlow 기준)
        #         upload_path = orig_path
        #         upload_name = orig_path.name                # 기본은 원본명

        #         # 1) HWP/HWPX면 변환해서 업로드명은 docx로
        #         if orig_ext in ("hwp", "hwpx"):
        #             print(f"[HWP] {orig_path.name} → DOCX로 변환")
        #             docx_path = hwp_adapter.to_docx(str(orig_path))
        #             upload_path = Path(docx_path).resolve()
        #             if not upload_path.exists():
        #                 raise RuntimeError(f"HWP 변환 실패: {upload_path}")

        #             upload_name = f"{orig_path.stem}{upload_path.suffix.lower()}"   # ✅ 규정01.docx

        #         ext = upload_path.suffix.lower().lstrip(".")  # ✅ 이후 로직은 "업로드 파일" 기준 확장자

        #         # 2) replace면 삭제도 분리 (배치에서도 적용)
        #         if replace_flag:
        #             # ✅ RAGFlow 삭제: 업로드명 기준(.docx)
        #             try:
        #                 deleted = delete_ragflow_document_by_name(dataset, upload_name)
        #                 print(f"[REPLACE] RAGFlow delete(upload_name={upload_name}) -> {deleted}")
        #             except Exception as e:
        #                 print(f"[REPLACE][WARN] RAGFlow delete failed: {e}")

        #             # ✅ Milvus 삭제: 원본키 기준(.hwp)
        #             if milvus and MILVUS_DATASET_ID:
        #                 try:
        #                     milvus.delete_file(dataset_id=MILVUS_DATASET_ID, doc_id=system_doc_id)
        #                     print(f"[REPLACE] Milvus delete(system_doc_id={system_doc_id}) OK")
        #                 except Exception as e:
        #                     print(f"[REPLACE][WARN] Milvus delete failed: {e}")

        #         # 3) 업로드는 upload_path + upload_name으로!
        #         with open(upload_path, "rb") as fb:
        #             blob = fb.read()

        #         doc = dataset.upload_documents(
        #             [{"display_name": upload_name, "name": upload_name, "blob": blob}]
        #         )[0]
        #         print(f"→ 업로드 완료 (doc.id={doc.id}, upload_name={upload_name})")

        #         document_title = orig_path.stem
        #         department = infer_department_from_filename(orig_path.name)

        #         print(f"[DEPT] filename={orig_path.name} -> department={department}")

        #         # -----------------------------
        #         # PDF / PPT / PPTX
        #         # -----------------------------
        #         if ext in ("pdf", "ppt", "pptx"):
        #             if ext == "pdf":
        #                 doc_type = classifier.classify(str(upload_path))
        #             else:
        #                 doc_type = "ppt"

        #             print(f"→ 문서 타입: {doc_type}")

        #             if doc_type == "text_pdf":
        #                 print("→ [텍스트 PDF] 로컬 규정형 청킹 사용")
        #                 chunks = chunk_text_pdf(upload_path)
        #                 compare_with_solution(dataset_dir, upload_path, chunks)

        #                 chunks_added += int(
        #                     add_chunks_safe(
        #                         doc,
        #                         chunks,
        #                         milvus=milvus,
        #                         dataset_id=domain,                 # (배치 Milvus ON이면 MILVUS_DATASET_ID로 바꿔)
        #                         doc_id=system_doc_id,              # ✅ Milvus docId는 원본키
        #                         embedding_model=EMBEDDING_MODEL_SELECTED,
        #                         experiment_tag=EXPERIMENT_TAG,
        #                         department=department,
        #                         document_title=document_title,
        #                     )
        #                 )
        #             else:
        #                 print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")
        #                 pipeline_result = preprocess_pipeline.run(str(upload_path))
        #                 chunks = pipeline_result.get("chunks", []) if isinstance(pipeline_result, dict) else []

        #                 chunks_added += int(
        #                     add_chunks_safe(
        #                         doc,
        #                         chunks,
        #                         milvus=milvus,
        #                         dataset_id=domain,                 # (배치 Milvus ON이면 MILVUS_DATASET_ID로 바꿔)
        #                         doc_id=system_doc_id,              # ✅ Milvus docId는 원본키
        #                         embedding_model=EMBEDDING_MODEL_SELECTED,
        #                         experiment_tag=EXPERIMENT_TAG,
        #                         department=department,
        #                         document_title=document_title,
        #                     )
        #                 )
        #                 print(f"→ 파이프라인 청크 {len(chunks)}개 반환")

        #             continue

        #         # -----------------------------
        #         # CSV / DOCX / TXT
        #         # -----------------------------
        #         if ext in ("csv", "docx", "txt"):
        #             print("→ [CSV/DOCX/TXT] 기존 규정형 청킹 사용")

        #             if ext == "docx":
        #                 print("→ DOCX blocks 기반 처리 + 조단위(제 n 조) 청킹 적용")
        #                 blocks = extract_docx_blocks(upload_path)
        #                 docx_chunks = chunk_docx_blocks_with_rules(blocks)

        #                 chunks_added += int(
        #                     add_chunks_safe(
        #                         doc,
        #                         docx_chunks,
        #                         milvus=milvus,
        #                         dataset_id=domain,                 # (배치 Milvus ON이면 MILVUS_DATASET_ID로 바꿔)
        #                         doc_id=system_doc_id,              # ✅ Milvus docId는 원본키
        #                         embedding_model=EMBEDDING_MODEL_SELECTED,
        #                         experiment_tag=EXPERIMENT_TAG,
        #                         department=department,
        #                         document_title=document_title,
        #                     )
        #                 )
        #                 continue

        #             chunks = chunk_document(upload_path)
        #             compare_with_solution(dataset_dir, upload_path, chunks)

        #             chunks_added += int(
        #                 add_chunks_safe(
        #                     doc,
        #                     chunks,
        #                     milvus=milvus,
        #                     dataset_id=domain,                     # (배치 Milvus ON이면 MILVUS_DATASET_ID로 바꿔)
        #                     doc_id=system_doc_id,                  # ✅ Milvus docId는 원본키
        #                     embedding_model=EMBEDDING_MODEL_SELECTED,
        #                     experiment_tag=EXPERIMENT_TAG,
        #                     department=department,
        #                     document_title=document_title,
        #                 )
        #             )
        #             continue


        #         print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")


        #     print_step(6, f"[{domain}] 검색 테스트")
        #     query = f"{domain} 관련 문서의 목적은 무엇인가?"
        #     results = rag.retrieve(dataset_ids=[dataset.id], question=query, top_k=3)
        #     for i, r in enumerate(results, 1):
        #         print(f"\n[검색 {i}]")
        #         print(r.content[:200] + "...")

        # stats_extra["status"] = "BATCH_COMPLETED"
        # print_ingest_stats(chunks_added, extra=stats_extra)
        # return

    except Exception as e:
        # ✅ 실패여도 worker가 stdout에서 stats 뽑을 수 있게 남김
        stats_extra["status"] = "FAILED"
        stats_extra["error"] = str(e)[:500]
        print_ingest_stats(0, extra=stats_extra)
        # worker가 stderr로 잡을 수 있게 re-raise
        raise

# 수행
if __name__ == "__main__":
    main()