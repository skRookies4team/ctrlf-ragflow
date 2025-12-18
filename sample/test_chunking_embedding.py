"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
HWP / PDF / PPT / DOCX / TXT / CSV 자동 처리 + 문서 타입 판별 + 자동 패턴 감지 완전판
"""

import sys
import os
import time
import requests
import re
import json
import csv
import pdfplumber
import hashlib
from pathlib import Path
from typing import List, Sequence
from dotenv import load_dotenv
from difflib import SequenceMatcher
import google.generativeai as genai  # Gemini SDK

# =======================
# 0. 경로/환경 설정
# =======================
BASE_DIR = Path(__file__).resolve().parent.parent

# ragflow 루트를 파이썬 모듈 경로에 추가 → preprocessing 패키지 import 가능
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# milvus 환경설정
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# RAGFLOW_EMBEDDING_MODEL 이 "text-embedding-004@Gemini" 라고 되어 있으니까,
# 실제 Gemini 모델 이름은 아래처럼 쓸게.
GEMINI_EMBED_MODEL = os.getenv(
    "GEMINI_EMBED_MODEL",
    "models/text-embedding-004",
)
# 벡터 차원 (Milvus dim과 반드시 일치해야 함)
GEMINI_EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("⚠ GEMINI_API_KEY 가 .env에 없습니다. embed_text() 호출 시 에러가 납니다.")


# =======================
# 1. RAGFlow SDK import
# =======================
try:
    from ragflow_sdk import RAGFlow
except ImportError:
    # sdk/python 폴더를 경로에 추가 후 재시도
    sys.path.insert(0, str(BASE_DIR / "sdk" / "python"))
    from ragflow_sdk import RAGFlow

from ragflow_sdk.modules.dataset import DataSet
from milvus_proxy import MilvusProxy

# =======================
# 2. 커스텀 전처리 모듈 import
# =======================
from preprocessing.coverters.hwp_to_docx import HwpAdapter
from preprocessing.classifier.document_classifier import DocumentClassifier
from preprocessing.pipeline import PreprocessPipeline

# =======================
# 3. 환경 변수
# =======================
HOST_ADDRESS = os.getenv("RAGFLOW_HOST", "http://localhost")
API_KEY = os.getenv("RAGFLOW_API_KEY")
EMBEDDING_MODEL = os.getenv(
    "RAGFLOW_EMBEDDING_MODEL",
    "text-embedding-004@Gemini"
)

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
# ==========================================
# HWP / 슬라이드형 PDF 전처리 파이프라인 래퍼
# ==========================================
def preprocess_to_chunks(path: Path, chunk_size: int = 1200) -> list[str]:
    """
    PreprocessPipeline을 실행해서 청크 리스트(list[str])로 변환.
    - result 구조:
        {
          "run_id": "...",
          "page_count": ...,
          "avg_quality": ...,
          "pages": [...],
          "chunks": [
             {"text": "...", "page_index": ..., ...},
             ...
          ]
        }
    를 가정하고 안전하게 파싱.
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

    # 2) data에서 실제 chunk 목록 꺼내기
    items = []

    if isinstance(data, dict):
        # case 1: {"result_json": {...}} (구버전 호환)
        if "result_json" in data:
            rj = data["result_json"]
            if isinstance(rj, dict):
                if "chunks" in rj and isinstance(rj["chunks"], list):
                    items = rj["chunks"]
            elif isinstance(rj, list):
                items = rj
        # case 2: {"chunks": [...]} (현재 버전)
        elif "chunks" in data and isinstance(data["chunks"], list):
            items = data["chunks"]

    elif isinstance(data, list):
        items = data

    if not items:
        return []

    # 3) 각 item에서 텍스트만 뽑기
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
    """
    CER = (삽입 + 삭제 + 교체) / 정답 글자 수
    edit distance 기반
    """
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
                dp[i - 1][j] + 1,         # 삭제
                dp[i][j - 1] + 1,         # 삽입
                dp[i - 1][j - 1] + cost   # 교체
            )

    return dp[len(t)][len(p)] / max(1, len(t))


def eval_cer_for_pdf_text(pdf_path: Path, extracted_text: str) -> None:
    """
    원본 PDF에 대응하는 정답 텍스트(.txt)를 읽어서 CER 출력.

    정답 파일 위치:
        sample/dataset/solution/<pdf파일명>.txt
        예) 이사회규정.pdf → solution/이사회규정.txt
    """
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
#    (PDF/HWP/PPT는 상위 루프에서 처리)
# ===========================================================
def extract_text_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def extract_text_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def extract_text_csv(path: Path) -> str:
    """
    CSV 파일을 TEXT 문서처럼 변환하여 반환.
    검색 품질을 높이기 위해 "col: value" 형식으로 변환.
    """
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
        lines.append("")  # 행과 행 사이 공백 라인

    return "\n".join(lines).strip()


def chunk_document(path: Path) -> List[str]:
    """
    DOCX / TXT / CSV 전용 청킹
    (PDF/HWP/HWPX/PPT 는 상위 for 루프에서 별도 처리)
    """
    ext = path.suffix.lower()

    if ext == ".docx":
        raw = extract_text_docx(path)
    elif ext == ".csv":
        raw = extract_text_csv(path)
    else:
        raw = extract_text_txt(path)

    if not raw.strip():
        return []

    # 문서 타입 분석 후 기존 규정 청킹
    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "regulation":
        return split_text_by_rules(raw, patterns, max_chars=999999, strict_heading_only=True)
    elif doc_type == "structured":
        return split_text_by_rules(raw, patterns, max_chars=2000)
    else:
        # 일반 보고서 스타일
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
    """
    텍스트 기반 PDF → pdfplumber로 텍스트 추출 후
    DOCX랑 똑같은 규정/구조 문서 청킹 로직 적용
    """
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")

    raw = "\n".join(pages)

    # 문서 타입 분석
    doc_type = detect_document_type(raw)
    patterns = detect_heading_patterns_from_text(raw)

    if doc_type == "regulation":
        # 제 1 조, 제 2 조 단위로 나누고 길이는 웬만하면 안 자름
        return split_text_by_rules(raw, patterns, max_chars=999999, strict_heading_only=True)
    elif doc_type == "structured":
        # 숫자 헤더/목차가 많은 경우: 조금 더 잘게
        return split_text_by_rules(raw, patterns, max_chars=2000)
    else:
        # 일반 보고서 스타일 → DOCX랑 동일한 단락 기반 청킹
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

# =========================
# 텍스트 임베딩 함수 (Milvus 용, Gemini text-embedding-004)
# =========================
def embed_text(text: str) -> list[float]:
    """
    Gemini text-embedding-004을 호출해서 임베딩을 생성한다.
    - dim: 기본 768 (GEMINI_EMBED_DIM과 MilvusProxy dim이 반드시 같아야 함)
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY 가 설정되어 있지 않습니다.")

    # 긴 문서는 잘라서 평균내고 싶으면 여기서 전처리하면 됨.
    # 일단은 전체 텍스트 한 번에Embedding.
    response = genai.embed_content(
        model=GEMINI_EMBED_MODEL,
        content=text,
        task_type="RETRIEVAL_DOCUMENT",
    )

    # text-embedding-004 응답은 {"embedding": [...]} 형태
    embedding = response.get("embedding") or response["embeddings"][0]

    if len(embedding) != GEMINI_EMBED_DIM:
        # 예상 차원과 다르면 경고만 찍고 그대로 반환
        print(
            f"⚠ Gemini 임베딩 차원({len(embedding)})이 "
            f"GEMINI_EMBED_DIM({GEMINI_EMBED_DIM})과 다릅니다."
        )

    return embedding

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
# 청크 추가 유틸 (RAGFlow + (옵션) Milvus 동시 저장 버전)
# ===========================================================
def add_chunks_safe(
    doc,
    chunks,
    milvus=None,
    dataset_id: str | None = None,
    doc_id: str | None = None,
):
    """
    RAGFlow: 무조건 chunk 추가
    Milvus: chunk_hash 기준 중복 차단
    0개 청크 문서: 재시도 가능하도록 안전 처리
    """

    print(f"→ 생성된 청크 수: {len(chunks)}")

    # ----------------------------
    # 0개 청크 보호 로직
    # ----------------------------
    if not chunks:
        print("⚠ 청크 0개 → 이번 실행에서는 아무 것도 저장하지 않음 (재시도 가능)")
        return

    milvus_payload = []
    added = 0

    for idx, chunk in enumerate(chunks, start=1):
        # ----------------------------
        # chunk 파싱
        # ----------------------------
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

        # ----------------------------
        # RAGFlow 저장 (항상)
        # ----------------------------
        doc.add_chunk(content=text)
        added += 1

        # ----------------------------
        #Milvus 중복 체크 (있으면 skip)
        # ----------------------------
        if not (milvus and dataset_id and doc_id):
            continue

        if milvus.exists_chunk_hash(dataset_id, chash):
            # ⚠ Milvus만 스킵 (RAGFlow는 이미 저장됨)
            continue

        # ----------------------------
        #Milvus payload 적재
        # ----------------------------
        try:
            embedding = embed_text(text)
            milvus_payload.append({
                "dataset_id": dataset_id,
                "doc_id": doc_id,
                "chunk_id": idx,
                "chunk_hash": chash,
                "text": text,
                "embedding": embedding,
                "metadata": metadata,
            })
        except Exception as e:
            print(f"⚠ embedding 실패 (chunk {idx}): {e}")

        if idx <= 2:
            print(f"\n[미리보기 {idx}] ({chunk_type})")
            print(text[:200] + ("..." if len(text) > 200 else ""))

    print(f"→ RAGFlow 청크 추가 완료: {added}개")

    # ----------------------------
    #Milvus 일괄 insert
    # ----------------------------
    if milvus_payload:
        milvus.insert_chunks(
            dataset_id=dataset_id,
            chunks=milvus_payload,
        )
        print(f"→ Milvus 적재 완료: {len(milvus_payload)}개")
    else:
        print("→ Milvus 적재 없음 (중복 또는 embedding 실패)")



        # ----------------------------
        # 2) Milvus 저장
        # ----------------------------
        if milvus and dataset_id and doc_id:
            try:
                embedding = embed_text(text)
                milvus_payload.append({
                    "dataset_id": dataset_id,
                    "doc_id": doc_id,
                    "chunk_id": idx,
                    "chunk_hash": chash,
                    "type": chunk_type,
                    "text": text,
                    "embedding": embedding,
                    "metadata": metadata,
                })
            except Exception as e:
                print(f"⚠️ embedding 실패 (chunk {idx}): {e}")

        # ----------------------------
        # 미리보기
        # ----------------------------
        if idx <= 2:
            print(f"\n[미리보기 {idx}] ({chunk_type})")
            print(text[:200] + ("..." if len(text) > 200 else ""))

    print(f"→ 총 {added}개 청크 추가 완료")

    # ----------------------------
    # Milvus 일괄 insert
    # ----------------------------
    if milvus and milvus_payload:
        milvus.insert_chunks(
            dataset_id=dataset_id,
            chunks=milvus_payload,
        )
        

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
    
    # ★ 여기 추가: Milvus 연결
        print_step(1, "Milvus 연결")
    try:
        milvus = MilvusProxy(
            host=MILVUS_HOST,
            port=MILVUS_PORT,
            collection_name=MILVUS_COLLECTION,
            dim=GEMINI_EMBED_DIM,   # Gemini 임베딩 차원과 반드시 동일
        )
        print("✅ Milvus 연결/컬렉션 준비 완료")

    except Exception as e:
        print(f"❌ Milvus 연결 실패 (일단 RAGFlow만 진행): {e}")
        milvus = None

    # ==============================================
    # ★ 도메인별로 Dataset을 순차적으로 구성 ★
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

            # ─────────────────────────────
            # 4-1. HWP/HWPX → DOCX로 변환
            # ─────────────────────────────
            if ext in ("hwp", "hwpx"):
                print(f"[HWP] {fpath.name} → DOCX로 변환")
                docx_path = hwp_adapter.to_docx(str(fpath))
                fpath = Path(docx_path)
                ext = "docx"

            # ─────────────────────────────
            # 4-2. PDF / PPT / PPTX 처리
            # ─────────────────────────────
            if ext in ("pdf", "ppt", "pptx"):
                if ext == "pdf":
                    doc_type = classifier.classify(str(fpath))
                else:
                    doc_type = "ppt"

                print(f"→ 문서 타입: {doc_type}")

                # 1-1) 텍스트 기반 PDF → 우리 규정형 청킹 사용
                if doc_type == "text_pdf":
                    print("→ [텍스트 PDF] 로컬 규정형 청킹 사용")

                    with open(fpath, "rb") as fb:
                        blob = fb.read()

                    doc = dataset.upload_documents(
                        [{"display_name": fpath.name, "blob": blob}]
                    )[0]
                    print(f"→ 업로드 완료 (doc.id={doc.id})")

                    chunks = chunk_text_pdf(fpath)

                    # ★ solution/ 정답 txt가 있으면 유사도 체크
                    compare_with_solution(dataset_dir, fpath, chunks)

                    # RAGFlow + (옵션) Milvus 동시 저장
                    add_chunks_safe(
                        doc,
                        chunks,
                        milvus=milvus,
                        dataset_id=domain,   # 도메인별 dataset 이름을 dataset_id로 사용
                        doc_id=fpath.name,
                    )
                    continue

                # 1-2) 이미지 기반 PDF / PPT → PreprocessPipeline + add_chunk
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

                add_chunks_safe(
                    doc,
                    chunks,
                    milvus=milvus,
                    dataset_id=domain,
                    doc_id=fpath.name,
                )
                continue

            # ─────────────────────────────
            # 4-3. CSV / DOCX / TXT → 기존 규정형 청킹 사용
            # ─────────────────────────────
            if ext in ("csv", "docx", "txt"):
                print("→ [CSV/DOCX/TXT] 기존 규정형 청킹 사용")

                with open(fpath, "rb") as fb:
                    blob = fb.read()

                doc = dataset.upload_documents(
                    [{"display_name": fpath.name, "blob": blob}]
                )[0]
                print(f"→ 업로드 완료 (doc.id={doc.id})")

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

            # 기타 확장자는 스킵
            print(f"⚠️ 지원하지 않는 확장자입니다: .{ext} (스킵)")

        # ------------------------------------
        # 5) 도메인별 검색 테스트 (간단히 1번만)
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


if __name__ == "__main__":
    main()
