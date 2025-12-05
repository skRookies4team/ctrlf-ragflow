"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
HWP / PDF / PPT / DOCX / TXT 자동 처리 + 문서 타입 판별 + 자동 패턴 감지 완전판
"""

import os
import sys
import time
import requests
import re
import json
import pdfplumber
from pathlib import Path
from typing import List, Sequence
from dotenv import load_dotenv
from difflib import SequenceMatcher

# =======================
# 0. 경로/환경 설정
# =======================
BASE_DIR = Path(__file__).resolve().parent.parent

# ragflow 루트를 파이썬 모듈 경로에 추가 → preprocessing 패키지 import 가능
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

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
def split_text_by_rules(raw_text: str,
                        heading_patterns: Sequence[str],
                        max_chars: int,
                        strict_heading_only: bool = False) -> List[str]:
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
          "result_json": {
             "num_chunks": 3,
             "chunks": [
                 {"text": "...", "meta": {...}},
                 ...
             ],
             "meta": {...}
          }
        }
    이런 형태를 가정하고 안전하게 파싱.
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
        # case 1: {"result_json": {...}}
        if "result_json" in data:
            rj = data["result_json"]
            if isinstance(rj, dict):
                # {"num_chunks": n, "chunks": [...], "meta": {...}}
                if "chunks" in rj and isinstance(rj["chunks"], list):
                    items = rj["chunks"]
            elif isinstance(rj, list):
                items = rj

        # case 2: {"chunks": [...]} 형태
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
                dp[i - 1][j] + 1,      # 삭제
                dp[i][j - 1] + 1,      # 삽입
                dp[i - 1][j - 1] + cost  # 교체
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
# 5. DOCX / TXT 전용 chunk 함수
#    (PDF/HWP/PPT는 상위 루프에서 처리)
# ===========================================================
def extract_text_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    paras = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paras)


def extract_text_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_document(path: Path) -> List[str]:
    """
    DOCX / TXT 전용 청킹
    (PDF/HWP/HWPX/PPT 는 상위 for 루프에서 별도 처리)
    """
    ext = path.suffix.lower()

    if ext == ".docx":
        raw = extract_text_docx(path)
    else:
        raw = extract_text_txt(path)

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
    pages = []
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
        chunks = []
        buf = []

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
# 메인
# ===========================================================
MAX_CHUNK_LEN = 8000  # 너무 긴 청크 방지용 (필요하면 4000~6000 정도로 줄여도 됨)

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
# 청크 추가 유틸 (심플 버전)
# ===========================================================
def add_chunks_safe(doc, chunks):
    """
    청크들을 RAGFlow doc에 추가.
    - 예전처럼: 생성된 청크 수 + 미리보기 1,2번만 출력
    """
    print(f"→ 생성된 청크 수: {len(chunks)}")

    for idx, c in enumerate(chunks, start=1):
        if not c or not c.strip():
            continue

        # 필요하면 여기서 길이 체크해서 잘라 넣을 수도 있지만
        # 지금은 다 짧으니까 그대로 추가
        doc.add_chunk(content=c)

        # 미리보기는 앞의 두 개만
        if idx <= 2:
            print(f"\n  [미리보기 청크 {idx}]")
            print(c[:200] + "...")

    print(f"→ 총 {len(chunks)}개 청크 추가 완료")



# ===========================================================
# 메인
# ===========================================================
def main():
    print_section("RAGFlow 커스텀 청킹 + add_chunk (HWP/PDF/PPT/DOCX/TXT 포함)")

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
    # 2) dataset 폴더 검색
    # ------------------------------------
    print_step(2, "dataset 폴더 스캔")
    dataset_dir = Path(__file__).parent / "dataset"

    pdfs = list(dataset_dir.glob("*.pdf"))
    ppts = list(dataset_dir.glob("*.ppt")) + list(dataset_dir.glob("*.pptx"))
    hwps = list(dataset_dir.glob("*.hwp")) + list(dataset_dir.glob("*.hwpx"))
    docxs = list(dataset_dir.glob("*.docx"))
    txts = list(dataset_dir.glob("*.txt"))

    files = sorted(pdfs + ppts + hwps + docxs + txts)

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
        description="자동 청킹 (HWP/슬라이드/텍스트 PDF/DOCX/TXT 혼합)",
        chunk_method="manual",
        embedding_model=EMBEDDING_MODEL,
        parser_config=parser_config,
    )

    print(f"✅ Dataset 생성 완료: {dataset.id}")

    # ------------------------------------
    # 4) 파일별 업로드 + 청킹
    # ------------------------------------
    print_step(4, "파일 업로드 + 청킹")

    for fpath in files:
        fpath = fpath.resolve()
        ext = fpath.suffix.lower().lstrip(".")
        print(f"\n======= {fpath.name} 처리 =======")

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

                # ★ PDF도 solution txt와 유사도 확인
                compare_with_solution(dataset_dir, fpath, chunks)

                add_chunks_safe(doc, chunks)
                continue
            # 1-2) 이미지 기반 PDF / PPT → PreprocessPipeline + add_chunk
            print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")

            print("→ [이미지/슬라이드] PreprocessPipeline + add_chunk 사용")

            # 🟢 1) 먼저 RAGFlow에 문서 업로드해서 doc 생성
            with open(fpath, "rb") as fb:
                blob = fb.read()

            doc = dataset.upload_documents(
                [{"display_name": fpath.name, "blob": blob}]
            )[0]
            print(f"→ 업로드 완료 (doc.id={doc.id})")

            # 🟢 2) PreprocessPipeline 실행
            pipeline_result = preprocess_pipeline.run(
                str(fpath),      # input_pdf
                chunk_size=1200, # 필요하면 조절
            )

            print("→ PreprocessPipeline 완료")

            # 🟢 3) 파이프라인에서 나온 청크 뽑기
            chunks = [c["text"] for c in pipeline_result["result_json"]["chunks"]]
            print(f"→ 파이프라인 청크 {len(chunks)}개 반환")

            # 🟢 4) RAGFlow doc에 add_chunk
            for idx, c in enumerate(chunks, 1):
                doc.add_chunk(content=c)
                if idx <= 2:
                    print(f"\n  [미리보기 청크 {idx}]")
                    print(c[:200] + ("..." if len(c) > 200 else ""))

            print(f"→ 총 {len(chunks)}개 청크 추가 완료")

            # 이 파일은 파이프라인으로 끝났으니까 다음 파일로 넘어감
            continue

        # ─────────────────────────────
        # 4-3. DOCX / TXT → 기존 규정형 청킹 사용
        # ─────────────────────────────
        print("→ [DOCX/TXT] 기존 규정형 청킹 사용")

        with open(fpath, "rb") as fb:
            blob = fb.read()

        doc = dataset.upload_documents(
            [{"display_name": fpath.name, "blob": blob}]
        )[0]
        print(f"→ 업로드 완료 (doc.id={doc.id})")

        chunks = chunk_document(fpath)
        compare_with_solution(dataset_dir, fpath, chunks)
        add_chunks_safe(doc, chunks)

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
