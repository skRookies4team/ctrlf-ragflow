#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding 모델 성능 평가 스크립트 (실제 문서 기반)

테스트 모델:
1. nlpai-lab/KURE-v1
2. BAAI/bge-m3
3. dragonkue/multilingual-e5-small-ko
4. jhgan/ko-sroberta-multitask
5. BM-K/KoSimCSE-roberta-multitask

평가 지표:
- Precision@K (K=1,3,5)
- Recall@K (K=1,3,5)
- MRR (Mean Reciprocal Rank)
- NDCG@5
- Hit Rate@K
- Latency (임베딩 생성 시간)
"""

import os
import sys
import csv
import json
import time
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from openai import OpenAI

# PDF 파싱
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF 미설치. PDF 파일은 건너뜁니다. 설치: pip install pymupdf")

# 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# vLLM 서버 설정
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "1237"))

# 청킹 설정
CHUNK_SIZE = 500  # 문자 수
CHUNK_OVERLAP = 50  # 오버랩


@dataclass
class EvalResult:
    """평가 결과 데이터 클래스"""
    model_name: str
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr: float
    ndcg_at_5: float
    hit_rate_at_1: float
    hit_rate_at_3: float
    hit_rate_at_5: float
    avg_latency_ms: float
    total_questions: int
    total_chunks: int
    timestamp: str


def print_section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def load_test_data(csv_path: Path) -> List[Dict]:
    """테스트 데이터 로드"""
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'id': row['Q_ID'],
                'domain': row['도메인'],
                'role': row['사용자_롤'],
                'difficulty': row['난이도'],
                'question': row['질문'],
                'ground_truth': row['모범답안'],
                'source_doc': row['출처_문서_ID']
            })
    return questions


def read_txt_file(file_path: Path) -> str:
    """TXT 파일 읽기"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp949') as f:
            return f.read()


def read_pdf_file(file_path: Path) -> str:
    """PDF 파일에서 텍스트 추출"""
    if not HAS_PYMUPDF:
        return ""

    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"  PDF 읽기 실패 {file_path.name}: {e}")
        return ""


def chunk_text(text: str, doc_id: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """텍스트를 청크로 분할"""
    # 텍스트 정리
    text = re.sub(r'\s+', ' ', text).strip()

    if not text:
        return []

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # 문장 끝에서 자르기 시도
        if end < len(text):
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            cut_point = max(last_period, last_newline)
            if cut_point > chunk_size * 0.5:  # 50% 이상이면 거기서 자름
                chunk_text = chunk_text[:cut_point + 1]
                end = start + cut_point + 1

        if chunk_text.strip():
            chunks.append({
                'id': f"{doc_id}_chunk_{chunk_idx}",
                'doc_id': doc_id,
                'content': chunk_text.strip()
            })
            chunk_idx += 1

        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break

    return chunks


def load_documents(dataset_dir: Path) -> List[Dict]:
    """모든 문서를 로드하고 청킹"""
    all_chunks = []

    # TXT 파일 로드
    print("  TXT 파일 로드 중...")
    txt_files = list(dataset_dir.rglob("*.txt"))
    for txt_file in txt_files:
        doc_id = txt_file.stem  # 파일명을 문서 ID로 사용
        text = read_txt_file(txt_file)
        if text:
            chunks = chunk_text(text, doc_id)
            all_chunks.extend(chunks)
            print(f"    {doc_id}: {len(chunks)}개 청크")

    # PDF 파일 로드
    if HAS_PYMUPDF:
        print("  PDF 파일 로드 중...")
        pdf_files = list(dataset_dir.rglob("*.pdf"))
        for pdf_file in pdf_files:
            doc_id = pdf_file.stem
            text = read_pdf_file(pdf_file)
            if text:
                chunks = chunk_text(text, doc_id)
                all_chunks.extend(chunks)
                print(f"    {doc_id}: {len(chunks)}개 청크")

    return all_chunks


def get_embedding(client: OpenAI, model: str, text: str) -> Tuple[List[float], float]:
    """텍스트 임베딩 생성 및 시간 측정"""
    start_time = time.time()

    response = client.embeddings.create(
        model=model,
        input=[text],
    )

    latency = (time.time() - start_time) * 1000  # ms
    embedding = response.data[0].embedding

    return embedding, latency


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """코사인 유사도 계산"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def calculate_ndcg(relevances: List[int], k: int) -> float:
    """NDCG@K 계산"""
    if not relevances or k == 0:
        return 0.0

    relevances = relevances[:k]

    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    # IDCG (이상적인 DCG)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def get_doc_keywords(doc_id: str) -> List[str]:
    """문서 ID에서 매칭용 키워드 추출"""
    # DOC-xxx 형식에서 키워드 추출
    doc_id_lower = doc_id.lower().replace("doc-", "").replace("_", "")

    # 키워드 매핑 테이블
    keyword_map = {
        "복무규정": ["복무", "근무", "출퇴근", "휴가", "연차"],
        "인사규정": ["인사", "채용", "승진", "평가", "인사위원회"],
        "성희롱규정": ["성희롱"],
        "성희롱예방교육": ["성희롱", "예방교육", "성폭력"],
        "직장내괴롭힘규정": ["괴롭힘", "직장내괴롭힘"],
        "장애인식교육": ["장애", "장애인", "인식개선"],
        "개인정보보호규정": ["개인정보", "정보보호"],
        "개인정보처리방침": ["개인정보", "처리방침"],
        "개인정보사고대응절차": ["개인정보", "사고대응"],
        "정보보안규정": ["정보보안", "보안"],
        "정보보안정책": ["정보보안", "보안정책"],
        "사규관리규정": ["사규", "규정관리"],
        "징계규정": ["징계", "상벌"],
        "당직규정": ["당직"],
        "출장규정": ["출장"],
        "출산육아규정": ["출산", "육아", "휴직"],
        "경조사규정": ["경조사"],
        "윤리경영규정": ["윤리", "윤리경영"],
        "데이터활용지침": ["데이터", "활용"],
        "ai이용지침": ["ai", "인공지능"],
        "계정보안지침": ["계정", "비밀번호"],
        "로그관리지침": ["로그"],
        "메일보안지침": ["메일", "이메일"],
        "영상정보처리기기지침": ["영상", "cctv"],
        "오픈소스정책": ["오픈소스"],
        "외주관리지침": ["외주"],
        "원격접속지침": ["원격"],
        "이동식매체관리지침": ["usb", "이동식"],
        "인프라보안지침": ["인프라", "서버"],
        "저작권지침": ["저작권"],
        "출력물관리지침": ["출력물"],
        "콜센터운영지침": ["콜센터", "고객센터"],
    }

    # 매핑된 키워드 반환
    for key, keywords in keyword_map.items():
        if key in doc_id_lower:
            return keywords

    # 매핑이 없으면 문서ID 자체를 키워드로
    return [doc_id_lower.replace("규정", "").replace("지침", "").replace("정책", "")]


def is_doc_relevant(ground_truth_doc: str, doc_id: str) -> bool:
    """문서 관련성 판단"""
    gt_lower = ground_truth_doc.lower().replace("doc-", "")
    doc_lower = doc_id.lower()

    # 키워드 추출
    keywords = get_doc_keywords(ground_truth_doc)

    # 1. 직접 매칭
    if gt_lower in doc_lower or doc_lower in gt_lower:
        return True

    # 2. 키워드 매칭
    for keyword in keywords:
        if keyword in doc_lower:
            return True

    # 3. 사규.txt는 복무/인사/징계/당직/출장/경조사/윤리경영 관련 질문과 매칭
    if doc_lower == "사규":
        general_keywords = ["복무", "인사", "징계", "당직", "출장", "경조사", "윤리", "사규"]
        for kw in general_keywords:
            if kw in gt_lower:
                return True

    return False


def evaluate_retrieval(
    question_embedding: List[float],
    doc_embeddings: List[Tuple[str, str, List[float]]],  # (chunk_id, doc_id, embedding)
    ground_truth_doc: str,
    k_values: List[int] = [1, 3, 5]
) -> Dict:
    """검색 평가 수행"""
    # 유사도 계산 및 정렬
    similarities = []
    for chunk_id, doc_id, doc_emb in doc_embeddings:
        sim = cosine_similarity(question_embedding, doc_emb)
        similarities.append((chunk_id, doc_id, sim))

    similarities.sort(key=lambda x: x[2], reverse=True)

    # 관련성 판단 (키워드 기반 매칭)
    relevances = []
    for chunk_id, doc_id, _ in similarities:
        is_relevant = is_doc_relevant(ground_truth_doc, doc_id)
        relevances.append(1 if is_relevant else 0)

    results = {}

    # Precision@K, Recall@K, Hit Rate@K
    for k in k_values:
        top_k_relevances = relevances[:k]
        relevant_count = sum(top_k_relevances)

        results[f'precision_at_{k}'] = relevant_count / k if k > 0 else 0
        results[f'recall_at_{k}'] = 1.0 if relevant_count > 0 else 0  # 단일 관련 문서 가정
        results[f'hit_rate_at_{k}'] = 1.0 if relevant_count > 0 else 0

    # MRR
    mrr = 0.0
    for i, rel in enumerate(relevances):
        if rel == 1:
            mrr = 1.0 / (i + 1)
            break
    results['mrr'] = mrr

    # NDCG@5
    results['ndcg_at_5'] = calculate_ndcg(relevances, 5)

    return results


def run_evaluation(
    model_name: str,
    test_questions: List[Dict],
    doc_chunks: List[Dict],  # [{'id': 'xxx_chunk_0', 'doc_id': 'xxx', 'content': '...'}]
    client: OpenAI
) -> EvalResult:
    """단일 모델 평가 실행"""
    print(f"\n모델: {model_name}")
    print("-" * 50)

    # 1. 문서 청크 임베딩 생성
    print("문서 청크 임베딩 생성 중...")
    doc_embeddings = []
    for i, doc in enumerate(doc_chunks):
        try:
            emb, _ = get_embedding(client, model_name, doc['content'])
            doc_embeddings.append((doc['id'], doc['doc_id'], emb))
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(doc_chunks)} 완료")
        except Exception as e:
            print(f"  청크 {doc['id']} 임베딩 실패: {e}")

    print(f"  총 {len(doc_embeddings)}개 청크 임베딩 완료")

    # 2. 질문별 평가
    print("질문 평가 중...")
    all_results = []
    total_latency = 0

    for i, q in enumerate(test_questions):
        try:
            # 질문 임베딩
            q_emb, latency = get_embedding(client, model_name, q['question'])
            total_latency += latency

            # 검색 평가
            eval_result = evaluate_retrieval(
                q_emb,
                doc_embeddings,
                q['source_doc']
            )
            all_results.append(eval_result)

            if (i + 1) % 30 == 0:
                print(f"  {i + 1}/{len(test_questions)} 완료")

        except Exception as e:
            print(f"  질문 {q['id']} 평가 실패: {e}")

    # 3. 결과 집계
    n = len(all_results)
    if n == 0:
        print("  평가 결과 없음!")
        return None

    avg_latency = total_latency / len(test_questions)

    result = EvalResult(
        model_name=model_name,
        precision_at_1=sum(r['precision_at_1'] for r in all_results) / n,
        precision_at_3=sum(r['precision_at_3'] for r in all_results) / n,
        precision_at_5=sum(r['precision_at_5'] for r in all_results) / n,
        recall_at_1=sum(r['recall_at_1'] for r in all_results) / n,
        recall_at_3=sum(r['recall_at_3'] for r in all_results) / n,
        recall_at_5=sum(r['recall_at_5'] for r in all_results) / n,
        mrr=sum(r['mrr'] for r in all_results) / n,
        ndcg_at_5=sum(r['ndcg_at_5'] for r in all_results) / n,
        hit_rate_at_1=sum(r['hit_rate_at_1'] for r in all_results) / n,
        hit_rate_at_3=sum(r['hit_rate_at_3'] for r in all_results) / n,
        hit_rate_at_5=sum(r['hit_rate_at_5'] for r in all_results) / n,
        avg_latency_ms=avg_latency,
        total_questions=n,
        total_chunks=len(doc_embeddings),
        timestamp=datetime.now().isoformat()
    )

    return result


def print_result(result: EvalResult):
    """결과 출력"""
    print(f"\n{'='*60}")
    print(f"  모델: {result.model_name}")
    print(f"{'='*60}")
    print(f"\n[정확도 지표]")
    print(f"  Precision@1: {result.precision_at_1:.4f}")
    print(f"  Precision@3: {result.precision_at_3:.4f}")
    print(f"  Precision@5: {result.precision_at_5:.4f}")
    print(f"\n[재현율 지표]")
    print(f"  Recall@1:    {result.recall_at_1:.4f}")
    print(f"  Recall@3:    {result.recall_at_3:.4f}")
    print(f"  Recall@5:    {result.recall_at_5:.4f}")
    print(f"\n[순위 지표]")
    print(f"  MRR:         {result.mrr:.4f}")
    print(f"  NDCG@5:      {result.ndcg_at_5:.4f}")
    print(f"\n[Hit Rate]")
    print(f"  Hit@1:       {result.hit_rate_at_1:.4f}")
    print(f"  Hit@3:       {result.hit_rate_at_3:.4f}")
    print(f"  Hit@5:       {result.hit_rate_at_5:.4f}")
    print(f"\n[성능]")
    print(f"  평균 Latency: {result.avg_latency_ms:.2f} ms")
    print(f"  테스트 질문수: {result.total_questions}")
    print(f"  총 청크 수: {result.total_chunks}")


def save_result(result: EvalResult, output_dir: Path):
    """결과 저장"""
    # 모델 이름에서 파일명 생성
    model_safe_name = result.model_name.split('/')[-1].replace('--', '_')[:50]
    json_path = output_dir / f"{model_safe_name}_result.json"

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: {json_path}")


def main():
    print_section("Embedding 모델 성능 평가 (실제 문서 기반)")

    # 1. 테스트 데이터 로드
    print("\n[1] 테스트 데이터 로드...")
    csv_path = EVAL_DIR / "test_questions.csv"
    test_questions = load_test_data(csv_path)
    print(f"  총 {len(test_questions)}개 질문 로드")

    # 2. 실제 문서 로드 및 청킹
    print("\n[2] 문서 로드 및 청킹...")
    doc_chunks = load_documents(DATASET_DIR)
    print(f"\n  총 {len(doc_chunks)}개 청크 생성")

    if not doc_chunks:
        print("  문서가 없습니다!")
        return

    # 3. vLLM 클라이언트 연결
    print("\n[3] vLLM 서버 연결...")
    client = OpenAI(
        base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        api_key="not-needed"
    )

    # 서버에서 사용 가능한 모델 확인
    try:
        models = client.models.list()
        available_model = models.data[0].id if models.data else None
        print(f"  사용 가능한 모델: {available_model}")
    except Exception as e:
        print(f"  서버 연결 실패: {e}")
        return

    if not available_model:
        print("  사용 가능한 모델이 없습니다.")
        return

    # 4. 평가 실행
    print_section("평가 실행")

    result = run_evaluation(
        model_name=available_model,
        test_questions=test_questions,
        doc_chunks=doc_chunks,
        client=client
    )

    if result:
        # 5. 결과 출력 및 저장
        print_result(result)
        save_result(result, RESULTS_DIR)

    print_section("평가 완료")
    print(f"\n다음 모델 테스트하려면:")
    print(f"  1. 서버에서 다른 모델로 교체")
    print(f"  2. 이 스크립트 다시 실행")
    print(f"\n결과 파일 위치: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
