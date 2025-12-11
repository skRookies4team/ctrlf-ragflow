"""
RAG 검색 결과 생성 및 저장 스크립트

1단계: 임베딩 모델로 문서 청킹 + 검색 결과 저장
- LLM 없이 임베딩 모델만 사용
- 나중에 LLM 답변 생성 시 재사용
"""

import os
import csv
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from openai import OpenAI

# PDF 파싱
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF 미설치. pip install pymupdf")

# 경로 설정
EVAL_DIR = Path(__file__).resolve().parent
DATASET_DIR = EVAL_DIR.parent / "dataset"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# vLLM 서버 설정
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "1237"))

# 청킹 설정
CHUNK_SIZE = 300  # 토큰 제한 고려하여 축소
CHUNK_OVERLAP = 30


def load_questions(csv_path: Path) -> List[Dict]:
    """Q세트 로드"""
    questions = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append({
                'q_id': row['Q_ID'],
                'domain': row['도메인'],
                'role': row['사용자_롤'],
                'difficulty': row['난이도'],
                'question': row['질문'],
                'ground_truth': row['모범답안'],
                'source_doc': row['출처_문서_ID']
            })
    return questions


def extract_text_from_pdf(pdf_path: Path) -> str:
    """PDF에서 텍스트 추출"""
    if not HAS_PYMUPDF:
        return ""
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"PDF 읽기 실패 {pdf_path}: {e}")
        return ""


def extract_text_from_txt(txt_path: Path) -> str:
    """TXT 파일에서 텍스트 추출"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"TXT 읽기 실패 {txt_path}: {e}")
        return ""


def load_documents(dataset_dir: Path) -> List[Dict]:
    """문서 로드"""
    documents = []

    for folder in dataset_dir.iterdir():
        if not folder.is_dir():
            continue

        for file_path in folder.iterdir():
            text = ""
            if file_path.suffix.lower() == '.pdf':
                text = extract_text_from_pdf(file_path)
            elif file_path.suffix.lower() == '.txt':
                text = extract_text_from_txt(file_path)
            # HWP는 별도 라이브러리 필요, 일단 스킵

            if text.strip():
                documents.append({
                    'source': file_path.stem,
                    'folder': folder.name,
                    'path': str(file_path),
                    'text': text
                })

    # solution 폴더의 txt 파일도 로드
    solution_dir = dataset_dir / "solution"
    if solution_dir.exists():
        for txt_file in solution_dir.glob("*.txt"):
            text = extract_text_from_txt(txt_file)
            if text.strip():
                documents.append({
                    'source': txt_file.stem,
                    'folder': 'solution',
                    'path': str(txt_file),
                    'text': text
                })

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """텍스트를 청크로 분할"""
    # 문단 단위로 먼저 분리
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += "\n" + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # 긴 문단은 문장 단위로 분할
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) <= chunk_size:
                        current_chunk += " " + sent if current_chunk else sent
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
            else:
                current_chunk = para

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks_from_documents(documents: List[Dict]) -> List[Dict]:
    """문서들을 청크로 변환"""
    all_chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc['text'])
        for i, chunk in enumerate(text_chunks):
            if len(chunk) < 50:  # 너무 짧은 청크 제외
                continue
            all_chunks.append({
                'id': f"{doc['source']}_{i}",
                'source': doc['source'],
                'folder': doc['folder'],
                'text': chunk
            })

    return all_chunks


class EmbeddingRetriever:
    """임베딩 기반 검색기"""

    def __init__(self, host: str = VLLM_HOST, port: int = VLLM_PORT):
        self.base_url = f"http://{host}:{port}/v1"
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="dummy"
        )
        self.chunks = []
        self.embeddings = None
        self.model_id = self._get_model_id()

    def _get_model_id(self) -> str:
        """서버에서 실제 모델 ID 조회"""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=10)
            data = resp.json()
            if data.get("data"):
                model_id = data["data"][0]["id"]
                print(f"임베딩 모델 ID: {model_id}")
                return model_id
        except Exception as e:
            print(f"모델 ID 조회 실패: {e}")
        return "jhgan/ko-sroberta-multitask"

    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        # 길이 제한 (128 토큰 ≈ 약 150자)
        text = text[:150] if len(text) > 150 else text
        response = self.client.embeddings.create(
            model=self.model_id,
            input=text
        )
        return response.data[0].embedding

    def get_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """배치 임베딩 생성"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            # 텍스트 길이 제한 (128 토큰 ≈ 약 150자)
            batch = [t[:150] if len(t) > 150 else t for t in batch]

            try:
                response = self.client.embeddings.create(
                    model=self.model_id,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"  배치 {i} 오류: {e}")
                # 개별 처리
                for t in batch:
                    try:
                        resp = self.client.embeddings.create(
                            model=self.model_id,
                            input=[t[:100]]
                        )
                        all_embeddings.append(resp.data[0].embedding)
                    except:
                        # 빈 임베딩 추가
                        all_embeddings.append([0.0] * 768)

            if (i + batch_size) % 100 == 0:
                print(f"  임베딩 진행: {min(i+batch_size, len(texts))}/{len(texts)}")

        return all_embeddings

    def index_chunks(self, chunks: List[Dict]):
        """청크 인덱싱"""
        self.chunks = chunks
        texts = [c['text'] for c in chunks]

        print(f"총 {len(texts)}개 청크 임베딩 생성 중...")
        start = time.time()
        embeddings = self.get_embeddings_batch(texts)
        elapsed = time.time() - start
        print(f"임베딩 완료: {elapsed:.1f}초")

        self.embeddings = np.array(embeddings)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """쿼리로 검색"""
        query_embedding = np.array(self.get_embedding(query))

        # 코사인 유사도 계산
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 상위 k개 선택
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': self.chunks[idx]['id'],
                'source': self.chunks[idx]['source'],
                'folder': self.chunks[idx]['folder'],
                'text': self.chunks[idx]['text'],
                'score': float(similarities[idx])
            })

        return results


def generate_retrieval_results(
    questions: List[Dict],
    retriever: EmbeddingRetriever,
    output_path: Path,
    top_k: int = 5
):
    """모든 질문에 대해 검색 수행 및 저장"""

    results = []
    total = len(questions)

    print(f"\n{'='*60}")
    print(f"검색 결과 생성 시작")
    print(f"총 질문 수: {total}")
    print(f"Top-K: {top_k}")
    print(f"{'='*60}\n")

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{total}] {q['q_id']}: {q['question'][:40]}...")

        start = time.time()
        search_results = retriever.search(q['question'], top_k=top_k)
        elapsed = time.time() - start

        result = {
            'q_id': q['q_id'],
            'domain': q['domain'],
            'role': q['role'],
            'difficulty': q['difficulty'],
            'question': q['question'],
            'ground_truth': q['ground_truth'],
            'source_doc': q['source_doc'],
            'retrieved_contexts': [r['text'] for r in search_results],
            'retrieved_scores': [r['score'] for r in search_results],
            'retrieved_sources': [r['source'] for r in search_results],
            'retrieval_time': round(elapsed, 4)
        }
        results.append(result)

        print(f"    검색 시간: {elapsed:.3f}s | Top1 스코어: {search_results[0]['score']:.3f}")

    # 결과 저장
    output_data = {
        'embedding_model': 'jhgan/ko-sroberta-multitask',
        'top_k': top_k,
        'total_questions': len(results),
        'generated_at': datetime.now().isoformat(),
        'results': results
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장 완료: {output_path}")
    return results


def main():
    print("=" * 60)
    print("RAG 검색 결과 생성기")
    print("=" * 60)

    # 1. Q세트 로드
    questions_path = EVAL_DIR / "test_questions.csv"
    questions = load_questions(questions_path)
    print(f"\n로드된 질문: {len(questions)}개")

    # 2. 문서 로드
    print("\n문서 로드 중...")
    documents = load_documents(DATASET_DIR)
    print(f"로드된 문서: {len(documents)}개")

    for doc in documents:
        print(f"  - {doc['folder']}/{doc['source']}: {len(doc['text'])}자")

    # 3. 청킹
    print("\n청킹 중...")
    chunks = create_chunks_from_documents(documents)
    print(f"생성된 청크: {len(chunks)}개")

    # 4. 임베딩 및 인덱싱
    print("\n임베딩 모델 초기화...")
    retriever = EmbeddingRetriever()
    retriever.index_chunks(chunks)

    # 5. 검색 결과 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"retrieval_results_{timestamp}.json"

    generate_retrieval_results(
        questions=questions,
        retriever=retriever,
        output_path=output_path,
        top_k=5
    )

    print("\n완료!")


if __name__ == "__main__":
    main()
