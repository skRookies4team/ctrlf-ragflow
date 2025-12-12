# RAG 평가 도구 모음

RAG 시스템의 임베딩, LLM, 검색 품질을 평가하기 위한 스크립트들입니다.

## 환경 설정

`.env` 파일 생성:
```bash
VLLM_HOST=your-server-ip
VLLM_PORT=1237
```

## 평가 스크립트

### 1. 임베딩 모델 평가
```bash
python embedding_eval.py
```
- Precision@K, Recall@K, MRR, NDCG@5, Hit Rate 측정
- 결과: `results/*_result.json`

### 2. LLM Q&A 생성
```bash
python llm_qa_generator.py
```
- 테스트 질문에 대한 LLM 답변 생성
- 결과: `results/*_pure_llm_*.json`

### 3. 검색 결과 생성
```bash
python retrieval_generator.py
```
- 질문별 검색(retrieval) 결과 생성
- 결과: `results/retrieval_results_*.json`

### 4. RAG 답변 생성
```bash
python rag_answer_generator.py
```
- 검색 결과 + LLM으로 RAG 답변 생성
- 결과: `results/rag_answers_*.json`

### 5. RAGAS 배치 평가 (권장)
```bash
# 기본 (10개씩 배치, 이어서 진행)
python ragas_batch_evaluator.py

# 배치 크기 변경 (예: 15개씩)
python ragas_batch_evaluator.py --batch-size 15

# 처음부터 다시
python ragas_batch_evaluator.py --no-resume

# 실패한 배치만 재시도
python ragas_batch_evaluator.py --retry-failed

# 특정 파일만 처리
python ragas_batch_evaluator.py --file rag_answers_20251209_153940.json
```
- Faithfulness, Answer Relevancy, Context Precision, Context Recall 측정
- 배치별 중간 결과 저장 (`results/batches/`)
- 서버 타임아웃 시 이어서 진행 가능

### 6. 간단 품질 평가
```bash
python simple_quality_evaluator.py
```
- LLM-as-Judge 방식의 간단 품질 평가

### 7. 보고서 생성
```bash
python generate_report.py
```
- 모든 결과를 종합한 마크다운 보고서 생성

## 평가 순서

```
1. embedding_eval.py      → 임베딩 모델 선정
2. llm_qa_generator.py    → LLM 모델별 답변 생성
3. retrieval_generator.py → 검색 결과 생성
4. rag_answer_generator.py → RAG 답변 생성
5. ragas_batch_evaluator.py → RAGAS 품질 평가
6. generate_report.py     → 최종 보고서
```

## 결과 파일 구조

```
results/
├── batches/                    # RAGAS 배치 중간 결과
│   └── rag_answers_*_batch_*.json
├── *_result.json               # 임베딩 평가 결과
├── *_pure_llm_*.json           # LLM 답변 결과
├── retrieval_results_*.json    # 검색 결과
├── rag_answers_*.json          # RAG 답변 결과
├── ragas_batch_final_*.json    # RAGAS 최종 결과
└── quality_evaluation_*.json   # 품질 평가 결과
```

## 선정 결과

- **임베딩 모델**: `jhgan/ko-sroberta-multitask` (MRR 0.749, P@1 0.731)
- **LLM 모델**: `Qwen2.5-7B-Instruct` (정확도 62.3%)

자세한 내용은 `embedding_comparison_report.md`, `LLM_comparison_report.md` 참고.
