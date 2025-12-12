# RAGFlow /search API 추가 보고서

## 1. 개요

AI Gateway에서 재사용 가능한 경량 검색 API `/v1/chunk/search`를 RAGFlow에 추가했습니다.

### 1.1 목표

- 기존 `/retrieval_test` API의 기능을 유지하면서 코어 로직 공유
- AI Gateway용 단순화된 인터페이스 제공
- 빠른 응답을 위한 경량화된 요청/응답 구조

### 1.2 작업 범위

| 항목 | 내용 |
|------|------|
| 수정 파일 | `api/apps/chunk_app.py` |
| 추가 함수 | `_run_retrieval()`, `search_simple()` |
| 신규 엔드포인트 | `POST /v1/chunk/search` |

---

## 2. 구현 내용

### 2.1 코어 검색 로직 분리 (`_run_retrieval`)

기존 `retrieval_test` 함수의 검색 로직을 헬퍼 함수로 분리했습니다.

```python
async def _run_retrieval(
    question: str,
    kb_ids: list,
    tenant_ids: list,
    page: int = 1,
    size: int = 30,
    similarity_threshold: float = 0.0,
    vector_similarity_weight: float = 0.3,
    top_k: int = 1024,
    doc_ids: list = None,
    rerank_id: str = None,
    use_kg: bool = False,
    highlight: bool = False,
    cross_languages: list = None,
    keyword: bool = False,
) -> dict:
    """
    검색 코어 로직을 담당하는 헬퍼 함수.
    Returns: {"chunks": [...], "labels": [...], ...}
    """
```

**주요 기능:**
- 임베딩 모델 로드 및 쿼리 인코딩
- Rerank 모델 지원 (선택적)
- Knowledge Graph 검색 지원 (선택적)
- 다국어 처리 지원
- 키워드 추출 지원

### 2.2 `/retrieval_test` 리팩토링

기존 `retrieval_test` 함수를 `_run_retrieval` 호출 방식으로 변경했습니다.

**변경 전:**
```python
# 검색 로직이 함수 내부에 직접 구현됨
ranks = settings.retriever.retrieval(question, embd_mdl, ...)
```

**변경 후:**
```python
# 코어 검색 로직 호출
ranks = await _run_retrieval(
    question=question,
    kb_ids=kb_ids,
    tenant_ids=tenant_ids,
    ...
)
```

**기존 스펙 완전 유지:**
- 요청 파라미터 동일
- 응답 형식 동일
- 에러 처리 동일

### 2.3 새 `/search` 엔드포인트

AI Gateway용 경량 검색 API를 추가했습니다.

#### Request

```
POST /v1/chunk/search
Content-Type: application/json
```

```json
{
    "query": "검색 질문 (필수)",
    "top_k": 5,
    "dataset": "kb_id (필수)"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| query | string | O | 검색 질문 |
| top_k | int | X | 반환할 최대 결과 수 (기본값: 5) |
| dataset | string | O | Knowledge Base ID |

#### Response

```json
{
    "code": 0,
    "data": {
        "results": [
            {
                "doc_id": "chunk_id",
                "title": "문서명",
                "page": 3,
                "score": 0.87,
                "snippet": "내용 일부 (최대 500자)"
            }
        ]
    }
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| doc_id | string | 청크 또는 문서 식별자 |
| title | string | 문서 이름 |
| page | int/null | 페이지 번호 (없으면 null) |
| score | float | 유사도 점수 |
| snippet | string | 내용 일부 (최대 500자) |

---

## 3. 에러 처리

### 3.1 필수 파라미터 누락

```json
// query 누락
{"code": 102, "message": "query is required"}

// dataset 누락
{"code": 102, "message": "dataset (kb_id) is required"}
```

### 3.2 잘못된 파라미터

```json
// top_k가 숫자가 아닌 경우
{"code": 102, "message": "top_k must be a positive integer"}

// 존재하지 않는 dataset
{"code": 102, "message": "Dataset not found"}
```

### 3.3 검색 결과 없음

```json
{
    "code": 102,
    "data": {"results": []},
    "message": "No chunk found"
}
```

---

## 4. 사용 예시

### 4.1 curl 예시

```bash
curl -X POST "http://localhost:9380/v1/chunk/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "4대 필수교육 미이수 시 패널티",
    "top_k": 3,
    "dataset": "d580f9bc65b911f0a5d20242ac120006"
  }'
```

### 4.2 Python 예시

```python
import requests

response = requests.post(
    "http://localhost:9380/v1/chunk/search",
    json={
        "query": "직장 내 괴롭힘 신고 절차",
        "top_k": 5,
        "dataset": "your_kb_id_here"
    }
)

results = response.json()["data"]["results"]
for r in results:
    print(f"[{r['score']:.2f}] {r['title']}: {r['snippet'][:100]}...")
```

### 4.3 AI Gateway 연동 예시

```python
class RagflowClient:
    def __init__(self, base_url: str, kb_ids: dict):
        self.base_url = base_url.rstrip("/")
        self.kb_ids = kb_ids  # {"policy": "kb_id_1", "training": "kb_id_2"}

    def search(self, query: str, dataset: str, top_k: int = 5):
        payload = {
            "query": query,
            "top_k": top_k,
            "dataset": self.kb_ids.get(dataset, dataset)
        }
        resp = requests.post(
            f"{self.base_url}/v1/chunk/search",
            json=payload,
            timeout=5
        )
        return resp.json()["data"]["results"]

# 사용
client = RagflowClient(
    "http://ragflow:9380",
    {"policy": "kb_policy_id", "training": "kb_training_id"}
)
results = client.search("연차 사용 방법", "policy", top_k=3)
```

---

## 5. 검증 체크리스트

| 항목 | 상태 |
|------|------|
| `/retrieval_test` 기존 스펙 유지 | O |
| `/search` 신규 엔드포인트 추가 | O |
| 코어 검색 로직 공유 (`_run_retrieval`) | O |
| 필수 파라미터 검증 (query, dataset) | O |
| top_k 기본값 처리 | O |
| 응답 포맷 변환 (doc_id, title, page, score, snippet) | O |
| 에러 처리 (400, 500) | O |
| 기존 코드 스타일 준수 | O |

---

## 6. 파일 변경 요약

### `api/apps/chunk_app.py`

| 라인 | 변경 내용 |
|------|----------|
| 44-137 | `_run_retrieval()` 헬퍼 함수 추가 |
| 429-445 | `retrieval_test()` 리팩토링 - `_run_retrieval()` 호출 |
| 455-568 | `search_simple()` 신규 엔드포인트 추가 |

**총 변경: +130 라인 (신규), ~30 라인 (리팩토링)**

---

## 7. 향후 개선 사항 (선택)

1. **인증 추가**: API Key 기반 인증 (`Authorization: Bearer <API_KEY>`)
2. **캐싱**: 자주 검색되는 쿼리에 대한 결과 캐싱
3. **Rate Limiting**: API 호출 제한
4. **메트릭**: 검색 성능 모니터링

---

## 8. 결론

- 기존 `/retrieval_test` API의 기능과 스펙을 100% 유지
- 새로운 `/search` API로 AI Gateway 연동 간소화
- 코어 검색 로직 공유로 유지보수성 향상
