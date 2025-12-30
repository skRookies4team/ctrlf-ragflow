📄 RAGFlow 문서 업로드 & Ingest 실행 가이드

본 문서는 RAGFlow + Milvus 기반 문서 업로드(ingest) 실행 방법을 설명합니다.
RAGFlow는 Docker Compose 기반 멀티 서비스 구조이므로, 환경 변수 설정 → 컨테이너 실행 → 문서 업로드 실행 순서로 진행해야 합니다.

1️⃣ 환경 설정 (.env)
RAGFlow 실행에 필요한 환경 변수를 .env 파일에 설정합니다.
이 단계에서 어떤 벡터 DB(Milvus) 와 어떤 임베딩/LLM 모델을 사용할지가 결정됩니다.

1) 필수 환경 변수
#### Milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

#### RAGFlow
RAGFLOW_HOST=http://localhost
RAGFLOW_API_KEY=your_ragflow_api_key

#### OpenAI (openai 임베딩 사용 시)
OPENAI_API_KEY=your_openai_api_key

(선택) 임베딩 A/B 테스트용
#### 실행 시 환경 변수로 덮어씀
EMBEDDING_MODEL_SELECTED=openai  # or sroberta

👉 EMBEDDING_MODEL_SELECTED 값에 따라

openai → Milvus ragflow_chunks

sroberta → Milvus ragflow_chunks_sroberta
컬렉션이 자동으로 선택됩니다.


2️⃣ Docker Compose로 서비스 실행
환경 설정이 끝나면 Docker Compose로 RAGFlow 스택을 실행합니다.

docker compose up -d


👉 RAGFlow는 단일 서버가 아니라 RAGFlow + Milvus + MySQL + Redis 등
여러 서비스가 결합된 구조이므로 Compose 기반 실행이 필수입니다.

3️⃣ 실행 상태 확인
컨테이너가 정상적으로 실행 중인지 확인합니다.

docker ps

- 확인 항목:

RAGFlow 컨테이너: Up

Milvus / MySQL 컨테이너: Up

문제가 있을 경우 로그 확인:

docker compose logs ragflow

4️⃣ RAGFlow 문서 처리 흐름
RAGFlow에서 문서가 처리되는 전체 흐름은 다음과 같습니다.

문서 업로드
 → 텍스트 추출
 → 규칙 기반 청킹(Chunking)
 → 임베딩 생성
 → Milvus 벡터 DB 저장

5️⃣ 단일 문서 파일 업로드 실행
⚠️ 밑 예시는 도메인명(domain)과 doc_id 규칙은 Git main 문서 기준과 동일하고, 별도로 도메인명과 사용하고자 하는 문서를 수정해야 함

(A) OpenAI 임베딩 → Milvus ragflow_chunks
cd C:\Ragflow_test\ragflow\sample
$env:EMBEDDING_MODEL_SELECTED="openai"

python .\main.py `
  --input "C:\Ragflow_test\ragflow\sample\dataset_사내규정\이사회규정.pdf" `
  --domain "사내규정" `
  --doc_id "POL-EDU-015" `
  --version 1 `
  --replace false

(B) sRoBERTa 임베딩 → Milvus ragflow_chunks_sroberta
cd C:\Ragflow_test\ragflow\sample
$env:EMBEDDING_MODEL_SELECTED="sroberta"

python .\main.py `
  --input "C:\Ragflow_test\ragflow\sample\dataset_장애인인식개선\직장내괴롭힘예방조치교육자료_근로자용.pdf" `
  --domain "장애인인식개선교육" `
  --doc_id "HR-ANTI-BULLY-001" `
  --version 1 `
  --replace false

✅ doc_id는 무엇을 넣어야 하나?

원칙: AI 서버에서 내려오는 docId 그대로 사용
예: POL-EDU-015

로컬 테스트: 유니크한 식별자 사용 가능
예: HR-ANTI-BULLY-001

핵심 규칙:
👉 replace 테스트 시 반드시 같은 doc_id를 사용해야 함

6️⃣ replace(교체) 동작 테스트
같은 doc_id로 2번 실행하여 기존 문서가 교체되는지 확인합니다.

1️) 최초 업로드 (replace=false)
$env:EMBEDDING_MODEL_SELECTED="sroberta"

python .\main.py `
  --input "C:\...\직장내괴롭힘예방조치교육자료_근로자용.pdf" `
  --domain "장애인인식개선교육" `
  --doc_id "HR-ANTI-BULLY-001" `
  --version 1 `
  --replace false

2) 동일 doc_id 재업로드 (replace=true)
$env:EMBEDDING_MODEL_SELECTED="sroberta"

python .\main.py `
  --input "C:\...\직장내괴롭힘예방조치교육자료_근로자용.pdf" `
  --domain "장애인인식개선교육" `
  --doc_id "HR-ANTI-BULLY-001" `
  --version 2 `
  --replace true

✅ 성공 기준 (콘솔 로그)
replace=true 실행 시 아래 로그 흐름이 나오면 교체 성공입니다.

[MilvusProxy] Deleted chunks for dataset_id=..., doc_id=...
→ Milvus 적재 완료: ...

삭제 로그 없이 바로 insert만 되면 ❌
delete 후 다시 insert 되면 ✅


📌 요약 체크리스트

 .env 환경 변수 설정 완료

 docker compose up -d 실행

 docker ps로 컨테이너 Up 확인

 EMBEDDING_MODEL_SELECTED 설정 확인

 doc_id 동일 여부 확인 (replace 테스트)