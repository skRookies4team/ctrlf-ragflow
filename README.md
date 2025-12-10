📁 .env 파일 생성

프로젝트 루트(예: RAGFLOW/)에 .env 파일을 생성하고 다음 내용을 붙여넣습니다.

⚠️ .env 파일은 절대 Git에 올리지 마세요.
.gitignore에 *.env 또는 .env 가 등록되어 있어야 합니다.

###########################################
# 1. RAGFlow API 설정
###########################################

# Ragflow Personal Access Token
RAGFLOW_API_KEY=your_ragflow_api_key_here

# Ragflow Server Host
# Docker로 실행 중이면 보통 http://localhost
RAGFLOW_HOST=http://localhost

# Ragflow에서 사용할 임베딩 모델 이름
RAGFLOW_EMBEDDING_MODEL=


###########################################
# 2. Gemini 임베딩 모델 / LLM 설정
###########################################

# Google Gemini API Key --> 임시로 Gemini 사용
GEMINI_API_KEY=your_gemini_api_key_here

# Gemini 임베딩 모델 이름
GEMINI_EMBED_MODEL=models/text-embedding-004

# Gemini 임베딩 모델 차원
GEMINI_EMBED_DIM=768


###########################################
# 3. Milvus 벡터DB 설정
###########################################

# milvus standalone docker 실행 시 host/port
MILVUS_HOST=
MILVUS_PORT=

# 사용할 Milvus 컬렉션 이름
MILVUS_COLLECTION=ragflow_chunks

# 임베딩 차원 (임베딩 차원과 반드시 동일)
EMBED_DIM=

# Milvus 미러링 기능 활성화 여부
ENABLE_MILVUS_MIRROR=true

