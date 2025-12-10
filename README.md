📁 .env 파일 예시
# RAGFlow API 설정
RAGFLOW_API_KEY=your_ragflow_api_key_here
RAGFLOW_HOST=http://localhost
RAGFLOW_EMBEDDING_MODEL=

# Gemini 설정 --> 임시로 임베딩 시 gemini 사용
GEMINI_API_KEY=
GEMINI_EMBED_MODEL=models/text-embedding-004
GEMINI_EMBED_DIM=

# Milvus 설정
MILVUS_HOST=
MILVUS_PORT=
MILVUS_COLLECTION=
EMBED_DIM=
ENABLE_MILVUS_MIRROR=true

📌 README에 추가할 안내문
> ⚠️ **주의:** `.env` 파일은 절대 Git에 커밋되면 안 됩니다.  
> `.gitignore` 에 아래 항목이 포함되어 있어야 합니다:

    .env
    *.env