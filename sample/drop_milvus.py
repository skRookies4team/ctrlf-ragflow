
import os

from dotenv import load_dotenv
from pymilvus import connections, utility

# =========================
# 1. 환경 변수 로드
# =========================
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

# ✅ 1. Milvus에 먼저 연결 (docker 기준)
connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

# ✅ 2. 컬렉션 존재 여부 확인
exists = utility.has_collection("ragflow_chunks_openai")
print("Before drop:", exists)

# ✅ 3. 컬렉션 삭제
if exists:
    utility.drop_collection("ragflow_chunks_openai")
    print("✅ ragflow_chunks_openai 컬렉션 삭제 완료")
else:
    print("ℹ ragflow_chunks_openai 컬렉션이 이미 없음")

# ✅ 4. 다시 확인
print("After drop:", utility.has_collection("ragflow_chunks_openai"))
