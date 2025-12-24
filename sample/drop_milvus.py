
from pymilvus import connections, utility
from dotenv import load_dotenv
import os
from milvus_proxy import MilvusProxy

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
exists = utility.has_collection("ragflow_chunks")
print("Before drop:", exists)

# ✅ 3. 컬렉션 삭제
if exists:
    utility.drop_collection("ragflow_chunks")
    print("✅ ragflow_chunks 컬렉션 삭제 완료")
else:
    print("ℹ ragflow_chunks 컬렉션이 이미 없음")

# ✅ 4. 다시 확인
print("After drop:", utility.has_collection("ragflow_chunks"))
