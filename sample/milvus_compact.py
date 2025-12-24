from dotenv import load_dotenv
import os
from pymilvus import connections, Collection

# =========================
# 1. 환경 변수 로드
# =========================
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")

# =========================
# 2. Milvus 연결
# =========================
connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT,
)

# =========================
# 3. compaction 실행
# =========================
collection = Collection(MILVUS_COLLECTION)

print("⏳ Milvus compaction 시작...")
collection.compact()
collection.flush()

print("🧹 Milvus compaction 완료")
