# test_milvus_conn.py
import os

from dotenv import load_dotenv
from pymilvus import connections, utility

# .env 로드
load_dotenv()

# milvus 환경설정
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")

print(f"👉 connecting to Milvus on {MILVUS_HOST}:{MILVUS_PORT}...")

try:
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT,
    )
    print("✅ connected!")
except Exception as e:
    print("❌ connection failed:", e)
    exit(1)

try:
    cols = utility.list_collections()
    print("collections:", cols)
except Exception as e:
    print("❌ Failed to list collections:", e)
