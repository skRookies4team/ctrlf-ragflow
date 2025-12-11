from pymilvus import connections, Collection, list_collections
from dotenv import load_dotenv
import os

# .env 로드
load_dotenv()

# milvus 환경설정
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")


connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

print("📌 Milvus 연결 성공!")
print("📂 Collections:", list_collections())

col = Collection("ragflow_chunks")
print("🧱 Schema:", col.schema)
print("🔢 총 엔티티:", col.num_entities)

print("\n▶ 샘플 데이터 조회")
res = col.query(
    expr="chunk_id >= 0",
    output_fields=["dataset_id", "doc_id", "chunk_id", "text"],
    limit=3
)
print(res)
