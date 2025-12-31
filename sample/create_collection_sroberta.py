import os
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")

NEW_COLLECTION = os.getenv("MILVUS_COLLECTION", "ragflow_chunks_sroberta")
DIM = int(os.getenv("SROBERTA_DIM", "768"))  # 보통 768

# 1) 연결
connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)

# 2) 이미 있으면 스킵
if utility.has_collection(NEW_COLLECTION):
    col = Collection(NEW_COLLECTION)
    print(f"✅ already exists: {NEW_COLLECTION}")
    print("schema:", col.schema)
    print("entities:", col.num_entities)
    raise SystemExit(0)

# 3) 스키마 정의 (너가 쓰는 필드 구조에 맞춤)
fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="dataset_id", dtype=DataType.VARCHAR, max_length=128),
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="chunk_id", dtype=DataType.INT64),
    FieldSchema(name="chunk_hash", dtype=DataType.VARCHAR, max_length=80),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
]

schema = CollectionSchema(fields, description="sroberta embeddings")

# 4) 컬렉션 생성
col = Collection(NEW_COLLECTION, schema=schema)

# 5) 인덱스 생성 (추천)
index_params = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}
col.create_index("embedding", index_params)
col.load()

print(f"✅ created: {NEW_COLLECTION} (dim={DIM})")
