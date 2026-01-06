import os
from collections import defaultdict

from dotenv import load_dotenv
from pymilvus import Collection, connections, list_collections

# =========================
# 1. 환경 로드
# =========================
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ragflow_chunks_openai")

# =========================
# 2. Milvus 연결
# =========================
connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

print("📌 Milvus 연결 성공!")
print("📂 Collections:", list_collections())

col = Collection(MILVUS_COLLECTION)

print("\n🧱 Schema:")
print(col.schema)

print("\n🔢 총 엔티티 수:", col.num_entities)

# =========================
# 3. dataset_id / doc_id 기준으로 조회
# =========================
print("\n📦 dataset_id (폴더) / doc_id (파일) 목록 조회")

# 넉넉하게 가져오기 (엔티티 많으면 limit 키우면 됨)
res = col.query(
    expr="chunk_id >= 0",
    output_fields=["dataset_id", "doc_id"],
    limit=5000
)

tree = defaultdict(set)

for r in res:
    dataset_id = r.get("dataset_id", "UNKNOWN_DATASET")
    doc_id = r.get("doc_id", "UNKNOWN_DOC")
    tree[dataset_id].add(doc_id)

# =========================
# 4. 폴더 구조처럼 출력
# =========================
for dataset_id, docs in tree.items():
    print(f"\n📁 dataset_id: {dataset_id}")
    for d in sorted(docs):
        print(f"   ├─ 📄 {d}")

# =========================
# 5. 필드 목록 + 값 예시(짧게)
# =========================
print("\n🧾 Fields & sample values (short)")

# 스키마에서 필드명 추출
fields = [f.name for f in col.schema.fields]

sample = col.query(
    expr="chunk_id >= 0",
    output_fields=fields,
    limit=1
)

if not sample:
    print("⚠ 데이터 없음")
else:
    row = sample[0]
    for f in fields:
        v = row.get(f)

        # 값 짧게 요약
        if v is None:
            short = "None"
        elif isinstance(v, list):
            short = f"<list len={len(v)}>"
        elif isinstance(v, str):
            short = v[:40] + ("..." if len(v) > 40 else "")
        else:
            short = str(v)

        print(f" - {f:20s}: {short}")
