from dotenv import load_dotenv
import os
from milvus_proxy import MilvusProxy

# =========================
# 1. 환경 변수 로드
# =========================
load_dotenv()

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
EMBED_DIM = int(os.getenv("OPENAI_EMBED_DIM", "3072"))

# =========================
# 2. Milvus 연결
# =========================
milvus = MilvusProxy(
    host=MILVUS_HOST,
    port=MILVUS_PORT,
    collection_name=MILVUS_COLLECTION,
    dim=EMBED_DIM,
)

print("✅ Milvus 연결 완료")

# =========================
# 3. 여기만 수정 
# =========================
DATASET_ID = "8e7fd878dc9a11f0a241dac8bbbfe4c3"
DOC_ID = "사규.docx"

# =========================
# 4. 삭제 실행
# =========================
milvus.delete_file(
    dataset_id=DATASET_ID,
    doc_id=DOC_ID,
)

print("🧹 파일 단위 삭제 완료")
