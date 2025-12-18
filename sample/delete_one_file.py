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
EMBED_DIM = int(os.getenv("GEMINI_EMBED_DIM", "768"))

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
DATASET_ID = "정보보안교육"
DOC_ID = "개인정보_보호법_시행령_일부개정_2025.09.23_대통령령_제35780호_시행_2025.10.2._ 개인정보보호위원회.docx"

# =========================
# 4. 삭제 실행
# =========================
milvus.delete_file(
    dataset_id=DATASET_ID,
    doc_id=DOC_ID,
)

print("🧹 파일 단위 삭제 완료")
