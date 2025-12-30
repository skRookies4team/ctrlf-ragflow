# sample/milvus_proxy.py
import os
import hashlib
from typing import List, Dict, Any

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


class MilvusProxy:
    """
    ✅ 패치 요약 (최소 수정)
    - expr에 들어가는 문자열 escape 처리 (_escape)
    - VARCHAR(text) max_length(8192) 초과 방지: insert 전에 SAFE_TEXT_MAX로 컷
      (한글 UTF-8 바이트 이슈 때문에 len() 기준으로 더 여유 있게 자름)
    - delete_file 로그 강화
    """

    # ✅ Milvus 스키마에서 text max_length=8192 이므로 여유 있게 잘라서 안전 확보
    SAFE_TEXT_MAX = int(os.getenv("MILVUS_TEXT_MAX", "7500"))

    def __init__(
        self,
        host: str | None = None,
        port: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
    ):
        """
        Milvus 연결 및 컬렉션 헬퍼

        환경변수 우선순위:
        - MILVUS_HOST (default: localhost)
        - MILVUS_PORT (default: 19530)
        - MILVUS_COLLECTION (default: ragflow_chunks)
        - OPENAI_EMBED_DIM (권장)
        - EMBED_DIM (fallback)
        """

        host = host or os.getenv("MILVUS_HOST", "localhost")
        port = port or os.getenv("MILVUS_PORT", "19530")
        collection_name = collection_name or os.getenv(
            "MILVUS_COLLECTION",
            "ragflow_chunks",
        )

        if dim is None:
            dim = int(
                os.getenv("OPENAI_EMBED_DIM")
                or os.getenv("EMBED_DIM", "768")
            )

        self.collection_name = collection_name
        self.dim = dim

        # Milvus 연결
        connections.connect("default", host=host, port=port)

        # 컬렉션 준비
        if not utility.has_collection(collection_name):
            self._create_collection()
        else:
            self.collection = Collection(collection_name)
            self.collection.load()

        print(
            f"[MilvusProxy] Connected to collection '{self.collection_name}' "
            f"(dim={self.dim})"
        )

        # =========================================================
        # 🔍 DEBUG: 실제 Milvus 컬렉션 스키마 진단
        # =========================================================
        print("\n[DEBUG] Milvus collection schema check")
        for i, f in enumerate(self.collection.schema.fields):
            max_len = getattr(f, "max_length", None)
            dim = getattr(f, "dim", None)
            print(
                f"  [{i}] name={f.name}, "
                f"type={f.dtype}, "
                f"max_length={max_len}, "
                f"dim={dim}"
            )
        print()

    # =========================================================
    # ✅ expr 안전 처리
    # =========================================================
    @staticmethod
    def _escape(s: str) -> str:
        """
        Milvus expr 문자열용 최소 escape
        - 역슬래시/따옴표 깨짐 방지
        """
        if s is None:
            return ""
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    # =========================================================
    # 컬렉션 생성
    # =========================================================
    def _create_collection(self):
        """
        ragflow_chunks 컬렉션 스키마 정의 + 인덱스 생성
        """
        fields = [
            FieldSchema(
                name="pk",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=True,
            ),
            FieldSchema(
                name="dataset_id",
                dtype=DataType.VARCHAR,
                max_length=128,
            ),
            FieldSchema(
                name="doc_id",
                dtype=DataType.VARCHAR,
                max_length=256,
            ),
            FieldSchema(
                name="chunk_id",
                dtype=DataType.INT64,
            ),
            FieldSchema(
                name="text",
                dtype=DataType.VARCHAR,
                max_length=8192,
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
            FieldSchema(
                name="chunk_hash",
                dtype=DataType.VARCHAR,
                max_length=64,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAGFlow custom chunks mirror",
        )

        collection = Collection(
            name=self.collection_name,
            schema=schema,
        )

        # 벡터 인덱스 생성
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(
            field_name="embedding",
            index_params=index_params,
        )

        collection.load()
        self.collection = collection

        print(
            f"[MilvusProxy] Created collection '{self.collection_name}' "
            f"(dim={self.dim})"
        )

    # =========================================================
    # 청크 삽입
    # =========================================================
    def insert_chunks(self, dataset_id: str, chunks: List[Dict[str, Any]]):
        if not chunks:
            return

        rows = []
        for c in chunks:
            raw_text = (c.get("text") or "").strip()

            # ✅ text 길이 방어 (UTF-8 이슈 포함해서 여유 있게 컷)
            if len(raw_text) > self.SAFE_TEXT_MAX:
                raw_text = raw_text[: self.SAFE_TEXT_MAX]

            ch = c.get("chunk_hash")
            if not ch:
                ch = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

            rows.append({
                "dataset_id": str(dataset_id),
                "doc_id": str(c["doc_id"]),
                "chunk_id": int(c["chunk_id"]),
                "text": raw_text,
                "embedding": c["embedding"],
                "chunk_hash": ch,
            })

        # ✅ dict-list로 insert → 순서 문제 제거
        self.collection.insert(rows)
        self.collection.flush()

    # =========================================================
    # 중복 체크 (chunk_hash)
    # =========================================================
    def exists_chunk_hash(self, dataset_id: str, doc_id: str, chunk_hash: str) -> bool:
        """
        동일 dataset_id(도메인) + doc_id(파일) + chunk_hash(내용) 가 이미 있는지 확인
        - 도메인 누적은 유지하면서
        - 같은 파일에서만 중복 삽입 방지
        """
        ds = self._escape(dataset_id)
        di = self._escape(doc_id)
        ch = self._escape(chunk_hash)

        expr = (
            f'dataset_id == "{ds}" && '
            f'doc_id == "{di}" && '
            f'chunk_hash == "{ch}"'
        )
        res = self.collection.query(
            expr=expr,
            output_fields=["chunk_hash"],
            limit=1,
        )
        return len(res) > 0

    # =========================================================
    # 파일 단위 삭제 (dataset_id + doc_id)
    # =========================================================
    def delete_file(self, dataset_id: str, doc_id: str) -> int:
        """
        파일 단위 삭제 (dataset_id + doc_id 기준)

        ✅ 주의:
        - pymilvus 버전에 따라 delete 결과에 deleted_count가 0으로 나오거나 없을 수 있음
        - "삭제가 실제로 됐는지"는 후속 query로 검증하는 게 가장 확실
        """
        ds = self._escape(dataset_id)
        di = self._escape(doc_id)

        expr = f'dataset_id == "{ds}" && doc_id == "{di}"'

        # (선택) 삭제 전 샘플 확인용
        try:
            before_sample = self.collection.query(expr=expr, output_fields=["pk"], limit=1)
            print(f"[MilvusProxy] delete_file before_sample_count={len(before_sample)} expr={expr}")
        except Exception as e:
            print(f"[MilvusProxy] delete_file precheck failed: {e}")

        res = self.collection.delete(expr)
        self.collection.flush()

        deleted = 0
        if res is not None:
            deleted = (
                getattr(res, "delete_count", None)
                or getattr(res, "deleted_count", None)
                or 0
            )

        # (선택) 삭제 후 샘플 확인용
        try:
            after_sample = self.collection.query(expr=expr, output_fields=["pk"], limit=1)
            print(f"[MilvusProxy] delete_file after_sample_count={len(after_sample)} expr={expr}")
        except Exception as e:
            print(f"[MilvusProxy] delete_file postcheck failed: {e}")

        print(
            f"[MilvusProxy] Deleted chunks for dataset_id={dataset_id}, doc_id={doc_id} "
            f"(reported_deleted={deleted})"
        )
        return int(deleted)

    # =========================================================
    # ✅ alias: services.py에서 delete_doc으로 부르고 싶을 때
    # =========================================================
    def delete_doc(self, dataset_id: str, doc_id: str) -> int:
        return self.delete_file(dataset_id, doc_id)
