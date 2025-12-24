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
            ch = c.get("chunk_hash")
            if not ch:
                ch = hashlib.sha256(c["text"].strip().encode("utf-8")).hexdigest()

            rows.append({
                "dataset_id": dataset_id,
                "doc_id": c["doc_id"],
                "chunk_id": int(c["chunk_id"]),
                "text": c["text"],
                "embedding": c["embedding"],
                "chunk_hash": ch,
            })

        # ✅ dict-list로 insert → 순서 문제 제거
        self.collection.insert(rows)
        self.collection.flush()


    # =========================================================
    # 중복 체크 (chunk_hash)  ✅ 수정본
    # =========================================================
    def exists_chunk_hash(self, dataset_id: str, doc_id: str, chunk_hash: str) -> bool:
        """
        동일 dataset_id(도메인) + doc_id(파일) + chunk_hash(내용) 가 이미 있는지 확인
        - 도메인 누적은 유지하면서
        - 같은 파일에서만 중복 삽입 방지
        """
        expr = (
            f'dataset_id == "{dataset_id}" && '
            f'doc_id == "{doc_id}" && '
            f'chunk_hash == "{chunk_hash}"'
        )
        res = self.collection.query(
            expr=expr,
            output_fields=["chunk_hash"],
            limit=1,
        )
        return len(res) > 0

    # =========================================================
    # 파일 단위 삭제
    # =========================================================
    def delete_file(self, dataset_id: str, doc_id: str):
        """
        파일 단위 삭제 (dataset_id + doc_id 기준)
        """
        expr = f'dataset_id == "{dataset_id}" && doc_id == "{doc_id}"'
        self.collection.delete(expr)
        self.collection.flush()
        print(
            f"[MilvusProxy] Deleted chunks for dataset_id={dataset_id}, doc_id={doc_id}"
        )