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
    def insert_chunks(
        self,
        dataset_id: str,
        chunks: List[Dict[str, Any]],
    ):
        """
        chunks 예시:
        [
          {
            "doc_id": "...",
            "chunk_id": 0,
            "text": "...",
            "embedding": [...],
            "chunk_hash": "...",  # optional
          },
        ]
        """
        if not chunks:
            return

        dataset_ids = [dataset_id] * len(chunks)
        doc_ids = [c["doc_id"] for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        embeddings = [c["embedding"] for c in chunks]

        # ✅ chunk_hash 없으면 text 기반으로 생성
        chunk_hashes: List[str] = []
        for c in chunks:
            if c.get("chunk_hash"):
                chunk_hashes.append(c["chunk_hash"])
            else:
                h = hashlib.sha256(
                    c["text"].strip().encode("utf-8")
                ).hexdigest()
                chunk_hashes.append(h)

        # ⚠️ auto_id(pk)는 넣지 않음
        # 스키마 순서:
        # pk(auto) | dataset_id | doc_id | chunk_id | text | embedding | chunk_hash
        self.collection.insert(
            [
                dataset_ids,   # dataset_id
                doc_ids,       # doc_id
                chunk_ids,     # chunk_id
                texts,         # text
                embeddings,    # embedding
                chunk_hashes,  # chunk_hash
            ]
        )

        self.collection.flush()
        print(
            f"[MilvusProxy] Inserted {len(chunks)} chunks into '{self.collection_name}'"
        )

    # =========================================================
    # 중복 체크 (chunk_hash)
    # =========================================================
    def exists_chunk_hash(self, dataset_id: str, chunk_hash: str) -> bool:
        """
        동일 dataset_id + chunk_hash 청크가 이미 있는지 확인
        """
        expr = f'dataset_id == "{dataset_id}" && chunk_hash == "{chunk_hash}"'
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