# sample/milvus_proxy.py
import os
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
        - 기본값은 환경변수(.env)에서 가져옴
          - MILVUS_HOST (기본: localhost)
          - MILVUS_PORT (기본: 19530)
          - MILVUS_COLLECTION (기본: ragflow_chunks)
          - EMBED_DIM (기본: 768)
        - test_chunking_embedding.py에서 인자로 넘기면 그 값이 우선
        """

        host = host or os.getenv("MILVUS_HOST", "localhost")
        port = port or os.getenv("MILVUS_PORT", "19530")
        collection_name = collection_name or os.getenv(
            "MILVUS_COLLECTION",
            "ragflow_chunks",
        )
        if dim is None:
            dim = int(os.getenv("EMBED_DIM", "768"))

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
                max_length=8192,  # 긴 청크 대비 여유 있게
            ),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dim,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="RAGFlow custom chunks mirror",
        )
        collection = Collection(
            name=self.collection_name,
            schema=schema,
            # consistency_level 지정하고 싶으면 여기서 추가 가능
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
            f"[MilvusProxy] Created collection with index: {self.collection_name} "
            f"(dim={self.dim})"
        )

    def insert_chunks(
        self,
        dataset_id: str,
        chunks: List[Dict[str, Any]],
    ):
        """
        chunks 예시:
        [
          {
            "doc_id": "이사회규정.pdf",
            "chunk_id": 0,
            "text": "...",
            "embedding": [...],
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

        self.collection.insert(
            [
                dataset_ids,
                doc_ids,
                chunk_ids,
                texts,
                embeddings,
            ]
        )
        self.collection.flush()
        print(
            f"[MilvusProxy] Inserted {len(chunks)} chunks into {self.collection_name}"
        )
