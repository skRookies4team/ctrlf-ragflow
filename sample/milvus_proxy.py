# sample/milvus_proxy.py
import os
import hashlib
import time
import traceback
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

class MilvusProxy:
    """
    ✅ 패치 요약 (ChatSource 메타 + department)
    - 스키마에 department(VARCHAR, max_length=32) 추가
    - ✅ 스키마에 ChatSource 메타 추가:
      - document_title(VARCHAR, 256)
      - page_num(INT64)
      - section(VARCHAR, 128)
      - section_path(VARCHAR, 256)
    - insert 시 위 메타 값 저장 (없으면 안전 기본값/None)
    - expr에 들어가는 문자열 escape 처리 (_escape)
    - VARCHAR(text) max_length(8192) 초과 방지: insert 전에 SAFE_TEXT_MAX로 컷
    - delete_file 로그 강화
    """

    SAFE_TEXT_MAX = int(os.getenv("MILVUS_TEXT_MAX", "7500"))

    DEFAULT_DEPARTMENT = os.getenv("MILVUS_DEFAULT_DEPARTMENT", "ALL").strip() or "ALL"
    DEPT_INDEX_TYPE = os.getenv("MILVUS_DEPT_INDEX_TYPE", "BITMAP").strip() or "BITMAP"
    DEPT_INDEX_NAME = os.getenv("MILVUS_DEPT_INDEX_NAME", "idx_department").strip() or "idx_department"

    # (선택) 메타 인덱스 이름
    META_INDEX_PREFIX = os.getenv("MILVUS_META_INDEX_PREFIX", "idx_").strip() or "idx_"

    def __init__(
        self,
        host: str | None = None,
        port: str | None = None,
        collection_name: str | None = None,
        dim: int | None = None,
    ):
        host = host or os.getenv("MILVUS_HOST", "localhost")
        port = port or os.getenv("MILVUS_PORT", "19530")
        collection_name = collection_name or os.getenv("MILVUS_COLLECTION", "ragflow_chunks_openai")

        if dim is None:
            dim = int(os.getenv("OPENAI_EMBED_DIM") or os.getenv("EMBED_DIM", "3072"))

        self.collection_name = collection_name
        self.dim = dim

        connections.connect("default", host=host, port=port)

        if not utility.has_collection(collection_name):
            self._create_collection()
        else:
            self.collection = Collection(collection_name)
            self.collection.load()

        print(f"[MilvusProxy] Connected to collection '{self.collection_name}' (dim={self.dim})")

        # DEBUG
        print("\n[DEBUG] Milvus collection schema check")
        for i, f in enumerate(self.collection.schema.fields):
            max_len = getattr(f, "max_length", None)
            dim = getattr(f, "dim", None)
            print(f"  [{i}] name={f.name}, type={f.dtype}, max_length={max_len}, dim={dim}")
        print()

    @staticmethod
    def _escape(s: str) -> str:
        if s is None:
            return ""
        return str(s).replace("\\", "\\\\").replace('"', '\\"')

    def _has_field(self, field_name: str) -> bool:
        try:
            return any(f.name == field_name for f in self.collection.schema.fields)
        except Exception:
            return False

    def _create_collection(self):
        """
        ragflow_chunks_openai 컬렉션 스키마 정의 + 인덱스 생성
        """
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),

            FieldSchema(name="dataset_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),

            # ✅ department (필터링)
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=32),

            # ✅ ChatSource 메타 (AI UX/근거 표시에 중요)
            FieldSchema(name="document_title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="page_num", dtype=DataType.INT64, nullable=True),
            FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=1024),

            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="chunk_hash", dtype=DataType.VARCHAR, max_length=64),
        ]

        schema = CollectionSchema(fields=fields, description="RAGFlow custom chunks mirror (+department +chat_source_meta)")
        collection = Collection(name=self.collection_name, schema=schema)

        # 벡터 인덱스
        vec_index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="embedding", index_params=vec_index_params)

        # ✅ department 스칼라 인덱스 (필터 성능)
        try:
            collection.create_index(
                field_name="department",
                index_params={"index_type": self.DEPT_INDEX_TYPE},
                index_name=self.DEPT_INDEX_NAME,
            )
            print(f"[MilvusProxy] Created scalar index: field=department index_type={self.DEPT_INDEX_TYPE}")
        except Exception as e:
            print(f"[MilvusProxy][WARN] department index create failed (skip): {repr(e)}")

        # (선택) 메타 필드 인덱스: 자주 expr로 필터/정렬할 거면 도움
        # - index_type 지원은 Milvus 버전마다 다를 수 있어 실패해도 무시
        for meta_field in ("doc_id", "dataset_id", "document_title", "section_path"):
            try:
                collection.create_index(
                    field_name=meta_field,
                    index_params={"index_type": "BITMAP"},
                    index_name=f"{self.META_INDEX_PREFIX}{meta_field}",
                )
            except Exception:
                pass

        collection.load()
        self.collection = collection
        print(f"[MilvusProxy] Created collection '{self.collection_name}' (dim={self.dim})")

    def insert_chunks(self, dataset_id: str, chunks: List[Dict[str, Any]]):
        if not chunks:
            print("[MILVUS] insert_chunks: chunks=0 -> return")
            return

        t0 = time.time()
        col_name = getattr(self.collection, "name", None)
        print(f"[MILVUS] insert_chunks start: collection={col_name}, dataset_id={dataset_id}, chunks={len(chunks)}")

        # ----------------------------
        # 1) 스키마 기반 max_length 맵 구성
        # ----------------------------
        schema_fields = list(self.collection.schema.fields)

        # auto_id(pk) 제외하고 insert할 필드만 추림
        insert_fields = [f for f in schema_fields if not getattr(f, "auto_id", False)]
        insert_field_names = [f.name for f in insert_fields]

        # VARCHAR max_length 딕셔너리
        varchar_max: Dict[str, int] = {}
        for f in schema_fields:
            if getattr(f, "dtype", None) == DataType.VARCHAR:
                ml = getattr(f, "max_length", None)
                if ml is not None:
                    varchar_max[f.name] = int(ml)

        # (기존 컬렉션 호환) 존재하는지 체크
        has_department = self._has_field("department")
        has_document_title = self._has_field("document_title")
        has_page_num = self._has_field("page_num")
        has_section = self._has_field("section")
        has_section_path = self._has_field("section_path")

        # ----------------------------
        # 2) row 빌드 (안전 컷/정규화)
        # ----------------------------
        rows: List[Dict[str, Any]] = []
        skipped = 0
        expected_dim = getattr(self, "dim", None)

        def _cut(field: str, val: Any) -> str:
            """VARCHAR 필드면 스키마 max_length로 안전 컷"""
            s = "" if val is None else str(val).strip()
            ml = varchar_max.get(field)
            if ml is not None and len(s) > ml:
                return s[:ml]
            return s

        for idx, c in enumerate(chunks, start=1):
            try:
                raw_text = (c.get("text") or "").strip()
                # text는 SAFE_TEXT_MAX로 1차 컷 + 스키마로 2차 컷
                if isinstance(self.SAFE_TEXT_MAX, int) and len(raw_text) > self.SAFE_TEXT_MAX:
                    raw_text = raw_text[: self.SAFE_TEXT_MAX]
                raw_text = _cut("text", raw_text)

                ch = c.get("chunk_hash") or hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

                emb = c.get("embedding")
                if emb is None:
                    skipped += 1
                    continue

                # embedding dim 체크(경고)
                try:
                    emb_len = len(emb)
                    if expected_dim and int(expected_dim) != int(emb_len):
                        print(f"[MILVUS][WARN] embedding dim mismatch: got={emb_len}, expected={expected_dim} (row={idx})")
                except Exception:
                    pass

                # department
                dept = (c.get("department") or self.DEFAULT_DEPARTMENT or "ALL")
                dept = _cut("department", dept) if dept else "ALL"

                # ChatSource meta
                document_title = _cut("document_title", c.get("document_title") or "")
                section = _cut("section", c.get("section") or "")
                section_path = _cut("section_path", c.get("section_path") or "")

                page_num_val = c.get("page_num", None)
                page_num: Optional[int] = None
                if page_num_val is not None and page_num_val != "":
                    try:
                        page_num = int(page_num_val)
                    except Exception:
                        page_num = None

                row: Dict[str, Any] = {
                    "dataset_id": _cut("dataset_id", dataset_id),
                    "doc_id": _cut("doc_id", c.get("doc_id") or ""),
                    "chunk_id": int(c.get("chunk_id", 0)),
                    "text": raw_text,
                    "embedding": emb,
                    "chunk_hash": _cut("chunk_hash", ch),
                }

                if has_department:
                    row["department"] = dept
                if has_document_title:
                    row["document_title"] = document_title
                if has_page_num:
                    row["page_num"] = page_num
                if has_section:
                    row["section"] = section
                if has_section_path:
                    row["section_path"] = section_path

                # ✅ 최종 방어: section이 128 넘으면 즉시 로그(여기 걸리면 upstream 값이 이미 문제)
                if has_section and row.get("section") and len(row["section"]) > varchar_max.get("section", 10**9):
                    print(f"[MILVUS][DEBUG] section too long BEFORE insert row={idx} len={len(row['section'])}")
                    print(f"  preview={row['section'][:200]}")

                rows.append(row)

            except Exception as e:
                skipped += 1
                print(f"[MILVUS][WARN] row build failed (row={idx}): {repr(e)}")

        print(f"[MILVUS] rows built: rows={len(rows)}, skipped={skipped}")
        if not rows:
            print("[MILVUS] no rows -> return")
            return

        # ----------------------------
        # 3) ✅ 컬럼 기반 insert (dict 매칭 꼬임 방지)
        # ----------------------------
        try:
            # auto_id 필드는 insert_data에서 제외되어야 함
            columns: Dict[str, List[Any]] = {name: [] for name in insert_field_names}

            for i, r in enumerate(rows, start=1):
                for name in insert_field_names:
                    columns[name].append(r.get(name))

            insert_data = [columns[name] for name in insert_field_names]

            # 디버그: 첫 row 샘플
            first = rows[0]
            print(
                "[MILVUS] insert (column-mode) fields="
                + ",".join(insert_field_names)
            )
            print(
                f"[MILVUS] sample row: doc_id={first.get('doc_id')}, chunk_id={first.get('chunk_id')}, "
                f"dept={first.get('department')}, title={first.get('document_title')}, page={first.get('page_num')}, "
                f"section_len={len(first.get('section') or '')}, section={str(first.get('section') or '')[:80]!r}, "
                f"text_len={len(first.get('text') or '')}"
            )

            self.collection.insert(insert_data)
            print(f"[MILVUS] insert OK: inserted={len(rows)} (elapsed={time.time() - t0:.2f}s)")

        except Exception as e:
            print(f"[MILVUS][ERROR] insert failed: {repr(e)}")
            # 실패했을 때 원인 row를 찾기 위한 추가 덤프
            # (Milvus가 row number를 주는 케이스가 많아서, 너 로그랑 같이 보면 바로 찾음)
            try:
                # 길이 초과 후보를 쫙 찍어줌
                for j, r in enumerate(rows, start=1):
                    sec = r.get("section") or ""
                    if "section" in varchar_max and len(sec) > varchar_max["section"]:
                        print(f"[MILVUS][DUMP] row={j} section_len={len(sec)} preview={sec[:200]!r}")

                    sp = r.get("section_path") or ""
                    if "section_path" in varchar_max and len(sp) > varchar_max["section_path"]:
                        print(f"[MILVUS][DUMP] row={j} section_path_len={len(sp)} preview={sp[:200]!r}")

                    dt = r.get("document_title") or ""
                    if "document_title" in varchar_max and len(dt) > varchar_max["document_title"]:
                        print(f"[MILVUS][DUMP] row={j} title_len={len(dt)} preview={dt[:200]!r}")

                    tx = r.get("text") or ""
                    if "text" in varchar_max and len(tx) > varchar_max["text"]:
                        print(f"[MILVUS][DUMP] row={j} text_len={len(tx)} preview={tx[:200]!r}")
            except Exception:
                pass

            print(traceback.format_exc()[-2000:])
            raise

        # ----------------------------
        # 4) flush
        # ----------------------------
        try:
            self.collection.flush()
            print(f"[MILVUS] flush OK (total_elapsed={time.time() - t0:.2f}s)")
        except Exception as e:
            print(f"[MILVUS][ERROR] flush failed: {repr(e)}")
            print(traceback.format_exc()[-1200:])
            raise


    def exists_chunk_hash(self, dataset_id: str, doc_id: str, chunk_hash: str, department: str | None = None) -> bool:
        ds = self._escape(dataset_id)
        di = self._escape(doc_id)
        ch = self._escape(chunk_hash)

        expr = (
            f'dataset_id == "{ds}" && '
            f'doc_id == "{di}" && '
            f'chunk_hash == "{ch}"'
        )

        if department is not None and self._has_field("department"):
            dept = self._escape(department)
            expr += f' && department == "{dept}"'

        res = self.collection.query(expr=expr, output_fields=["chunk_hash"], limit=1)
        return len(res) > 0

    def delete_file(self, dataset_id: str, doc_id: str) -> int:
        ds = self._escape(dataset_id)
        di = self._escape(doc_id)
        expr = f'dataset_id == "{ds}" && doc_id == "{di}"'

        try:
            before_sample = self.collection.query(expr=expr, output_fields=["pk"], limit=1)
            print(f"[MilvusProxy] delete_file before_sample_count={len(before_sample)} expr={expr}")
        except Exception as e:
            print(f"[MilvusProxy] delete_file precheck failed: {e}")

        res = self.collection.delete(expr)
        self.collection.flush()

        deleted = 0
        if res is not None:
            deleted = getattr(res, "delete_count", None) or getattr(res, "deleted_count", None) or 0

        try:
            after_sample = self.collection.query(expr=expr, output_fields=["pk"], limit=1)
            print(f"[MilvusProxy] delete_file after_sample_count={len(after_sample)} expr={expr}")
        except Exception as e:
            print(f"[MilvusProxy] delete_file postcheck failed: {e}")

        print(f"[MilvusProxy] Deleted chunks for dataset_id={dataset_id}, doc_id={doc_id} (reported_deleted={deleted})")
        return int(deleted)

    def delete_doc(self, dataset_id: str, doc_id: str) -> int:
        return self.delete_file(dataset_id, doc_id)
