# ============================================================
# apps/api-server/admin/server/services/dataset_service.py
# ============================================================
import uuid
import subprocess
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Optional

BASE_DIR = Path(__file__).resolve().parents[3]
SAMPLE_MAIN = BASE_DIR / "sample" / "main.py"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatasetMgr:
    """
    dataset_id = 업로드 '작업' 1건을 식별하는 UUID (status.json/폴더키)
    - uploads: sample/uploads/<dataset_id>/
    - status : sample/uploads/<dataset_id>/status.json

    ✅ 삭제는 두 종류를 제공:
    1) delete_upload(dataset_id): uploads/<dataset_id> 폴더(파일+status) 삭제
    2) purge_index(dataset_id, ...): (선택) Milvus에서 해당 doc_id까지 삭제
       - 이건 MilvusProxy에 delete_doc()가 있어야 한다.
    """

    # -----------------------
    # paths
    # -----------------------
    @staticmethod
    def _upload_dir(dataset_id: str) -> Path:
        return BASE_DIR / "sample" / "uploads" / dataset_id

    @staticmethod
    def _status_path(dataset_id: str) -> Path:
        return DatasetMgr._upload_dir(dataset_id) / "status.json"

    # -----------------------
    # status I/O
    # -----------------------
    @staticmethod
    def _read_status(dataset_id: str) -> Dict[str, Any]:
        p = DatasetMgr._status_path(dataset_id)
        if not p.exists():
            return {"status": "unknown", "dataset_id": dataset_id}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            # 파일이 깨졌거나 부분 기록된 경우
            return {
                "status": "unknown",
                "dataset_id": dataset_id,
                "error": "status.json parse failed",
            }

    @staticmethod
    def _write_status(dataset_id: str, payload: Dict[str, Any]) -> None:
        upload_dir = DatasetMgr._upload_dir(dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        p = DatasetMgr._status_path(dataset_id)

        # 기존 상태가 있으면 merge (없으면 새로)
        current: Dict[str, Any] = {}
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        merged = {
            **current,
            **payload,
            "dataset_id": dataset_id,
            "updated_at": _now_iso(),
        }
        if "created_at" not in merged:
            merged["created_at"] = _now_iso()

        # 원자적 저장(부분기록 방지): temp -> replace
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(p)

    # -----------------------
    # public APIs
    # -----------------------
    @staticmethod
    def save_dataset(file, dataset_name: str) -> str:
        """
        업로드 파일을 sample/uploads/<dataset_id>/ 아래에 저장하고
        status.json 생성 후 dataset_id 반환.
        """
        dataset_id = str(uuid.uuid4())

        upload_dir = DatasetMgr._upload_dir(dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        file.save(file_path)

        DatasetMgr._write_status(
            dataset_id,
            {
                "status": "uploaded",
                "file": str(file_path),
                "filename": file.filename,
                # form의 dataset_name을 domain/논리그룹으로 활용
                "dataset_name": dataset_name,
            },
        )

        return dataset_id

    @staticmethod
    def process_dataset(dataset_id: str) -> None:
        """
        status.json에서 file 경로 + dataset_name(domain)을 읽어서
        main.py를 '업로드된 그 파일만' 처리하도록 실행.
        """
        info = DatasetMgr._read_status(dataset_id)
        file_path = info.get("file")
        domain = info.get("dataset_name", "default")  # 업로드 폼의 dataset_name을 domain으로 사용

        if not file_path:
            DatasetMgr._write_status(
                dataset_id,
                {"status": "failed", "error": "file path missing in status.json"},
            )
            return

        fp = Path(file_path)
        if not fp.exists():
            DatasetMgr._write_status(
                dataset_id,
                {"status": "failed", "error": f"file not found: {file_path}"},
            )
            return

        DatasetMgr._write_status(dataset_id, {"status": "processing"})

        try:
            subprocess.run(
                [
                    sys.executable,
                    str(SAMPLE_MAIN),
                    "--input",
                    str(fp),
                    "--domain",
                    str(domain),
                ],
                cwd=str(SAMPLE_MAIN.parent),
                check=True,
            )
            DatasetMgr._write_status(dataset_id, {"status": "done"})
        except Exception as e:
            DatasetMgr._write_status(
                dataset_id,
                {"status": "failed", "error": str(e)},
            )

    @staticmethod
    def get_status(dataset_id: str) -> Dict[str, Any]:
        return DatasetMgr._read_status(dataset_id)

    # ============================================================
    # ✅ 삭제 1) 업로드 작업 삭제(uploads/<dataset_id> 폴더 삭제)
    # ============================================================
    @staticmethod
    def delete_upload(dataset_id: str) -> Dict[str, Any]:
        """
        sample/uploads/<dataset_id>/ 폴더 자체를 삭제.
        - processing 중이면 삭제 금지(안전)
        """
        upload_dir = DatasetMgr._upload_dir(dataset_id)

        if not upload_dir.exists():
            return {
                "success": False,
                "dataset_id": dataset_id,
                "message": "upload dir not found",
            }

        st = DatasetMgr._read_status(dataset_id).get("status")
        if st == "processing":
            return {
                "success": False,
                "dataset_id": dataset_id,
                "message": "processing in progress; cannot delete now",
            }

        shutil.rmtree(upload_dir, ignore_errors=True)
        return {
            "success": True,
            "dataset_id": dataset_id,
            "message": "upload deleted",
        }

    # ============================================================
    # ✅ 삭제 2) (선택) Milvus 인덱스까지 purge
    # ============================================================
    @staticmethod
    def purge_index(self, dataset_id: str, milvus=None, collection_name: Optional[str] = None) -> Dict[str, Any]:
        info = DatasetMgr._read_status(dataset_id)

        domain = info.get("dataset_name")
        file_path = info.get("file")
        filename = info.get("filename") or (Path(file_path).name if file_path else None)

        if not domain or not filename:
            return {
                "success": False,
                "dataset_id": dataset_id,
                "message": "missing dataset_name(domain) or filename in status.json",
            }

        st = info.get("status")
        if st == "processing":
            return {
                "success": False,
                "dataset_id": dataset_id,
                "message": "processing in progress; cannot purge now",
            }

        if milvus is None:
            try:
                from milvus_proxy import MilvusProxy
                import os
                from dotenv import load_dotenv

                load_dotenv(BASE_DIR / ".env")
                host = os.getenv("MILVUS_HOST")
                port = os.getenv("MILVUS_PORT")
                col = collection_name or os.getenv("MILVUS_COLLECTION", "ragflow_chunks")

                milvus = MilvusProxy(host=host, port=port, collection_name=col)
            except Exception as e:
                return {
                    "success": False,
                    "dataset_id": dataset_id,
                    "message": f"failed to init milvus: {e}",
                }

        try:
            # ✅ 너 proxy는 delete_file이 정식 메서드
            deleted_count = milvus.delete_file(dataset_id=domain, doc_id=filename)

            DatasetMgr._write_status(
                dataset_id,
                {
                    "purged": True,
                    "purged_at": _now_iso(),
                    "purged_domain": domain,
                    "purged_doc_id": filename,
                    "purged_deleted_count": deleted_count,
                },
            )
            return {
                "success": True,
                "dataset_id": dataset_id,
                "message": "index purged",
                "domain": domain,
                "doc_id": filename,
                "deleted_count": deleted_count,
            }
        except Exception as e:
            DatasetMgr._write_status(dataset_id, {"purged": False, "purge_error": str(e)})
            return {
                "success": False,
                "dataset_id": dataset_id,
                "message": f"purge failed: {e}",
                "domain": domain,
                "doc_id": filename,
            }