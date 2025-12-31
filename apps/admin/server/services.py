# ============================================================
# apps/api-server/admin/server/services.py   (or services/dataset_service.py)
# ============================================================
"""
DatasetMgr (Admin/API Server side)

✅ 지원 범위
1) (기존) 업로드 기반 작업
   - save_dataset(file, dataset_name) -> dataset_id(UUID)
   - process_dataset(dataset_id) : sample/main.py --input <file> --domain <dataset_name>
   - get_status(dataset_id)

2) (신규) AI -> RAGFlow ingest 작업
   - create_ingest_job(payload) -> ingest_id
   - ingest_from_ai(ingest_id)
     - fileUrl 다운로드
     - replace=true 이면 (Milvus에서) doc_id 기준 기존 청크 삭제
     - sample/main.py 를 단일파일 모드로 실행 (가능하면 --doc_id/--replace 전달)
     - (선택) AI 콜백 POST /internal/ai/callbacks/ragflow/ingest

⚠️ 주의
- 이 파일은 “Flask 프로세스” 안에서 돌아감. 따라서 subprocess 실행은 blocking이라
  routes.py에서 thread로 돌리게 해둠.
"""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import requests

# (선택) 콜백이 필요하면 사용
# requests.post 로 처리함

BASE_DIR = Path(__file__).resolve().parents[3]
SAMPLE_MAIN = BASE_DIR / "sample" / "main.py"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# DatasetMgr
# ============================================================
class DatasetMgr:
    """
    저장 구조

    (A) 업로드 작업
      sample/uploads/<dataset_id>/status.json

    (B) ingest 작업(AI->RAGFlow)
      sample/ingest_jobs/<ingest_id>/status.json
      sample/ingest_jobs/<ingest_id>/payload.json
      sample/ingest_jobs/<ingest_id>/download/<파일>
    """

    # -------------------------
    # (A) 업로드 작업 저장소
    # -------------------------
    @staticmethod
    def _upload_dir(dataset_id: str) -> Path:
        return BASE_DIR / "sample" / "uploads" / dataset_id

    @staticmethod
    def _status_path(dataset_id: str) -> Path:
        return DatasetMgr._upload_dir(dataset_id) / "status.json"

    @staticmethod
    def _read_status(dataset_id: str) -> Dict[str, Any]:
        p = DatasetMgr._status_path(dataset_id)
        if not p.exists():
            return {"status": "unknown", "dataset_id": dataset_id}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "unknown", "dataset_id": dataset_id, "error": "status.json parse failed"}

    @staticmethod
    def _write_status(dataset_id: str, payload: Dict[str, Any]) -> None:
        upload_dir = DatasetMgr._upload_dir(dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        p = DatasetMgr._status_path(dataset_id)

        current: Dict[str, Any] = {}
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        merged = {**current, **payload, "dataset_id": dataset_id, "updated_at": _now_iso()}
        if "created_at" not in merged:
            merged["created_at"] = _now_iso()

        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    @staticmethod
    def save_dataset(file, dataset_name: str) -> str:
        """
        form-data 업로드 기반:
          - dataset_id(UUID) 생성
          - sample/uploads/<dataset_id>/ 아래로 저장
          - status.json 생성
        """
        dataset_id = str(uuid.uuid4())
        upload_dir = DatasetMgr._upload_dir(dataset_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        file.save(file_path)

        DatasetMgr._write_status(dataset_id, {
            "status": "uploaded",
            "file": str(file_path),
            "dataset_name": dataset_name,
        })
        return dataset_id

    @staticmethod
    def process_dataset(dataset_id: str) -> None:
        """
        save_dataset()로 저장된 단일 파일을 sample/main.py로 처리
        """
        info = DatasetMgr._read_status(dataset_id)
        file_path = info.get("file")
        domain = info.get("dataset_name", "default")

        if not file_path:
            DatasetMgr._write_status(dataset_id, {"status": "failed", "error": "file path missing"})
            return

        fp = Path(file_path)
        if not fp.exists():
            DatasetMgr._write_status(dataset_id, {"status": "failed", "error": f"file not found: {file_path}"})
            return

        DatasetMgr._write_status(dataset_id, {"status": "processing"})

        try:
            subprocess.run(
                [sys.executable, str(SAMPLE_MAIN), "--input", str(fp), "--domain", str(domain)],
                cwd=str(SAMPLE_MAIN.parent),
                check=True,
            )
            DatasetMgr._write_status(dataset_id, {"status": "done"})
        except Exception as e:
            DatasetMgr._write_status(dataset_id, {"status": "failed", "error": str(e)})

    @staticmethod
    def get_status(dataset_id: str) -> Dict[str, Any]:
        return DatasetMgr._read_status(dataset_id)

    # -------------------------
    # (A-추가) 업로드 작업 삭제(정리)
    # -------------------------
    @staticmethod
    def delete_upload_job(dataset_id: str) -> Dict[str, Any]:
        """
        ✅ “작업 폴더/상태” 정리용
        - Milvus 삭제는 여기서 안 함(현재 dataset_id가 '작업ID'라서 도메인/문서와 매핑이 애매함)
        """
        upload_dir = DatasetMgr._upload_dir(dataset_id)
        if not upload_dir.exists():
            return {"success": False, "message": "upload job not found"}

        # 폴더 삭제
        for p in sorted(upload_dir.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except Exception:
                pass
        try:
            upload_dir.rmdir()
        except Exception:
            pass

        return {"success": True, "message": f"upload job deleted: {dataset_id}"}

    # ============================================================
    # (B) ingest 작업 저장소
    # ============================================================
    @staticmethod
    def _ingest_dir(ingest_id: str) -> Path:
        return BASE_DIR / "sample" / "ingest_jobs" / ingest_id

    @staticmethod
    def _ingest_status_path(ingest_id: str) -> Path:
        return DatasetMgr._ingest_dir(ingest_id) / "status.json"

    @staticmethod
    def _ingest_payload_path(ingest_id: str) -> Path:
        return DatasetMgr._ingest_dir(ingest_id) / "payload.json"

    @staticmethod
    def _read_ingest_status(ingest_id: str) -> Dict[str, Any]:
        p = DatasetMgr._ingest_status_path(ingest_id)
        if not p.exists():
            return {"status": "unknown", "ingestId": ingest_id}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {"status": "unknown", "ingestId": ingest_id, "error": "status.json parse failed"}

    @staticmethod
    def _write_ingest_status(ingest_id: str, payload: Dict[str, Any]) -> None:
        d = DatasetMgr._ingest_dir(ingest_id)
        d.mkdir(parents=True, exist_ok=True)
        p = DatasetMgr._ingest_status_path(ingest_id)

        current: Dict[str, Any] = {}
        if p.exists():
            try:
                current = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                current = {}

        merged = {**current, **payload, "ingestId": ingest_id, "updated_at": _now_iso()}
        if "created_at" not in merged:
            merged["created_at"] = _now_iso()

        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    @staticmethod
    def _read_ingest_payload(ingest_id: str) -> Dict[str, Any]:
        p = DatasetMgr._ingest_payload_path(ingest_id)
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    @staticmethod
    def create_ingest_job(payload: Dict[str, Any]) -> str:
        """
        routes.py에서 호출:
        - ingestId 생성
        - payload 저장
        - status=QUEUED 기록
        """
        ingest_id = str(uuid.uuid4())
        d = DatasetMgr._ingest_dir(ingest_id)
        d.mkdir(parents=True, exist_ok=True)

        DatasetMgr._ingest_payload_path(ingest_id).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        DatasetMgr._write_ingest_status(ingest_id, {
            "status": "QUEUED",
            "docId": payload.get("docId"),
            "datasetId": payload.get("datasetId"),
            "version": payload.get("version"),
            "replace": bool(payload.get("replace", False)),
        })

        return ingest_id

    # ============================================================
    # (B) ingest 실행
    # ============================================================
    @staticmethod
    def ingest_from_ai(ingest_id: str) -> None:
        """
        비동기 실행 대상(스레드)

        동작:
        1) payload 로드
        2) fileUrl 다운로드
        3) replace=true면 docId 기준 기존 Milvus 삭제 (main.py에서도 하게 만들 수 있음)
        4) sample/main.py 단일파일 모드 실행
        5) (옵션) 콜백 호출
        """
        payload = DatasetMgr._read_ingest_payload(ingest_id)
        if not payload:
            DatasetMgr._write_ingest_status(ingest_id, {"status": "FAILED", "failReason": "payload missing"})
            return

        dataset_id = payload.get("datasetId")  # = domain (예: 사내규정)
        doc_id = payload.get("docId")          # = 문서 식별자(권장)
        version = payload.get("version")
        file_url = payload.get("fileUrl")
        replace = bool(payload.get("replace", False))
        meta = payload.get("meta") or {}

        if not dataset_id or not doc_id or not file_url:
            DatasetMgr._write_ingest_status(ingest_id, {
                "status": "FAILED",
                "failReason": "datasetId/docId/fileUrl required",
            })
            return

        DatasetMgr._write_ingest_status(ingest_id, {"status": "DOWNLOADING"})

        # 다운로드 저장 위치
        dl_dir = DatasetMgr._ingest_dir(ingest_id) / "download"
        dl_dir.mkdir(parents=True, exist_ok=True)

        # file name 추정
        fname = os.path.basename(file_url.split("?")[0]) or f"{doc_id}.bin"
        out_path = dl_dir / fname

        try:
            DatasetMgr._download_file(file_url, out_path)
        except Exception as e:
            DatasetMgr._write_ingest_status(ingest_id, {"status": "FAILED", "failReason": f"download failed: {e}"})
            DatasetMgr._maybe_callback(ingest_id, payload, status="FAILED", fail_reason=str(e), chunks=None)
            return

        DatasetMgr._write_ingest_status(ingest_id, {"status": "PROCESSING", "file": str(out_path)})

        # ✅ replace=true면 “docId 기준 교체”
        #   - main.py에서 처리하도록 인자 넘겨서 main.py가 Milvus delete_file()을 호출하게 만드는 게 정석
        #   - 지금 당장은 main.py가 --doc_id/--replace를 안 받으니까,
        #     (1) main.py를 수정할 거면 아래 args에 추가해주면 됨.
        #     (2) main.py 수정 전이라면, 여기서 Milvus 직접 삭제 로직을 넣어도 됨(추천은 main.py에 넣기).
        #
        # 여기서는 "main.py 수정 가능" 기준으로 args를 준비해둠.
        args = [
            sys.executable,
            str(SAMPLE_MAIN),
            "--input", str(out_path),
            "--domain", str(dataset_id),
        ]

        # main.py가 아래 인자를 받도록 바꾸면 그대로 동작
        if doc_id:
            args += ["--doc_id", str(doc_id)]
        if version is not None:
            args += ["--version", str(version)]
        if replace:
            args += ["--replace", "true"]

        try:
            subprocess.run(args, cwd=str(SAMPLE_MAIN.parent), check=True)
        except Exception as e:
            DatasetMgr._write_ingest_status(ingest_id, {"status": "FAILED", "failReason": str(e)})
            DatasetMgr._maybe_callback(ingest_id, payload, status="FAILED", fail_reason=str(e), chunks=None)
            return

        # 완료
        DatasetMgr._write_ingest_status(ingest_id, {"status": "COMPLETED"})
        DatasetMgr._maybe_callback(ingest_id, payload, status="COMPLETED", fail_reason=None, chunks=None)

    # ============================================================
    # 파일 다운로드
    # ============================================================
    @staticmethod
    def _download_file(url: str, out_path: Path, timeout_sec: int = 60) -> None:
        # 큰 파일도 올 수 있으니 stream
        with requests.get(url, stream=True, timeout=timeout_sec) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

    # ============================================================
    # (옵션) 콜백 호출
    # ============================================================
    @staticmethod
    def _maybe_callback(
        ingest_id: str,
        payload: Dict[str, Any],
        status: str,
        fail_reason: Optional[str],
        chunks: Optional[int],
    ) -> None:
        """
        Spec:
        POST /internal/ai/callbacks/ragflow/ingest
        Headers:
          X-Internal-Token: <RAGFLOW_TO_AI_TOKEN>

        환경변수:
          AI_CALLBACK_URL  : 예) http://ai-service/internal/ai/callbacks/ragflow/ingest
          RAGFLOW_TO_AI_TOKEN : 토큰
        """
        callback_url = os.getenv("AI_CALLBACK_URL")
        token = os.getenv("RAGFLOW_TO_AI_TOKEN")
        if not callback_url or not token:
            return  # 콜백 비활성

        body = {
            "ingestId": ingest_id,
            "docId": payload.get("docId"),
            "version": payload.get("version"),
            "status": status,
            "processedAt": _now_iso(),
            "failReason": fail_reason,
            "meta": payload.get("meta") or {},
            "stats": {"chunks": chunks} if chunks is not None else {"chunks": None},
        }

        try:
            requests.post(
                callback_url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "X-Internal-Token": token,
                },
                timeout=10,
            ).raise_for_status()
        except Exception:
            # 콜백 실패는 ingest 자체 실패로 치면 안 되므로 조용히 상태에만 남김
            DatasetMgr._write_ingest_status(ingest_id, {"callback_error": "callback failed"})

    # ============================================================
    # (B) ingest 작업 삭제 (정리)
    # ============================================================
    @staticmethod
    def delete_ingest_job(ingest_id: str) -> Dict[str, Any]:
        d = DatasetMgr._ingest_dir(ingest_id)
        if not d.exists():
            return {"success": False, "message": "ingest job not found"}

        for p in sorted(d.rglob("*"), reverse=True):
            try:
                if p.is_file():
                    p.unlink()
                else:
                    p.rmdir()
            except Exception:
                pass
        try:
            d.rmdir()
        except Exception:
            pass

        return {"success": True, "message": f"ingest job deleted: {ingest_id}"}
