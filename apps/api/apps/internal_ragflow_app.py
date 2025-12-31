import os, tempfile
from typing import Any, Dict, List, Optional, Union
import httpx
from quart import request
import logging

logger = logging.getLogger("internal_ragflow")

def _expected_token() -> str:
    return os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""

def _ragflow_api_key() -> str:
    return os.getenv("RAGFLOW_API_KEY") or ""

def _rf_headers() -> Dict[str, str]:
    return {"Authorization": f"Bearer {_ragflow_api_key()}"}

def _rf_base() -> str:
    return os.getenv("RAGFLOW_BASE_URL", "http://ragflow:9380").rstrip("/")

def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}

def _extract_datasets(payload: Any) -> List[dict]:
    """
    RAGFlow 응답이 환경마다 달라서 두 케이스 모두 대응:
    1) {"code":0,"data":{"datasets":[...]}}
    2) {"code":0,"data":[...]}   <- 너희 현재 이 케이스
    """
    if not isinstance(payload, dict):
        return []

    data = payload.get("data")
    if isinstance(data, dict):
        ds = data.get("datasets", [])
        return ds if isinstance(ds, list) else []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []

async def _rf_get_dataset_id_by_name(client: httpx.AsyncClient, name: str) -> Optional[str]:
    url = f"{_rf_base()}/api/v1/datasets?page=1&page_size=200"
    r = await client.get(url, headers=_rf_headers())
    r.raise_for_status()
    payload = _safe_json(r)

    logger.error(f"[RF:list_datasets] GET {url} -> {r.status_code} body={r.text}")

    if isinstance(payload, dict) and payload.get("code") not in (0, "0", None):
        return None

    datasets = _extract_datasets(payload)
    for ds in datasets:
        if ds.get("name") == name:
            return ds.get("id")
    return None

async def _rf_create_dataset(client: httpx.AsyncClient, name: str) -> str:
    url = f"{_rf_base()}/api/v1/datasets"
    r = await client.post(
        url,
        headers={**_rf_headers(), "Content-Type": "application/json"},
        json={"name": name},
    )
    r.raise_for_status()
    payload = _safe_json(r)

    logger.error(f"[RF:create_dataset] POST {url} -> {r.status_code} body={r.text}")

    if not isinstance(payload, dict) or payload.get("code") != 0:
        raise RuntimeError(f"create dataset failed: {payload}")
    return payload["data"]["id"]

@manager.route("/internal/ragflow/ingest", methods=["POST"])
async def internal_ragflow_ingest():
    # 1) 내부 인증 (그대로)
    got = request.headers.get("X-Internal-Token", "")
    if got != _expected_token():
        return {"code": 401, "data": False, "message": "Unauthorized internal request"}, 401

    body = await request.get_json(force=True, silent=True) or {}
    dataset_name = body.get("datasetId")
    doc_id = body.get("docId")
    file_url = body.get("fileUrl")
    replace = bool(body.get("replace", False))
    meta = body.get("meta") or {}

    if not dataset_name or not doc_id or not file_url:
        return {"code": 400, "data": False, "message": "datasetId, docId, fileUrl required"}, 400

    # 2) main.py 실행 (여기서 청킹/임베딩/Milvus 적재까지 끝내게 만들기)
    #    main.py가 URL을 직접 받든, 다운로드를 내부에서 하든 너희 구현에 맞춰 args 구성
    cmd = [
        "python",
        "/ragflow/sample/main.py",  # <- 너희 main.py 경로로 바꿔
        "--dataset", dataset_name,
        "--doc-id", doc_id,
        "--file-url", file_url,
        "--replace", "1" if replace else "0",
        "--meta", json.dumps(meta, ensure_ascii=False),
    ]

    # 로그에 커맨드/결과 남기기 (네가 원한 "URL 찍는 로그" 포함)
    print("[INTERNAL_INGEST] run main.py:", " ".join(shlex.quote(x) for x in cmd), flush=True)

    p = subprocess.run(cmd, capture_output=True, text=True)
    print("[INTERNAL_INGEST] main.py rc=", p.returncode, flush=True)
    if p.stdout:
        print("[INTERNAL_INGEST][STDOUT]\n", p.stdout, flush=True)
    if p.stderr:
        print("[INTERNAL_INGEST][STDERR]\n", p.stderr, flush=True)

    if p.returncode != 0:
        return {"code": 500, "data": {"rc": p.returncode}, "message": "main.py failed"}, 500

    return {"code": 0, "data": True, "message": "OK"}, 200
