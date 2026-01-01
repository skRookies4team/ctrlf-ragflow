# /ragflow/api/apps/internal_ragflow_app.py
import os
import json
import logging
from typing import Any, Dict, Optional

import httpx
from quart import request

logger = logging.getLogger("internal_ragflow")


# ----------------------------
# config helpers
# ----------------------------
def _expected_token() -> str:
    return os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""


def _worker_url() -> str:
    # docker-compose에서: INGEST_WORKER_URL=http://ingest-worker:9001
    return os.getenv("INGEST_WORKER_URL", "http://ingest-worker:9001").rstrip("/")


def _timeout() -> float:
    # worker에서 main.py가 오래 돌 수 있으니 충분히 크게
    return float(os.getenv("INGEST_WORKER_TIMEOUT", "1800"))  # 30min default


def _safe_json(resp: httpx.Response) -> Dict[str, Any]:
    try:
        return resp.json()
    except Exception:
        return {"_raw": resp.text}


# ----------------------------
# route
# ----------------------------
@manager.route("/internal/ragflow/ingest", methods=["POST"])
async def internal_ragflow_ingest():
    """
    RAGFlow 내부 엔드포인트:
    - 토큰 검증
    - JSON 바디 검증
    - ingest-worker(/ingest)로 그대로 전달
    """
    # 1) 내부 토큰 검증
    got = request.headers.get("X-Internal-Token", "")
    expected = _expected_token()
    if not expected:
        return {"code": 500, "data": False, "message": "AI_TO_RAGFLOW_TOKEN not configured"}, 500
    if got != expected:
        return {"code": 401, "data": False, "message": "Unauthorized internal request"}, 401

    # 2) JSON 바디 파싱
    if not request.is_json:
        return {"code": 415, "data": False, "message": "Content-Type must be application/json"}, 415

    body: Dict[str, Any] = await request.get_json(force=True, silent=True) or {}

    # 필수값 체크(최소)
    dataset_name = body.get("datasetId")
    doc_id = body.get("docId")
    file_url = body.get("fileUrl")
    if not dataset_name or not doc_id or not file_url:
        return {"code": 400, "data": False, "message": "datasetId, docId, fileUrl required"}, 400

    # 3) worker로 전달
    worker_endpoint = f"{_worker_url()}/ingest"

    # worker에서도 같은 토큰 검증 가능하도록 헤더 전달
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Internal-Token": got,
    }

    timeout = httpx.Timeout(_timeout(), connect=10.0)

    logger.warning("[INTERNAL_INGEST] -> worker=%s body_keys=%s", worker_endpoint, list(body.keys()))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(worker_endpoint, headers=headers, json=body)

        payload = _safe_json(resp)

        # worker가 이미 {code, data, message} 형태로 주면 그대로 프록시
        # (status code도 그대로)
        return payload, resp.status_code

    except httpx.TimeoutException:
        logger.exception("[INTERNAL_INGEST] worker timeout")
        return {"code": 504, "data": False, "message": "ingest-worker timeout"}, 504

    except httpx.RequestError as e:
        logger.exception("[INTERNAL_INGEST] worker request error: %s", e)
        return {"code": 502, "data": False, "message": f"ingest-worker unreachable: {e}"}, 502

    except Exception as e:
        logger.exception("[INTERNAL_INGEST] unexpected error: %s", e)
        return {"code": 500, "data": False, "message": f"internal error: {e}"}, 500
