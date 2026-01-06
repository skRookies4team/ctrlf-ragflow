# /ragflow/api/apps/internal_ragflow_app.py
import os
import json
import logging
import unicodedata
import asyncio
from uuid import uuid4
from typing import Any, Dict, Optional

import httpx
from quart import request

logger = logging.getLogger("internal_ragflow")

# ----------------------------
# config helpers
# ----------------------------
def _worker_ingest_path() -> str:
    return os.getenv("INGEST_WORKER_PATH", "/ingest")

def _enable_callback_receiver() -> bool:
    return os.getenv("ENABLE_CALLBACK_RECEIVER", "0").strip() in ("1","true","yes","y","on")

def _expected_token() -> str:
    # AI -> RAGFlow 전용 토큰 우선
    return os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""


def _expected_callback_token() -> str:
    # ✅ RAGFlow(ingest-worker) -> AI callback token
    # (테스트/운영 환경에선 "AI 서버"가 이 토큰으로 검증해야 함)
    return os.getenv("AI_CALLBACK_TOKEN") or ""


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


def _normalize_unicode(obj: Any) -> Any:
    """
    들어오는 JSON에서 문자열 값들을 NFC로 정규화(한글 조합형/분해형 섞임 방지).
    """
    if isinstance(obj, str):
        return unicodedata.normalize("NFC", obj)
    if isinstance(obj, list):
        return [_normalize_unicode(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _normalize_unicode(v) for k, v in obj.items()}
    return obj


async def _parse_json_body() -> Dict[str, Any]:
    """
    Quart의 request.get_json이 환경/클라이언트 인코딩 영향으로 깨질 때가 있어서
    raw bytes를 직접 읽고 utf-8 우선으로 안전 파싱한다.
    - 1차: request.get_json()
    - 2차: raw bytes -> utf-8-sig / utf-8 / cp949 순으로 디코드 후 json.loads
    """
    # 1) 정상 경로
    try:
        if request.is_json:
            body = await request.get_json(force=True, silent=True)
            if isinstance(body, dict):
                return body
    except Exception:
        pass

    # 2) 폴백: raw bytes 직접 파싱
    raw = await request.get_data()  # bytes
    if not raw:
        return {}

    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            s = raw.decode(enc)
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            return {"_json": parsed}
        except Exception:
            continue

    return {"_raw_bytes_len": len(raw)}


def _coerce_str(v: Any) -> str:
    """
    bytes/bytearray로 들어온 값은 안전하게 str로 복구.
    그 외에는 str()로 강제 변환하되 None은 "".
    """
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        for enc in ("utf-8", "utf-8-sig", "cp949"):
            try:
                return v.decode(enc)
            except Exception:
                continue
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, (dict, list)):
        return ""
    return str(v)


def _bool(v: Any, default: bool = False) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _int(v: Any, default: int | None = None) -> int | None:
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _pick(body: Dict[str, Any], key: str, aliases: list[str]) -> Any:
    """
    key가 없으면 aliases를 순서대로 찾아서 반환.
    (예: docId / doc_id / doc-id)
    """
    if key in body:
        return body.get(key)
    for a in aliases:
        if a in body:
            return body.get(a)
    return None


def _canonicalize_ingest_body(body: Dict[str, Any]) -> Dict[str, Any]:
    """
    ingest-worker가 기대하는 키로 정규화 + alias 흡수 + 타입 보정.
    - datasetId/docId/fileUrl 3개는 반드시 canonical key로 맞춘다.
    - replace/version/meta/ingestId 같은 옵션 키도 정규화해서 worker로 넘긴다.
    """
    dataset_id = _pick(body, "datasetId", ["dataset_id", "dataset", "datasetName", "dataset_name"])
    doc_id = _pick(body, "docId", ["doc_id", "doc-id", "documentId", "document_id", "doc"])
    file_url = _pick(body, "fileUrl", ["file_url", "file", "url", "fileURL"])

    # ✅ ingestId도 alias 흡수 (AI가 이미 준 경우 유지)
    ingest_id = _pick(body, "ingestId", ["ingest_id", "ingest-id", "requestId", "request_id"])

    out: Dict[str, Any] = dict(body)  # 원래 payload 유지(필요한 확장키 유지)

    out["datasetId"] = _coerce_str(dataset_id).strip()
    out["docId"] = _coerce_str(doc_id).strip()
    out["fileUrl"] = _coerce_str(file_url).strip()

    if ingest_id is not None:
        out["ingestId"] = _coerce_str(ingest_id).strip()

    # 옵션들 정규화
    if "replace" in out or "isReplace" in out or "replace_doc" in out:
        out["replace"] = _bool(_pick(out, "replace", ["isReplace", "replace_doc"]), default=False)
    else:
        # ✅ worker/main.py가 bool로 받기 쉬우라고 기본값을 명시(선택)
        out.setdefault("replace", False)

    if "version" in out or "ver" in out or "docVersion" in out:
        out["version"] = _int(_pick(out, "version", ["ver", "docVersion"]), default=None)

    # meta는 dict 보장
    if "meta" in out and out["meta"] is not None and not isinstance(out["meta"], dict):
        try:
            if isinstance(out["meta"], str):
                out["meta"] = json.loads(out["meta"])
            else:
                out["meta"] = {"_meta_raw": str(out["meta"])}
        except Exception:
            out["meta"] = {"_meta_raw": str(out["meta"])}

    return out


def _make_ingest_id() -> str:
    return str(uuid4())


def _get_sync_mode() -> bool:
    return os.getenv("INGEST_SYNC_MODE", "0").strip() in ("1", "true", "yes", "y", "on")


async def _post_to_worker_background(
    worker_endpoint: str,
    headers: Dict[str, str],
    payload_bytes: bytes,
    *,
    ingest_id: str,
    dataset_id: str,
    doc_id: str,
) -> None:
    """
    202로 즉시 응답한 뒤, 백그라운드에서 worker로 실제 ingest 요청 전달.
    실패해도 API 응답은 이미 나갔으므로 여기서는 로그로만 남긴다.
    """
    timeout = httpx.Timeout(_timeout(), connect=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(worker_endpoint, headers=headers, content=payload_bytes)

        payload = _safe_json(resp)
        ok = resp.status_code < 400

        logger.warning(
            "[INTERNAL_INGEST_BG] ingestId=%s ok=%s status=%s datasetId=%r docId=%r resp=%s",
            ingest_id,
            ok,
            resp.status_code,
            dataset_id,
            doc_id,
            payload if isinstance(payload, dict) else {"_raw": str(payload)},
        )
    except Exception as e:
        logger.exception(
            "[INTERNAL_INGEST_BG] FAILED ingestId=%s datasetId=%r docId=%r err=%s",
            ingest_id,
            dataset_id,
            doc_id,
            e,
        )


# ----------------------------
# route
# ----------------------------
@manager.route("internal/ragflow/ingest", methods=["POST"])
async def internal_ragflow_ingest():
    """
    AI -> RAGFlow Ingest 실행 API

    - 토큰 검증
    - JSON 바디 검증/정규화
    - ingestId 확정(요청에 있으면 재사용, 없으면 생성)
    - 202 Accepted 즉시 응답 (QUEUED)
    - 실제 ingest는 ingest-worker(/ingest)로 비동기 전달
    """
    # 1) 내부 토큰 검증
    got = request.headers.get("X-Internal-Token", "")
    expected = _expected_token()
    if not expected:
        return {"code": 500, "data": False, "message": "AI_TO_RAGFLOW_TOKEN not configured"}, 500
    if got != expected:
        return {"code": 401, "data": False, "message": "Unauthorized internal request"}, 401

    # 2) JSON 바디 파싱 (강건 버전)
    body: Dict[str, Any] = await _parse_json_body()
    body = _normalize_unicode(body)

    if not isinstance(body, dict):
        return {"code": 400, "data": False, "message": "Invalid JSON body"}, 400

    # 2-1) key/alias/타입 정규화
    body = _canonicalize_ingest_body(body)

    # 필수값 체크(최소)
    dataset_id = body.get("datasetId")
    doc_id = body.get("docId")
    file_url = body.get("fileUrl")

    if not dataset_id or not doc_id or not file_url:
        return {"code": 400, "data": False, "message": "datasetId, docId, fileUrl required"}, 400

    # 3) ingestId 확정
    ingest_id = (body.get("ingestId") or "").strip() or _make_ingest_id()
    body["ingestId"] = ingest_id

    # (선택) meta에 ingestId 주입
    meta = body.get("meta")
    if isinstance(meta, dict):
        meta.setdefault("ingestId", ingest_id)
        body["meta"] = meta

    # 4) worker로 전달 준비
    worker_endpoint = f"{_worker_url()}{_worker_ingest_path()}"

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Internal-Token": _expected_token(),   # worker에서도 동일 토큰 검증
        "X-From": "ragflow-api",
        "X-Ingest-Id": ingest_id,
    }

    # ✅ UTF-8 bytes 고정
    payload_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    logger.warning(
        "[INTERNAL_INGEST] QUEUE ingestId=%s worker=%s datasetId=%r docId=%r replace=%r version=%r keys=%s",
        ingest_id,
        worker_endpoint,
        dataset_id,
        doc_id,
        body.get("replace"),
        body.get("version"),
        list(body.keys()),
    )

    # 5) 202 즉시 반환 + (옵션) 동기 디버깅
    if _get_sync_mode():
        timeout = httpx.Timeout(_timeout(), connect=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(worker_endpoint, headers=headers, content=payload_bytes)

            payload = _safe_json(resp)
            if isinstance(payload, dict) and "code" not in payload:
                payload = {
                    "code": 100 if resp.status_code < 400 else resp.status_code,
                    "data": payload,
                    "message": "ok" if resp.status_code < 400 else "error",
                }
            return payload, resp.status_code

        except httpx.TimeoutException:
            logger.exception("[INTERNAL_INGEST] worker timeout (sync)")
            return {"code": 504, "data": False, "message": "ingest-worker timeout"}, 504
        except httpx.RequestError as e:
            logger.exception("[INTERNAL_INGEST] worker request error (sync): %s", e)
            return {"code": 502, "data": False, "message": f"ingest-worker unreachable: {e}"}, 502
        except Exception as e:
            logger.exception("[INTERNAL_INGEST] unexpected error (sync): %s", e)
            return {"code": 500, "data": False, "message": f"internal error: {e}"}, 500

    # 비동기 모드(기본): 백그라운드로 worker 호출 후 즉시 202
    try:
        asyncio.create_task(
            _post_to_worker_background(
                worker_endpoint,
                headers,
                payload_bytes,
                ingest_id=ingest_id,
                dataset_id=dataset_id,
                doc_id=doc_id,
            )
        )
    except Exception as e:
        logger.exception("[INTERNAL_INGEST] failed to schedule background task ingestId=%s err=%s", ingest_id, e)
        return {"code": 500, "data": False, "message": "failed to queue ingest task"}, 500

    return {
        "received": True,
        "ingestId": ingest_id,
        "status": "QUEUED",
    }, 202


# ============================================================
# ✅ (추가) ingest-worker -> AI callback receiver (테스트용)
# ============================================================
if _enable_callback_receiver():
    @manager.route("/ai/callbacks/ragflow/ingest", methods=["POST"])
    async def internal_ai_callback_ragflow_ingest():
        """
        ingest-worker가 결과를 콜백하는 엔드포인트 (테스트/운영 시 AI 서버에서 구현해야 함)

        URL:
        /v1/internal_ragflow/internal/ai/callbacks/ragflow/ingest

        Headers:
        X-Internal-Token: {AI_CALLBACK_TOKEN}

        Body:
        {
            ingestId, docId, version, status, processedAt, failReason, meta, stats
        }
        """
        got = request.headers.get("X-Internal-Token", "")
        expected = _expected_callback_token()
        if not expected:
            return {"code": 500, "data": False, "message": "AI_CALLBACK_TOKEN not configured"}, 500
        if got != expected:
            return {"code": 401, "data": False, "message": "Unauthorized callback"}, 401

        body: Dict[str, Any] = await _parse_json_body()
        body = _normalize_unicode(body)

        ingest_id = (body.get("ingestId") or "").strip()
        doc_id = (body.get("docId") or "").strip()
        status = (body.get("status") or "").strip()

        logger.warning("[AI_CALLBACK_RX] ingestId=%s docId=%s status=%s body=%s", ingest_id, doc_id, status, body)

        # 여기서 DB 업데이트 / 상태 저장 / 이벤트 발행 등 처리하면 됨
        return {"code": 100, "data": {"received": True}, "message": "ok"}, 200
