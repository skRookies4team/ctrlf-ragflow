# /ragflow/api/apps/internal_ragflow_app.py
import os
import json
import logging
import unicodedata
from typing import Any, Dict

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
        # 아래 raw 파싱으로 폴백
        pass

    # 2) 폴백: raw bytes 직접 파싱
    raw = await request.get_data()  # bytes
    if not raw:
        return {}

    # 디코딩 후보(utf-8 우선)
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            s = raw.decode(enc)
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed
            return {"_json": parsed}
        except Exception:
            continue

    # 마지막: 완전 실패면 bytes 정보를 남김
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
        # 구조형이면 문자열 강제 X (상위에서 체크)
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
    - replace/version/meta 같은 옵션 키도 정규화해서 worker로 넘긴다.
    """
    dataset_id = _pick(body, "datasetId", ["dataset_id", "dataset", "datasetName", "dataset_name"])
    doc_id = _pick(body, "docId", ["doc_id", "doc-id", "documentId", "document_id", "doc"])
    file_url = _pick(body, "fileUrl", ["file_url", "file", "url", "fileURL"])

    out: Dict[str, Any] = dict(body)  # 원래 payload 유지(필요한 확장키 유지)

    out["datasetId"] = _coerce_str(dataset_id).strip()
    out["docId"] = _coerce_str(doc_id).strip()
    out["fileUrl"] = _coerce_str(file_url).strip()

    # 옵션들 정규화 (있으면 canonical 형태로 유지)
    if "replace" in out or "isReplace" in out or "replace_doc" in out:
        out["replace"] = _bool(_pick(out, "replace", ["isReplace", "replace_doc"]), default=False)

    if "version" in out or "ver" in out or "docVersion" in out:
        out["version"] = _int(_pick(out, "version", ["ver", "docVersion"]), default=None)

    # meta는 dict 보장
    if "meta" in out and out["meta"] is not None and not isinstance(out["meta"], dict):
        # meta가 문자열로 들어오면 JSON 파싱을 시도
        try:
            if isinstance(out["meta"], str):
                out["meta"] = json.loads(out["meta"])
            else:
                out["meta"] = {"_meta_raw": str(out["meta"])}
        except Exception:
            out["meta"] = {"_meta_raw": str(out["meta"])}

    return out


# ----------------------------
# route
# ----------------------------
@manager.route("/internal/ragflow/ingest", methods=["POST"])
async def internal_ragflow_ingest():
    """
    RAGFlow 내부 엔드포인트:
    - 토큰 검증
    - JSON 바디 검증
    - ingest-worker(/ingest)로 전달 (UTF-8로 강제)

    기대 바디(최소):
      {
        "datasetId": "사내규정",
        "docId": "TEST-001",
        "fileUrl": "https://....pdf",
        "replace": true,          # optional
        "version": 3,             # optional
        "meta": {...}             # optional
      }
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
        # ✅ 추적/관측용: upstream 식별 헤더(있어도 무해)
        "X-From": "ragflow-api",
    }

    timeout = httpx.Timeout(_timeout(), connect=10.0)

    # ✅ 핵심: 워커로 넘길 JSON을 "ensure_ascii=False"로 UTF-8 bytes로 강제
    payload_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")

    logger.warning(
        "[INTERNAL_INGEST] -> worker=%s datasetId=%r docId=%r replace=%r version=%r keys=%s",
        worker_endpoint,
        dataset_name,
        doc_id,
        body.get("replace"),
        body.get("version"),
        list(body.keys()),
    )

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(worker_endpoint, headers=headers, content=payload_bytes)

        payload = _safe_json(resp)

        # ✅ 표준화: worker가 raw를 주더라도 code/message 형태로 감싸기
        if isinstance(payload, dict) and "code" not in payload:
            payload = {"code": 100 if resp.status_code < 400 else resp.status_code, "data": payload, "message": "ok" if resp.status_code < 400 else "error"}

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
