# workers/ingest-worker/app.py
import os
import json
import shlex
import sys
import uuid
import asyncio
import logging
import subprocess
from datetime import datetime, timezone, timedelta

from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from contextvars import ContextVar

# ---- request-scoped context ----
_ctx_ingest_id: ContextVar[str] = ContextVar("ingest_id", default="-")
_ctx_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")
_ctx_doc_id: ContextVar[str] = ContextVar("doc_id", default="-")

class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.ingestId = _ctx_ingest_id.get()
        record.traceId = _ctx_trace_id.get()
        record.docId = _ctx_doc_id.get()
        return True

def _kst_now_iso() -> str:
    kst = timezone(timedelta(hours=9))
    return datetime.now(kst).isoformat(timespec="seconds")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)sZ %(levelname)s %(name)s "
           "[ingestId=%(ingestId)s traceId=%(traceId)s docId=%(docId)s] "
           "%(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

_ctx_filter = RequestContextFilter()

root = logging.getLogger()
root.addFilter(_ctx_filter)

logger = logging.getLogger("ingest-worker")
logger.addFilter(_ctx_filter)

logging.getLogger("httpx").addFilter(_ctx_filter)
logging.getLogger("httpcore").addFilter(_ctx_filter)


app = FastAPI()

# ----------------------------
# Config
# ----------------------------
# AI -> RAGFlow(ingest 요청) 토큰
EXPECTED = os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""

# RAGFlow -> AI(콜백) 토큰
CALLBACK_TOKEN = os.getenv("AI_CALLBACK_TOKEN") or ""

# main.py 경로
MAIN_PATH = os.getenv("INGEST_MAIN_PATH", "/workspace/sample/main.py")

# 콜백 URL (없으면 기본값)
base = (os.getenv("AI_CALLBACK_URL") or "").rstrip("/")
path = (os.getenv("AI_CALLBACK_PATH") or "").strip()

CALLBACK_URL = f"{base}{path}" if base and path else ""

# TMP_DIR
TMP_DIR = Path(os.getenv("INGEST_TMP_DIR", "/tmp/ingest"))
TMP_DIR.mkdir(parents=True, exist_ok=True)

# 다운로드 timeout / 콜백 timeout
DL_TIMEOUT = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=15.0)
DL_LIMITS = httpx.Limits(max_keepalive_connections=10, max_connections=20)
CALLBACK_TIMEOUT_SEC = float(os.getenv("AI_CALLBACK_TIMEOUT", "10"))


# ----------------------------
# Models
# ----------------------------
class IngestReq(BaseModel):
    datasetId: str
    docId: str
    fileUrl: str
    replace: bool = False
    version: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    # ✅ 선택: 상위(AI)가 ingestId를 주면 그대로 사용하고, 없으면 worker가 생성
    ingestId: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _safe_filename(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").strip()

def _guess_ext_from_url(file_url: str) -> str:
    try:
        path = urlparse(file_url).path
        ext = Path(path).suffix.lower()
        if ext and len(ext) <= 10:
            return ext
    except Exception:
        pass
    return ""


async def _download_to_tmp(file_url: str, doc_id: str) -> Path:
    """
    fileUrl(S3 presigned 등)을 다운로드해서 TMP_DIR 아래에 저장.
    ✅ 저장 파일명은 docId(원본 파일명) 그대로 유지 (SSOT/분기 안정)
    """
    fname = _safe_filename(doc_id)

    # docId에 확장자가 아예 없을 때만 URL 확장자를 보강
    if not Path(fname).suffix:
        ext = _guess_ext_from_url(file_url) or ".bin"
        fname = fname + ext

    out_path = TMP_DIR / fname

    async with httpx.AsyncClient(timeout=DL_TIMEOUT, limits=DL_LIMITS, follow_redirects=True) as client:
        async with client.stream("GET", file_url) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        f.write(chunk)

    return out_path

# --- Config 추가 ---
MAIN_TIMEOUT_SEC = int(os.getenv("INGEST_MAIN_TIMEOUT", "1800"))  # 30min

async def _run_subprocess(cmd: list[str], extra_env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    """
    FastAPI async handler를 막지 않게 thread에서 subprocess 실행.
    """
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    def _run():
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=MAIN_TIMEOUT_SEC,
        )

    return await asyncio.to_thread(_run)


def _parse_ingest_stats(stdout: str) -> Dict[str, Any]:
    """
    main.py가 stdout에 남긴 라인 중
    'INGEST_STATS_JSON={...}' 형태를 찾아 JSON 파싱
    """
    if not stdout:
        return {}
    marker = "INGEST_STATS_JSON="
    for line in stdout.splitlines():
        if line.startswith(marker):
            raw = line[len(marker):].strip()
            try:
                obj = json.loads(raw)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {}
    return {}


async def _send_callback(payload: Dict[str, Any]) -> None:
    """
    RAGFlow -> AI 콜백 전송.
    실패해도 ingest 프로세스 자체는 계속 진행(로그만 남김).
    """
    if not CALLBACK_URL:
        logger.warning("[CALLBACK] CALLBACK_URL empty -> skip")
        return
    if not CALLBACK_TOKEN:
        logger.warning("[CALLBACK] AI_CALLBACK_TOKEN empty -> skip (payload keys=%s)", list(payload.keys()))
        return

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Internal-Token": CALLBACK_TOKEN,
    }

    for attempt in range(1, 4):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(CALLBACK_TIMEOUT_SEC)) as client:
                resp = await client.post(CALLBACK_URL, headers=headers, json=payload)
            logger.info("[CALLBACK] -> %s status=%s http=%s", CALLBACK_URL, payload.get("status"), resp.status_code)
            if resp.status_code < 400:
                return
            logger.warning("[CALLBACK] non-2xx attempt=%s http=%s body=%s", attempt, resp.status_code, resp.text[:500])
        except Exception as e:
            logger.warning("[CALLBACK] attempt=%s failed: %r", attempt, e)
        await asyncio.sleep(0.5 * attempt)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(req: IngestReq, x_internal_token: str = Header(default="")):
    if not EXPECTED:
        raise HTTPException(status_code=500, detail="AI_TO_RAGFLOW_TOKEN/INTERNAL_TOKEN not configured")
    if x_internal_token != EXPECTED:
        raise HTTPException(status_code=401, detail="Unauthorized")

    main_path = Path(MAIN_PATH)
    if not main_path.exists():
        raise HTTPException(status_code=500, detail=f"main.py not found at {MAIN_PATH}")

    ingest_id = (req.ingestId or "").strip() or str(uuid.uuid4())

    meta = dict(req.meta or {})
    meta["ingestId"] = ingest_id

    # ✅ 여기부터: 요청 컨텍스트 세팅 (요청 단위 분리 핵심)
    trace_id = str(meta.get("traceId") or meta.get("trace_id") or "-")
    trace_id_for_log = trace_id if trace_id is not None else "-"

    _ctx_ingest_id.set(ingest_id)
    _ctx_trace_id.set(trace_id)
    _ctx_doc_id.set(req.docId or "-")

    # ✅ 요청 시작 로그(시간 포함)
    logger.info(
        "[INGEST_START] utc=%s kst=%s datasetId=%s version=%s replace=%s fileUrl=%s",
        _utc_now_iso(), _kst_now_iso(), req.datasetId, req.version, req.replace, req.fileUrl
    )

    local_path: Optional[Path] = None
    try:
        # 1) domain 체크 (여기서도 콜백 보장)
        domain = (req.datasetId or "").strip()
        if not domain:
            callback_payload = {
                "ingestId": ingest_id,
                "docId": req.docId,
                "version": req.version,
                "status": "FAILED",
                "processedAt": _utc_now_iso(),
                "failReason": "datasetId is required",
                "meta": meta,
                "stats": None,
            }
            await _send_callback(callback_payload)
            raise HTTPException(status_code=400, detail="datasetId is required")

        # 2) 다운로드 (실패해도 콜백 보장)
        try:
            logger.info("[DOWNLOAD_START] url=%s", req.fileUrl)
            local_path = await _download_to_tmp(req.fileUrl, req.docId)
            logger.info("[DOWNLOAD_OK] path=%s size=%s", local_path, local_path.stat().st_size)

        except httpx.HTTPError as e:
            callback_payload = {
                "ingestId": ingest_id,
                "docId": req.docId,
                "version": req.version,
                "status": "FAILED",
                "processedAt": _utc_now_iso(),
                "failReason": f"Download failed: {repr(e)}",
                "meta": meta,
                "stats": None,
            }
            await _send_callback(callback_payload)
            raise HTTPException(status_code=400, detail=f"Download failed: {repr(e)}")
        except Exception as e:
            callback_payload = {
                "ingestId": ingest_id,
                "docId": req.docId,
                "version": req.version,
                "status": "FAILED",
                "processedAt": _utc_now_iso(),
                "failReason": f"Download error: {repr(e)}",
                "meta": meta,
                "stats": None,
            }
            await _send_callback(callback_payload)
            raise HTTPException(status_code=500, detail=f"Download error: {repr(e)}")

        # 3) main.py 인자
        replace_flag = "true" if req.replace else "false"
        meta_json = json.dumps(meta, ensure_ascii=False)

        cmd = [
            sys.executable,
            str(main_path),
            "--input", str(local_path),
            "--domain", domain,
            "--doc_id", str(req.docId),   # ✅ 추가 (원본 파일명 docId SSOT)
            "--replace", replace_flag,
            "--ingest-id", ingest_id,
            "--meta-json", meta_json,
        ]
        if req.version is not None:
            cmd += ["--version", str(req.version)]

        extra_env = {
            "INGEST_META_JSON": meta_json,
            "INGEST_DATASET_ID": str(req.datasetId),
            "INGEST_DOC_ID": str(req.docId),
            "INGEST_ID": ingest_id,
            "INGEST_VERSION": "" if req.version is None else str(req.version),
            "INGEST_REPLACE": replace_flag,
        }

        logger.info(
            "[WORKER] ingestId=%s datasetId=%s docId=%s version=%s replace=%s downloaded=%s size=%s",
            ingest_id, req.datasetId, req.docId, req.version, req.replace,
            local_path, local_path.stat().st_size if local_path.exists() else -1
        )
        logger.info("[WORKER] run: %s", " ".join(shlex.quote(x) for x in cmd))

        # 4) 실행
        p = await _run_subprocess(cmd, extra_env=extra_env)

        stdout = p.stdout or ""
        stderr = p.stderr or ""

        logger.info("[WORKER] main.py returncode=%s", p.returncode)
        logger.info("[WORKER] main.py stdout(last 2000):\n%s", stdout[-2000:])
        logger.info("[WORKER] main.py stderr(last 2000):\n%s", stderr[-2000:])

        stats_obj = _parse_ingest_stats(stdout)
        meta_from_main = stats_obj.get("meta") or {}
        if not isinstance(meta_from_main, dict):
            meta_from_main = {"_main_meta_raw": meta_from_main}

        # chunks
        stats = stats_obj.get("stats") or {}
        if isinstance(stats, dict):
            chunks = stats.get("chunks", None)
        else:
            chunks = None

        if chunks is None:
            # 구 스키마 호환: {"chunks":123}
            chunks = stats_obj.get("chunks", 0)

        chunks = int(chunks or 0)

        
        # returncode 기준 최종 status
        status = "COMPLETED" if p.returncode == 0 else "FAILED"

        # failReason는 main.py가 준 error가 있으면 그걸 우선
        fail_reason = None
        if status != "COMPLETED":
            fail_reason = (
                meta_from_main.get("error")
                or (stderr or "main.py failed")[-2000:]
            )

        
        callback_payload = {
            "ingestId": ingest_id,
            "docId": req.docId,
            "version": req.version,
            "status": status,
            "processedAt": _utc_now_iso(),
            "failReason": None if status == "COMPLETED" else fail_reason,
            "meta": {**meta, **meta_from_main},   # ✅ merge 핵심
            "stats": {"chunks": chunks} if status == "COMPLETED" else None,
        }
        await _send_callback(callback_payload)

        if p.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail={
                    "received": True,
                    "ingestId": ingest_id,
                    "status": "FAILED",
                    "ok": False,
                    "rc": p.returncode,
                    "stderr": (stderr or "")[-4000:],
                    "stdout": (stdout or "")[-4000:],
                },
            )

        return {
            "received": True,
            "ingestId": ingest_id,
            "status": "COMPLETED",
            "ok": True,
            "rc": 0,
            "stdout": (stdout or "")[-4000:],
        }

    except subprocess.TimeoutExpired as e:
        callback_payload = {
            "ingestId": ingest_id,
            "docId": req.docId,
            "version": req.version,
            "status": "FAILED",
            "processedAt": _utc_now_iso(),
            "failReason": f"main.py timeout after {MAIN_TIMEOUT_SEC}s",
            "meta": {**meta, "timeout": True},
            "stats": None,
        }
        await _send_callback(callback_payload)
        raise HTTPException(status_code=504, detail="main.py timeout")

    finally:
        logger.info("[INGEST_END] utc=%s kst=%s", _utc_now_iso(), _kst_now_iso())

        # ✅ 무조건 tmp 삭제
        try:
            if local_path and local_path.exists():
                local_path.unlink()
        except Exception as e:
            logger.warning("[WORKER] failed to remove tmp file: %s err=%r", local_path, e)

