# workers/ingest-worker/app.py
import os
import json
import shlex
import sys
import uuid
import asyncio
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-worker")

app = FastAPI()

# ----------------------------
# Config
# ----------------------------
# AI -> RAGFlow(ingest 요청) 토큰
EXPECTED = os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""

# RAGFlow -> AI(콜백) 토큰
CALLBACK_TOKEN = os.getenv("RAGFLOW_TO_AI_TOKEN") or ""

# main.py 경로
MAIN_PATH = os.getenv("INGEST_MAIN_PATH", "/workspace/sample/main.py")

# 콜백 URL (없으면 기본값)
DEFAULT_CALLBACK_URL = "http://192.168.0.112:8765/v1/internal_ragflow/internal/ai/callbacks/ragflow/ingest"
CALLBACK_URL = (os.getenv("AI_CALLBACK_URL") or DEFAULT_CALLBACK_URL).strip()

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


def _guess_ext_from_url(file_url: str) -> str:
    try:
        path = urlparse(file_url).path
        ext = Path(path).suffix.lower()
        if ext and len(ext) <= 10:
            return ext
    except Exception:
        pass
    return ""


async def _download_to_tmp(file_url: str) -> Path:
    """
    fileUrl(S3 presigned 등)을 다운로드해서 TMP_DIR 아래에 저장.
    """
    ext = _guess_ext_from_url(file_url) or ".bin"
    out_path = TMP_DIR / f"{uuid.uuid4().hex}{ext}"

    async with httpx.AsyncClient(timeout=DL_TIMEOUT, limits=DL_LIMITS, follow_redirects=True) as client:
        async with client.stream("GET", file_url) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                async for chunk in resp.aiter_bytes():
                    if chunk:
                        f.write(chunk)

    return out_path


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
        logger.warning("[CALLBACK] RAGFLOW_TO_AI_TOKEN empty -> skip")
        return

    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Internal-Token": CALLBACK_TOKEN,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(CALLBACK_TIMEOUT_SEC)) as client:
            resp = await client.post(CALLBACK_URL, headers=headers, json=payload)
        logger.info("[CALLBACK] -> %s status=%s http=%s", CALLBACK_URL, payload.get("status"), resp.status_code)
    except Exception as e:
        logger.exception("[CALLBACK] failed: %r", e)


# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/ingest")
async def ingest(req: IngestReq, x_internal_token: str = Header(default="")):
    # 0) 토큰 검증
    if not EXPECTED:
        raise HTTPException(status_code=500, detail="AI_TO_RAGFLOW_TOKEN/INTERNAL_TOKEN not configured")
    if x_internal_token != EXPECTED:
        raise HTTPException(status_code=401, detail="Unauthorized")

    main_path = Path(MAIN_PATH)
    if not main_path.exists():
        raise HTTPException(status_code=500, detail=f"main.py not found at {MAIN_PATH}")

    # 1) ingestId 결정 (AI가 주면 그대로)
    ingest_id = (req.ingestId or "").strip() or str(uuid.uuid4())

    # meta에 ingestId 강제 주입(추적 통일)
    meta = dict(req.meta or {})
    meta["ingestId"] = ingest_id

    # 2) 다운로드
    try:
        local_path = await _download_to_tmp(req.fileUrl)
    except httpx.HTTPError as e:
        # 다운로드 실패도 콜백 FAILED
        callback_payload = {
            "ingestId": ingest_id,
            "docId": req.docId,
            "version": req.version,
            "status": "FAILED",
            "processedAt": _utc_now_iso(),
            "failReason": f"Download failed: {repr(e)}",
            "meta": meta,
            "stats": {"chunks": 0},
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
            "stats": {"chunks": 0},
        }
        await _send_callback(callback_payload)
        raise HTTPException(status_code=500, detail=f"Download error: {repr(e)}")

    # 3) main.py 인자 매핑
    replace_flag = "true" if req.replace else "false"
    meta_json = json.dumps(meta, ensure_ascii=False)

    cmd = [
        sys.executable,
        str(main_path),
        "--input",
        str(local_path),
        "--domain",
        str(req.datasetId),
        "--doc_id",
        str(req.docId),
        "--replace",
        replace_flag,
        "--ingest-id",
        ingest_id,
        "--meta-json",
        meta_json,
    ]
    if req.version is not None:
        cmd += ["--version", str(req.version)]

    # env도 남김(디버깅/하위호환)
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
        ingest_id,
        req.datasetId,
        req.docId,
        req.version,
        req.replace,
        local_path,
        local_path.stat().st_size if local_path.exists() else -1,
    )
    logger.info("[WORKER] run: %s", " ".join(shlex.quote(x) for x in cmd))

    # 4) 실행
    p = await _run_subprocess(cmd, extra_env=extra_env)

    # 5) 임시파일 정리
    try:
        if local_path.exists():
            local_path.unlink()
    except Exception:
        logger.warning("[WORKER] failed to remove tmp file: %s", local_path)

    stdout = p.stdout or ""
    stderr = p.stderr or ""
    stats_obj = _parse_ingest_stats(stdout)
    chunks = int(stats_obj.get("chunks", 0) or 0)

    # 6) 콜백 payload 구성 + 전송
    if p.returncode == 0:
        callback_payload = {
            "ingestId": ingest_id,
            "docId": req.docId,
            "version": req.version,
            "status": "COMPLETED",
            "processedAt": _utc_now_iso(),
            "failReason": None,
            "meta": meta,
            "stats": {"chunks": chunks},
        }
    else:
        callback_payload = {
            "ingestId": ingest_id,
            "docId": req.docId,
            "version": req.version,
            "status": "FAILED",
            "processedAt": _utc_now_iso(),
            "failReason": (stderr or "main.py failed")[-2000:],
            "meta": meta,
            "stats": {"chunks": chunks},
        }

    await _send_callback(callback_payload)

    # 7) worker 응답 (너가 쓰던 형태 유지)
    if p.returncode != 0:
        return {
            "received": True,
            "ingestId": ingest_id,
            "status": "FAILED",
            "ok": False,
            "rc": p.returncode,
            "stderr": (stderr or "")[-4000:],
            "stdout": (stdout or "")[-4000:],
        }

    return {
        "received": True,
        "ingestId": ingest_id,
        "status": "SUCCEEDED",
        "ok": True,
        "rc": 0,
        "stdout": (stdout or "")[-4000:],
    }
