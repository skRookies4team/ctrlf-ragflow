# worker/app.py
import os
import json
import shlex
import sys
import uuid
import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest-worker")

app = FastAPI()

EXPECTED = os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""
MAIN_PATH = os.getenv("INGEST_MAIN_PATH", "/workspace/sample/main.py")

TMP_DIR = Path(os.getenv("INGEST_TMP_DIR", "/tmp/ingest"))
TMP_DIR.mkdir(parents=True, exist_ok=True)


class IngestReq(BaseModel):
    datasetId: str
    docId: str
    fileUrl: str
    replace: bool = False
    version: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


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
    ext = _guess_ext_from_url(file_url)
    if not ext:
        ext = ".bin"

    out_path = TMP_DIR / f"{uuid.uuid4().hex}{ext}"

    timeout = httpx.Timeout(connect=15.0, read=120.0, write=120.0, pool=15.0)
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
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

    # 1) 다운로드
    try:
        local_path = await _download_to_tmp(req.fileUrl)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {repr(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download error: {repr(e)}")

    # 2) main.py 인자 매핑
    # - datasetId -> --domain
    # - docId -> --doc_id
    # - fileUrl -> 다운로드된 파일 경로 -> --input
    # - replace/version 그대로
    replace_flag = "true" if req.replace else "false"

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
    ]
    if req.version is not None:
        cmd += ["--version", str(req.version)]

    # meta는 main이 직접 안 받으니 env로만 전달(추후 필요하면 main 확장)
    extra_env = {
        "INGEST_META_JSON": json.dumps(req.meta or {}, ensure_ascii=False),
        "INGEST_DATASET_ID": str(req.datasetId),
        "INGEST_DOC_ID": str(req.docId),
    }

    logger.info("[WORKER] downloaded=%s size=%s", local_path, local_path.stat().st_size if local_path.exists() else -1)
    logger.info("[WORKER] run: %s", " ".join(shlex.quote(x) for x in cmd))

    # 3) 실행
    p = await _run_subprocess(cmd, extra_env=extra_env)

    # 4) 임시파일 정리(실패해도 지우는 게 보통 맞음)
    try:
        if local_path.exists():
            local_path.unlink()
    except Exception:
        logger.warning("[WORKER] failed to remove tmp file: %s", local_path)

    if p.returncode != 0:
        return {
            "ok": False,
            "rc": p.returncode,
            "stderr": (p.stderr or "")[-2000:],
            "stdout": (p.stdout or "")[-2000:],
        }

    return {
        "ok": True,
        "rc": 0,
        "stdout": (p.stdout or "")[-2000:],
    }
