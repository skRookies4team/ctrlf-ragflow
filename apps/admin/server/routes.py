# ============================================================
# apps/api-server/admin/server/routes.py
# ============================================================
"""
Admin API routes

✅ 포함:
- 기존 /api/v1/admin/* (login/logout/users/services/version)
- 신규 Internal API: /internal/ragflow/ingest
  - AI → RAGFlow ingest 요청 수신
  - 202 Accepted 반환
  - 실제 처리는 services.py(DatasetMgr.ingest_from_ai)에서 비동기 실행
"""

import json
import os
import secrets
import threading
from typing import Any, Dict, Optional

from .auth import check_admin_auth, login_admin, login_verify
from flask import Blueprint, request
from flask_login import current_user, login_required, logout_user
from .responses import error_response, success_response
# ✅ DatasetMgr 위치: 네 프로젝트 구조 유지
# - 너는 "from services.dataset_service import DatasetMgr"를 쓰고 있었음
# - 만약 dataset_service.py가 아니라 services.py 안에 DatasetMgr가 있다면 import만 바꾸면 됨
# 기존 서비스들
from .services import DatasetMgr, ServiceMgr, UserMgr, UserServiceMgr

from libs.common.versions import get_ragflow_version

admin_bp = Blueprint("admin", __name__, url_prefix="/api/v1/admin")

# internal blueprint (url_prefix 없음: 완전 별도 경로)
internal_bp = Blueprint("internal", __name__, url_prefix="")

# ============================================================
# Helpers
# ============================================================

def _get_internal_token_header() -> str:
    """
    AI -> RAGFlow ingest 요청에 포함되는 토큰 헤더
    Spec: X-Internal-Token
    """
    return request.headers.get("X-Internal-Token", "")

def _require_internal_token() -> Optional[Any]:
    """
    내부 토큰 검증
    - 실패 시 error_response 반환(Flask response)
    - 성공 시 None
    """
    expected = os.getenv("AI_TO_RAGFLOW_TOKEN") or os.getenv("INTERNAL_TOKEN") or ""
    got = _get_internal_token_header()

    if not expected:
        # 운영에서는 반드시 설정해야 하지만, 개발 편의를 위해 명확히 에러로 터뜨림
        return error_response("Server internal token not configured (AI_TO_RAGFLOW_TOKEN)", 500)

    if not got or got != expected:
        return error_response("Unauthorized internal request", 401)

    return None

# ============================================================
# Auth
# ============================================================

@admin_bp.route("/login", methods=["POST"])
def login():
    if not request.json:
        return error_response("Authorize admin failed.", 400)

    try:
        return login_admin(
            request.json.get("email", ""),
            request.json.get("password", ""),
        )
    except Exception as e:
        return error_response(str(e), 500)


@admin_bp.route("/logout", methods=["GET"])
@login_required
def logout():
    try:
        current_user.access_token = f"INVALID_{secrets.token_hex(16)}"
        current_user.save()
        logout_user()
        return success_response(True)
    except Exception as e:
        return error_response(str(e), 500)


@admin_bp.route("/auth", methods=["GET"])
@login_verify
def auth_admin():
    return success_response(None, "Admin is authorized", 0)


# ============================================================
# User
# ============================================================

@admin_bp.route("/users", methods=["GET"])
@login_required
@check_admin_auth
def list_users():
    return success_response(UserMgr.get_all_users())


@admin_bp.route("/users", methods=["POST"])
@login_required
@check_admin_auth
def create_user():
    data = request.get_json()
    if not data:
        return error_response("Invalid body", 400)

    res = UserMgr.create_user(
        data.get("username", ""),
        data.get("password", ""),
        data.get("role", "user"),
    )
    if isinstance(res, dict) and "user_info" in res:
        res["user_info"].pop("password", None)
        return success_response(res["user_info"])
    return success_response(res)


@admin_bp.route("/users/<username>/datasets", methods=["GET"])
@login_required
@check_admin_auth
def get_user_datasets(username):
    return success_response(UserServiceMgr.get_user_datasets(username))


# ============================================================
# Dataset (기존 업로드/프로세스/status)
# ============================================================

@admin_bp.route("/datasets", methods=["POST"])
@login_required
@check_admin_auth
def upload_dataset():
    if "file" not in request.files:
        return error_response("file is required", 400)

    file = request.files["file"]
    dataset_name = request.form.get("dataset_name", "default")

    dataset_id = DatasetMgr.save_dataset(file, dataset_name)
    return success_response({"dataset_id": dataset_id}, "dataset uploaded")


@admin_bp.route("/datasets/<dataset_id>/process", methods=["POST"])
@login_required
@check_admin_auth
def process_dataset(dataset_id):
    threading.Thread(
        target=DatasetMgr.process_dataset,
        args=(dataset_id,),
        daemon=True,
    ).start()

    return success_response({"dataset_id": dataset_id}, "processing started")


@admin_bp.route("/datasets/<dataset_id>/status", methods=["GET"])
@login_required
@check_admin_auth
def dataset_status(dataset_id):
    return success_response(DatasetMgr.get_status(dataset_id))


# ============================================================
# ✅ Internal API: AI → RAGFlow ingest 실행
# POST /internal/ragflow/ingest
# ============================================================

@internal_bp.route("/internal/ragflow/ingest", methods=["POST"])
def internal_ragflow_ingest():
    """
    Spec:
      Headers:
        Content-Type: application/json
        X-Internal-Token: <AI_TO_RAGFLOW_TOKEN>

      Body:
      {
        "datasetId": "사내규정",
        "docId": "POL-EDU-015",
        "version": 3,
        "fileUrl": "https://...pdf",
        "replace": true,
        "meta": {...}
      }

    Response: 202
      { "received": true, "ingestId": "...", "status": "QUEUED" }
    """
    auth_err = _require_internal_token()
    if auth_err is not None:
        return auth_err

    if not request.is_json:
        return error_response("Content-Type must be application/json", 415)

    body: Dict[str, Any] = request.get_json(silent=True) or {}
    dataset_id = body.get("datasetId")
    doc_id = body.get("docId")
    file_url = body.get("fileUrl")

    if not dataset_id or not doc_id or not file_url:
        return error_response("datasetId, docId, fileUrl are required", 400)

    # ingest job 생성 + 상태 기록
    ingest_id = DatasetMgr.create_ingest_job(body)

    # 비동기 처리
    threading.Thread(
        target=DatasetMgr.ingest_from_ai,
        args=(ingest_id,),
        daemon=True,
    ).start()

    # 202 Accepted (비동기 큐잉)
    return (
        json.dumps(
            {"received": True, "ingestId": ingest_id, "status": "QUEUED"},
            ensure_ascii=False,
        ),
        202,
        {"Content-Type": "application/json; charset=utf-8"},
    )


# ============================================================
# Service / Version
# ============================================================

@admin_bp.route("/services", methods=["GET"])
@login_required
@check_admin_auth
def get_services():
    return success_response(ServiceMgr.get_all_services())


@admin_bp.route("/version", methods=["GET"])
@login_required
@check_admin_auth
def show_version():
    return success_response({"version": get_ragflow_version()})
