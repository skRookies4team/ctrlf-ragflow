import secrets
import threading

from flask import Blueprint, request
from flask_login import current_user, login_required, logout_user

from auth import login_verify, login_admin, check_admin_auth
from responses import success_response, error_response
from services import UserMgr, ServiceMgr, UserServiceMgr
from services.dataset_service import DatasetMgr
from roles import RoleMgr
from api.common.exceptions import AdminException
from common.versions import get_ragflow_version

admin_bp = Blueprint('admin', __name__, url_prefix='/api/v1/admin')

# ===============================
# Auth
# ===============================

@admin_bp.route('/login', methods=['POST'])
def login():
    if not request.json:
        return error_response('Authorize admin failed.', 400)

    return login_admin(
        request.json.get("email", ""),
        request.json.get("password", "")
    )


@admin_bp.route('/logout', methods=['GET'])
@login_required
def logout():
    current_user.access_token = f"INVALID_{secrets.token_hex(16)}"
    current_user.save()
    logout_user()
    return success_response(True)


@admin_bp.route('/auth', methods=['GET'])
@login_verify
def auth_admin():
    return success_response(None, "Admin is authorized", 0)

# ===============================
# User
# ===============================

@admin_bp.route('/users', methods=['GET'])
@login_required
@check_admin_auth
def list_users():
    return success_response(UserMgr.get_all_users())


@admin_bp.route('/users', methods=['POST'])
@login_required
@check_admin_auth
def create_user():
    data = request.get_json()
    if not data:
        return error_response("Invalid body", 400)

    res = UserMgr.create_user(
        data['username'],
        data['password'],
        data.get('role', 'user')
    )

    res["user_info"].pop("password", None)
    return success_response(res["user_info"])


@admin_bp.route('/users/<username>/datasets', methods=['GET'])
@login_required
@check_admin_auth
def get_user_datasets(username):
    return success_response(UserServiceMgr.get_user_datasets(username))

# ===============================
# ✅ Dataset (핵심)
# ===============================

@admin_bp.route('/datasets', methods=['POST'])
@login_required
@check_admin_auth
def upload_dataset():
    if 'file' not in request.files:
        return error_response("file is required", 400)

    file = request.files['file']
    dataset_name = request.form.get("dataset_name", "default")

    dataset_id = DatasetMgr.save_dataset(file, dataset_name)

    return success_response(
        {"dataset_id": dataset_id},
        "dataset uploaded"
    )


@admin_bp.route('/datasets/<dataset_id>/process', methods=['POST'])
@login_required
@check_admin_auth
def process_dataset(dataset_id):
    threading.Thread(
        target=DatasetMgr.process_dataset,
        args=(dataset_id,),
        daemon=True
    ).start()

    return success_response(
        {"dataset_id": dataset_id},
        "processing started"
    )


@admin_bp.route('/datasets/<dataset_id>/status', methods=['GET'])
@login_required
@check_admin_auth
def dataset_status(dataset_id):
    return success_response(
        DatasetMgr.get_status(dataset_id)
    )

# ===============================
# Service / Version
# ===============================

@admin_bp.route('/services', methods=['GET'])
@login_required
@check_admin_auth
def get_services():
    return success_response(ServiceMgr.get_all_services())


@admin_bp.route('/version', methods=['GET'])
@login_required
@check_admin_auth
def show_version():
    return success_response(
        {"version": get_ragflow_version()}
    )
