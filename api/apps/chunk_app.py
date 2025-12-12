#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import datetime
import json
import re

import xxhash
from quart import request

from api.db.services.dialog_service import meta_filter
from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.llm_service import LLMBundle
from api.db.services.search_service import SearchService
from api.db.services.user_service import UserTenantService
from api.utils.api_utils import get_data_error_result, get_json_result, server_error_response, validate_request, \
    request_json
from rag.app.qa import beAdoc, rmPrefix
from rag.app.tag import label_question
from rag.nlp import rag_tokenizer, search
from rag.prompts.generator import gen_meta_filter, cross_languages, keyword_extraction
from common.string_utils import remove_redundant_spaces
from common.constants import RetCode, LLMType, ParserType, PAGERANK_FLD
from common import settings
from api.apps import login_required, current_user


# ============================================================
# 코어 검색 로직 헬퍼 함수
# ============================================================
async def _run_retrieval(
    question: str,
    kb_ids: list,
    tenant_ids: list,
    page: int = 1,
    size: int = 30,
    similarity_threshold: float = 0.0,
    vector_similarity_weight: float = 0.3,
    top_k: int = 1024,
    doc_ids: list = None,
    rerank_id: str = None,
    use_kg: bool = False,
    highlight: bool = False,
    cross_languages: list = None,
    keyword: bool = False,
):
    """
    retrieval_test와 동일한 코어 검색 로직을 담당하는 헬퍼 함수.

    Args:
        question: 검색 질문
        kb_ids: Knowledge Base ID 리스트
        tenant_ids: Tenant ID 리스트
        page: 페이지 번호 (기본값: 1)
        size: 페이지당 결과 수 (기본값: 30)
        similarity_threshold: 유사도 임계값 (기본값: 0.0)
        vector_similarity_weight: 벡터 유사도 가중치 (기본값: 0.3)
        top_k: 최대 반환 수 (기본값: 1024)
        doc_ids: 문서 ID 필터 (선택)
        rerank_id: Rerank 모델 ID (선택)
        use_kg: Knowledge Graph 사용 여부
        highlight: 하이라이트 여부
        cross_languages: 다국어 지원 언어 리스트
        keyword: 키워드 추출 사용 여부

    Returns:
        dict: {"chunks": [...], "labels": [...], ...} 형식의 검색 결과
    """
    from rag.prompts.generator import cross_languages as do_cross_languages, keyword_extraction

    e, kb = KnowledgebaseService.get_by_id(kb_ids[0])
    if not e:
        raise Exception("Knowledgebase not found!")

    # 다국어 처리
    if cross_languages:
        question = do_cross_languages(kb.tenant_id, None, question, cross_languages)

    # 임베딩 모델 로드
    embd_mdl = LLMBundle(kb.tenant_id, LLMType.EMBEDDING.value, llm_name=kb.embd_id)

    # Rerank 모델 로드 (선택)
    rerank_mdl = None
    if rerank_id:
        rerank_mdl = LLMBundle(kb.tenant_id, LLMType.RERANK.value, llm_name=rerank_id)

    # 키워드 추출 (선택)
    if keyword:
        chat_mdl = LLMBundle(kb.tenant_id, LLMType.CHAT)
        question += keyword_extraction(chat_mdl, question)

    # 라벨링
    labels = label_question(question, [kb])

    # 검색 실행
    ranks = settings.retriever.retrieval(
        question, embd_mdl, tenant_ids, kb_ids, page, size,
        similarity_threshold,
        vector_similarity_weight,
        top_k,
        doc_ids,
        rerank_mdl=rerank_mdl,
        highlight=highlight,
        rank_feature=labels
    )

    # Knowledge Graph 검색 (선택)
    if use_kg:
        ck = settings.kg_retriever.retrieval(
            question,
            tenant_ids,
            kb_ids,
            embd_mdl,
            LLMBundle(kb.tenant_id, LLMType.CHAT)
        )
        if ck["content_with_weight"]:
            ranks["chunks"].insert(0, ck)

    # 벡터 필드 제거
    for c in ranks["chunks"]:
        c.pop("vector", None)

    ranks["labels"] = labels
    return ranks


@manager.route('/list', methods=['POST'])  # noqa: F821
@login_required
@validate_request("doc_id")
async def list_chunk():
    req = await request_json()
    doc_id = req["doc_id"]
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    question = req.get("keywords", "")
    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(message="Tenant not found!")
        e, doc = DocumentService.get_by_id(doc_id)
        if not e:
            return get_data_error_result(message="Document not found!")
        kb_ids = KnowledgebaseService.get_kb_ids(tenant_id)
        query = {
            "doc_ids": [doc_id], "page": page, "size": size, "question": question, "sort": True
        }
        if "available_int" in req:
            query["available_int"] = int(req["available_int"])
        sres = settings.retriever.search(query, search.index_name(tenant_id), kb_ids, highlight=["content_ltks"])
        res = {"total": sres.total, "chunks": [], "doc": doc.to_dict()}
        for id in sres.ids:
            d = {
                "chunk_id": id,
                "content_with_weight": remove_redundant_spaces(sres.highlight[id]) if question and id in sres.highlight else sres.field[
                    id].get(
                    "content_with_weight", ""),
                "doc_id": sres.field[id]["doc_id"],
                "docnm_kwd": sres.field[id]["docnm_kwd"],
                "important_kwd": sres.field[id].get("important_kwd", []),
                "question_kwd": sres.field[id].get("question_kwd", []),
                "image_id": sres.field[id].get("img_id", ""),
                "available_int": int(sres.field[id].get("available_int", 1)),
                "positions": sres.field[id].get("position_int", []),
            }
            assert isinstance(d["positions"], list)
            assert len(d["positions"]) == 0 or (isinstance(d["positions"][0], list) and len(d["positions"][0]) == 5)
            res["chunks"].append(d)
        return get_json_result(data=res)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, message='No chunk found!',
                                   code=RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/get', methods=['GET'])  # noqa: F821
@login_required
def get():
    chunk_id = request.args["chunk_id"]
    try:
        chunk = None
        tenants = UserTenantService.query(user_id=current_user.id)
        if not tenants:
            return get_data_error_result(message="Tenant not found!")
        for tenant in tenants:
            kb_ids = KnowledgebaseService.get_kb_ids(tenant.tenant_id)
            chunk = settings.docStoreConn.get(chunk_id, search.index_name(tenant.tenant_id), kb_ids)
            if chunk:
                break
        if chunk is None:
            return server_error_response(Exception("Chunk not found"))

        k = []
        for n in chunk.keys():
            if re.search(r"(_vec$|_sm_|_tks|_ltks)", n):
                k.append(n)
        for n in k:
            del chunk[n]

        return get_json_result(data=chunk)
    except Exception as e:
        if str(e).find("NotFoundError") >= 0:
            return get_json_result(data=False, message='Chunk not found!',
                                   code=RetCode.DATA_ERROR)
        return server_error_response(e)


@manager.route('/set', methods=['POST'])  # noqa: F821
@login_required
@validate_request("doc_id", "chunk_id", "content_with_weight")
async def set():
    req = await request_json()
    d = {
        "id": req["chunk_id"],
        "content_with_weight": req["content_with_weight"]}
    d["content_ltks"] = rag_tokenizer.tokenize(req["content_with_weight"])
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    if "important_kwd" in req:
        if not isinstance(req["important_kwd"], list):
            return get_data_error_result(message="`important_kwd` should be a list")
        d["important_kwd"] = req["important_kwd"]
        d["important_tks"] = rag_tokenizer.tokenize(" ".join(req["important_kwd"]))
    if "question_kwd" in req:
        if not isinstance(req["question_kwd"], list):
            return get_data_error_result(message="`question_kwd` should be a list")
        d["question_kwd"] = req["question_kwd"]
        d["question_tks"] = rag_tokenizer.tokenize("\n".join(req["question_kwd"]))
    if "tag_kwd" in req:
        d["tag_kwd"] = req["tag_kwd"]
    if "tag_feas" in req:
        d["tag_feas"] = req["tag_feas"]
    if "available_int" in req:
        d["available_int"] = req["available_int"]

    try:
        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(message="Tenant not found!")

        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING, embd_id)

        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")

        if doc.parser_id == ParserType.QA:
            arr = [
                t for t in re.split(
                    r"[\n\t]",
                    req["content_with_weight"]) if len(t) > 1]
            q, a = rmPrefix(arr[0]), rmPrefix("\n".join(arr[1:]))
            d = beAdoc(d, q, a, not any(
                [rag_tokenizer.is_chinese(t) for t in q + a]))

        v, c = embd_mdl.encode([doc.name, req["content_with_weight"] if not d.get("question_kwd") else "\n".join(d["question_kwd"])])
        v = 0.1 * v[0] + 0.9 * v[1] if doc.parser_id != ParserType.QA else v[1]
        d["q_%d_vec" % len(v)] = v.tolist()
        settings.docStoreConn.update({"id": req["chunk_id"]}, d, search.index_name(tenant_id), doc.kb_id)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/switch', methods=['POST'])  # noqa: F821
@login_required
@validate_request("chunk_ids", "available_int", "doc_id")
async def switch():
    req = await request_json()
    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        for cid in req["chunk_ids"]:
            if not settings.docStoreConn.update({"id": cid},
                                                {"available_int": int(req["available_int"])},
                                                search.index_name(DocumentService.get_tenant_id(req["doc_id"])),
                                                doc.kb_id):
                return get_data_error_result(message="Index updating failure")
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/rm', methods=['POST'])  # noqa: F821
@login_required
@validate_request("chunk_ids", "doc_id")
async def rm():
    req = await request_json()
    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        if not settings.docStoreConn.delete({"id": req["chunk_ids"]},
                                            search.index_name(DocumentService.get_tenant_id(req["doc_id"])),
                                            doc.kb_id):
            return get_data_error_result(message="Chunk deleting failure")
        deleted_chunk_ids = req["chunk_ids"]
        chunk_number = len(deleted_chunk_ids)
        DocumentService.decrement_chunk_num(doc.id, doc.kb_id, 1, chunk_number, 0)
        for cid in deleted_chunk_ids:
            if settings.STORAGE_IMPL.obj_exist(doc.kb_id, cid):
                settings.STORAGE_IMPL.rm(doc.kb_id, cid)
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)


@manager.route('/create', methods=['POST'])  # noqa: F821
@login_required
@validate_request("doc_id", "content_with_weight")
async def create():
    req = await request_json()
    chunck_id = xxhash.xxh64((req["content_with_weight"] + req["doc_id"]).encode("utf-8")).hexdigest()
    d = {"id": chunck_id, "content_ltks": rag_tokenizer.tokenize(req["content_with_weight"]),
         "content_with_weight": req["content_with_weight"]}
    d["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(d["content_ltks"])
    d["important_kwd"] = req.get("important_kwd", [])
    if not isinstance(d["important_kwd"], list):
        return get_data_error_result(message="`important_kwd` is required to be a list")
    d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))
    d["question_kwd"] = req.get("question_kwd", [])
    if not isinstance(d["question_kwd"], list):
        return get_data_error_result(message="`question_kwd` is required to be a list")
    d["question_tks"] = rag_tokenizer.tokenize("\n".join(d["question_kwd"]))
    d["create_time"] = str(datetime.datetime.now()).replace("T", " ")[:19]
    d["create_timestamp_flt"] = datetime.datetime.now().timestamp()
    if "tag_feas" in req:
        d["tag_feas"] = req["tag_feas"]
    if "tag_feas" in req:
        d["tag_feas"] = req["tag_feas"]

    try:
        e, doc = DocumentService.get_by_id(req["doc_id"])
        if not e:
            return get_data_error_result(message="Document not found!")
        d["kb_id"] = [doc.kb_id]
        d["docnm_kwd"] = doc.name
        d["title_tks"] = rag_tokenizer.tokenize(doc.name)
        d["doc_id"] = doc.id

        tenant_id = DocumentService.get_tenant_id(req["doc_id"])
        if not tenant_id:
            return get_data_error_result(message="Tenant not found!")

        e, kb = KnowledgebaseService.get_by_id(doc.kb_id)
        if not e:
            return get_data_error_result(message="Knowledgebase not found!")
        if kb.pagerank:
            d[PAGERANK_FLD] = kb.pagerank

        embd_id = DocumentService.get_embd_id(req["doc_id"])
        embd_mdl = LLMBundle(tenant_id, LLMType.EMBEDDING.value, embd_id)

        v, c = embd_mdl.encode([doc.name, req["content_with_weight"] if not d["question_kwd"] else "\n".join(d["question_kwd"])])
        v = 0.1 * v[0] + 0.9 * v[1]
        d["q_%d_vec" % len(v)] = v.tolist()
        settings.docStoreConn.insert([d], search.index_name(tenant_id), doc.kb_id)

        DocumentService.increment_chunk_num(
            doc.id, doc.kb_id, c, 1, 0)
        return get_json_result(data={"chunk_id": chunck_id})
    except Exception as e:
        return server_error_response(e)


@manager.route('/retrieval_test', methods=['POST'])  # noqa: F821
@login_required
@validate_request("kb_id", "question")
async def retrieval_test():
    req = await request_json()
    page = int(req.get("page", 1))
    size = int(req.get("size", 30))
    question = req["question"]
    kb_ids = req["kb_id"]
    if isinstance(kb_ids, str):
        kb_ids = [kb_ids]
    if not kb_ids:
        return get_json_result(data=False, message='Please specify dataset firstly.',
                               code=RetCode.DATA_ERROR)

    doc_ids = req.get("doc_ids", [])
    use_kg = req.get("use_kg", False)
    top = int(req.get("top_k", 1024))
    langs = req.get("cross_languages", [])
    tenant_ids = []

    if req.get("search_id", ""):
        search_config = SearchService.get_detail(req.get("search_id", "")).get("search_config", {})
        meta_data_filter = search_config.get("meta_data_filter", {})
        metas = DocumentService.get_meta_by_kbs(kb_ids)
        if meta_data_filter.get("method") == "auto":
            chat_mdl = LLMBundle(current_user.id, LLMType.CHAT, llm_name=search_config.get("chat_id", ""))
            filters: dict = gen_meta_filter(chat_mdl, metas, question)
            doc_ids.extend(meta_filter(metas, filters["conditions"], filters.get("logic", "and")))
            if not doc_ids:
                doc_ids = None
        elif meta_data_filter.get("method") == "manual":
            doc_ids.extend(meta_filter(metas, meta_data_filter["manual"], meta_data_filter.get("logic", "and")))
            if meta_data_filter["manual"] and not doc_ids:
                doc_ids = ["-999"]

    try:
        tenants = UserTenantService.query(user_id=current_user.id)
        for kb_id in kb_ids:
            for tenant in tenants:
                if KnowledgebaseService.query(
                        tenant_id=tenant.tenant_id, id=kb_id):
                    tenant_ids.append(tenant.tenant_id)
                    break
            else:
                return get_json_result(
                    data=False, message='Only owner of knowledgebase authorized for this operation.',
                    code=RetCode.OPERATING_ERROR)

        # 코어 검색 로직 호출
        ranks = await _run_retrieval(
            question=question,
            kb_ids=kb_ids,
            tenant_ids=tenant_ids,
            page=page,
            size=size,
            similarity_threshold=float(req.get("similarity_threshold", 0.0)),
            vector_similarity_weight=float(req.get("vector_similarity_weight", 0.3)),
            top_k=top,
            doc_ids=doc_ids if doc_ids else None,
            rerank_id=req.get("rerank_id"),
            use_kg=use_kg,
            highlight=req.get("highlight", False),
            cross_languages=langs if langs else None,
            keyword=req.get("keyword", False),
        )

        return get_json_result(data=ranks)
    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(data=False, message='No chunk found! Check the chunk status please!',
                                   code=RetCode.DATA_ERROR)
        return server_error_response(e)


# ============================================================
# AI Gateway용 경량 검색 API
# ============================================================
@manager.route('/search', methods=['POST'])  # noqa: F821
async def search_simple():
    """
    AI Gateway용 경량 검색 API

    URL: POST /v1/chunk/search

    Request Body:
        {
            "query": "검색 질문 (필수)",
            "top_k": 5,  (선택, 기본값 5)
            "dataset": "kb_id 문자열 (필수)"
        }

    Response Body:
        {
            "results": [
                {
                    "doc_id": "chunk_id",
                    "title": "문서명",
                    "page": 페이지번호 또는 null,
                    "score": 0.87,
                    "snippet": "내용 일부 (최대 500자)"
                }
            ]
        }

    Example:
        curl -X POST "http://localhost:9380/v1/chunk/search" \\
          -H "Content-Type: application/json" \\
          -d '{
            "query": "4대 필수교육 미이수 시 패널티",
            "top_k": 3,
            "dataset": "실제_kb_id"
          }'
    """
    try:
        req = await request_json()
    except Exception:
        return get_data_error_result(message="Invalid JSON body")

    # 필수 파라미터 검증
    query = (req.get("query") or "").strip()
    if not query:
        return get_data_error_result(message="query is required")

    dataset = req.get("dataset")
    if not dataset:
        return get_data_error_result(message="dataset (kb_id) is required")

    # top_k 파라미터 처리
    try:
        top_k = int(req.get("top_k", 5))
        if top_k <= 0:
            top_k = 5
    except (ValueError, TypeError):
        return get_data_error_result(message="top_k must be a positive integer")

    kb_ids = [dataset]

    try:
        # Knowledge Base 존재 확인 및 tenant_id 조회
        e, kb = KnowledgebaseService.get_by_id(dataset)
        if not e:
            return get_data_error_result(message="Dataset not found")

        tenant_ids = [kb.tenant_id]

        # 코어 검색 로직 호출
        ranks = await _run_retrieval(
            question=query,
            kb_ids=kb_ids,
            tenant_ids=tenant_ids,
            page=1,
            size=top_k,
            similarity_threshold=0.0,
            vector_similarity_weight=0.3,
            top_k=top_k,
        )

        # 응답 포맷 변환
        results = []
        for chunk in ranks.get("chunks", []):
            # 필드 매핑
            doc_id = chunk.get("chunk_id") or chunk.get("id", "")
            title = chunk.get("docnm_kwd") or chunk.get("doc_name") or chunk.get("title", "")
            page = chunk.get("page_num_int") or chunk.get("page_num") or chunk.get("page")
            score = chunk.get("similarity") or chunk.get("score") or 0.0
            content = chunk.get("content_with_weight") or chunk.get("content", "")

            # snippet은 최대 500자
            snippet = content[:500] if content else ""

            results.append({
                "doc_id": doc_id,
                "title": title,
                "page": page,
                "score": round(float(score), 4) if score else 0.0,
                "snippet": snippet,
            })

        return get_json_result(data={"results": results})

    except Exception as e:
        if str(e).find("not_found") > 0:
            return get_json_result(
                data={"results": []},
                message='No chunk found',
                code=RetCode.DATA_ERROR
            )
        return server_error_response(e)


@manager.route('/knowledge_graph', methods=['GET'])  # noqa: F821
@login_required
def knowledge_graph():
    doc_id = request.args["doc_id"]
    tenant_id = DocumentService.get_tenant_id(doc_id)
    kb_ids = KnowledgebaseService.get_kb_ids(tenant_id)
    req = {
        "doc_ids": [doc_id],
        "knowledge_graph_kwd": ["graph", "mind_map"]
    }
    sres = settings.retriever.search(req, search.index_name(tenant_id), kb_ids)
    obj = {"graph": {}, "mind_map": {}}
    for id in sres.ids[:2]:
        ty = sres.field[id]["knowledge_graph_kwd"]
        try:
            content_json = json.loads(sres.field[id]["content_with_weight"])
        except Exception:
            continue

        if ty == 'mind_map':
            node_dict = {}

            def repeat_deal(content_json, node_dict):
                if 'id' in content_json:
                    if content_json['id'] in node_dict:
                        node_name = content_json['id']
                        content_json['id'] += f"({node_dict[content_json['id']]})"
                        node_dict[node_name] += 1
                    else:
                        node_dict[content_json['id']] = 1
                if 'children' in content_json and content_json['children']:
                    for item in content_json['children']:
                        repeat_deal(item, node_dict)

            repeat_deal(content_json, node_dict)

        obj[ty] = content_json

    return get_json_result(data=obj)
