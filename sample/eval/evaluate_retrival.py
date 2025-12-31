import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
from dotenv import load_dotenv
from pymilvus import Collection, connections

BASE_DIR = Path(__file__).resolve().parents[2]  # ragflow/
load_dotenv(BASE_DIR / ".env")

# -----------------------
# ENV
# -----------------------
MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ragflow_chunks_sroberta")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
if not VLLM_BASE_URL:
    raise RuntimeError("VLLM_BASE_URL is not set. Check your .env file.")

VLLM_BASE_URL = VLLM_BASE_URL.rstrip("/")
if not VLLM_BASE_URL.endswith("/v1"):
    VLLM_BASE_URL += "/v1"

EMBEDDING_ENDPOINT = f"{VLLM_BASE_URL}/embeddings"
VLLM_MODEL_ID = os.getenv("VLLM_EMBED_MODEL_ID")

# metrics
TOPKS = [1, 3, 5]
NPROBE = 16

# ✅ (TOP3-1) 검색 후보 확장
SEARCH_K = 50  # 검색은 넉넉히, 평가는 TOPKS로만

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "results", "eval_questions.json")


# -----------------------
# helpers
# -----------------------
def embed_one(text: str) -> List[float]:
    text = (text or "").strip()
    r = requests.post(
        EMBEDDING_ENDPOINT,
        json={"input": [text], "model": VLLM_MODEL_ID},
        timeout=60
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]


def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_docname(s: str) -> str:
    """
    doc_id / filename normalize
    - path 제거
    - 앞쪽 __ 같은 underscore 제거
    - 공백 제거
    """
    s = (s or "").strip()
    s = s.replace("\\", "/").split("/")[-1]      # path -> filename
    s = re.sub(r"^_+", "", s)                    # leading underscores 제거
    s = s.replace(" ", "")                       # 공백 제거
    return s


def hit_by_keywords(text: str, keywords: List[str]) -> bool:
    t = norm(text)
    if not keywords:
        return False
    return all(norm(k) in t for k in keywords if norm(k))


def calc_mrr_and_recall(ranks: List[int]) -> Tuple[Dict[int, float], float]:
    recall = {}
    for k in TOPKS:
        recall[k] = sum(1 for r in ranks if 1 <= r <= k) / max(1, len(ranks))
    mrr = sum((1.0 / r) for r in ranks if r > 0) / max(1, len(ranks))
    return recall, mrr


def get_answer_docs(item: Dict[str, Any]) -> List[str]:
    """
    eval_questions.json에서 정답 문서 목록을 복수로 받는다.
    지원:
      - answer_docs: ["a.pdf","b.pdf"]
      - answer_doc: "a.pdf"  (단일)
      - gt_doc_id / doc_id (단일 fallback)
    """
    docs = item.get("answer_docs")
    if isinstance(docs, list) and docs:
        return [norm_docname(x) for x in docs if str(x).strip()]

    one = (
        item.get("answer_doc")
        or item.get("gt_doc_id")
        or item.get("doc_id")
        or ""
    )
    one = one.strip()
    return [norm_docname(one)] if one else []


# =========================================================
# ✅ (TOP3-2) 리플릿/간이 문서 패널티
# =========================================================
def doc_penalty(doc_id: str) -> float:
    """
    리플릿/간이/요약 같은 문서가 벡터에서 과도하게 상위권을 먹는 경우 방지.
    metadata가 있으면 doc_type로 바꾸는 게 더 좋지만, 지금은 doc_id 휴리스틱.
    """
    d = (doc_id or "").lower()
    # 한글 키워드는 lower 영향 없지만, 영문 대비로 lower 유지
    if ("리플릿" in d) or ("간이" in d) or ("요약" in d) or ("leaflet" in d) or ("brochure" in d):
        return 0.92  # 8% 패널티
    return 1.0


# =========================================================
# ✅ (TOP3-3) 숫자/법적 질문 휴리스틱 rerank
# =========================================================
_RULE_Q_PAT = re.compile(r"(몇\s*회|몇\s*시간|법적|의무|기준|차별|조항|근거|벌칙|과태료|시간\s*이상|연\s*\d+|회\s*이상)")
def is_rule_question(q: str) -> bool:
    return bool(_RULE_Q_PAT.search(q or ""))


def rerank_bonus(text: str) -> float:
    """
    법령/의무성 질문에서 유리한 chunk 특징에 보너스:
    - 숫자 포함 (횟수/시간/조항)
    - 충분히 긴 설명(짧은 요약보다 본문/법령에 가산)
    """
    t = text or ""
    bonus = 1.0
    if re.search(r"\d", t):
        bonus += 0.05
    if len(t) > 300:
        bonus += 0.05
    return bonus


def rerank_hits(q: str, hits: List[Any]) -> List[Any]:
    """
    Milvus raw hits -> (패널티/보너스 반영) 재정렬
    주의: 여기서 hit.score를 직접 바꾸지 않고, 별도 계산 점수로 정렬만 함.
    """
    if not hits:
        return hits

    rule_q = is_rule_question(q)

    scored = []
    for h in hits:
        e = h.entity
        doc_id = e.get("doc_id") or ""
        text = e.get("text") or ""

        score = float(h.score)

        # 리플릿 패널티
        score *= doc_penalty(doc_id)

        # rule 질문이면 추가 보너스
        if rule_q:
            score *= rerank_bonus(text)

        scored.append((score, h))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [h for _, h in scored]


def main():
    # sanity check
    if not MILVUS_HOST or not MILVUS_PORT:
        raise RuntimeError(f"MILVUS_HOST/MILVUS_PORT not set. host={MILVUS_HOST}, port={MILVUS_PORT}")
    if not VLLM_MODEL_ID:
        raise RuntimeError("VLLM_EMBED_MODEL_ID not set in .env")
    if not os.path.exists(QUESTIONS_PATH):
        raise FileNotFoundError(f"eval_questions.json not found: {QUESTIONS_PATH}")

    # 1) load questions
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or not questions:
        raise ValueError("eval_questions.json must be a non-empty list")

    # 2) milvus connect
    connections.connect("default", host=str(MILVUS_HOST), port=str(MILVUS_PORT))
    col = Collection(MILVUS_COLLECTION)
    col.load()

    print("=" * 80)
    print("Collection:", MILVUS_COLLECTION)
    print("VLLM:", VLLM_BASE_URL)
    print("Model:", VLLM_MODEL_ID)
    print("Questions:", len(questions))
    print("SearchK:", SEARCH_K, "| EvalTopKs:", TOPKS, "| nprobe:", NPROBE)
    print("=" * 80)

    chunk_ranks: List[int] = []
    doc_ranks: List[int] = []
    details: List[Dict[str, Any]] = []

    for i, item in enumerate(questions, 1):
        q = item.get("q") or item.get("query") or item.get("question")
        q = norm(q)
        if not q:
            continue

        domain = item.get("domain") or item.get("dataset_id") or ""

        # ✅ 복수 문서 GT 지원
        gt_docs = get_answer_docs(item)  # normalized list
        gt_hash = item.get("gt_chunk_hash") or item.get("chunk_hash") or ""
        gt_keywords = item.get("gt") or item.get("keywords") or item.get("answer_keywords") or []
        if isinstance(gt_keywords, str):
            gt_keywords = [gt_keywords]

        expr = f'dataset_id == "{domain}"' if domain else None

        vec = embed_one(q)
        res = col.search(
            data=[vec],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": NPROBE}},
            # ✅ 후보 확장
            limit=SEARCH_K,
            expr=expr,
            output_fields=["dataset_id", "doc_id", "chunk_id", "chunk_hash", "text"],
        )

        # domain filter가 틀려서 0개면 재검색
        if (not res) or (not res[0]):
            res = col.search(
                data=[vec],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": NPROBE}},
                limit=SEARCH_K,
                expr=None,
                output_fields=["dataset_id", "doc_id", "chunk_id", "chunk_hash", "text"],
            )

        # ✅ rerank 적용 (리플릿 패널티 + rule 질문 보너스)
        hits = rerank_hits(q, res[0] if (res and res[0]) else [])
        # 평가/출력은 TOPKS까지만 보면 됨
        eval_hits = hits[:max(TOPKS)]

        # -----------------------
        # CHUNK HIT: gt_chunk_hash(정확) 우선
        # -----------------------
        found_chunk_rank = 0
        chunk_reason = "miss"

        if eval_hits:
            for r_i, hit in enumerate(eval_hits, 1):
                ent = hit.entity
                chash = ent.get("chunk_hash") or ""
                text = ent.get("text") or ""

                if gt_hash and chash == gt_hash:
                    found_chunk_rank = r_i
                    chunk_reason = "gt_chunk_hash"
                    break

                if gt_keywords and hit_by_keywords(text, gt_keywords):
                    found_chunk_rank = r_i
                    chunk_reason = "keywords"
                    break

        chunk_ranks.append(found_chunk_rank)

        # -----------------------
        # DOC HIT: gt_docs(복수) 중 하나만 맞으면 OK
        # -----------------------
        found_doc_rank = 0
        doc_reason = "miss"

        if eval_hits and gt_docs:
            for r_i, hit in enumerate(eval_hits, 1):
                ent = hit.entity
                doc_id = norm_docname(ent.get("doc_id") or "")
                if doc_id in gt_docs:
                    found_doc_rank = r_i
                    doc_reason = "gt_doc_id_multi"
                    break

        # fallback: gt_docs 없으면(옛 포맷) 단일 doc_id 키 매칭
        if (not gt_docs) and eval_hits:
            single = (
                item.get("answer_doc")
                or item.get("gt_doc_id")
                or item.get("doc_id")
                or ""
            )
            single_norm = norm_docname(single) if single else ""
            if single_norm:
                for r_i, hit in enumerate(eval_hits, 1):
                    ent = hit.entity
                    doc_id = norm_docname(ent.get("doc_id") or "")
                    if doc_id == single_norm:
                        found_doc_rank = r_i
                        doc_reason = "gt_doc_id_norm"
                        break

        doc_ranks.append(found_doc_rank)

        # sample print
        if i <= 6:
            print("\n--- SAMPLE ---")
            print("Q:", q)
            print("domain:", domain)
            print("rule_q:", is_rule_question(q))
            print("gt_docs:", gt_docs)
            print("gt_hash:", (gt_hash[:12] + "...") if gt_hash else "")
            print("chunk_rank:", found_chunk_rank, "reason:", chunk_reason)
            print("doc_rank:", found_doc_rank, "reason:", doc_reason)

            if eval_hits:
                print("top hits (reranked):")
                for j, h in enumerate(eval_hits[:3], 1):
                    e = h.entity
                    # 원 점수 표시 + 패널티/보너스 설명까지 보고 싶으면 아래처럼 계산 가능
                    raw = float(h.score)
                    pen = doc_penalty(e.get("doc_id") or "")
                    bonus = rerank_bonus(e.get("text") or "") if is_rule_question(q) else 1.0
                    adj = raw * pen * bonus
                    print(f"  {j}) raw={raw:.4f} adj={adj:.4f} doc_id={e.get('doc_id')} chunk_id={e.get('chunk_id')}")

        details.append({
            "q": q,
            "domain": domain,
            "gt_docs": gt_docs,
            "gt_chunk_hash": gt_hash,
            "gt_keywords": gt_keywords,
            "chunk_rank": found_chunk_rank,
            "chunk_reason": chunk_reason,
            "doc_rank": found_doc_rank,
            "doc_reason": doc_reason,
            "rule_question": is_rule_question(q),
            "search_k": SEARCH_K,
        })

    chunk_recall, chunk_mrr = calc_mrr_and_recall(chunk_ranks)
    doc_recall, doc_mrr = calc_mrr_and_recall(doc_ranks)

    print("\n" + "=" * 80)
    print("[Chunk metrics]")
    for k in TOPKS:
        print(f"Recall@{k}: {chunk_recall[k]*100:.2f}%")
    print(f"MRR: {chunk_mrr:.4f}")

    print("\n[Doc metrics] (answer_docs multi-match)")
    for k in TOPKS:
        print(f"Recall@{k}: {doc_recall[k]*100:.2f}%")
    print(f"MRR: {doc_mrr:.4f}")
    print("=" * 80)

    # save results
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_result.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "collection": MILVUS_COLLECTION,
            "vllm_base_url": VLLM_BASE_URL,
            "vllm_model_id": VLLM_MODEL_ID,
            "topks": TOPKS,
            "nprobe": NPROBE,
            "search_k": SEARCH_K,
            "count": len(chunk_ranks),
            "chunk_metrics": {"recall": chunk_recall, "mrr": chunk_mrr},
            "doc_metrics": {"recall": doc_recall, "mrr": doc_mrr},
            "details": details[:500],
        }, f, ensure_ascii=False, indent=2)

    print("✅ saved:", out_path)


if __name__ == "__main__":
    main()
