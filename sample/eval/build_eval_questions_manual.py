import json
from pathlib import Path

# =========================
# 고정 설정 (⚠ 네 doc_chunks.json 기준)
# =========================
DOMAIN = "장애인인식개선교육"
ANSWER_DOC = "2023년장애인인식개선교육자료배포.pdf"

BASE_DIR = Path(__file__).resolve().parent
DOC_CHUNKS_PATH = BASE_DIR / "results" / "doc_chunks.json"
OUT_PATH = BASE_DIR / "results" / "eval_questions.json"

# =========================
# 질문 정의 (✅ chunk_id 1~6만 사용)
# =========================
QUESTIONS = [
    {
        "question": "직장 내 장애인 인식개선교육을 무료로 지원받을 수 있는 제도에는 무엇이 있는가?",
        "chunk_id": 1,  # 무료 강사지원 / 이러닝 교육
    },
    {
        "question": "직장 내 장애인 인식개선교육은 연 몇 회, 몇 시간 이상 실시해야 하는가?",
        "chunk_id": 2,  # 연 1회, 1시간 이상
    },
    {
        "question": "직장 내 장애인 인식개선교육의 법적 근거는 무엇인가?",
        "chunk_id": 2,  # 장애인고용촉진 및 직업재활법 제5조의2
    },
    {
        "question": "직장 내 장애인 인식개선교육의 주요 목적은 무엇인가?",
        "chunk_id": 3,  # 장애 이해, 차이 수용
    },
    {
        "question": "장애를 바라보는 관점은 과거와 현재 어떻게 달라졌는가?",
        "chunk_id": 4,  # 의료적 관점 → 사회적 관점
    },
    {
        "question": "채용 과정에서 청각장애인에게 듣기 평가를 요구하는 것은 어떤 차별에 해당하는가?",
        "chunk_id": 6,  # 간접차별
    },
]

def main():
    if not DOC_CHUNKS_PATH.exists():
        raise FileNotFoundError(f"❌ doc_chunks.json not found: {DOC_CHUNKS_PATH}")

    chunks = json.loads(DOC_CHUNKS_PATH.read_text(encoding="utf-8"))

    # chunk_id → chunk 매핑
    chunk_map = {int(c["chunk_id"]): c for c in chunks}

    eval_items = []
    for q in QUESTIONS:
        cid = int(q["chunk_id"])
        if cid not in chunk_map:
            raise ValueError(
                f"❌ chunk_id {cid} not found. "
                f"available={sorted(chunk_map.keys())}"
            )

        eval_items.append({
            "domain": DOMAIN,
            "question": q["question"],
            "answer_doc": ANSWER_DOC,
            "gt_chunk_hash": chunk_map[cid]["chunk_hash"],
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(eval_items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"✅ saved: {OUT_PATH}")
    print(f"✅ questions: {len(eval_items)}")
    print("▶ chunk_ids used:", [q["chunk_id"] for q in QUESTIONS])

if __name__ == "__main__":
    main()
