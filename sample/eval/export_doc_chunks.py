import os, json
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import connections, Collection

BASE_DIR = Path(__file__).resolve().parents[2]  # ragflow/
load_dotenv(BASE_DIR / ".env")

MILVUS_HOST = os.getenv("MILVUS_HOST")
MILVUS_PORT = os.getenv("MILVUS_PORT")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "ragflow_chunks_sroberta")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, help='dataset_id, e.g. 장애인인식개선교육')
    ap.add_argument("--doc_id", required=True, help='exact doc_id stored in Milvus')
    ap.add_argument("--out", default="doc_chunks.json")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    connections.connect("default", host=str(MILVUS_HOST), port=str(MILVUS_PORT))
    col = Collection(MILVUS_COLLECTION)

    expr = f'dataset_id == "{args.domain}" && doc_id == "{args.doc_id}"'
    rows = col.query(
        expr=expr,
        output_fields=["dataset_id", "doc_id", "chunk_id", "chunk_hash", "text"],
        limit=args.limit,
    )

    # chunk_id 기준 정렬
    rows = sorted(rows, key=lambda x: int(x.get("chunk_id", 0)))

    out = []
    for r in rows:
        out.append({
            "dataset_id": r["dataset_id"],
            "doc_id": r["doc_id"],
            "chunk_id": r["chunk_id"],
            "chunk_hash": r["chunk_hash"],
            "preview": (r.get("text") or "")[:400]
        })

    out_path = Path(__file__).resolve().parent / "results" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ saved: {out_path} (chunks={len(out)})")
    for x in out[:5]:
        print(f"- chunk_id={x['chunk_id']} hash={x['chunk_hash'][:10]} preview={x['preview'][:80]}")

if __name__ == "__main__":
    main()
