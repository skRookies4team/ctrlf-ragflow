"""
RAGAS 배치 평가 스크립트
- 130문항을 배치로 나눠서 처리 (타임아웃 방지)
- 중간 결과 저장 (이어서 평가 가능)
- 실패한 배치만 재시도 가능
"""

import json
import os
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import httpx

# 환경 변수 설정
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "1237")
os.environ["OPENAI_API_BASE"] = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
os.environ["OPENAI_API_KEY"] = "not-needed"

from datasets import Dataset
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig

# 경로 설정
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
BATCH_DIR = RESULTS_DIR / "batches"
BATCH_DIR.mkdir(parents=True, exist_ok=True)


def get_model_id() -> str:
    """vLLM 서버에서 실제 모델 ID 조회"""
    import requests
    try:
        resp = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models", timeout=10)
        data = resp.json()
        if data.get("data"):
            model_id = data["data"][0]["id"]
            print(f"Judge LLM: {model_id}")
            return model_id
    except Exception as e:
        print(f"모델 ID 조회 실패: {e}")
    return "Qwen/Qwen2.5-7B-Instruct"


def load_rag_results(file_path: Path) -> Dict:
    """RAG 답변 결과 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset_from_items(items: List[Dict]) -> Dataset:
    """아이템 리스트에서 RAGAS 데이터셋 생성"""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in items:
        data["question"].append(item["question"])
        data["answer"].append(item["rag_answer"])
        contexts = item.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(data)


def run_batch_evaluation(dataset: Dataset, model_id: str, llm, embeddings) -> Dict:
    """단일 배치 RAGAS 평가"""
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    run_config = RunConfig(
        max_workers=2,
        max_wait=300,
        max_retries=3,
        timeout=300,
    )

    try:
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config,
        )
        return {
            "faithfulness": float(results["faithfulness"]) if results["faithfulness"] else 0,
            "answer_relevancy": float(results["answer_relevancy"]) if results["answer_relevancy"] else 0,
            "context_precision": float(results["context_precision"]) if results["context_precision"] else 0,
            "context_recall": float(results["context_recall"]) if results["context_recall"] else 0,
            "status": "success"
        }
    except Exception as e:
        print(f"  [ERROR] {e}")
        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_precision": 0,
            "context_recall": 0,
            "status": "failed",
            "error": str(e)
        }


def get_batch_file_path(rag_file_name: str, batch_idx: int) -> Path:
    """배치 결과 파일 경로"""
    base_name = rag_file_name.replace(".json", "")
    return BATCH_DIR / f"{base_name}_batch_{batch_idx:03d}.json"


def load_existing_batches(rag_file_name: str) -> Dict[int, Dict]:
    """기존 배치 결과 로드"""
    existing = {}
    base_name = rag_file_name.replace(".json", "")
    for batch_file in BATCH_DIR.glob(f"{base_name}_batch_*.json"):
        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                batch_idx = data.get("batch_idx", 0)
                if data.get("status") == "success":
                    existing[batch_idx] = data
        except:
            pass
    return existing


def save_batch_result(rag_file_name: str, batch_idx: int, result: Dict, items: List[Dict]):
    """배치 결과 저장"""
    batch_file = get_batch_file_path(rag_file_name, batch_idx)
    result["batch_idx"] = batch_idx
    result["item_count"] = len(items)
    result["timestamp"] = datetime.now().isoformat()

    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def merge_batch_results(batch_results: List[Dict]) -> Dict:
    """배치 결과들을 하나로 합침 (가중 평균)"""
    total_items = sum(b.get("item_count", 0) for b in batch_results)
    if total_items == 0:
        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_precision": 0,
            "context_recall": 0,
        }

    merged = {
        "faithfulness": 0,
        "answer_relevancy": 0,
        "context_precision": 0,
        "context_recall": 0,
    }

    for batch in batch_results:
        weight = batch.get("item_count", 0) / total_items
        for key in merged.keys():
            merged[key] += batch.get(key, 0) * weight

    return merged


def run_batched_evaluation(
    rag_file: Path,
    model_id: str,
    batch_size: int = 10,
    resume: bool = True,
    only_failed: bool = False
):
    """배치 단위 RAGAS 평가 실행"""

    print(f"\n{'='*60}")
    print(f"파일: {rag_file.name}")
    print(f"배치 크기: {batch_size}")
    print(f"{'='*60}")

    # RAG 결과 로드
    rag_results = load_rag_results(rag_file)
    items = rag_results["results"]
    total_items = len(items)
    total_batches = (total_items + batch_size - 1) // batch_size

    print(f"총 문항: {total_items}개")
    print(f"총 배치: {total_batches}개")

    # 기존 결과 로드
    existing_batches = {}
    if resume:
        existing_batches = load_existing_batches(rag_file.name)
        if existing_batches:
            print(f"기존 완료 배치: {len(existing_batches)}개 (이어서 진행)")

    # LLM, Embeddings 초기화 (한 번만)
    http_client = httpx.Client(
        timeout=httpx.Timeout(300.0, connect=60.0),
    )

    llm = ChatOpenAI(
        model=model_id,
        openai_api_base=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        openai_api_key="not-needed",
        temperature=0.0,
        max_tokens=2048,
        max_retries=3,
        http_client=http_client,
        model_kwargs={"stop": ["<|im_end|>"]},
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 배치별 처리
    all_batch_results = dict(existing_batches)  # 기존 결과 포함

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        batch_items = items[start_idx:end_idx]

        # 이미 완료된 배치 스킵
        if batch_idx in existing_batches and not only_failed:
            print(f"\n[배치 {batch_idx+1}/{total_batches}] 이미 완료됨 - 스킵")
            continue

        print(f"\n[배치 {batch_idx+1}/{total_batches}] 처리 중... ({start_idx+1}~{end_idx}번 문항)")

        # 배치 데이터셋 생성
        batch_dataset = prepare_ragas_dataset_from_items(batch_items)

        # 평가 실행
        result = run_batch_evaluation(batch_dataset, model_id, llm, embeddings)

        # 결과 저장
        save_batch_result(rag_file.name, batch_idx, result, batch_items)
        all_batch_results[batch_idx] = result

        if result["status"] == "success":
            print(f"  -> 완료! Faith:{result['faithfulness']:.3f} Rel:{result['answer_relevancy']:.3f}")
        else:
            print(f"  -> 실패: {result.get('error', 'unknown')}")

    # 최종 결과 합산
    successful_batches = [b for b in all_batch_results.values() if b.get("status") == "success"]

    if successful_batches:
        final_result = merge_batch_results(successful_batches)

        print(f"\n{'='*60}")
        print("최종 결과 (가중 평균)")
        print(f"{'='*60}")
        print(f"  완료된 배치: {len(successful_batches)}/{total_batches}")
        print(f"  Faithfulness:      {final_result['faithfulness']:.4f}")
        print(f"  Answer Relevancy:  {final_result['answer_relevancy']:.4f}")
        print(f"  Context Precision: {final_result['context_precision']:.4f}")
        print(f"  Context Recall:    {final_result['context_recall']:.4f}")

        # 최종 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_file = RESULTS_DIR / f"ragas_batch_final_{timestamp}.json"

        with open(final_file, 'w', encoding='utf-8') as f:
            json.dump({
                "rag_file": rag_file.name,
                "model": rag_results["metadata"]["model"],
                "judge_llm": model_id,
                "batch_size": batch_size,
                "total_items": total_items,
                "completed_batches": len(successful_batches),
                "total_batches": total_batches,
                "metrics": final_result,
                "timestamp": timestamp,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장: {final_file}")
        return final_result
    else:
        print("\n완료된 배치가 없습니다!")
        return None


def main():
    parser = argparse.ArgumentParser(description="RAGAS 배치 평가")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="배치 크기 (기본: 10)")
    parser.add_argument("--no-resume", action="store_true", help="처음부터 다시 시작")
    parser.add_argument("--retry-failed", action="store_true", help="실패한 배치만 재시도")
    parser.add_argument("--file", "-f", type=str, help="특정 RAG 결과 파일만 처리")
    args = parser.parse_args()

    print("="*60)
    print("RAGAS 배치 평가기")
    print("="*60)
    print(f"vLLM 서버: {VLLM_HOST}:{VLLM_PORT}")
    print(f"배치 크기: {args.batch_size}")
    print(f"이어서 진행: {'아니오' if args.no_resume else '예'}")

    # Judge LLM 확인
    model_id = get_model_id()

    # RAG 파일 찾기
    if args.file:
        rag_files = [RESULTS_DIR / args.file]
    else:
        rag_files = sorted(RESULTS_DIR.glob("rag_answers_*.json"))

    if not rag_files:
        print("\nRAG 답변 파일이 없습니다!")
        print("먼저 rag_answer_generator.py를 실행하세요.")
        return

    print(f"\n처리할 파일: {len(rag_files)}개")
    for f in rag_files:
        print(f"  - {f.name}")

    # 각 파일 처리
    for rag_file in rag_files:
        if not rag_file.exists():
            print(f"\n[SKIP] 파일 없음: {rag_file}")
            continue

        run_batched_evaluation(
            rag_file,
            model_id,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            only_failed=args.retry_failed
        )


if __name__ == "__main__":
    main()
