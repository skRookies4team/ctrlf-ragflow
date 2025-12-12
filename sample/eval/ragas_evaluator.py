"""
RAGAS 평가 스크립트
- RAG 답변 결과를 RAGAS 지표로 평가
- Faithfulness, Answer Relevancy, Context Precision 등 측정
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import os

# RAGAS 관련 import
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False
    print("RAGAS 미설치. pip install ragas datasets")

# OpenAI API 설정 (RAGAS 내부에서 사용)
# vLLM 서버를 OpenAI 호환으로 사용
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "1237")
os.environ["OPENAI_API_BASE"] = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
os.environ["OPENAI_API_KEY"] = "not-needed"

# 경로 설정
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def load_rag_results(file_path: Path) -> Dict:
    """RAG 답변 결과 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_dataset(rag_results: Dict) -> Dataset:
    """RAGAS 평가용 데이터셋 준비"""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for item in rag_results["results"]:
        data["question"].append(item["question"])
        data["answer"].append(item["rag_answer"])
        data["contexts"].append(item.get("contexts", []))
        data["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(data)


def evaluate_with_ragas(dataset: Dataset, model_name: str) -> Dict:
    """RAGAS 평가 실행"""
    print(f"\n{'='*60}")
    print(f"RAGAS 평가 시작: {model_name}")
    print(f"{'='*60}\n")

    # 평가할 지표 선택
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    try:
        results = evaluate(dataset, metrics=metrics)
        return results.to_pandas().to_dict()
    except Exception as e:
        print(f"RAGAS 평가 오류: {e}")
        return None


def simple_evaluation(rag_results: Dict) -> Dict:
    """간단한 자체 평가 (RAGAS 없이)"""
    results = rag_results["results"]
    model_name = rag_results["metadata"]["model"]

    print(f"\n{'='*60}")
    print(f"자체 평가: {model_name}")
    print(f"{'='*60}\n")

    total = len(results)

    # 기본 통계
    total_response_time = sum(r["response_time"] for r in results)
    avg_response_time = total_response_time / total

    total_answer_length = sum(len(r["rag_answer"]) for r in results)
    avg_answer_length = total_answer_length / total

    # 답변 품질 간단 체크 (에러 답변, 빈 답변 등)
    error_count = sum(1 for r in results if "ERROR" in r["rag_answer"] or len(r["rag_answer"]) < 10)
    success_rate = (total - error_count) / total * 100

    # 도메인별 통계
    domain_stats = {}
    for r in results:
        domain = r["domain"]
        if domain not in domain_stats:
            domain_stats[domain] = {"count": 0, "total_time": 0, "total_length": 0}
        domain_stats[domain]["count"] += 1
        domain_stats[domain]["total_time"] += r["response_time"]
        domain_stats[domain]["total_length"] += len(r["rag_answer"])

    for domain in domain_stats:
        count = domain_stats[domain]["count"]
        domain_stats[domain]["avg_time"] = domain_stats[domain]["total_time"] / count
        domain_stats[domain]["avg_length"] = domain_stats[domain]["total_length"] / count

    return {
        "model": model_name,
        "total_questions": total,
        "avg_response_time": round(avg_response_time, 2),
        "avg_answer_length": round(avg_answer_length, 0),
        "success_rate": round(success_rate, 1),
        "error_count": error_count,
        "domain_stats": domain_stats,
    }


def main():
    # RAG 답변 파일 찾기
    rag_files = sorted(RESULTS_DIR.glob("rag_answers_*.json"))

    if not rag_files:
        print("RAG 답변 파일이 없습니다!")
        return

    print(f"발견된 RAG 답변 파일: {len(rag_files)}개")
    for f in rag_files:
        print(f"  - {f.name}")

    all_results = []

    for rag_file in rag_files:
        print(f"\n처리 중: {rag_file.name}")

        # RAG 결과 로드
        rag_results = load_rag_results(rag_file)
        model_name = rag_results["metadata"]["model"]

        # 간단한 자체 평가
        eval_result = simple_evaluation(rag_results)
        all_results.append(eval_result)

        print(f"\n  모델: {model_name}")
        print(f"  총 질문: {eval_result['total_questions']}")
        print(f"  평균 응답시간: {eval_result['avg_response_time']}s")
        print(f"  평균 답변길이: {eval_result['avg_answer_length']} chars")
        print(f"  성공률: {eval_result['success_rate']}%")

    # 결과 비교 출력
    print(f"\n\n{'='*80}")
    print("모델별 RAG 성능 비교")
    print(f"{'='*80}")
    print(f"{'모델':<35} {'응답시간':>10} {'답변길이':>10} {'성공률':>10}")
    print("-" * 80)

    for result in all_results:
        model = result["model"][:35]
        print(f"{model:<35} {result['avg_response_time']:>10.2f}s {result['avg_answer_length']:>10.0f} {result['success_rate']:>9.1f}%")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"ragas_evaluation_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "evaluations": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n평가 결과 저장: {output_file}")

    # 도메인별 상세 분석
    print(f"\n\n{'='*80}")
    print("도메인별 성능 분석")
    print(f"{'='*80}")

    # 모든 도메인 수집
    all_domains = set()
    for result in all_results:
        all_domains.update(result["domain_stats"].keys())

    for domain in sorted(all_domains):
        print(f"\n[{domain}]")
        for result in all_results:
            model = result["model"].split("/")[-1][:20]
            stats = result["domain_stats"].get(domain, {})
            if stats:
                print(f"  {model}: 응답시간 {stats['avg_time']:.2f}s, 답변길이 {stats['avg_length']:.0f} chars")


if __name__ == "__main__":
    main()
