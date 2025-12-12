"""
RAGAS 품질 평가 스크립트
- vLLM 서버의 Qwen2.5를 Judge LLM으로 사용
- HuggingFace 임베딩 모델 사용
- Faithfulness, Answer Relevancy, Context Precision 등 측정
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import httpx

# 환경 변수 설정 (RAGAS가 OpenAI 호환 API 사용하도록)
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
        # contexts는 리스트여야 함
        contexts = item.get("contexts", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        data["contexts"].append(contexts)
        data["ground_truth"].append(item["ground_truth"])

    return Dataset.from_dict(data)


def run_ragas_evaluation(dataset: Dataset, model_id: str, sample_size: int = None) -> Dict:
    """RAGAS 평가 실행"""

    # 샘플링 (전체 평가는 시간이 오래 걸림)
    if sample_size and len(dataset) > sample_size:
        dataset = dataset.select(range(sample_size))
        print(f"샘플 {sample_size}개로 평가 진행")

    print(f"\n{'='*60}")
    print(f"RAGAS 품질 평가 시작")
    print(f"평가 데이터: {len(dataset)}개")
    print(f"Judge LLM: {model_id}")
    print(f"{'='*60}\n")

    # vLLM을 OpenAI 호환으로 사용
    # 타임아웃을 크게 늘리고, 재시도 설정 추가
    http_client = httpx.Client(
        timeout=httpx.Timeout(300.0, connect=60.0),  # 전체 5분, 연결 1분
    )

    llm = ChatOpenAI(
        model=model_id,
        openai_api_base=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
        openai_api_key="not-needed",
        temperature=0.0,
        max_tokens=2048,
        max_retries=3,  # 최대 3회 재시도
        http_client=http_client,
        model_kwargs={
            "stop": ["<|im_end|>"],
        }
    )

    # HuggingFace 임베딩 모델 사용 (answer_relevancy에 필요)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 평가할 지표
    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    # 병렬 처리 제한 설정 (타임아웃 방지)
    run_config = RunConfig(
        max_workers=2,  # 동시 요청 2개로 제한 (기본값 16)
        max_wait=300,   # 최대 대기 시간 5분
        max_retries=3,  # 재시도 3회
        timeout=300,    # 개별 요청 타임아웃 5분
    )

    try:
        results = evaluate(
            dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
            run_config=run_config,  # 병렬 처리 제한 적용
        )
        # EvaluationResult 객체에서 결과 추출
        return {
            "faithfulness": results["faithfulness"] if "faithfulness" in results else 0,
            "answer_relevancy": results["answer_relevancy"] if "answer_relevancy" in results else 0,
            "context_precision": results["context_precision"] if "context_precision" in results else 0,
            "context_recall": results["context_recall"] if "context_recall" in results else 0,
        }
    except Exception as e:
        print(f"RAGAS 평가 오류: {e}")
        import traceback
        traceback.print_exc()
        # 오류 발생해도 부분 결과 반환 시도
        return {
            "faithfulness": 0,
            "answer_relevancy": 0,
            "context_precision": 0,
            "context_recall": 0,
            "error": str(e)
        }


def main():
    print("="*60)
    print("RAGAS 품질 평가기 (vLLM + Qwen2.5)")
    print("="*60)

    # Judge LLM 모델 ID 확인
    model_id = get_model_id()

    # RAG 답변 파일 찾기
    rag_files = sorted(RESULTS_DIR.glob("rag_answers_*.json"))

    if not rag_files:
        print("RAG 답변 파일이 없습니다!")
        return

    print(f"\n발견된 RAG 답변 파일: {len(rag_files)}개")
    for f in rag_files:
        print(f"  - {f.name}")

    all_results = []

    for rag_file in rag_files:
        print(f"\n{'='*60}")
        print(f"처리 중: {rag_file.name}")
        print(f"{'='*60}")

        # RAG 결과 로드
        rag_results = load_rag_results(rag_file)
        model_name = rag_results["metadata"]["model"]

        print(f"평가 대상 모델: {model_name}")

        # RAGAS 데이터셋 준비
        dataset = prepare_ragas_dataset(rag_results)

        # RAGAS 평가 실행 (130개 전체 평가)
        ragas_results = run_ragas_evaluation(dataset, model_id, sample_size=130)

        if ragas_results:
            # 결과 정리
            result_dict = {
                "model": model_name,
                "judge_llm": model_id,
                "sample_size": min(130, len(dataset)),
                "metrics": {
                    "faithfulness": float(ragas_results["faithfulness"]) if ragas_results["faithfulness"] else 0,
                    "answer_relevancy": float(ragas_results["answer_relevancy"]) if ragas_results["answer_relevancy"] else 0,
                    "context_precision": float(ragas_results["context_precision"]) if ragas_results["context_precision"] else 0,
                    "context_recall": float(ragas_results["context_recall"]) if ragas_results["context_recall"] else 0,
                }
            }
            all_results.append(result_dict)

            print(f"\n결과:")
            print(f"  Faithfulness: {result_dict['metrics']['faithfulness']:.3f}")
            print(f"  Answer Relevancy: {result_dict['metrics']['answer_relevancy']:.3f}")
            print(f"  Context Precision: {result_dict['metrics']['context_precision']:.3f}")
            print(f"  Context Recall: {result_dict['metrics']['context_recall']:.3f}")

    # 결과 저장
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"ragas_quality_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": timestamp,
                "judge_llm": model_id,
                "evaluations": all_results,
            }, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*60}")
        print(f"RAGAS 품질 평가 완료!")
        print(f"결과 파일: {output_file}")
        print(f"{'='*60}")

        # 비교 테이블 출력
        print(f"\n{'='*80}")
        print("모델별 RAGAS 품질 지표 비교")
        print(f"{'='*80}")
        print(f"{'모델':<35} {'Faithful':>10} {'Relevancy':>10} {'Precision':>10} {'Recall':>10}")
        print("-" * 80)

        for result in all_results:
            model = result["model"].split("/")[-1][:30]
            m = result["metrics"]
            print(f"{model:<35} {m['faithfulness']:>10.3f} {m['answer_relevancy']:>10.3f} {m['context_precision']:>10.3f} {m['context_recall']:>10.3f}")


if __name__ == "__main__":
    main()
