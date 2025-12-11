"""
간단한 RAG 품질 평가 스크립트
- vLLM 서버의 Qwen2.5를 Judge LLM으로 직접 사용
- Faithfulness, Relevancy 등을 LLM에게 직접 평가 요청
"""

import os
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 경로 설정
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# vLLM 서버 설정
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "1237"))


def get_model_id() -> str:
    """vLLM 서버에서 실제 모델 ID 조회"""
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


def call_llm(model_id: str, prompt: str, max_tokens: int = 512) -> str:
    """vLLM 서버에 질의"""
    # Qwen 형식 프롬프트
    full_prompt = f"""<|im_start|>system
You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
Evaluate the given answer based on the provided criteria.
Always respond in the exact format requested.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

    payload = {
        "model": model_id,
        "prompt": full_prompt,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stop": ["<|im_end|>"],
    }

    try:
        response = requests.post(
            f"http://{VLLM_HOST}:{VLLM_PORT}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["text"].strip()
    except Exception as e:
        print(f"LLM 호출 오류: {e}")
        return ""


def evaluate_faithfulness(model_id: str, answer: str, contexts: List[str]) -> float:
    """답변이 컨텍스트에 충실한지 평가 (0-1)"""
    context_text = "\n---\n".join(contexts[:3]) if contexts else "컨텍스트 없음"

    prompt = f"""다음 답변이 주어진 컨텍스트에 충실한지 평가해주세요.

컨텍스트:
{context_text}

답변:
{answer}

평가 기준:
- 답변의 모든 주장이 컨텍스트에서 뒷받침되는가?
- 컨텍스트에 없는 정보를 지어내지 않았는가?

점수만 숫자로 답변하세요 (0.0 ~ 1.0):
- 1.0: 완전히 충실함
- 0.5: 부분적으로 충실함
- 0.0: 충실하지 않음

점수:"""

    response = call_llm(model_id, prompt, max_tokens=50)
    try:
        # 숫자만 추출
        score = float(''.join(c for c in response if c.isdigit() or c == '.'))
        return min(1.0, max(0.0, score))
    except:
        return 0.5


def evaluate_relevancy(model_id: str, question: str, answer: str) -> float:
    """답변이 질문에 관련있는지 평가 (0-1)"""
    prompt = f"""다음 답변이 질문에 적절하게 답하고 있는지 평가해주세요.

질문:
{question}

답변:
{answer}

평가 기준:
- 답변이 질문에 직접적으로 답하고 있는가?
- 답변이 질문과 관련 있는가?

점수만 숫자로 답변하세요 (0.0 ~ 1.0):
- 1.0: 매우 관련 있음
- 0.5: 부분적으로 관련 있음
- 0.0: 관련 없음

점수:"""

    response = call_llm(model_id, prompt, max_tokens=50)
    try:
        score = float(''.join(c for c in response if c.isdigit() or c == '.'))
        return min(1.0, max(0.0, score))
    except:
        return 0.5


def evaluate_correctness(model_id: str, answer: str, ground_truth: str) -> float:
    """답변이 정답과 일치하는지 평가 (0-1)"""
    prompt = f"""다음 답변이 모범답안과 얼마나 일치하는지 평가해주세요.

모범답안:
{ground_truth}

평가할 답변:
{answer}

평가 기준:
- 핵심 정보가 포함되어 있는가?
- 정보가 정확한가?

점수만 숫자로 답변하세요 (0.0 ~ 1.0):
- 1.0: 완전히 일치
- 0.5: 부분적으로 일치
- 0.0: 일치하지 않음

점수:"""

    response = call_llm(model_id, prompt, max_tokens=50)
    try:
        score = float(''.join(c for c in response if c.isdigit() or c == '.'))
        return min(1.0, max(0.0, score))
    except:
        return 0.5


def load_rag_results(file_path: Path) -> Dict:
    """RAG 답변 결과 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_model(rag_file: Path, judge_model_id: str, sample_size: int = 20) -> Dict:
    """단일 모델 평가"""
    rag_results = load_rag_results(rag_file)
    model_name = rag_results["metadata"]["model"]
    results = rag_results["results"][:sample_size]

    print(f"\n평가 대상: {model_name}")
    print(f"샘플 수: {len(results)}")

    faithfulness_scores = []
    relevancy_scores = []
    correctness_scores = []

    for i, item in enumerate(results):
        print(f"  [{i+1}/{len(results)}] 평가 중...")

        # Faithfulness 평가
        f_score = evaluate_faithfulness(
            judge_model_id,
            item["rag_answer"],
            item.get("contexts", [])
        )
        faithfulness_scores.append(f_score)

        # Relevancy 평가
        r_score = evaluate_relevancy(
            judge_model_id,
            item["question"],
            item["rag_answer"]
        )
        relevancy_scores.append(r_score)

        # Correctness 평가
        c_score = evaluate_correctness(
            judge_model_id,
            item["rag_answer"],
            item["ground_truth"]
        )
        correctness_scores.append(c_score)

        print(f"      F:{f_score:.2f} R:{r_score:.2f} C:{c_score:.2f}")

    return {
        "model": model_name,
        "sample_size": len(results),
        "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "relevancy": sum(relevancy_scores) / len(relevancy_scores),
        "correctness": sum(correctness_scores) / len(correctness_scores),
    }


def main():
    print("="*60)
    print("RAG 품질 평가기 (LLM-as-Judge)")
    print("="*60)

    # Judge LLM 모델 ID 확인
    judge_model_id = get_model_id()

    # RAG 답변 파일 찾기
    rag_files = sorted(RESULTS_DIR.glob("rag_answers_*.json"))

    if not rag_files:
        print("RAG 답변 파일이 없습니다!")
        return

    print(f"\n발견된 RAG 답변 파일: {len(rag_files)}개")

    all_results = []

    for rag_file in rag_files:
        print(f"\n{'='*60}")
        print(f"처리 중: {rag_file.name}")
        print(f"{'='*60}")

        result = evaluate_model(rag_file, judge_model_id, sample_size=130)
        all_results.append(result)

        print(f"\n결과:")
        print(f"  Faithfulness: {result['faithfulness']:.3f}")
        print(f"  Relevancy: {result['relevancy']:.3f}")
        print(f"  Correctness: {result['correctness']:.3f}")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"quality_evaluation_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "judge_llm": judge_model_id,
            "evaluations": all_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*80}")
    print("모델별 RAG 품질 지표 비교")
    print(f"{'='*80}")
    print(f"{'모델':<35} {'Faithfulness':>12} {'Relevancy':>12} {'Correctness':>12}")
    print("-" * 80)

    for result in all_results:
        model = result["model"].split("/")[-1][:30]
        print(f"{model:<35} {result['faithfulness']:>12.3f} {result['relevancy']:>12.3f} {result['correctness']:>12.3f}")

    print(f"\n결과 저장: {output_file}")


if __name__ == "__main__":
    main()
