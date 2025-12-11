"""
RAG 답변 생성 스크립트
- 저장된 검색 결과(retrieval_results)를 사용해서 LLM 답변 생성
"""

import os
import json
import requests
import time
from datetime import datetime
from pathlib import Path


# 경로 설정
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

# vLLM 서버 설정
VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = int(os.getenv("VLLM_PORT", "1237"))


class RAGAnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        self.base_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
        self.model_name = model_name
        self.actual_model_id = self._get_actual_model_id()

    def _get_actual_model_id(self) -> str:
        """vLLM 서버에서 실제 모델 ID 조회"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                actual_id = data["data"][0]["id"]
                print(f"모델 ID: {actual_id}")
                return actual_id
        except Exception as e:
            print(f"모델 ID 조회 실패: {e}")
        return self.model_name

    def _build_qwen_prompt(self, system_prompt: str, user_content: str) -> str:
        """Qwen Instruct 형식의 프롬프트 생성"""
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""

    def _build_llama3_prompt(self, system_prompt: str, user_content: str) -> str:
        """Llama 3 Instruct 형식의 프롬프트 생성"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def _build_prompt(self, system_prompt: str, user_content: str) -> str:
        """모델에 맞는 프롬프트 형식 선택"""
        if "llama" in self.actual_model_id.lower():
            return self._build_llama3_prompt(system_prompt, user_content)
        else:
            return self._build_qwen_prompt(system_prompt, user_content)

    def call_llm(self, question: str, context: str, max_tokens: int = 1024) -> str:
        """vLLM 서버에 RAG 질의"""
        system_prompt = """당신은 회사 내규 및 정책에 대해 답변하는 AI 어시스턴트입니다.
주어진 참고 문서를 기반으로 질문에 정확하게 답변해주세요.
참고 문서에 없는 내용은 추측하지 말고, 문서에 기반한 답변만 제공하세요.
반드시 한국어로 답변해주세요."""

        user_content = f"""참고 문서:
{context}

질문: {question}

위 참고 문서를 기반으로 질문에 답변해주세요."""

        prompt = self._build_prompt(system_prompt, user_content)

        # 모델별 stop 토큰
        if "llama" in self.actual_model_id.lower():
            stop_tokens = ["<|eot_id|>"]
        else:
            stop_tokens = ["<|im_end|>"]

        payload = {
            "model": self.actual_model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stop": stop_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return f"ERROR: {str(e)}"


def load_retrieval_results(file_path: Path) -> dict:
    """검색 결과 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # 가장 최근 검색 결과 파일 찾기
    retrieval_files = list(RESULTS_DIR.glob("retrieval_results_*.json"))
    if not retrieval_files:
        print("검색 결과 파일이 없습니다!")
        return

    latest_file = max(retrieval_files, key=lambda x: x.stat().st_mtime)
    print(f"검색 결과 파일: {latest_file.name}")

    # 검색 결과 로드
    data = load_retrieval_results(latest_file)
    questions = data["results"]

    print(f"\n{'='*60}")
    print("RAG 답변 생성 시작")
    print(f"총 질문 수: {len(questions)}")
    print(f"{'='*60}\n")

    # LLM 초기화
    generator = RAGAnswerGenerator()

    # RAG 답변 생성
    results = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['q_id']}: {q['question'][:50]}...")

        # 검색된 context 결합
        contexts = q.get("retrieved_contexts", [])
        if contexts:
            context = "\n\n---\n\n".join(contexts[:3])  # Top-3 사용
        else:
            context = "관련 문서를 찾을 수 없습니다."

        start_time = time.time()
        answer = generator.call_llm(q["question"], context)
        elapsed = time.time() - start_time

        result = {
            "q_id": q["q_id"],
            "domain": q["domain"],
            "question": q["question"],
            "ground_truth": q["ground_truth"],
            "contexts": contexts[:3],
            "rag_answer": answer,
            "response_time": round(elapsed, 2),
            "model": generator.model_name,
        }
        results.append(result)

        print(f"    응답 시간: {elapsed:.2f}s | 답변 길이: {len(answer)} chars")

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RESULTS_DIR / f"rag_answers_{timestamp}.json"

    output_data = {
        "metadata": {
            "model": generator.model_name,
            "total_questions": len(results),
            "timestamp": timestamp,
            "retrieval_file": latest_file.name,
        },
        "results": results,
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"RAG 답변 생성 완료!")
    print(f"결과 파일: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
