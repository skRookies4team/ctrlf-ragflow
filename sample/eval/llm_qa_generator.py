"""
LLM 모델별 Q&A 리스트 생성기
- 순수 LLM 답변 생성 (검색 없이)
- RAG 파이프라인 답변 생성 (검색 + 생성)

RAGAS 평가를 위한 데이터셋 생성용
"""

import os
import csv
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse
import time


# 테스트할 모델 목록
MODELS = {
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2-7b": "Qwen/Qwen2-7B-Instruct",
    "gemma3-12b": "google/gemma-3-12b-it",
}


class LLMQAGenerator:
    def __init__(
        self,
        vllm_host: str = None,
        vllm_port: int = None,
        model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    ):
        vllm_host = vllm_host or os.getenv("VLLM_HOST", "localhost")
        vllm_port = vllm_port or int(os.getenv("VLLM_PORT", "1237"))
        self.base_url = f"http://{vllm_host}:{vllm_port}/v1"
        self.vllm_url = f"{self.base_url}/completions"  # completions 엔드포인트 사용
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
                print(f"실제 모델 ID: {actual_id}")
                return actual_id
        except Exception as e:
            print(f"모델 ID 조회 실패: {e}")
        return self.model_name

    def load_questions(self, csv_path: str) -> list[dict]:
        """CSV에서 질문 세트 로드"""
        questions = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append({
                    "q_id": row["Q_ID"],
                    "domain": row["도메인"],
                    "role": row["사용자_롤"],
                    "difficulty": row["난이도"],
                    "question": row["질문"],
                    "ground_truth": row["모범답안"],
                    "source_doc": row["출처_문서_ID"],
                })
        return questions

    def _build_llama3_prompt(self, system_prompt: str, user_content: str) -> str:
        """Llama 3 Instruct 형식의 프롬프트 생성"""
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

    def _build_qwen_prompt(self, system_prompt: str, user_content: str) -> str:
        """Qwen Instruct 형식의 프롬프트 생성"""
        return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""

    def _build_gemma_prompt(self, system_prompt: str, user_content: str) -> str:
        """Gemma 형식의 프롬프트 생성"""
        return f"""<start_of_turn>user
{system_prompt}

{user_content}<end_of_turn>
<start_of_turn>model
"""

    def _build_prompt(self, system_prompt: str, user_content: str) -> str:
        """모델에 맞는 프롬프트 형식 선택"""
        model_lower = self.model_name.lower()
        if "llama" in model_lower:
            return self._build_llama3_prompt(system_prompt, user_content)
        elif "qwen" in model_lower:
            return self._build_qwen_prompt(system_prompt, user_content)
        elif "gemma" in model_lower:
            return self._build_gemma_prompt(system_prompt, user_content)
        else:
            # 기본: Llama 형식
            return self._build_llama3_prompt(system_prompt, user_content)

    def call_llm(
        self,
        question: str,
        context: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> str:
        """vLLM 서버에 질의"""

        if context:
            # RAG 모드: context 포함
            system_prompt = """당신은 회사 내규 및 정책에 대해 답변하는 AI 어시스턴트입니다.
주어진 참고 문서를 기반으로 질문에 정확하게 답변해주세요.
참고 문서에 없는 내용은 추측하지 말고, 문서에 기반한 답변만 제공하세요.
반드시 한국어로 답변해주세요."""

            user_content = f"""참고 문서:
{context}

질문: {question}

위 참고 문서를 기반으로 질문에 답변해주세요."""
        else:
            # 순수 LLM 모드: context 없이
            system_prompt = """당신은 회사 내규 및 정책에 대해 답변하는 AI 어시스턴트입니다.
질문에 대해 일반적인 기업 관행과 한국 노동법을 기반으로 답변해주세요.
반드시 한국어로 답변해주세요."""

            user_content = question

        # 모델에 맞는 프롬프트 생성
        prompt = self._build_prompt(system_prompt, user_content)

        payload = {
            "model": self.actual_model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["<|eot_id|>", "<|im_end|>", "<end_of_turn>"],
        }

        try:
            response = requests.post(
                self.vllm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"].strip()
        except requests.exceptions.RequestException as e:
            print(f"LLM 호출 오류: {e}")
            return f"ERROR: {str(e)}"

    def generate_pure_llm_answers(
        self,
        questions: list[dict],
        output_path: str,
    ) -> list[dict]:
        """순수 LLM 답변 생성 (검색 없이)"""
        results = []
        total = len(questions)

        print(f"\n{'='*60}")
        print(f"순수 LLM 답변 생성 시작")
        print(f"모델: {self.model_name}")
        print(f"총 질문 수: {total}")
        print(f"{'='*60}\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{total}] {q['q_id']}: {q['question'][:50]}...")

            start_time = time.time()
            answer = self.call_llm(q["question"])
            elapsed = time.time() - start_time

            result = {
                "q_id": q["q_id"],
                "domain": q["domain"],
                "role": q["role"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "source_doc": q["source_doc"],
                "llm_answer": answer,
                "response_time": round(elapsed, 2),
                "mode": "pure_llm",
                "model": self.model_name,
            }
            results.append(result)

            print(f"    응답 시간: {elapsed:.2f}s")
            print(f"    답변 길이: {len(answer)} chars\n")

        # 결과 저장
        self._save_results(results, output_path)
        return results

    def generate_rag_answers(
        self,
        questions: list[dict],
        output_path: str,
        retriever_func=None,
    ) -> list[dict]:
        """RAG 파이프라인 답변 생성 (검색 + 생성)"""
        results = []
        total = len(questions)

        print(f"\n{'='*60}")
        print(f"RAG 파이프라인 답변 생성 시작")
        print(f"모델: {self.model_name}")
        print(f"총 질문 수: {total}")
        print(f"{'='*60}\n")

        for i, q in enumerate(questions, 1):
            print(f"[{i}/{total}] {q['q_id']}: {q['question'][:50]}...")

            # 검색 수행
            start_time = time.time()

            if retriever_func:
                context, retrieved_docs = retriever_func(q["question"])
            else:
                # retriever가 없으면 ground_truth를 context로 사용 (테스트용)
                context = f"[참고] {q['ground_truth']}"
                retrieved_docs = [{"content": q["ground_truth"], "source": q["source_doc"]}]

            retrieval_time = time.time() - start_time

            # LLM 답변 생성
            llm_start = time.time()
            answer = self.call_llm(q["question"], context=context)
            llm_time = time.time() - llm_start

            total_time = time.time() - start_time

            result = {
                "q_id": q["q_id"],
                "domain": q["domain"],
                "role": q["role"],
                "difficulty": q["difficulty"],
                "question": q["question"],
                "ground_truth": q["ground_truth"],
                "source_doc": q["source_doc"],
                "retrieved_context": context,
                "retrieved_docs": retrieved_docs,
                "llm_answer": answer,
                "retrieval_time": round(retrieval_time, 2),
                "llm_time": round(llm_time, 2),
                "total_time": round(total_time, 2),
                "mode": "rag",
                "model": self.model_name,
            }
            results.append(result)

            print(f"    검색 시간: {retrieval_time:.2f}s | LLM 시간: {llm_time:.2f}s | 총: {total_time:.2f}s")
            print(f"    답변 길이: {len(answer)} chars\n")

        # 결과 저장
        self._save_results(results, output_path)
        return results

    def _save_results(self, results: list[dict], output_path: str):
        """결과를 JSON 파일로 저장"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model": self.model_name,
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(results),
            "results": results,
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"\n결과 저장 완료: {output_path}")


def get_model_short_name(model_name: str) -> str:
    """모델 이름에서 짧은 이름 추출"""
    for short, full in MODELS.items():
        if full == model_name:
            return short
    # 기본: 모델 이름에서 마지막 부분 사용
    return model_name.split("/")[-1].lower().replace("-", "_")


def main():
    parser = argparse.ArgumentParser(description="LLM 모델별 Q&A 생성기")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="사용할 모델 이름",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="vLLM 서버 호스트 (기본값: VLLM_HOST 환경변수 또는 localhost)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=None,
        help="vLLM 서버 포트 (기본값: VLLM_PORT 환경변수 또는 1237)",
    )
    parser.add_argument(
        "--questions", "-q",
        type=str,
        default="sample/eval/test_questions.csv",
        help="질문 CSV 파일 경로",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="sample/eval/results",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pure", "rag", "both"],
        default="pure",
        help="실행 모드: pure(순수 LLM), rag(RAG 파이프라인), both(둘 다)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="테스트할 질문 수 제한 (디버깅용)",
    )

    args = parser.parse_args()

    # 생성기 초기화
    generator = LLMQAGenerator(
        vllm_host=args.host,
        vllm_port=args.port,
        model_name=args.model,
    )

    # 질문 로드
    questions = generator.load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]

    print(f"로드된 질문 수: {len(questions)}")

    # 모델 짧은 이름
    model_short = get_model_short_name(args.model)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 순수 LLM 모드
    if args.mode in ["pure", "both"]:
        output_path = f"{args.output_dir}/{model_short}_pure_llm_{timestamp}.json"
        generator.generate_pure_llm_answers(questions, output_path)

    # RAG 모드
    if args.mode in ["rag", "both"]:
        output_path = f"{args.output_dir}/{model_short}_rag_{timestamp}.json"
        # TODO: RAGFlow retriever 연결 필요
        generator.generate_rag_answers(questions, output_path)


if __name__ == "__main__":
    main()
