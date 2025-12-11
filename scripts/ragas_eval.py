"""
RAGFlow + RAGAs 연동 평가 스크립트

사용법:
    # 외부 API (OpenAI) 사용
    python scripts/ragas_eval.py --api-key YOUR_RAGFLOW_API_KEY --chat-id YOUR_CHAT_ID

    # 로컬 LLM (Ollama) 사용 - 폐쇄망용
    python scripts/ragas_eval.py --api-key YOUR_API_KEY --chat-id YOUR_CHAT_ID --local-llm --ollama-url http://localhost:11434

    # 테스트 질문 파일 사용
    python scripts/ragas_eval.py --api-key YOUR_API_KEY --chat-id YOUR_CHAT_ID --questions questions.json

설치:
    pip install ragas datasets langchain-community requests
"""

import argparse
import json
import requests
from typing import Optional
from datetime import datetime


def install_dependencies():
    """필요한 패키지 설치 안내"""
    print("""
필요한 패키지를 설치하세요:
    pip install ragas datasets langchain-community requests

폐쇄망에서 Ollama 사용시:
    pip install langchain-ollama
""")


try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
except ImportError:
    install_dependencies()
    exit(1)


class RAGFlowClient:
    """RAGFlow API 클라이언트"""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def chat(self, chat_id: str, question: str, session_id: Optional[str] = None) -> dict:
        """RAGFlow 채팅 API 호출"""
        url = f"{self.base_url}/v1/chats/{chat_id}/completions"

        payload = {
            "question": question,
            "stream": False
        }
        if session_id:
            payload["session_id"] = session_id

        try:
            resp = requests.post(url, headers=self.headers, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"API 호출 실패: {e}")
            return None

    def extract_response(self, api_response: dict) -> dict:
        """API 응답에서 답변과 컨텍스트 추출"""
        if not api_response or "data" not in api_response:
            return {"answer": "", "contexts": []}

        data = api_response.get("data", {})
        answer = data.get("answer", "")

        # 컨텍스트 추출
        contexts = []
        reference = data.get("reference", {})
        chunks = reference.get("chunks", [])

        for chunk in chunks:
            content = chunk.get("content", "") or chunk.get("content_with_weight", "")
            if content:
                contexts.append(content)

        return {
            "answer": answer,
            "contexts": contexts
        }


class RAGAsEvaluator:
    """RAGAs 평가기"""

    def __init__(self, use_local_llm: bool = False, ollama_url: str = "http://localhost:11434", model_name: str = "llama3.1:8b"):
        self.use_local_llm = use_local_llm
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.llm = None
        self.embeddings = None

        if use_local_llm:
            self._setup_local_llm()

    def _setup_local_llm(self):
        """로컬 LLM 설정 (Ollama)"""
        try:
            from langchain_ollama import OllamaLLM, OllamaEmbeddings

            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=self.ollama_url,
            )
            self.embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.ollama_url,
            )
            print(f"✅ 로컬 LLM 설정 완료: {self.model_name} @ {self.ollama_url}")
        except ImportError:
            print("langchain-ollama 설치 필요: pip install langchain-ollama")
            exit(1)
        except Exception as e:
            print(f"로컬 LLM 설정 실패: {e}")
            exit(1)

    def evaluate(self, questions: list, answers: list, contexts: list, ground_truths: list = None) -> dict:
        """RAGAs 평가 실행"""

        # 데이터셋 구성
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        # 사용할 메트릭 선택
        metrics = [faithfulness, answer_relevancy, context_precision]

        # ground_truth가 있으면 context_recall도 측정
        if ground_truths and all(gt for gt in ground_truths):
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)

        print(f"\n📊 평가 시작 (질문 {len(questions)}개)...")
        print(f"   메트릭: {[m.name for m in metrics]}")

        # 평가 실행
        if self.use_local_llm and self.llm:
            result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
        else:
            # OpenAI API 사용 (OPENAI_API_KEY 환경변수 필요)
            result = evaluate(
                dataset,
                metrics=metrics,
            )

        return result


def load_questions(file_path: str) -> list:
    """질문 파일 로드 (JSON 형식)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 다양한 형식 지원
    if isinstance(data, list):
        if isinstance(data[0], str):
            return [{"question": q} for q in data]
        return data
    elif isinstance(data, dict) and "questions" in data:
        return data["questions"]

    return data


def save_results(results: dict, output_path: str):
    """결과 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 결과 저장: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAGFlow + RAGAs 평가 스크립트")

    # 필수 인자
    parser.add_argument("--api-key", required=True, help="RAGFlow API 키")
    parser.add_argument("--chat-id", required=True, help="RAGFlow Chat ID")

    # 선택 인자
    parser.add_argument("--base-url", default="http://localhost:9380", help="RAGFlow API URL")
    parser.add_argument("--questions", help="질문 파일 경로 (JSON)")
    parser.add_argument("--output", help="결과 저장 경로")

    # 로컬 LLM 옵션
    parser.add_argument("--local-llm", action="store_true", help="로컬 LLM 사용 (Ollama)")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama 서버 URL")
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama 모델명")

    args = parser.parse_args()

    # 기본 테스트 질문
    default_questions = [
        {"question": "RAGFlow의 주요 기능은 무엇인가요?"},
        {"question": "문서 청킹은 어떻게 동작하나요?"},
        {"question": "지원하는 파일 형식은 무엇인가요?"},
    ]

    # 질문 로드
    if args.questions:
        questions_data = load_questions(args.questions)
    else:
        print("⚠️  질문 파일 미지정. 기본 테스트 질문 사용.")
        questions_data = default_questions

    # RAGFlow 클라이언트 초기화
    client = RAGFlowClient(args.base_url, args.api_key)

    # RAGAs 평가기 초기화
    evaluator = RAGAsEvaluator(
        use_local_llm=args.local_llm,
        ollama_url=args.ollama_url,
        model_name=args.model,
    )

    # 데이터 수집
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("\n🔍 RAGFlow에서 응답 수집 중...")

    for i, q_data in enumerate(questions_data):
        question = q_data["question"] if isinstance(q_data, dict) else q_data
        ground_truth = q_data.get("ground_truth", "") if isinstance(q_data, dict) else ""

        print(f"   [{i+1}/{len(questions_data)}] {question[:50]}...")

        # RAGFlow API 호출
        response = client.chat(args.chat_id, question)
        result = client.extract_response(response)

        if result["answer"]:
            questions.append(question)
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(ground_truth)
        else:
            print(f"   ⚠️  응답 없음: {question[:30]}...")

    if not questions:
        print("❌ 수집된 응답이 없습니다.")
        return

    # RAGAs 평가 실행
    try:
        scores = evaluator.evaluate(questions, answers, contexts, ground_truths)

        # 결과 출력
        print("\n" + "="*60)
        print("📊 RAGAs 평가 결과")
        print("="*60)

        for metric, score in scores.items():
            if isinstance(score, (int, float)):
                print(f"   {metric}: {score:.4f}")

        print("="*60)

        # 결과 저장
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ragas_result_{timestamp}.json"

        save_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": args.base_url,
                "chat_id": args.chat_id,
                "local_llm": args.local_llm,
                "model": args.model if args.local_llm else "openai",
            },
            "scores": {k: float(v) if isinstance(v, (int, float)) else v for k, v in scores.items()},
            "details": [
                {
                    "question": q,
                    "answer": a[:200] + "..." if len(a) > 200 else a,
                    "context_count": len(c),
                }
                for q, a, c in zip(questions, answers, contexts)
            ]
        }

        save_results(save_data, output_path)

    except Exception as e:
        print(f"❌ 평가 실패: {e}")
        raise


if __name__ == "__main__":
    main()
