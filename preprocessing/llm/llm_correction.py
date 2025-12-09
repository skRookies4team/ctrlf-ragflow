# llm_correction.py
import requests
import json

LLM_SERVER = "http://your-qwen-server:8000/v1/chat/completions"
LLM_TOKEN = "INTERNAL_TOKEN"   # 필요 없으면 None

def llm_correct_text(text: str) -> str:
    """
    Qwen2.5-7B-Instruct를 이용한 완전 자동 문맥 기반 문장 교정기
    (오타, 잘못된 띄어쓰기, 단어 복원, 문장 재구성)
    """

    if not text or len(text.strip()) == 0:
        return text

    prompt = f"""
다음 텍스트는 OCR 결과이며 오타가 많습니다.
문맥에 맞게 자연스럽게 교정하세요.
내용(의미)은 변경하지 마세요.
문장 순서도 유지하세요.
출력은 '교정된 문장만' 주세요.

OCR 텍스트:
{text}
"""

    headers = {
        "Content-Type": "application/json"
    }
    if LLM_TOKEN:
        headers["Authorization"] = f"Bearer {LLM_TOKEN}"

    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [
            {"role": "system", "content": "너는 한국어 OCR 텍스트를 교정하는 전문가다."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 2048
    }

    try:
        res = requests.post(LLM_SERVER, headers=headers, json=payload, timeout=30)
        res.raise_for_status()
        corrected = res.json()["choices"][0]["message"]["content"]
        return corrected.strip()
    except Exception as e:
        print(f"[LLM Correction Error] {e}")
        return text  # 실패 시 원본 반환
