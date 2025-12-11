# vLLM 서버 연결 가이드

## 환경변수 설정

`.env` 파일에 다음 설정을 추가하세요:
```bash
VLLM_HOST=your-server-ip
VLLM_PORT=1237
```

## 1. 연결 확인

```cmd
curl http://${VLLM_HOST}:${VLLM_PORT}/v1/models
```

---

## 2. LLM 텍스트 생성

### Python
```python
import os
from openai import OpenAI

VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "1237")

client = OpenAI(
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="not-needed"
)

response = client.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    prompt="안녕하세요",
    max_tokens=100
)

print(response.choices[0].text)
```

---

## 3. 임베딩 생성

### Python
```python
import os
from openai import OpenAI

VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "1237")

client = OpenAI(
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="not-needed"
)

response = client.embeddings.create(
    model="jhgan/ko-sroberta-multitask",
    input="테스트 문장"
)

print(response.data[0].embedding)
```

---

## 4. 모델 ID 모를 때

```python
import os
import requests

VLLM_HOST = os.getenv("VLLM_HOST", "localhost")
VLLM_PORT = os.getenv("VLLM_PORT", "1237")

resp = requests.get(f"http://{VLLM_HOST}:{VLLM_PORT}/v1/models")
model_id = resp.json()["data"][0]["id"]
print(model_id)
```

---

## 5. 필요한 패키지

```bash
pip install openai requests python-dotenv
```
