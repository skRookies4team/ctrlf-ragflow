FROM python:3.10-slim

# ---- OS deps (OCR / LibreOffice / PDF) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    tesseract-ocr \
    tesseract-ocr-kor \
    poppler-utils \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
  && rm -rf /var/lib/apt/lists/*


# ---- workspace code ----
WORKDIR /workspace

# 1) worker requirements
COPY workers/ingest-worker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 2) RAGFlow SDK (repo 내부 SDK를 이미지에 포함 + editable install)
COPY libs/sdk/python /opt/ragflow_sdk
RUN pip install --no-cache-dir -e /opt/ragflow_sdk

# 3) 너희 파이프라인 코드 포함 (서버에서도 동일 동작)
COPY sample /workspace/sample
COPY core /workspace/core
COPY libs /workspace/libs
COPY apps /workspace/apps

# 4) worker app
COPY workers/ingest-worker/app.py /workspace/app.py

ENV PYTHONUNBUFFERED=1

EXPOSE 9001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9001"]
