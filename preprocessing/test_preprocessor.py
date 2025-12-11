"""
RAGFlow 커스텀 청킹(Chunking) + add_chunk
HWP / PDF / PPT / DOCX / TXT / CSV 자동 처리 + 문서 타입 판별 + 자동 패턴 감지 완전판
"""

import os
import sys
import time
import requests
import re
import json
import csv
import pdfplumber
from pathlib import Path
from typing import List, Sequence
from dotenv import load_dotenv
from difflib import SequenceMatcher

# =======================
# 0. 경로/환경 설정
# =======================
BASE_DIR = Path(__file__).resolve().parent.parent

# ragflow 루트를 파이썬 모듈 경로에 추가 → preprocessing 패키지 import 가능
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

# =======================
# 1. RAGFlow SDK import
# =======================
try:
    from ragflow_sdk import RAGFlow
except ImportError:
    # sdk/python 폴더를 경로에 추가 후 재시도
    sys.path.insert(0, str(BASE_DIR / "sdk" / "python"))
    from ragflow_sdk import RAGFlow

from ragflow_sdk.modules.dataset import DataSet

# =======================
# 2. 커스텀 전처리 모듈 import
# =======================
from preprocessing.coverters.hwp_to_docx import HwpAdapter
from preprocessing.classifier.document_classifier import DocumentClassifier
from preprocessing.pipeline import PreprocessPipeline

def test_all():
    pdf = "sample.pdf"

    print("\n================= Preprocess Pipeline (V1) =================")
    pipeline = PreprocessPipeline()
    result = pipeline.run(pdf)

    # 기본 요약 출력
    print("\n===== RESULT SUMMARY =====")
    print("페이지 수:", result.get("page_count"))
    print("평균 품질:", result.get("avg_quality"))

    # 교정된 텍스트 미리보기
    print("\n===== FIRST PAGE PREVIEW =====")
    first_page = result["pages"][0]["text"]
    print(first_page[:500], "...\n")

    # JSON 저장 (원할 경우)
    with open("pipeline_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("JSON 저장 완료 → pipeline_output.json")

if __name__ == "__main__":
    test_all()
