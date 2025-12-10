import re
import logging

logger = logging.getLogger(__name__)


def clean_ocr_text(text: str) -> str:
    """
    OCR 텍스트를 안전하게 클리닝하는 버전.
    - 절대 문장 삭제 금지
    - 불필요한 장식·도형만 제거
    - 개행 및 문단은 유지
    """

    if not text:
        return ""

    # 1) 도형 / 장식 기호 제거 (본문 한글에는 영향 없음)
    text = re.sub(r"[│┃◆●■□▪▫▣▤▥▦◈○◎◇☆★▷▶]", " ", text)
    text = re.sub(r"[~_<>{}=]{2,}", " ", text)

    # 2) OCR 개행은 최대한 유지
    text = text.replace("\t", " ")

    # 3) 중복 공백 줄이기 (개행은 유지)
    text = re.sub(r" {2,}", " ", text)

    # 4) 한글-숫자/영문 붙은 경우만 최소한으로 분리
    text = re.sub(r"([0-9])([가-힣A-Za-z])", r"\1 \2", text)

    # 5) 짧은 라인/노이즈 삭제 절대 금지 → 문단 유지
    lines = text.split("\n")
    cleaned_lines = [l.rstrip() for l in lines]

    return "\n".join(cleaned_lines).strip()
