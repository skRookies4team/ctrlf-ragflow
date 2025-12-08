import re
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    @staticmethod
    def clean(raw: str) -> str:
        """기존 규칙 기반 클리닝 (법률/정관 등 일반 문서용)"""
        if not raw:
            return ""

        text = raw.replace("\u3000", " ")

        # 조문 경계 통일
        text = re.sub(r"\s*제\s*(\d+)\s*조", r"\n\n제\1조", text)

        # 괄호 내부 공백 정리
        text = re.sub(r"【\s*", "【", text)
        text = re.sub(r"\s*】", "】", text)

        # 다중 공백 축소 (줄바꿈 2개는 유지)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

        return text.strip()

    @staticmethod
    def clean_raw(raw: str) -> str:
        return TextCleaner.clean(raw)


# -------------------------------------------------------------
# 🔥 OCR 전용 클리너 (노이즈 제거 + 문단 복원)
# -------------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    """
    OCR 노이즈 제거 + 문단 복원 + 줄바꿈 정리
    """
    if not text:
        return ""

    # -------------------------------
    # 1) OCR 노이즈 패턴 제거
    # -------------------------------
    noise_patterns = [
        r"\b[A-Z]{2,5}\b",              # ASO, HES 같은 슬라이드 노이즈
        r"[-=]{2,}",                    # ----, ==== 등의 라인 노이즈
        r"[│┃◆●■□▪▫▩▣▤▥▦▧▨◈○◎◇☆★▷▶]"  # 도형 글자 제거
    ]

    for pat in noise_patterns:
        text = re.sub(pat, " ", text)

    # -------------------------------
    # 2) 여러 공백 하나로 정리
    # -------------------------------
    text = re.sub(r"\s+", " ", text)

    # -------------------------------
    # 3) 문단 단위로 자연스럽게 병합
    # -------------------------------
    lines = text.split("\n")
    merged = []
    buf = ""

    for line in lines:
        line = line.strip()

        if len(line) == 0:
            # 공백줄 → 하나의 문단 종료
            if buf:
                merged.append(buf)
                buf = ""
        else:
            # 문단 구성
            buf += " " + line

    # 마지막 문단 추가
    if buf:
        merged.append(buf)

    # 문단은 2줄 띄어쓰기
    return "\n\n".join(merged).strip()
