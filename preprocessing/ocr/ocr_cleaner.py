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


def clean_ocr_text(text: str) -> str:
    """
    OCR 노이즈 제거 + 문장 구조 보존 + LLM 교정 친화적 클리닝
    """
    if not text:
        return ""

    # -------------------------------
    # 1) 슬라이드/도형 노이즈 제거
    # -------------------------------
    noise_patterns = [
        r"\b[A-Z]{3,5}\b",                  # ASO, HES 등
        r"[│┃◆●■□▪▫▩▣▤▥▦▧▨◈○◎◇☆★▷▶]",     # 도형 문자
        r"[~_<>]{2,}",                      # 장식 기호
        r"[^\x00-\x7F]{1}\s?[^\x00-\x7F]{1}\s?[^\x00-\x7F]{1}"  # 깨진 한글 뭉치
    ]
    for pat in noise_patterns:
        text = re.sub(pat, " ", text)

    # -------------------------------
    # 2) 줄바꿈 보존한 채 불필요한 공백 정리
    # -------------------------------
    # 탭 → 공백
    text = text.replace("\t", " ")

    # 연속 공백 줄이기 (단, 줄바꿈은 유지)
    text = re.sub(r" {2,}", " ", text)

    # -------------------------------
    # 3) OCR에서 붙어버린 문장 패턴 복원
    # -------------------------------
    # 예: "직장 내 괴롭힘 금지 제도는무엇인가요?"
    text = re.sub(r"([가-힣])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # 숫자 + 문자 붙은 패턴
    text = re.sub(r"(\d)([가-힣A-Za-z])", r"\1 \2", text)

    # -------------------------------
    # 4) 한글 단어가 붙은 대표적 슬라이드 오타 패턴 (가벼운 자동 보정)
    # -------------------------------
    common_merge_patterns = [
        (r"직장내", "직장 내"),
        (r"괴롭힘금지", "괴롭힘 금지"),
        (r"예방및", "예방 및"),
        (r"사용자에대한", "사용자에 대한"),
        (r"발생시", "발생 시"),
        (r"직장내괴롭힘", "직장 내 괴롭힘")
    ]
    for wrong, right in common_merge_patterns:
        text = re.sub(wrong, right, text)

    # -------------------------------
    # 5) 문단 단위 정리
    # -------------------------------
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        l = line.strip()

        # 아무 내용 없는 줄은 문단 구분 유지
        if len(l) == 0:
            cleaned_lines.append("")
            continue

        # OCR 라인 조각 제거 (길이 1~2)
        if len(l) <= 2:
            continue

        cleaned_lines.append(l)

    # -------------------------------
    # 6) 문단 병합 (두 줄 띄어쓰기 유지)
    # -------------------------------
    paragraphs = []
    buf = []

    for line in cleaned_lines:
        if line == "":
            if buf:
                paragraphs.append(" ".join(buf))
                buf = []
        else:
            buf.append(line)

    if buf:
        paragraphs.append(" ".join(buf))

    return "\n\n".join(paragraphs).strip()
