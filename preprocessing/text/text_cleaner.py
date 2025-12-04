import re
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    @staticmethod
    def clean(raw: str) -> str:
        if not raw:
            return ""
        text = raw.replace("\u3000", " ")

        # 법률/정관 공백 깨짐 보정 + 조문 경계 통일
        text = re.sub(r"\s*제\s*(\d+)\s*조", r"\n\n제\1조", text)

        # 괄호 내부 공백만 최소 정리
        text = re.sub(r"【\s*", "【", text)
        text = re.sub(r"\s*】", "】", text)

        # 다중 공백 1개로 축소하되 \n\n은 유지
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # 단독 newline은 공백으로

        return text.strip()

    @staticmethod
    def clean_raw(raw: str) -> str:
        return TextCleaner.clean(raw)
