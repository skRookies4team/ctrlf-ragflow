import re
import logging
import json
from typing import List, Dict, Any

from preprocessing.ocr.ocr_cleaner import TextCleaner
from serialization.json_serializer import JSONSerializer

logger = logging.getLogger(__name__)

class Chunker:
    @staticmethod
    def chunk(text: str, max_len=1200, overlap=100) -> List[str]:
        cleaned = TextCleaner.clean_raw(text)

        # 1) semantic (제X조) 기준 split
        articles = re.split(r"\n{2,}(?=제\d+조)", cleaned)
        articles = [a.strip() for a in articles if a.strip()]

        logger.info(f"[Chunker] Detected article pieces: {len(articles)}")

        # 2) 과분할 방지 → size cap으로 의미 덩어리 묶기
        chunks, buf = [], ""
        for art in articles:
            if not buf:
                buf = art
            elif len(buf) + len(art) + 1 <= max_len:
                buf += " " + art
            else:
                chunks.append(buf)
                # overlap 유지하면서 다음 buf 시작
                buf = buf[-overlap:] + " " + art if overlap < len(buf) else art
        if buf:
            chunks.append(buf)

        logger.info(f"[Chunker] Final semantic chunks: {len(chunks)}")
        return chunks
