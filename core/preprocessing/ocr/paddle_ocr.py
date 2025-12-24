import logging
import numpy as np
import cv2
from paddleocr import PaddleOCR

logger = logging.getLogger("paddle_ocr")


# PaddleOCR 초기화 (show_log 제거!)
paddle = PaddleOCR(
    lang="korean",
    use_angle_cls=True
)


def run_paddle_ocr(img):
    """
    PaddleOCR 실행
    img: OpenCV BGR 배열 또는 PIL 이미지
    """
    try:
        if img is None:
            return ""

        # PIL → OpenCV 또는 OpenCV → 그대로
        if not isinstance(img, np.ndarray):
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        result = paddle.ocr(img, cls=True)

        if not result:
            return ""

        # 텍스트만 추출
        lines = []
        for res in result:
            for line in res:
                text = line[1][0]
                lines.append(text)

        return "\n".join(lines).strip()

    except Exception as e:
        logger.error(f"[Paddle OCR] 실패: {e}")
        return ""
