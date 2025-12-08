import cv2
import numpy as np

def preprocess_image(img):
    """
    슬라이드형 PDF에서 OCR 성능 극대화를 위한 이미지 전처리.
    """
    # 1) Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Noise 제거 + Sharpen
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # 3) Adaptive Threshold
    th = cv2.adaptiveThreshold(
        sharpen, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15, 10
    )

    # 4) Morphology - 글씨 두껍게
    kernel = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    # 5) Deskew (기울기 보정)
    coords = np.column_stack(np.where(morph > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = morph.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    deskewed = cv2.warpAffine(morph, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)

    return deskewed
