import cv2
import numpy as np
import easyocr
import pytesseract

# EasyOCR 한국어 엔진 준비
easy_reader = easyocr.Reader(["ko", "en"], gpu=False)

def ocr_tesseract(img, heavy=False):
    config = "--psm 6"
    if heavy:
        config += " -c tessedit_char_blacklist=§¶•—–“”…"
    return pytesseract.image_to_string(img, lang="kor+eng", config=config)

def ocr_easyocr(img):
    result = easy_reader.readtext(img)
    return " ".join([x[1] for x in result])

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def sharpen(img):
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    return cv2.filter2D(img, -1, kernel)

def binarize_otsu(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def adaptive_binary(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 35, 5)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

def denoise(img):
    return cv2.medianBlur(img, 3)

def super_auto_preprocess(img):
    preprocess_candidates = {
        "original": img,
        "sharpen": sharpen(img),
        "clahe": apply_clahe(img),
        "binary_otsu": binarize_otsu(img),
        "binary_adapt": adaptive_binary(img),
        "denoise": denoise(img),
    }

    engines = ["tesseract_light", "tesseract_heavy", "easyocr"]

    best_text = ""
    best_info = {"engine": None, "method": None, "len": 0}

    for method, processed in preprocess_candidates.items():
        for engine in engines:
            if engine == "tesseract_light":
                text = ocr_tesseract(processed, heavy=False)
            elif engine == "tesseract_heavy":
                text = ocr_tesseract(processed, heavy=True)
            else:
                text = ocr_easyocr(processed)

            text_len = len(text.strip())

            if text_len > best_info["len"]:
                best_info = {
                    "engine": engine,
                    "method": method,
                    "len": text_len
                }
                best_text = text

    return best_text, best_info
