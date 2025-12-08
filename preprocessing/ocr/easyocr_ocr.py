try:
    import easyocr
    reader = easyocr.Reader(['ko', 'en'])
    EASY_AVAILABLE = True
except:
    EASY_AVAILABLE = False

def run_easyocr(img):
    if not EASY_AVAILABLE:
        return ""
    result = reader.readtext(img, detail=0)
    return "\n".join(result)
