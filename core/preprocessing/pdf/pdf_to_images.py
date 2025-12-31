import fitz  # PyMuPDF
from PIL import Image


class PDFImageExtractor:
    @staticmethod
    def extract(pdf_path: str, dpi=300):
        """
        PDF 각 페이지를 고해상도 이미지(PIL.Image) 리스트로 반환
        OCR 품질 향상을 위해 DPI=300 기준으로 렌더링 수행
        """
        doc = fitz.open(pdf_path)
        imgs = []

        # 🔥 DPI → 확대 비율 계산 (PyMuPDF 기본 DPI 72)
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)

        for page in doc:
            # 🔥 get_pixmap(matrix=mat)를 사용해야 고 DPI 반영됨
            pix = page.get_pixmap(matrix=mat, alpha=False)

            # Pixmap → PIL.Image 변환
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

            imgs.append(img)

        return imgs
