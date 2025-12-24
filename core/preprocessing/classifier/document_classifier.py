import fitz  # PyMuPDF

class DocumentClassifier:
    """
    문서 타입 + PDF 내부 이미지 여부까지 판단하는 강화된 분류기
    """

    def classify(self, file_path: str) -> str:
        """
        분류 결과:
        - "hwp"
        - "ppt"
        - "image_pdf"  → 이미지 포함 PDF → 전처리
        - "text_pdf"   → 텍스트 중심 PDF → RAG 직행
        """
        ext = file_path.split(".")[-1].lower()

        if ext == "hwp":
            return "hwp"
        elif ext in ("ppt", "pptx"):
            return "ppt"
        elif ext == "pdf":
            return self._classify_pdf(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    def _classify_pdf(self, file_path: str) -> str:
        """
        PDF 내부 이미지 비율 기반 분류
        """
        try:
            doc = fitz.open(file_path)
        except Exception:
            return "image_pdf"  # 열리지도 않으면 이미지 기반으로 간주

        total_pages = len(doc)
        image_pages = 0
        text_pages = 0

        for page in doc:
            text = page.get_text().strip()
            images = page.get_images(full=True)

            if len(images) > 0:
                image_pages += 1
            if len(text) > 20:
                text_pages += 1

        # ─────────────────────────
        # 분류 규칙
        # ─────────────────────────
        if image_pages == 0:
            return "text_pdf"

        image_ratio = image_pages / total_pages

        # 이미지 비율이 높으면 → 이미지 기반 PDF
        if image_ratio >= 0.4:
            return "image_pdf"

        # 텍스트도 어느 정도 있지만 이미지도 있음 → 기본적으로 OCR 필요
        if text_pages < image_pages:
            return "image_pdf"

        return "text_pdf"
