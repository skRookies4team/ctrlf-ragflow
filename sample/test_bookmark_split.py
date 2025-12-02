import os
import re
from typing import List, Tuple

import fitz               # PyMuPDF  (책갈피 + 페이지 번호용)
from pypdf import PdfReader, PdfWriter   # pip install pypdf


def sanitize_title(title: str) -> str:
    """파일명으로 쓰기 좋게 제목 정리."""
    title = title.strip()
    # 맨 앞 번호 '1. 정관' 같은 거 제거
    title = re.sub(r'^\d+[\.\)]\s*', '', title)
    title = re.sub(r'\s+', ' ', title)
    title = re.sub(r'[\\/:*?"<>|]', '_', title)
    return title or "untitled"


def get_sections_from_bookmarks(doc: "fitz.Document",
                                min_level: int = 3) -> List[Tuple[str, int, int]]:
    """
    책갈피 기반 섹션 목록 생성.
    return: [(title, start_page, end_page), ...]   // page는 0-based
    """
    toc = doc.get_toc()
    if not toc:
        raise RuntimeError("책갈피(Toc)가 없는 PDF입니다.")

    # level >= 3 인 것만 “규정 하나”라고 본다 (지금 CTRLF 사규 구조 기준)
    filtered = [(lvl, title, page) for (lvl, title, page) in toc if lvl >= min_level]
    if not filtered:
        raise RuntimeError(f"level>={min_level} 책갈피가 없습니다. min_level을 조정해보세요.")

    sections = []
    page_count = doc.page_count

    for idx, (lvl, title, page) in enumerate(filtered):
        start_page = max(page - 1, 0)   # 1-based → 0-based
        if idx < len(filtered) - 1:
            _, _, next_page = filtered[idx + 1]
            end_page = max(min(next_page - 2, page_count - 1), start_page)
        else:
            end_page = page_count - 1

        sections.append((title, start_page, end_page))

    return sections


def split_pdf_by_bookmarks_to_pdfs(pdf_path: str,
                                   output_dir: str = "./CTRLF_rules_pdfs",
                                   min_level: int = 3) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # 1) PyMuPDF로 책갈피 + 페이지 범위 계산
    doc = fitz.open(pdf_path)
    sections = get_sections_from_bookmarks(doc, min_level=min_level)
    doc.close()

    # 2) pypdf로 실제 PDF를 규정별로 잘라내기
    reader = PdfReader(pdf_path)

    print(f"[INFO] 규정 섹션 {len(sections)}개 발견")

    for idx, (title, start_page, end_page) in enumerate(sections, start=1):
        pretty_title = sanitize_title(title)
        index_str = f"{idx:03d}"
        out_name = f"{index_str}_{pretty_title}.pdf"
        out_path = os.path.join(output_dir, out_name)

        writer = PdfWriter()
        for p in range(start_page, end_page + 1):
            # pypdf는 0-based 페이지 인덱스
            if 0 <= p < len(reader.pages):
                writer.add_page(reader.pages[p])

        with open(out_path, "wb") as f:
            writer.write(f)

        print(f"[{index_str}] '{title}' -> pages {start_page+1}~{end_page+1} -> {out_name}")

    print(f"\n[DONE] '{output_dir}' 폴더에 규정별 PDF가 생성되었습니다.")
    

if __name__ == "__main__":
    pdf_path = "./CTRLF 사규.pdf"   # 파일 이름 맞게
    split_pdf_by_bookmarks_to_pdfs(pdf_path)
