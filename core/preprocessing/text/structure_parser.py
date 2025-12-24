# preprocessing/text/structure_parser.py
import re

class StructureParser:

    @staticmethod
    def parse(text: str):
        headings = re.findall(r'^(#+|\d+\.)\s*(.+)$', text, flags=re.MULTILINE)
        return {"headings": headings}
