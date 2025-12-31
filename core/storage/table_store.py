import json
from pathlib import Path
from typing import Any, Dict, List


class TableStore:
    def __init__(self, base_dir: Path):
        """
        base_dir 예:
        Data_Preprocessing/sample/storage
        """
        self.base_dir = base_dir
        self.tables_dir = self.base_dir / "tables"
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.tables_dir / "index.json"
        if not self.index_path.exists():
            self._save_index({})

    # -------------------------
    # 내부 유틸
    # -------------------------
    def _load_index(self) -> Dict[str, str]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _save_index(self, index: Dict[str, str]):
        self.index_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # -------------------------
    # 표 저장
    # -------------------------
    def save_table(
        self,
        table_id: str,
        doc: str,
        page: int | None,
        headers: List[str],
        rows: List[List[Any]],
    ) -> Path:
        table_data = {
            "table_id": table_id,
            "doc": doc,
            "page": page,
            "headers": headers,
            "rows": rows,
        }

        table_path = self.tables_dir / f"{table_id}.json"
        table_path.write_text(
            json.dumps(table_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        index = self._load_index()
        index[table_id] = table_path.name
        self._save_index(index)

        return table_path

    # -------------------------
    # 표 로드
    # -------------------------
    def load_table(self, table_id: str) -> Dict[str, Any]:
        index = self._load_index()
        if table_id not in index:
            raise KeyError(f"Table not found: {table_id}")

        table_path = self.tables_dir / index[table_id]
        return json.loads(table_path.read_text(encoding="utf-8"))
