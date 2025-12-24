import uuid
import subprocess
import sys
from pathlib import Path

DATASET_STATUS = {}

BASE_DIR = Path(__file__).resolve().parents[3]
SAMPLE_MAIN = BASE_DIR / "sample" / "main.py"


class DatasetMgr:

    @staticmethod
    def save_dataset(file, dataset_name: str) -> str:
        dataset_id = str(uuid.uuid4())

        upload_dir = BASE_DIR / "sample" / "uploads" / dataset_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        file.save(file_path)

        DATASET_STATUS[dataset_id] = {
            "status": "uploaded",
            "file": str(file_path),
            "dataset_name": dataset_name,
        }

        return dataset_id

    @staticmethod
    def process_dataset(dataset_id: str):
        info = DATASET_STATUS.get(dataset_id)
        if not info:
            return

        DATASET_STATUS[dataset_id]["status"] = "processing"

        try:
            # ✅ sample/main.py 실행
            subprocess.run(
                [sys.executable, str(SAMPLE_MAIN)],
                cwd=str(SAMPLE_MAIN.parent),
                check=True
            )

            DATASET_STATUS[dataset_id]["status"] = "done"

        except Exception as e:
            DATASET_STATUS[dataset_id]["status"] = "failed"
            DATASET_STATUS[dataset_id]["error"] = str(e)

    @staticmethod
    def get_status(dataset_id: str):
        return DATASET_STATUS.get(dataset_id, {"status": "unknown"})
