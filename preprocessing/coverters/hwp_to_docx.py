from pathlib import Path
from .hwp_extract import convert_hwp_to_text
from .hwp_extract import _check_libreoffice_available, _get_soffice_cmd
import subprocess


class HwpAdapter:
    """
    통합 HWP 어댑터:
    - HWP → DOCX 변환 (파이프라인용)
    - 또는 HWP → 텍스트 직접 추출
    """

    def to_docx(self, hwp_path: str) -> str:
        """
        HWP → DOCX 변환 (파이프라인용)

        - 결과 DOCX는 HWP가 있는 dataset 폴더가 아니라
          sample 폴더 바로 아래의 hwptodocx 폴더에 저장함.
          예) sample/dataset/구매업무처리규정.hwp
              → sample/hwptodocx/구매업무처리규정.docx
        """
        hwp_path = Path(hwp_path).resolve()

        if not hwp_path.exists():
            raise FileNotFoundError(hwp_path)

        if not _check_libreoffice_available():
            raise RuntimeError("LibreOffice not available. Cannot convert HWP → DOCX.")

        cmd = _get_soffice_cmd()

        # ----------------------------------------
        # 1) 출력 폴더: sample/hwptodocx
        #    (hwp_path: .../sample/dataset/파일.hwp 기준)
        # ----------------------------------------
        # hwp_path.parents[0] = dataset
        # hwp_path.parents[1] = sample
        sample_dir = hwp_path.parents[1]
        out_dir = sample_dir / "hwptodocx"
        out_dir.mkdir(exist_ok=True)

        # ----------------------------------------
        # 2) LibreOffice로 HWP → DOCX 변환
        # ----------------------------------------
        result = subprocess.run(
            [
                cmd,
                "--headless",
                "--infilter=Hwp2002_File",
                "--convert-to", "docx",           # ★ 필터 이름 없이 docx로만
                str(hwp_path),
                "--outdir", str(out_dir),         # ★ 출력 위치 = hwptodocx
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        docx_file = out_dir / f"{hwp_path.stem}.docx"
        if not docx_file.exists():
            # 디버깅용으로 stdout/stderr 찍어두면 나중에 편함
            print("=== soffice stdout ===")
            print(result.stdout)
            print("=== soffice stderr ===")
            print(result.stderr)
            raise RuntimeError(f"Failed to convert HWP→DOCX: {result.stderr}")

        # 이제 변환된 DOCX 경로를 그대로 반환
        return str(docx_file)

    def to_text(self, hwp_path: str, method: str | None = None) -> str:
        """
        HWP → 텍스트 직접 추출 기능
        (테스트 / 유틸용)
        """
        return convert_hwp_to_text(hwp_path, method=method)
