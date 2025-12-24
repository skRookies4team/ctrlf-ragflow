import os
import shutil
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass


class HwpAdapter:
    def _get_soffice_cmd(self) -> str:
        env = os.getenv("SOFFICE_PATH") or os.getenv("LIBREOFFICE_PATH")
        if env:
            env = env.strip().strip('"')  # ✅ 공백/따옴표 제거
            if Path(env).exists():
                return env

        which = shutil.which("soffice") or shutil.which("soffice.com")
        if which:
            return which

        # ✅ com 우선
        candidates = [
            r"C:\Program Files\LibreOffice\program\soffice.com",
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.com",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c

        return "soffice"

    def _check_libreoffice_available(self) -> bool:
        try:
            cmd = self._get_soffice_cmd()
            r = subprocess.run(
                [cmd, "--version"],
                timeout=10,
                capture_output=True,
                text=True,
            )
            # 디버그 필요하면 아래 2줄만 잠깐 켜
            # print("[DEBUG] soffice stdout =", r.stdout.strip())
            # print("[DEBUG] soffice stderr =", r.stderr.strip())
            return r.returncode == 0
        except Exception:
            return False

    def to_docx(self, hwp_path: str) -> str:
        hwp_path = Path(hwp_path).resolve()
        if not hwp_path.exists():
            raise FileNotFoundError(hwp_path)

        cmd = self._get_soffice_cmd()
        print("[DEBUG] SOFFICE_PATH =", os.getenv("SOFFICE_PATH"))
        print("[DEBUG] cmd =", cmd)

        if not self._check_libreoffice_available():
            raise RuntimeError("LibreOffice not available. Cannot convert HWP → DOCX.")

        sample_dir = hwp_path.parents[1]
        out_dir = sample_dir / "hwptodocx"
        out_dir.mkdir(exist_ok=True)

        result = subprocess.run(
            [
                cmd,
                "--headless",
                "--convert-to", "docx",
                str(hwp_path),
                "--outdir", str(out_dir),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        docx_file = out_dir / f"{hwp_path.stem}.docx"
        if not docx_file.exists():
            print("=== soffice stdout ===")
            print(result.stdout)
            print("=== soffice stderr ===")
            print(result.stderr)
            raise RuntimeError(f"Failed to convert HWP→DOCX: {result.stderr}")

        return str(docx_file)
