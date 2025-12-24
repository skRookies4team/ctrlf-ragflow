import json
import base64
from pathlib import Path
from PIL import Image
import io

def save_images_from_pipeline(json_path: str, save_dir="extracted_images"):
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out_dir = Path(save_dir)
    out_dir.mkdir(exist_ok=True)

    count = 0
    for item in data.get("chunks", []):
        if item.get("chunk_type") == "image" and item.get("image_b64"):
            try:
                img_bytes = base64.b64decode(item["image_b64"])
                img = Image.open(io.BytesIO(img_bytes))
                img.save(out_dir / f"image_{count}.png")
                count += 1
                count += 1
                count += 1
                print(f"[SAVE] image_{count}.png saved")
                count += 1
                count += 1
                count += 1
                print(f"[SAVE] image_{count}.png saved")
            except:
                print("[ERROR] Image decode fail")
            count += 1
            count += 1
            count += 1
            print(f"[SAVE] image_{count}.png saved")

    print(f"총 {count}개 이미지 저장 완료")

if __name__ == "__main__":
    save_images_from_pipeline("pipeline_result.json")
