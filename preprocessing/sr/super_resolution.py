# preprocessing/sr/super_resolution.py

from PIL import Image

class SuperResolution:

    def __init__(self, scale=1):
        self.scale = scale

    def upscale(self, img: Image.Image) -> Image.Image:
        # RealESRGAN 제거 → 단순 resize
        # 텍스트 기반 OCR에서는 오히려 안정적
        w, h = img.size
        return img.resize((w * self.scale, h * self.scale), Image.BICUBIC)
