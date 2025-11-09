import os
from PIL import Image

TEMPLATES_DIR = "templates"
VARIANT_DIR = "template4"

ROTATIONS = list(range(0, 360, 15))  # 0, 15, 30, ..., 345
SCALES = [1.0]

def make_variants():
    os.makedirs(VARIANT_DIR, exist_ok=True)

    for filename in os.listdir(TEMPLATES_DIR):
        template_path = os.path.join(TEMPLATES_DIR, filename)
        if os.path.isfile(template_path):
            name, ext = os.path.splitext(filename)
            variant_folder = os.path.join(VARIANT_DIR, name)
            os.makedirs(variant_folder, exist_ok=True)

            try:
                img = Image.open(template_path)
            except Exception as e:
                print(f"Skipping {filename}: {e}")
                continue

            for rot in ROTATIONS:
                rotated = img.rotate(rot, expand=True)
                for scale in SCALES:
                    new_size = (int(rotated.width * scale), int(rotated.height * scale))
                    resized = rotated.resize(new_size, Image.Resampling.LANCZOS)
                    variant_name = f"{name}_rot{rot}_scale{scale}.jpg"
                    variant_path = os.path.join(variant_folder, variant_name)
                    resized.save(variant_path)
                    print(f"Saved {variant_path}")

if __name__ == "__main__":
    make_variants()