import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
]


def create_subset():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        src = RAW_DIR / cls
        dst = OUT_DIR / cls

        if not src.exists():
            print(f"Missing: {src}")
            continue

        dst.mkdir(parents=True, exist_ok=True)

        images = (
            list(src.glob("*.jpg")) + list(src.glob("*.JPG")) + list(src.glob("*.png"))
        )

        # take only first 500 images per class (keep it small)
        for img in images[:500]:
            shutil.copy(img, dst / img.name)

        print(f"{cls}: {min(len(images), 500)} images copied")


if __name__ == "__main__":
    create_subset()
