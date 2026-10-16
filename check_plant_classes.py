
from pathlib import Path

dataset = Path(r"C:\Users\Sagar\Downloads\archive (5)\PlantVillage")

print("Checking available plant classes...\n")

if not dataset.exists():
    print("Dataset folder not found:", dataset)
else:
    for folder in sorted(dataset.iterdir()):
        if folder.is_dir():
            count = sum(
                1 for file in folder.rglob("*")
                if file.is_file()
                and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            )
            print(f"{folder.name}: {count} images")
