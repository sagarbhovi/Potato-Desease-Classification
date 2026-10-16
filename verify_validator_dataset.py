
from pathlib import Path

dataset = Path("leaf_validator_dataset")
classes = ["potato_leaf", "not_potato_leaf"]

print("Validator Dataset Verification")
print("-" * 35)

for class_name in classes:
    folder = dataset / class_name

    if not folder.is_dir():
        print(f"{class_name}: FOLDER MISSING")
        continue

    images = [
        file for file in folder.rglob("*")
        if file.is_file()
        and file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    print(f"{class_name}: {len(images)} images")

print("\nVerification finished.")
