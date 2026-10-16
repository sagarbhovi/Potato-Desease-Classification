
import random
import shutil
from pathlib import Path

# Extracted full dataset folder
SOURCE_DIR = Path(
    r"C:\Users\Sagar\Downloads\Plant_leaf_diseases_dataset_without_augmentation"
    r"\Plant_leave_diseases_dataset_without_augmentation"
)

# Destination for non-potato images
DEST_DIR = Path("leaf_validator_dataset") / "not_potato_leaf"

# Maximum number of images to copy from each non-potato category
IMAGES_PER_CLASS = 200

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)

if not SOURCE_DIR.is_dir():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{SOURCE_DIR}\n"
        "Check the extracted folder path."
    )

DEST_DIR.mkdir(parents=True, exist_ok=True)

# Find all image folders except potato categories
class_folders = [
    folder for folder in SOURCE_DIR.iterdir()
    if folder.is_dir()
    and not folder.name.lower().startswith("potato")
]

copied = 0

for folder in sorted(class_folders):
    images = [
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)
    images = images[:IMAGES_PER_CLASS]

    # Use unique filenames to avoid collisions
    safe_class_name = "".join(
        character if character.isalnum() else "_"
        for character in folder.name
    )

    for index, image_path in enumerate(images):
        destination = (
            DEST_DIR
            / f"{safe_class_name}_{index:04d}{image_path.suffix.lower()}"
        )
        shutil.copy2(image_path, destination)
        copied += 1

    print(f"{folder.name}: copied {len(images)} images")

print(f"\nTotal non-potato leaf images copied: {copied}")
print(f"Destination: {DEST_DIR}")
