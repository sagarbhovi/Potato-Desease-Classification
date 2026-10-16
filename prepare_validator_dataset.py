
import pickle
import random
import shutil
from pathlib import Path

# Original datasets
PLANT_DIR = Path("PlantVillage")
CIFAR_DIR = Path("cifar-10-batches-py")

# New dataset folders
OUTPUT_DIR = Path("leaf_validator_dataset")
POTATO_DIR = OUTPUT_DIR / "potato_leaf"
NOT_POTATO_DIR = OUTPUT_DIR / "not_potato_leaf"

POTATO_LIMIT = 2000
CIFAR_LIMIT = 2500

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
random.seed(42)

POTATO_DIR.mkdir(parents=True, exist_ok=True)
NOT_POTATO_DIR.mkdir(parents=True, exist_ok=True)


def clear_images(folder):
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            path.unlink()


# Clear only the generated images in our new dataset folders
# so repeated runs do not create duplicates.
clear_images(POTATO_DIR)
clear_images(NOT_POTATO_DIR)


# --------------------------------------------------
# 1. Copy potato-leaf images
# --------------------------------------------------
potato_images = []

for class_dir in PLANT_DIR.iterdir():
    if class_dir.is_dir() and class_dir.name.startswith("Potato___"):
        potato_images.extend(
            path for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

random.shuffle(potato_images)
potato_images = potato_images[:POTATO_LIMIT]

for index, image_path in enumerate(potato_images):
    destination = POTATO_DIR / f"potato_{index:05d}{image_path.suffix.lower()}"
    shutil.copy2(image_path, destination)

print(f"Potato images copied: {len(potato_images)}")


# --------------------------------------------------
# 2. Copy CIFAR-10 images as non-potato examples
# --------------------------------------------------
negative_images = []

for batch_number in range(1, 6):
    batch_path = CIFAR_DIR / f"data_batch_{batch_number}"

    with open(batch_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    for image_data in batch[b"data"]:
        negative_images.append(image_data)

random.shuffle(negative_images)
negative_images = negative_images[:CIFAR_LIMIT]

import numpy as np
from PIL import Image

for index, image_data in enumerate(negative_images):
    image_array = np.asarray(image_data, dtype=np.uint8)
    image_array = image_array.reshape(3, 32, 32).transpose(1, 2, 0)

    destination = NOT_POTATO_DIR / f"object_{index:05d}.png"
    Image.fromarray(image_array).save(destination)

print(f"Non-potato object images copied: {len(negative_images)}")
print("\nDataset preparation complete.")
print("Note: other plant leaves still need to be added.")
