
import tensorflow as tf
import numpy as np
from pathlib import Path

MODEL_PATH = "potato_leaf_validator.keras"
IMAGE_SIZE = (128, 128)

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        print(f"File not found: {image_path}")
        return

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    probability = float(model.predict(image, verbose=0)[0][0])

    print(f"\nImage: {image_path.name}")
    print(f"Potato-leaf score: {probability:.4f}")

    if probability >= 0.5:
        print("Validator result: POTATO LEAF")
    else:
        print("Validator result: NOT POTATO LEAF")

print("Potato Leaf Validator Test")
print("--------------------------")

leaf_path = input("Enter the path to a potato-leaf image: ").strip().strip('"')
predict_image(leaf_path)

other_path = input("\nEnter the path to a car or unrelated image: ").strip().strip('"')
predict_image(other_path)
