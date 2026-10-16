
import numpy as np
import tensorflow as tf
from pathlib import Path

IMAGE_SIZE = (128, 128)

MODEL_PATH = "potato_leaf_validator_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):
    image_path = Path(image_path.strip('"'))

    if not image_path.is_file():
        print(f"File not found: {image_path}")
        return

    image = tf.keras.utils.load_img(
        str(image_path),
        target_size=IMAGE_SIZE,
        color_mode="rgb"
    )

    image = tf.keras.utils.img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    score = float(model.predict(image, verbose=0)[0][0])

    print(f"\nImage: {image_path.name}")
    print(f"Potato-leaf score: {score:.4f}")

    if score >= 0.5:
        print("Result: POTATO LEAF")
    else:
        print("Result: NOT POTATO LEAF")


print("Potato Leaf Validator V2 Test")
print("--------------------------------")

leaf_path = input("Enter potato-leaf image path: ")
predict_image(leaf_path)

car_path = input("\nEnter car image path: ")
predict_image(car_path)
