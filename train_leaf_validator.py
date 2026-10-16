
import os
import pickle
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

# -------------------------------
# Configuration
# -------------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 32
SEED = 42

POTATO_DIR = "PlantVillage"
CIFAR_DIR = "cifar-10-batches-py"
MODEL_PATH = "potato_leaf_validator.keras"

# -------------------------------
# Load potato-leaf images
# -------------------------------
print("Loading potato-leaf images...")

potato_ds = tf.keras.utils.image_dataset_from_directory(
    POTATO_DIR,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=None,
    label_mode=None,
    shuffle=False
)

potato_images = np.concatenate(
    [image.numpy()[None, ...] for image in potato_ds],
    axis=0
).astype("float32") / 255.0

print("Potato images:", len(potato_images))

# -------------------------------
# Load CIFAR-10 negative images
# -------------------------------
print("Loading CIFAR-10 images...")

negative_images = []

for filename in [
    "data_batch_1",
    "data_batch_2",
    "data_batch_3",
    "data_batch_4",
    "data_batch_5",
]:
    with open(os.path.join(CIFAR_DIR, filename), "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"].reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    # Use a subset for the first experiment
    images = images[:500]
    images = tf.image.resize(
        images, (IMAGE_SIZE, IMAGE_SIZE)
    ).numpy()

    negative_images.append(images / 255.0)

negative_images = np.concatenate(
    negative_images, axis=0
).astype("float32")

print("Negative images:", len(negative_images))

# -------------------------------
# Create labels
# 1 = potato leaf
# 0 = unrelated image
# -------------------------------
X = np.concatenate([potato_images, negative_images])
y = np.concatenate([
    np.ones(len(potato_images), dtype="float32"),
    np.zeros(len(negative_images), dtype="float32")
])

# Split before training
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=SEED,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

# -------------------------------
# Build a small CNN
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    tf.keras.layers.Conv2D(
        16, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, 3, activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------------
# Train and save
# -------------------------------
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=BATCH_SIZE
)

model.save(MODEL_PATH)

print(f"Validator saved to: {MODEL_PATH}")
