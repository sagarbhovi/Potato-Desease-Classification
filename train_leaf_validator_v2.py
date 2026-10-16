
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATASET_DIR = Path("leaf_validator_dataset")
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

MODEL_OUTPUT = "potato_leaf_validator_v2.keras"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# 1. Collect image paths and binary labels
# --------------------------------------------------
class_names = ["not_potato_leaf", "potato_leaf"]
paths = []
labels = []

for label, class_name in enumerate(class_names):
    folder = DATASET_DIR / class_name

    if not folder.is_dir():
        raise FileNotFoundError(f"Missing dataset folder: {folder}")

    images = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    print(f"{class_name}: {len(images)} images")

    paths.extend(images)
    labels.extend([label] * len(images))

paths = np.array(paths, dtype=object)
labels = np.array(labels, dtype=np.int32)

if len(paths) == 0:
    raise ValueError("No images found in the dataset.")

# --------------------------------------------------
# 2. Split into train, validation, and test sets
# --------------------------------------------------
train_paths, temp_paths, y_train, y_temp = train_test_split(
    paths,
    labels,
    test_size=0.30,
    random_state=SEED,
    stratify=labels
)

val_paths, test_paths, y_val, y_test = train_test_split(
    temp_paths,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nDataset split:")
print(f"Training:   {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Testing:    {len(test_paths)}")

# --------------------------------------------------
# 3. Load and preprocess images
# --------------------------------------------------
def load_images(image_paths):
    images = []

    for index, image_path in enumerate(image_paths):
        image = tf.keras.utils.load_img(
            str(image_path),
            target_size=IMAGE_SIZE,
            color_mode="rgb"
        )

        image = tf.keras.utils.img_to_array(image)
        image = image / 255.0
        images.append(image)

        if (index + 1) % 1000 == 0:
            print(f"Loaded {index + 1}/{len(image_paths)} images")

    return np.asarray(images, dtype=np.float32)


print("\nLoading training images...")
X_train = load_images(train_paths)

print("\nLoading validation images...")
X_val = load_images(val_paths)

print("\nLoading test images...")
X_test = load_images(test_paths)

# --------------------------------------------------
# 4. Calculate class weights
# --------------------------------------------------
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)

# --------------------------------------------------
# 5. Build a CNN validator
# --------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        64, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        128, (3, 3), activation="relu", padding="same"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()

# --------------------------------------------------
# 6. Train with early stopping
# --------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=12,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# --------------------------------------------------
# 7. Evaluate once on the held-out test set
# --------------------------------------------------
print("\nFinal test evaluation:")
test_results = model.evaluate(
    X_test,
    y_test,
    batch_size=BATCH_SIZE,
    return_dict=True
)

for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

# --------------------------------------------------
# 8. Save the new model separately
# --------------------------------------------------
model.save(MODEL_OUTPUT)

print(f"\nNew validator saved as: {MODEL_OUTPUT}")
print("Existing models were not overwritten.")
