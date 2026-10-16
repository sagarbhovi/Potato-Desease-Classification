
import os
from io import BytesIO

import numpy as np
import tensorflow as tf
import uvicorn

from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Potato Disease Classification API",
    description="Validate a potato leaf image and predict its disease.",
    version="1.2.0"
)


# --------------------------------------------------
# 2. Configure CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. Model paths and loading
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "potato_disease_model.keras"
)

VALIDATOR_PATH = os.path.join(
    PROJECT_DIR,
    "potato_leaf_validator_v2.keras"
)

for path in [MODEL_PATH, VALIDATOR_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

MODEL = tf.keras.models.load_model(MODEL_PATH)
VALIDATOR = tf.keras.models.load_model(VALIDATOR_PATH)

print("Potato disease model loaded successfully!")
print("Potato leaf validator V2 loaded successfully!")


# --------------------------------------------------
# 4. Class names and settings
# --------------------------------------------------

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

DISEASE_IMAGE_SIZE = 256
VALIDATOR_IMAGE_SIZE = 128

# Initial experimental thresholds
LEAF_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.80


# --------------------------------------------------
# 5. Read and preprocess uploaded images
# --------------------------------------------------

def read_uploaded_image(data: bytes):
    try:
        with Image.open(BytesIO(data)) as img:
            image = img.convert("RGB")

            # Disease model preprocessing: 256 x 256, 0-255
            disease_image = image.resize(
                (DISEASE_IMAGE_SIZE, DISEASE_IMAGE_SIZE)
            )
            disease_array = np.array(disease_image)

            # Validator preprocessing: 128 x 128, normalized 0-1
            validator_image = image.resize(
                (VALIDATOR_IMAGE_SIZE, VALIDATOR_IMAGE_SIZE)
            )
            validator_array = (
                np.array(validator_image, dtype=np.float32) / 255.0
            )

        return disease_array, validator_array

    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image."
        ) from e


# --------------------------------------------------
# 6. Health-check endpoint
# --------------------------------------------------

@app.get("/ping")
async def ping():
    return {
        "message": "Hello, Potato Disease API is running!"
    }


# --------------------------------------------------
# 7. Predict potato disease
# --------------------------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    }

    try:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Please upload a JPG, JPEG, PNG, or WEBP image."
            )

        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Preprocess the same image for both models
        disease_image, validator_image = read_uploaded_image(image_bytes)

        disease_batch = np.expand_dims(disease_image, axis=0)
        validator_batch = np.expand_dims(validator_image, axis=0)

        # --------------------------------------------------
        # Stage 1: Check whether the image is a potato leaf
        # --------------------------------------------------

        leaf_predictions = VALIDATOR.predict(
            validator_batch,
            verbose=0
        )

        if (
            leaf_predictions.size != 1
            or not np.all(np.isfinite(leaf_predictions))
        ):
            raise HTTPException(
                status_code=500,
                detail="The leaf validator returned an invalid prediction."
            )

        leaf_score = float(leaf_predictions.reshape(-1)[0])

        if not 0.0 <= leaf_score <= 1.0:
            raise HTTPException(
                status_code=500,
                detail="The leaf validator returned an invalid score."
            )

        print("Potato-leaf score:", leaf_score)

        if leaf_score < LEAF_THRESHOLD:
            return {
                "filename": file.filename,
                "is_potato_leaf": False,
                "leaf_score": leaf_score,
                "class": "Not a potato leaf",
                "message": (
                    "This image was not recognized as a potato leaf. "
                    "Please upload a clear potato leaf image."
                )
            }

        # --------------------------------------------------
        # Stage 2: Predict the potato disease
        # --------------------------------------------------

        predictions = MODEL.predict(
            disease_batch,
            verbose=0
        )

        if (
            predictions.ndim != 2
            or predictions.shape[0] != 1
            or predictions.shape[1] != len(CLASS_NAMES)
            or not np.all(np.isfinite(predictions))
        ):
            raise HTTPException(
                status_code=500,
                detail="The disease model returned an invalid prediction."
            )

        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0]))

        print("Predicted class:", CLASS_NAMES[predicted_index])
        print("Disease model confidence:", confidence)

        if confidence < CONFIDENCE_THRESHOLD:
            return {
                "filename": file.filename,
                "is_potato_leaf": True,
                "leaf_score": leaf_score,
                "class": "Uncertain prediction",
                "confidence": confidence,
                "message": (
                    "The image passed leaf validation, but the disease "
                    "prediction confidence is too low. Please try a "
                    "clearer potato leaf image."
                )
            }

        return {
            "filename": file.filename,
            "is_potato_leaf": True,
            "leaf_score": leaf_score,
            "class": CLASS_NAMES[predicted_index],
            "confidence": confidence,
            "message": "Prediction completed successfully."
        }

    except HTTPException:
        raise

    except Exception as e:
        print("Prediction error:", str(e))
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the image."
        ) from e

    finally:
        await file.close()


# --------------------------------------------------
# 8. Run the application
# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
