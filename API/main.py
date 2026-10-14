import os
from io import BytesIO

import numpy as np
import tensorflow as tf
import uvicorn

from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# --------------------------------------------------
# 1. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Potato Disease Classification API",
    description="Upload a potato leaf image to predict its disease.",
    version="1.0.0"
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
# 3. Load the trained model
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "potato_disease_model.keras"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

MODEL = tf.keras.models.load_model(MODEL_PATH)

print("Potato disease model loaded successfully!")


# --------------------------------------------------
# 4. Define class names
# IMPORTANT: Keep the same order as during training
# --------------------------------------------------

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

IMAGE_SIZE = 256


# --------------------------------------------------
# 5. Convert uploaded image into a NumPy ndarray
# --------------------------------------------------

def read_file_as_image(data: bytes) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image_array = np.array(image)

    print("\nImage uploaded successfully!")
    print("Array shape:", image_array.shape)
    print("Array data type:", image_array.dtype)
    print("Array dimensions:", image_array.ndim)
    print("First 5x5 pixels:")
    print(image_array[:5, :5, :])

    return image_array


# --------------------------------------------------
# 6. Test endpoint
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

    # Validate the uploaded file type
    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/jpg",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    try:
        # Read the uploaded image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # Convert image to NumPy ndarray
        image = read_file_as_image(image_bytes)

        # Add batch dimension:
        # (256, 256, 3) -> (1, 256, 256, 3)
        img_batch = np.expand_dims(image, axis=0)

        print("Batch shape:", img_batch.shape)

        # Make prediction
        predictions = MODEL.predict(
            img_batch,
            verbose=0
        )

        # Get predicted class and confidence
        predicted_index = int(
            np.argmax(predictions[0])
        )

        predicted_class = CLASS_NAMES[predicted_index]

        confidence = float(
            np.max(predictions[0])
        )

        print("Predicted class:", predicted_class)
        print("Confidence:", confidence)

        # Return result to Postman
        return {
            "filename": file.filename,
            "class": predicted_class,
            "confidence": confidence,
            "image_shape": list(image.shape),
            "image_dtype": str(image.dtype),
            "batch_shape": list(img_batch.shape)
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {str(e)}"
        )


# --------------------------------------------------
# 8. Run the application
# --------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )