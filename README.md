
# Potato Disease Classification

A deep learning web application that classifies potato leaf images into three categories using a Convolutional Neural Network (CNN). The project combines TensorFlow/Keras, FastAPI, and React.

## Disease Classes

- **Potato Early Blight:** `Potato___Early_blight`
- **Potato Late Blight:** `Potato___Late_blight`
- **Healthy Potato Leaf:** `Potato___healthy`

## Technologies Used

- Python
- TensorFlow and Keras
- NumPy
- Pillow (PIL)
- FastAPI
- Uvicorn
- React

## Project Structure

```text
Potato-Desease-Classification/
├── API/
│   ├── main.py
│   └── requirements.txt
├── Models/
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
├── potato_disease_model.keras
├── training.ipynb
└── .gitignore
```

## How It Works

1. The user uploads a potato leaf image through the React interface.
2. The frontend sends the image to the FastAPI backend.
3. The backend converts the image to RGB and resizes it to 256 × 256 pixels.
4. The CNN model predicts the leaf category.
5. The API returns the predicted class and confidence score.
6. The frontend displays the prediction.

## Installation and Setup

### Prerequisites

- Python compatible with `API/requirements.txt`
- Node.js and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/sagarbhovi/Potato-Desease-Classification.git
cd Potato-Desease-Classification
```

### 2. Install Backend Dependencies

From the project root, activate your virtual environment and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r API/requirements.txt
```

### 3. Run the FastAPI Backend

From the project root:

```powershell
python -m uvicorn main:app --app-dir API --reload --host 127.0.0.1 --port 8000
```

API documentation: http://127.0.0.1:8000/docs

### 4. Run the React Frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm start
```

The frontend normally runs at http://localhost:3000.

Make sure the API URL in `frontend/src/home.js` points to the running backend, such as `http://127.0.0.1:8000/predict`.

## Model Training

The `training.ipynb` notebook contains the model-training workflow.

The `PlantVillage/` dataset is excluded from this repository. Obtain the dataset separately and place it in the location expected by the notebook if you want to retrain the model.

## Model Evaluation

During development, the model evaluation output showed approximately **95.70% accuracy** on the evaluated data.

This is an educational project. Model confidence does not guarantee correct predictions, and the application should not be used as the sole basis for agricultural treatment decisions.

## Future Improvements

- Test the model using independent field images.
- Improve API error handling.
- Add automated tests.
- Deploy the frontend and backend.

## Author

**Sagar Bhovi**

GitHub: https://github.com/sagarbhovi
