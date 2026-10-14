README
Potato Disease Classification
A deep learning web application that classifies potato leaf images into three categories using a Convolutional Neural Network (CNN). The project combines a TensorFlow/Keras model, a FastAPI backend, and a React frontend.
Disease Classes
The model predicts one of these classes:
Potato Early Blight (Potato___Early_blight)
Potato Late Blight (Potato___Late_blight)
Healthy Potato Leaf (Potato___healthy)
Tech Stack
Python
TensorFlow / Keras — model training and prediction
NumPy — image-array processing
Pillow (PIL) — image loading and preprocessing
FastAPI — prediction API
Uvicorn — ASGI server
React — web interface
Project Structure
Potato-Desease-Classification/
├── API/
│   ├── main.py
│   └── requirements.txt
├── Models/
│   ├── potato_disease_model.keras
│   └── potato_disease_model_1.keras
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── package-lock.json
├── potato_disease_model.keras
├── training.ipynb
└── .gitignore
​
The repository contains model files in more than one location. Check API/main.py to confirm which model path the backend loads, and keep only the model copies you actually need.
How It Works
The user selects a potato leaf image in the React interface.
The frontend sends the image to the FastAPI /predict endpoint.
The backend loads the image, converts it to RGB, and resizes it to 256 × 256 pixels.
The processed image is passed to the trained CNN model.
The API returns the predicted class and its model confidence score.
The frontend displays the prediction.
Prerequisites
Install the following before running the project:
Python (use a version compatible with the packages in API/requirements.txt)
Node.js and npm
Git (optional, for cloning the repository)
Setup and Run
Open a terminal in the project root.
1. Clone the repository
git clone <https://github.com/sagarbhovi/Potato-Desease-Classification.git>
cd Potato-Desease-Classification
​
2. Set up the Python backend
Create and activate a virtual environment.
Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
​
Install the backend dependencies:
pip install -r API/requirements.txt
​
3. Start the FastAPI server
From the project root, run:
python -m uvicorn main:app --app-dir API --reload --host 127.0.0.1 --port 8000
​
If your current working setup uses the API directory directly, you can instead run:
cd API
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
​
The API should be available at:
API base URL: http://127.0.0.1:8000
Interactive API documentation: http://127.0.0.1:8000/docs
Health check: http://127.0.0.1:8000/ping (if implemented in the current backend)
Keep this terminal running.
4. Set up and start the React frontend
Open a second terminal:
cd frontend
npm install
npm start
​
The React development server commonly opens at http://localhost:3000.
Keep the frontend and backend running in separate terminals. Ensure the API URL configured in frontend/src/home.js points to the backend address, such as http://127.0.0.1:8000/predict.
If the frontend fails to start because of a Node.js/OpenSSL compatibility error, check your Node.js version and the project's React scripts before applying a version-specific workaround.
API Endpoint
POST /predict
Send a potato leaf image to the prediction endpoint using the form field expected by API/main.py (commonly file).
Example with curl if the endpoint expects a field named file:
curl -X POST "<http://127.0.0.1:8000/predict>" \
  -F "file=@path/to/potato_leaf.jpg"
​
A typical response may look like this (the exact property names depend on API/main.py):
{
  "class": "Potato___Late_blight",
  "confidence": 0.99
}
​
Use http://127.0.0.1:8000/docs to inspect the actual request field and response schema for your running API.
Model Training
The notebook training.ipynb contains the model-training workflow. Open it in Jupyter Notebook, JupyterLab, or PyCharm to review the data preparation, model training, and evaluation steps.
The PlantVillage/ dataset is excluded from this repository through .gitignore, so it is not included when cloning the project. Obtain the dataset separately and place it in the location expected by the notebook before retraining.
Current Evaluation
During development, the model evaluation output showed approximately 95.70% accuracy for the evaluated data. The exact result depends on the evaluation split and dataset used.
This is an experimental educational project. Model confidence is not the same as verified diagnostic accuracy. Predictions may be incorrect on images that differ from the training data, and this application should not be used as the sole basis for agricultural treatment decisions.
Future Improvements
Add a clear loading state and friendlier error messages.
Add a prediction history or downloadable result.
Improve validation with independent, representative field images.
Add tests for image preprocessing and API responses.
Deploy the frontend and backend with appropriate security and configuration.
Author
Sagar Bhovi
GitHub: @sagarbhovi
If you find this project useful, consider giving the repository a star.
