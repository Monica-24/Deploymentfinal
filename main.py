# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load('iris_model.pkl')

# Load the class names
iris = load_iris()
class_names = iris.target_names

app = FastAPI()

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(features: IrisFeatures):
    # Prepare input data
    data = np.array([[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]])
    
    # Make a prediction
    prediction = model.predict(data)
    
    # Get the class name
    class_name = class_names[prediction[0]]
    
    return {"prediction": class_name}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris flower prediction API!"}