from src.mlproject.logger import logging
import sys
from src.mlproject.exception import CustomException
from pydantic import BaseModel, Field,ConfigDict
from fastapi import FastAPI, HTTPException
from criptos import Cripto
import numpy as np
import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "trained_model.pkl")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
#Create app object

app = FastAPI()
with open(MODEL_PATH, "rb") as pickle_in:
    classifier = joblib.load(pickle_in)


@app.get('/')
def index():
    return {'message' : 'Hello world'}

@app.get('/{name}')
def get_name(name : str):
    return {'criptocurrancy project by' : f'{name}'}

@app.post('/predict')
def predict_currancy(data:Cripto):
    try:
        # Convert input to dictionary using aliases
        input_data = data.dict(by_alias=True)

        # Extract features in the order the model expects
        features = np.array([
            [
                input_data["24h"],
                input_data["7d"],
                input_data["24h_volume"],
                input_data["mkt_cap"]
            ]
        ])

        # Predict using loaded model
        prediction = classifier.predict(features)[0]
        return {"predicted_liquidity_ratio": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    if __name__ == '__main__':
        uvicorn.run(app,host = '127.0.0.1',port=8000)