from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from typing import Union
import model
import pandas as pd
import numpy as np

app = FastAPI()

class  ModelInfo(BaseModel):
    """Model information structure"""
    mode_name: str
    model_type:str
    model_version:str
    model_features:list[str]
    model_author:str
    model_description:str
    model_accuracy:float
    model_precision:float
    model_recall:float
    model_f1_score:float

def load_model_pipeline():
    """Load the trained model pipeline from a file"""
    global model_pipeline
    model_pipeline = joblib.load('../model/titanic_model_pipeline.pkl')
    return model_pipeline    

@app.on_event("startup")
def startup_event():
    load_model_pipeline()
    print("Model Loaded Successfully")

@app.get("/")
def read_root():
    return {
            "Model API"     : "Titanic Survival Prediction Model API",
            "Version"       : "1.0.0",
            "Author"        : "Muniappan Mohanraj",
            "Description"   : "API for predicting survival on the Titanic dataset using a trained Logistic Regression model.",
            "Endpoints"     : {
                            "/docs"         : "/docs",
                            "/predict"      : "Predict survival based on input features.",
                            "/model_info"   : "Get information about the trained model."
                            }
        }            

@app.get("/model_info", response_model=ModelInfo)
def get_model_info():
    """Get teh info about trained model"""
    if not model_pipeline:
        return {"error": "Model not loaded"}
    else:
        return ModelInfo(
            mode_name         = "Titanic Survival Prediction Model",
            model_type        = "Logistic Regression",
            model_version     = "1.0.0",
            model_features    = model.all_model_features,
            model_author      = "Muniappan Mohanraj",
            model_description = "A Logistic Regression model trained to predict survival on the Titanic dataset.",
            model_accuracy    = model.test_accuracy,
            model_precision   = model.test_precision,
            model_recall      = model.test_recall,
            model_f1_score    = model.test_f1_score
        )

class PredictRequest(BaseModel):
    passenger_id: int
    Parch: int
    Fare: float
    Sex: str
    Cabin_class: str
    Is_alone: int

class PredictResponse(BaseModel):
    passenger_id: int
    survival_prediction: int

@app.post("/predict")
def predict_survival(data: PredictRequest) -> PredictResponse:
    """Predict survival based on input features"""
    if not model_pipeline:
        return {"error": "Model not loaded"}
    else:
        input_data = pd.DataFrame([data.dict()])
        prediction = model_pipeline.predict(input_data[model.all_model_features])
        return PredictResponse(
            passenger_id = data.passenger_id,
            survival_prediction = int(prediction[0])
            )
