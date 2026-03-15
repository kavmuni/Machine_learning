import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class input_params(BaseModel):
  city: object
  city_development_index: float
  gender: object
  relevent_experience: object
  enrolled_university: object
  education_level: object
  major_discipline: object
  experience: object
  company_size: object
  company_type: object
  last_new_job: object
  training_hours: int

class output_param(BaseModel):
  target:int

@app.post("/predict")
def predict(input: input_params) -> output_param:
  model = joblib.load('best_model_HR_analysis.pkl')
  input_data = pd.DataFrame([input.dict()])
  prediction = model.predict(input_data)
  return output_param(target=prediction[0])
  


