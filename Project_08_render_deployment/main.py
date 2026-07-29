from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib
import pandas as pd
from pathlib import Path

app = FastAPI(
    title="Adult Income Classification API",
    description="End-to-End ML Prediction Service for UCI Adult Income dataset deployed on Render.",
    version="1.0.0"
)

# Load model
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully at startup.")
    except Exception as e:
        print(f"Error loading model: {e}")

class IncomeFeatureInput(BaseModel):
    age: int = 37
    workclass: str = "Private"
    fnlwgt: int = 22245
    education: str = "Some-college"
    education_num: int = Field(10, alias="education-num")
    marital_status: str = Field("Divorced", alias="marital-status")
    occupation: str = "Sales"
    relationship: str = "Not-in-family"
    race: str = "White"
    sex: str = "Male"
    capital_gain: int = Field(0, alias="capital-gain")
    capital_loss: int = Field(0, alias="capital-loss")
    hours_per_week: int = Field(40, alias="hours-per-week")
    native_country: str = Field("Outlying-US(Guam-USVI-etc)", alias="native-country")

    class Config:
        populate_by_name = True

@app.get("/")
def root():
    return {
        "message": "Adult Income Prediction API is running.",
        "status": "online",
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict(payload: IncomeFeatureInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    input_dict = {
        "age": payload.age,
        "workclass": payload.workclass,
        "fnlwgt": payload.fnlwgt,
        "education": payload.education,
        "education-num": payload.education_num,
        "marital-status": payload.marital_status,
        "occupation": payload.occupation,
        "relationship": payload.relationship,
        "race": payload.race,
        "sex": payload.sex,
        "capital-gain": payload.capital_gain,
        "capital-loss": payload.capital_loss,
        "hours-per-week": payload.hours_per_week,
        "native-country": payload.native_country
    }
    
    df = pd.DataFrame([input_dict])
    
    prob = float(model.predict_proba(df)[0, 1])
    pred = int(prob >= 0.5)
    income_label = ">50K" if pred == 1 else "<=50K"
    
    return {
        "prediction": pred,
        "label": income_label,
        "probability_high_income": round(prob, 4),
        "input": input_dict
    }
