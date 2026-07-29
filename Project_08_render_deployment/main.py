from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import joblib
import pandas as pd
from pathlib import Path
import traceback

app = FastAPI(
    title="Adult Income Classification API",
    description="End-to-End ML Prediction Service for UCI Adult Income dataset deployed on Render.",
    version="1.0.0"
)

# Load model
MODEL_PATH = Path(__file__).parent / "model.joblib"
model = None

def get_model():
    global model
    if model is None:
        if MODEL_PATH.exists():
            try:
                model = joblib.load(MODEL_PATH)
                print("Model loaded successfully.")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("model.joblib not found. Attempting fallback training...")
            try:
                import train
                if MODEL_PATH.exists():
                    model = joblib.load(MODEL_PATH)
            except Exception as e:
                print(f"Fallback training failed: {e}")
    return model

@app.on_event("startup")
def startup_event():
    get_model()

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
    m = get_model()
    return {
        "message": "Adult Income Prediction API is running.",
        "status": "online",
        "model_loaded": m is not None,
        "docs_url": "/docs",
        "health_check": "/health"
    }

@app.get("/health")
def health_check():
    m = get_model()
    return {
        "status": "healthy",
        "model_loaded": m is not None
    }

@app.post("/predict")
def predict(payload: IncomeFeatureInput):
    m = get_model()
    if m is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    try:
        input_dict = {
            "age": payload.age,
            "workclass": str(payload.workclass),
            "fnlwgt": payload.fnlwgt,
            "education": str(payload.education),
            "education-num": payload.education_num,
            "marital-status": str(payload.marital_status),
            "occupation": str(payload.occupation),
            "relationship": str(payload.relationship),
            "race": str(payload.race),
            "sex": str(payload.sex),
            "capital-gain": payload.capital_gain,
            "capital-loss": payload.capital_loss,
            "hours-per-week": payload.hours_per_week,
            "native-country": str(payload.native_country)
        }
        
        df = pd.DataFrame([input_dict])
        
        num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col])
            
        prob = float(m.predict_proba(df)[0, 1])
        pred = int(prob >= 0.5)
        income_label = ">50K" if pred == 1 else "<=50K"
        
        return {
            "prediction": pred,
            "label": income_label,
            "probability_high_income": round(prob, 4),
            "input": input_dict
        }
    except Exception as e:
        err_msg = f"Inference Error: {str(e)}\n{traceback.format_exc()}"
        print(err_msg)
        raise HTTPException(status_code=500, detail=err_msg)
