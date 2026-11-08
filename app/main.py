from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

# ---------- Load model once at startup ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "addiction_model.joblib")
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Teen Phone Addiction Predictor",
    version="1.0.0",
    description="Predicts addiction class (Low / Medium / High) from teen usage data.",
)


# ---------- Request schema (matches your dataset columns exactly) ----------
class TeenInput(BaseModel):
    Age: int = Field(..., ge=10, le=25)
    Gender: str = Field(..., description="Male, Female, or Other")
    Location: str = Field(..., description="City name (unseen → falls back to most common)")
    School_Grade: str = Field(..., description="'9th', '10th', '11th', '12th', ...")
    Daily_Usage_Hours: float = Field(..., ge=0, le=24)
    Sleep_Hours: float = Field(..., ge=0, le=24)
    Academic_Performance: int = Field(..., ge=0, le=100)
    Social_Interactions: int = Field(..., ge=0, le=10)
    Exercise_Hours: float = Field(..., ge=0, le=24)
    Anxiety_Level: int = Field(..., ge=0, le=10)
    Depression_Level: int = Field(..., ge=0, le=10)
    Self_Esteem: int = Field(..., ge=0, le=10)
    Parental_Control: int = Field(..., ge=0, le=1)
    Screen_Time_Before_Bed: float = Field(..., ge=0, le=24)
    Phone_Checks_Per_Day: int = Field(..., ge=0)
    Apps_Used_Daily: int = Field(..., ge=0)
    Time_on_Social_Media: float = Field(..., ge=0, le=24)
    Time_on_Gaming: float = Field(..., ge=0, le=24)
    Time_on_Education: float = Field(..., ge=0, le=24)
    Phone_Usage_Purpose: str = Field(..., description="Browsing, Gaming, Social Media, Education, ...")
    Family_Communication: int = Field(..., ge=0, le=10)
    Weekend_Usage_Hours: float = Field(..., ge=0, le=24)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Age": 15, "Gender": "Female", "Location": "Hansonfort",
                "School_Grade": "9th", "Daily_Usage_Hours": 4.0, "Sleep_Hours": 6.1,
                "Academic_Performance": 78, "Social_Interactions": 5, "Exercise_Hours": 0.1,
                "Anxiety_Level": 10, "Depression_Level": 3, "Self_Esteem": 6,
                "Parental_Control": 0, "Screen_Time_Before_Bed": 1.4,
                "Phone_Checks_Per_Day": 86, "Apps_Used_Daily": 19,
                "Time_on_Social_Media": 3.6, "Time_on_Gaming": 1.7,
                "Time_on_Education": 1.2, "Phone_Usage_Purpose": "Browsing",
                "Family_Communication": 4, "Weekend_Usage_Hours": 8.7,
            }
        }
    }


# ---------- Routes ----------
@app.get("/")
def root():
    return {
        "service": "Teen Phone Addiction Predictor",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TeenInput):
    try:
        raw = pd.DataFrame([payload.model_dump()])
        pred = pipeline.predict(raw)[0]
        proba = pipeline.predict_proba(raw)[0]
        classes = pipeline.named_steps["clf"].classes_
        return {
            "predicted_class": str(pred),
            "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")