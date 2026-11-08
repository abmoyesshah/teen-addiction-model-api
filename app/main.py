import sys
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


class ColumnEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols or []
    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        self.encoders_ = {}
        self.fallbacks_ = {}
        for col in self.cat_cols:
            le = LabelEncoder().fit(X[col])
            self.encoders_[col] = le
            self.fallbacks_[col] = X[col].mode()[0]
        return self
    def transform(self, X):
        X = X[self.feature_names_in_].copy()
        for col in self.cat_cols:
            X[col] = [self._encode(col, v) for v in X[col]]
        return X.to_numpy()
    def _encode(self, col, val):
        le = self.encoders_[col]
        try:
            return int(le.transform([val])[0])
        except (ValueError, TypeError):
            return int(le.transform([self.fallbacks_[col]])[0])
    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)


sys.modules['__main__'].ColumnEncoder = ColumnEncoder

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "addiction_model.joblib")
pipeline = joblib.load(MODEL_PATH)

app = FastAPI(title="Teen Phone Addiction Predictor", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origin_regex=r"https://.*\.vercel\.app",
#     allow_origins=[
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         "https://teen-addiction-model-api.onrender.com",
#     ],
#     allow_methods=["GET", "POST", "OPTIONS"],
#     allow_headers=["*"],
#     allow_credentials=False,
# )
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


class TeenInput(BaseModel):
    Age: int = Field(..., ge=10, le=25)
    Gender: str
    Location: str
    School_Grade: str
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
    Phone_Usage_Purpose: str
    Family_Communication: int = Field(..., ge=0, le=10)
    Weekend_Usage_Hours: float = Field(..., ge=0, le=24)


@app.get("/")
def root():
    return {"service": "Teen Phone Addiction Predictor", "docs": "/docs"}

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






# import sys
# import os
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.preprocessing import LabelEncoder
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field


# # ---------------------------------------------------------------
# # ColumnEncoder MUST be defined before joblib.load() runs.
# # We also register it in __main__ so pickle can find it.
# # ---------------------------------------------------------------
# class ColumnEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, cat_cols=None):
#         self.cat_cols = cat_cols or []

#     def fit(self, X, y=None):
#         self.feature_names_in_ = list(X.columns)
#         self.encoders_ = {}
#         self.fallbacks_ = {}
#         for col in self.cat_cols:
#             le = LabelEncoder().fit(X[col])
#             self.encoders_[col] = le
#             self.fallbacks_[col] = X[col].mode()[0]
#         return self

#     def transform(self, X):
#         X = X[self.feature_names_in_].copy()
#         for col in self.cat_cols:
#             X[col] = [self._encode(col, v) for v in X[col]]
#         return X.to_numpy()

#     def _encode(self, col, val):
#         le = self.encoders_[col]
#         try:
#             return int(le.transform([val])[0])
#         except (ValueError, TypeError):
#             return int(le.transform([self.fallbacks_[col]])[0])

#     def get_feature_names_out(self, input_features=None):
#         return np.array(self.feature_names_in_)


# # Register under __main__ so pickle finds it
# sys.modules['__main__'].ColumnEncoder = ColumnEncoder


# # ---------------------------------------------------------------
# # Load pipeline (now works)
# # ---------------------------------------------------------------
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "addiction_model.joblib")
# pipeline = joblib.load(MODEL_PATH)

# app = FastAPI(title="Teen Phone Addiction Predictor", version="1.0.0")


# class TeenInput(BaseModel):
#     Age: int = Field(..., ge=10, le=25)
#     Gender: str
#     Location: str
#     School_Grade: str
#     Daily_Usage_Hours: float = Field(..., ge=0, le=24)
#     Sleep_Hours: float = Field(..., ge=0, le=24)
#     Academic_Performance: int = Field(..., ge=0, le=100)
#     Social_Interactions: int = Field(..., ge=0, le=10)
#     Exercise_Hours: float = Field(..., ge=0, le=24)
#     Anxiety_Level: int = Field(..., ge=0, le=10)
#     Depression_Level: int = Field(..., ge=0, le=10)
#     Self_Esteem: int = Field(..., ge=0, le=10)
#     Parental_Control: int = Field(..., ge=0, le=1)
#     Screen_Time_Before_Bed: float = Field(..., ge=0, le=24)
#     Phone_Checks_Per_Day: int = Field(..., ge=0)
#     Apps_Used_Daily: int = Field(..., ge=0)
#     Time_on_Social_Media: float = Field(..., ge=0, le=24)
#     Time_on_Gaming: float = Field(..., ge=0, le=24)
#     Time_on_Education: float = Field(..., ge=0, le=24)
#     Phone_Usage_Purpose: str
#     Family_Communication: int = Field(..., ge=0, le=10)
#     Weekend_Usage_Hours: float = Field(..., ge=0, le=24)


# @app.get("/")
# def root():
#     return {"service": "Teen Phone Addiction Predictor", "docs": "/docs"}


# @app.get("/health")
# def health():
#     return {"status": "ok"}


# @app.post("/predict")
# def predict(payload: TeenInput):
#     try:
#         raw = pd.DataFrame([payload.model_dump()])
#         pred = pipeline.predict(raw)[0]
#         proba = pipeline.predict_proba(raw)[0]
#         classes = pipeline.named_steps["clf"].classes_
#         return {
#             "predicted_class": str(pred),
#             "probabilities": {str(c): round(float(p), 4) for c, p in zip(classes, proba)},
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
