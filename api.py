import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ---------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)


# ---------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Netflix Content Prediction API",
    description="API for predicting whether a Netflix title is a Movie or TV Show",
    version="1.0"
)


# ---------------------------------------------------------
# Input data format
# ---------------------------------------------------------

class PredictionInput(BaseModel):
    release_year: int
    runtimeMinutes: float
    averageRating: float
    numVotes: int


# ---------------------------------------------------------
# Home endpoint
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Netflix Prediction API is running",
        "endpoint": "/predict"
    }


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    # Convert input into DataFrame
    input_data = pd.DataFrame([{
        "release_year": data.release_year,
        "runtimeMinutes": data.runtimeMinutes,
        "averageRating": data.averageRating,
        "numVotes": data.numVotes
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "prediction": prediction
    }