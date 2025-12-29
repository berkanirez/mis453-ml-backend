from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Sentiment Analysis API",
    description="ML-powered sentiment analysis service",
    version="1.0.0"
)

model = joblib.load("models/sentiment_model.joblib")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/")
def root():
    return {"message": "Sentiment API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(input: TextInput):
    pred = model.predict([input.text])[0]

    if isinstance(pred, str):
        label = pred.upper()
    else:
        label = "POSITIVE" if pred == 1 else "NEGATIVE"

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        confidence = float(model.predict_proba([input.text])[0].max())

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }
