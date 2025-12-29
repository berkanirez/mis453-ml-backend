from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple ML-powered sentiment analysis service",
    version="1.0.0"
)

# Model yükle (startup'ta)
model = joblib.load("models/sentiment_model.joblib")

class TextInput(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(input: TextInput):
    pred = model.predict([input.text])[0]
    proba = model.predict_proba([input.text])[0].max()

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    return {
        "label": label,
        "confidence": round(float(proba), 2)
    }
