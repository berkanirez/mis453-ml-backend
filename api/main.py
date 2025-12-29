from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Sentiment Analysis API",
    description="A simple ML-powered sentiment analysis service",
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

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(input: TextInput):
    prediction = model.predict([input.text])[0]

    # Label güvenli belirleme
    if isinstance(prediction, str):
        label = prediction.upper()
    else:
        label = "POSITIVE" if prediction == 1 else "NEGATIVE"

    # Confidence güvenli hesaplama
    confidence = 0.5
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba([input.text])[0]
        confidence = float(max(proba))

    return {
        "label": label,
        "confidence": round(confidence, 2)
    }

