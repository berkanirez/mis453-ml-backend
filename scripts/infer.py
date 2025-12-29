import joblib

model = joblib.load("models/sentiment_model.joblib")

while True:
    text = input("Enter text (or 'q' to quit): ")
    if text.lower() == "q":
        break

    pred = model.predict([text])[0]
    proba = model.predict_proba([text])[0].max()

    label = "POSITIVE" if pred == 1 else "NEGATIVE"
    print(f"Prediction: {label} (confidence={proba:.2f})")
