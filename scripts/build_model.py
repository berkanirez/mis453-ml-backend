import joblib
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import os

# 1) Dataset yükle (scikit-learn demo text dataset)
data = load_files("data", categories=["pos", "neg"], encoding="utf-8")

X = data.data
y = data.target

# 2) Train / Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3) Pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

# 4) Train
pipeline.fit(X_train, y_train)

# 5) Evaluation
train_preds = pipeline.predict(X_train)
val_preds = pipeline.predict(X_val)

print("Train Accuracy:", accuracy_score(y_train, train_preds))
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation F1:", f1_score(y_val, val_preds))

# 6) Model kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/sentiment_model.joblib")

print("Model saved to models/sentiment_model.joblib")
