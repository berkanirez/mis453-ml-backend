import joblib
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")



data = load_files(DATA_DIR, categories=["pos", "neg"], encoding="utf-8")


X = data.data
y = data.target


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])


pipeline.fit(X_train, y_train)


train_preds = pipeline.predict(X_train)
val_preds = pipeline.predict(X_val)

print("Train Accuracy:", accuracy_score(y_train, train_preds))
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print("Validation F1:", f1_score(y_val, val_preds))


os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(pipeline, os.path.join(MODELS_DIR, "sentiment_model.joblib"))


print("Model saved to models/sentiment_model.joblib")
