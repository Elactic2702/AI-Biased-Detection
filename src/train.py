import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv("data/processed/processed_data.csv")
X = df["cleaned_resume"]
y = df["label"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# TF-IDF vectorization
# -----------------------------
tfidf = TfidfVectorizer(max_features=3000)
X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)

# -----------------------------
# Train Logistic Regression
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vect, y_train)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = model.predict(X_test_vect)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# -----------------------------
# Save model and vectorizer
# -----------------------------
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/logreg_model.pkl")
joblib.dump(tfidf, "artifacts/tfidf_vectorizer.pkl")

print("✅ Training complete. Model saved in artifacts/")
