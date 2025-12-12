import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv("data/processed/processed_data.csv")

X = df["cleaned_resume"]
y = df["label"]
gender = df["gender"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    X, y, gender, test_size=0.2, random_state=42
)

# -----------------------------
# TF-IDF vectorization
# -----------------------------
tfidf = TfidfVectorizer(max_features=3000)
X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)

# -----------------------------
# Normal model
# -----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train_vect, y_train)

# -----------------------------
# Bias-mitigated model (REWEIGHTING)
# Give higher weight to unprivileged group (gender = 0)
# -----------------------------
weights = g_train.apply(lambda x: 2 if x == 0 else 1)

model_mitigated = LogisticRegression(max_iter=200)
model_mitigated.fit(
    X_train_vect,
    y_train,
    sample_weight=weights
)

# -----------------------------
# Save models
# -----------------------------
os.makedirs("artifacts", exist_ok=True)

joblib.dump(model, "artifacts/logreg_model.pkl")
joblib.dump(model_mitigated, "artifacts/logreg_model_mitigated.pkl")
joblib.dump(tfidf, "artifacts/tfidf_vectorizer.pkl")

print("✅ Both normal and bias-mitigated models saved.")
