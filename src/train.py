import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing

import joblib
import os

# -----------------------------
# Load processed data
# -----------------------------
df = pd.read_csv("data/processed/processed_data.csv")

X = df["cleaned_resume"]
y = df["label"]
gender = df["gender"]   # ✅ protected attribute

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

# ==========================================================
# 🔹 MODEL 1: NORMAL (NO BIAS MITIGATION)
# ==========================================================
model = LogisticRegression(max_iter=200)
model.fit(X_train_vect, y_train)

y_pred = model.predict(X_test_vect)

print("🔹 Normal Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save normal model
os.makedirs("artifacts", exist_ok=True)
joblib.dump(model, "artifacts/logreg_model.pkl")

# ==========================================================
# 🔹 MODEL 2: BIAS-MITIGATED (REWEIGHING)
# ==========================================================

# Prepare dataframe for AIF360
fair_df = pd.DataFrame({
    "label": y_train.values,
    "gender": g_train.values
})

# Add vectorized features
for i in range(X_train_vect.shape[1]):
    fair_df[f"f{i}"] = X_train_vect[:, i].toarray().ravel()

dataset = BinaryLabelDataset(
    df=fair_df,
    label_names=["label"],
    protected_attribute_names=["gender"]
)

# Apply reweighing
RW = Reweighing(
    unprivileged_groups=[{"gender": 0}],
    privileged_groups=[{"gender": 1}]
)

dataset_rw = RW.fit_transform(dataset)

# Train mitigated model
model_mitigated = LogisticRegression(max_iter=200)
model_mitigated.fit(
    dataset_rw.features,
    dataset_rw.labels.ravel(),
    sample_weight=dataset_rw.instance_weights
)

# Evaluate mitigated model
y_pred_mitigated = model_mitigated.predict(X_test_vect)

print("\n🔹 Mitigated Model")
print("Accuracy:", accuracy_score(y_test, y_pred_mitigated))
print("Precision:", precision_score(y_test, y_pred_mitigated))
print("Recall:", recall_score(y_test, y_pred_mitigated))
print("F1 Score:", f1_score(y_test, y_pred_mitigated))

# Save mitigated model
joblib.dump(model_mitigated, "artifacts/logreg_model_mitigated.pkl")

# Save vectorizer
joblib.dump(tfidf, "artifacts/tfidf_vectorizer.pkl")

print("\n✅ Training complete. Both models saved.")
