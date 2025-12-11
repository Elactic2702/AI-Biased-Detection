import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)
import joblib
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# LOAD PROCESSED DATA
# ---------------------------------------------------------
def load_data():
    df = pd.read_csv("data/processed/processed_data.csv")
    return df


# ---------------------------------------------------------
# LOAD TRAINED MODEL
# ---------------------------------------------------------
def load_model():
    model_path = "artifacts/logreg_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Trained model not found. Run train.py first.")
    return joblib.load(model_path)


# ---------------------------------------------------------
# RUN FAIRNESS METRICS
# ---------------------------------------------------------
def run_fairness_evaluation(model, df):

    X_text = df["cleaned_resume"]
    y_true = df["label"]
    sensitive = df["gender"]   # protected attribute

    # TF-IDF vectorization (same as training)
    tfidf = TfidfVectorizer(max_features=3000)
    X_vectorized = tfidf.fit_transform(X_text)

    y_pred = model.predict(X_vectorized)

    print("\n📊 MODEL PERFORMANCE")
    print("----------------------")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))

    # Fairlearn MetricFrame
    metrics = {
        "accuracy": accuracy_score,
        "selection_rate": selection_rate
    }

    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)

    print("\n🎯 GROUP-WISE METRICS (Fairlearn)")
    print(mf.by_group)
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    dp_ratio = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)


    print("\n⚖️ FAIRNESS METRICS")
    print("----------------------")
    print("Demographic Parity Difference:", dp_diff)
    print("Demographic Parity Ratio:", dp_ratio)
    print("Equalized Odds Difference:", eo_diff)

    # Save report
    os.makedirs("reports", exist_ok=True)
    report_path = "reports/bias_report.csv"
    mf.by_group.to_csv(report_path)
    print(f"\n📄 Fairness report saved to: {report_path}")

    # Plot
    plt.figure(figsize=(6, 4))
    mf.by_group["selection_rate"].plot(kind="bar")
    plt.title("Selection Rate by Gender")
    plt.ylabel("Rate")
    plt.savefig("reports/figures/selection_rate_by_gender.png")
    print("📊 Fairness plot saved to: reports/figures/selection_rate_by_gender.png")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("🔍 Loading processed data...")
    df = load_data()

    print("🧠 Loading trained model...")
    model = load_model()

    print("⚖️ Running fairness analysis...")
    run_fairness_evaluation(model, df)

    print("\n✅ Fairness evaluation complete!")
