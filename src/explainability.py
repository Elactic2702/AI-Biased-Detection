import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
import os

# Load processed data
df = pd.read_csv("data/processed/processed_data.csv")
X_text = df["cleaned_resume"]
y = df["label"]

# Load trained model and TF-IDF
model = joblib.load("artifacts/logreg_model.pkl")
tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
X_vect = tfidf.transform(X_text)

# --------------------------
# SHAP
# --------------------------
explainer = shap.LinearExplainer(model, X_vect, feature_perturbation="interventional")
shap_values = explainer.shap_values(X_vect)

# Plot summary
os.makedirs("reports/figures", exist_ok=True)
shap.summary_plot(shap_values, X_vect, feature_names=tfidf.get_feature_names_out(), show=False)
plt.savefig("reports/figures/shap_summary.png")
print("✅ SHAP summary plot saved at reports/figures/shap_summary.png")

# --------------------------
# LIME
# --------------------------
class_names = ["Rejected", "Selected"]
explainer_lime = LimeTextExplainer(class_names=class_names)

sample_idx = 0
exp = explainer_lime.explain_instance(X_text.iloc[sample_idx], 
                                      lambda x: model.predict_proba(tfidf.transform(x)), 
                                      num_features=10)
exp.save_to_file("reports/figures/lime_explanation.html")
print(f"✅ LIME explanation saved at reports/figures/lime_explanation.html")
