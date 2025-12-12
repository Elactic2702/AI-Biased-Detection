import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np

st.set_page_config(page_title="AI Recruitment Bias Demo", layout="wide")
st.title("AI Recruitment Bias Demo")

# ------------------ Load model + vectorizer ------------------
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("artifacts/logreg_model.pkl")
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model_and_vectorizer()

# --------------------- Input ---------------------
resume_text = st.text_area("Paste resume text here:")

# --------------------- Prediction ---------------------
if st.button("Predict") and resume_text:
    vect = tfidf.transform([resume_text]).toarray()  # convert sparse to dense
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0][1]

    st.write(f"Prediction: {'Selected ✅' if pred == 1 else 'Rejected ❌'}")
    st.write("Probability of being selected:", round(proba, 2))

    # Save vector for SHAP
    st.session_state.vect = vect

# --------------------- Fairness Metrics ---------------------
if st.checkbox("Show Fairness Metrics"):
    df = pd.read_csv("data/processed/processed_data.csv")
    st.subheader("Gender Distribution in Dataset")
    gender_counts = df['gender'].value_counts()
    st.bar_chart(gender_counts)

    st.subheader("Selection Rate by Gender")
    selection_by_gender = df.groupby('gender')['label'].mean()
    st.bar_chart(selection_by_gender)

# ---------------------- SHAP Explanation ----------------------
if resume_text:

    # Load explainer once
    if "explainer" not in st.session_state:
        df_train = pd.read_csv("data/processed/processed_data.csv")
        background_texts = df_train["cleaned_resume"].sample(100, random_state=42).tolist()

        # Use a Text masker for SHAP
        masker = shap.maskers.Text(tfidf)
        st.session_state.explainer = shap.Explainer(model.predict_proba, masker)

    # Compute SHAP values for current input
    shap_values = st.session_state.explainer([resume_text])

    st.subheader("Why this prediction?")
    st.write("The plot below shows the words that influenced the decision:")

    # Display SHAP plot
    shap_html = shap.plots.text(shap_values[0], display=False)
    st.components.v1.html(shap_html, height=300)

