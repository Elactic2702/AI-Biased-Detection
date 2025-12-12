import streamlit as st
import joblib
import pandas as pd
import shap

st.title("AI Recruitment Bias Demo")

# ------------------ Load model + vectorizer ------------------
model = joblib.load("artifacts/logreg_model.pkl")
tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")

# --------------------- Input ---------------------
resume_text = st.text_area("Paste resume text here:")

# --------------------- Prediction ---------------------
if st.button("Predict") and resume_text:
    vect = tfidf.transform([resume_text])
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0][1]

    st.write(f"Prediction: {'Selected ✅' if pred == 1 else 'Rejected ❌'}")
    st.write("Probability of being selected:", round(proba, 2))

    # Save vector for SHAP explanation
    st.session_state.vect = vect

# --------------------- Fairness Metrics ---------------------
if st.checkbox("Show Fairness Metrics"):
    df = pd.read_csv("data/processed/processed_data.csv")
    gender_counts = df['gender'].value_counts()
    st.bar_chart(gender_counts)

# ---------------------- SHAP Explanation ----------------------
if resume_text:
    # Load explainer once
    if "explainer" not in st.session_state:
        df_train = pd.read_csv("data/processed/processed_data.csv")
        # Take a sample of cleaned_resume and vectorize
        background_vect = tfidf.transform(df_train["cleaned_resume"].sample(100, random_state=42))
        st.session_state.explainer = shap.Explainer(model.predict_proba, background_vect)

    # Compute SHAP values for current input
    shap_values = st.session_state.explainer(st.session_state.vect)

    st.subheader("Why this prediction?")
    st.write("The plot below shows the words that influenced the decision:")

    shap_html = shap.plots.text(shap_values[0], display=False)
    st.components.v1.html(shap_html, height=300)
