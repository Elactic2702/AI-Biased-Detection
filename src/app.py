import streamlit as st
import joblib
import pandas as pd

st.title("AI Recruitment Bias Demo")

# Load model and vectorizer
model = joblib.load("artifacts/logreg_model.pkl")
tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")

# Input resume text
resume_text = st.text_area("Paste resume text here:")

if st.button("Predict"):
    vect = tfidf.transform([resume_text])
    pred = model.predict(vect)[0]
    proba = model.predict_proba(vect)[0][1]
    st.write("Prediction:", "Selected ✅" if pred==1 else "Rejected ❌")
    st.write("Probability of being selected:", round(proba, 2))

# Display fairness metrics
if st.checkbox("Show Fairness Metrics"):
    df = pd.read_csv("data/processed/processed_data.csv")
    gender_counts = df["gender"].value_counts()
    st.bar_chart(gender_counts)
