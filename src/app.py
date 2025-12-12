import streamlit as st
import joblib
import pandas as pd
import shap

st.set_page_config(page_title="AI Recruitment Bias Demo", layout="wide")

st.title("🤖 AI Recruitment Bias Demo")

# ------------------ Load model + vectorizer ------------------
@st.cache_resource
def load_model():
    model = joblib.load("artifacts/logreg_model.pkl")
    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model()

# --------------------- Layout ---------------------
col1, col2 = st.columns([1,2])

with col1:
    resume_text = st.text_area("📄 Paste Resume Text Here:", height=200)
    
    if st.button("Predict"):
        vect = tfidf.transform([resume_text])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0][1]

        # Save for SHAP
        st.session_state.vect = vect
        st.session_state.pred = pred
        st.session_state.proba = proba

with col2:
    if "pred" in st.session_state:
        st.markdown("### 🏷 Prediction")
        st.markdown(f"<h3 style='color: {'green' if st.session_state.pred==1 else 'red'};'>"
                    f"{'Selected ✅' if st.session_state.pred==1 else 'Rejected ❌'}</h3>",
                    unsafe_allow_html=True)
        st.progress(st.session_state.proba)

    with st.expander("🔍 Show SHAP Explanation"):
        if resume_text and "vect" in st.session_state:
            # Load explainer once
            if "explainer" not in st.session_state:
                df_train = pd.read_csv("data/processed/processed_data.csv")
                background_texts = df_train["cleaned_resume"].sample(50, random_state=42).tolist()
                st.session_state.explainer = shap.Explainer(
                    model.predict_proba, background_texts, vectorizer=tfidf
                )

            shap_values = st.session_state.explainer([resume_text])
            shap_html = shap.plots.text(shap_values[0], display=False)
            st.components.v1.html(shap_html, height=350)

with st.expander("📊 Fairness Metrics"):
    df = pd.read_csv("data/processed/processed_data.csv")
    gender_counts = df['gender'].value_counts()
    st.bar_chart(gender_counts)

    selection_rates = df.groupby("gender")["label"].mean()
    st.bar_chart(selection_rates, use_container_width=True)
