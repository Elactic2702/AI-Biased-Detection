import streamlit as st
import joblib
import pandas as pd
import shap

# ------------------ Page config ------------------
st.set_page_config(page_title="AI Recruitment Bias Demo", layout="wide")

st.title("🤖 AI Recruitment Bias Demo")

# ------------------ Bias Mitigation Toggle ------------------
use_mitigation = st.checkbox("⚖️ Apply Bias Mitigation")

# ------------------ Load model + vectorizer ------------------
@st.cache_resource
def load_model(use_mitigation: bool):
    if use_mitigation:
        model = joblib.load("artifacts/logreg_model_mitigated.pkl")
    else:
        model = joblib.load("artifacts/logreg_model.pkl")

    tfidf = joblib.load("artifacts/tfidf_vectorizer.pkl")
    return model, tfidf

model, tfidf = load_model(use_mitigation)

if use_mitigation:
    st.info("⚖️ Bias-mitigated model is active")
else:
    st.warning("⚠️ Baseline model is active (may contain bias)")

# --------------------- Layout ---------------------
col1, col2 = st.columns([1, 2])

# ================= LEFT COLUMN ===================
with col1:
    resume_text = st.text_area("📄 Paste Resume Text Here:", height=200)

    if st.button("Predict"):
        if resume_text.strip() == "":
            st.warning("Please paste resume text first.")
        else:
            vect = tfidf.transform([resume_text])
            pred = model.predict(vect)[0]
            proba = model.predict_proba(vect)[0][1]

            # Save for SHAP
            st.session_state.vect = vect
            st.session_state.pred = pred
            st.session_state.proba = proba

# ================= RIGHT COLUMN ==================
with col2:
    if "pred" in st.session_state:
        st.markdown("### 🏷 Prediction")

        st.markdown(
            f"<h3 style='color: {'green' if st.session_state.pred==1 else 'red'};'>"
            f"{'Selected ✅' if st.session_state.pred==1 else 'Rejected ❌'}"
            f"</h3>",
            unsafe_allow_html=True
        )

        st.progress(st.session_state.proba)

    # ------------------ SHAP Explanation ------------------
    with st.expander("🔍 Show SHAP Explanation"):
        if resume_text and "vect" in st.session_state:

            # Recreate explainer if model toggled
            if "explainer" not in st.session_state or st.session_state.get("explainer_model") != use_mitigation:
                df_train = pd.read_csv("data/processed/processed_data.csv")
                background_texts = df_train["cleaned_resume"].sample(50, random_state=42).tolist()
                background_vect = tfidf.transform(background_texts)

                st.session_state.explainer = shap.Explainer(
                    model.predict_proba,
                    background_vect
                )
                st.session_state.explainer_model = use_mitigation

            # Transform resume text for SHAP
            resume_vect = tfidf.transform([resume_text])
            shap_values = st.session_state.explainer(resume_vect)
            shap_html = shap.plots.text(shap_values[0], display=False)
            st.components.v1.html(shap_html, height=350)

# ------------------ Fairness Metrics ------------------
with st.expander("📊 Fairness Metrics"):
    df = pd.read_csv("data/processed/processed_data.csv")

    st.markdown("#### Gender Distribution")
    gender_counts = df["gender"].value_counts()
    st.bar_chart(gender_counts)

    st.markdown("#### Selection Rate by Gender")
    selection_rates = df.groupby("gender")["label"].mean()
    st.bar_chart(selection_rates, use_container_width=True)

    gap = abs(selection_rates.max() - selection_rates.min())
    if gap > 0.2:
        st.error(f"⚠️ High bias detected (gap = {gap:.2f})")
    else:
        st.success(f"✅ Bias within acceptable limits (gap = {gap:.2f})")
