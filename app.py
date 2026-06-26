"""
Streamlit Demo App — Phishing URL Detection Framework
-------------------------------------------------------
Paste a URL, get an instant phishing / legitimate prediction using the
trained Random Forest model from the lexical-feature-engineering pipeline.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy for free (so you can show examiners a live link):
    1. Push this folder to a public GitHub repo
    2. Go to https://share.streamlit.io , sign in with GitHub
    3. Point it at app.py in your repo — it builds and hosts it automatically
"""

import streamlit as st
import pandas as pd
import joblib
from feature_extraction import extract_features

st.set_page_config(page_title="Phishing URL Detector", page_icon="🔍", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("🔍 Phishing URL Detector")
st.caption("An Intelligent Framework for Phishing URL Detection using Lexical Feature Engineering and Machine Learning")

url_input = st.text_input("Enter a URL to check:", placeholder="https://example.com/login")

if st.button("Analyze URL", type="primary") and url_input.strip():
    raw_features = extract_features(url_input)
    feature_df = pd.DataFrame([raw_features])[feature_columns]
    scaled_features = scaler.transform(feature_df)

    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0]
    phishing_confidence = probability[1] * 100

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ Likely PHISHING — {phishing_confidence:.1f}% confidence")
    else:
        st.success(f"✅ Likely LEGITIMATE — {100 - phishing_confidence:.1f}% confidence")

    st.progress(min(int(phishing_confidence), 100), text=f"Phishing risk score: {phishing_confidence:.1f}%")

    with st.expander("See extracted lexical features"):
        st.dataframe(
            pd.DataFrame(raw_features.items(), columns=["Feature", "Value"]),
            hide_index=True,
            use_container_width=True,
        )

st.divider()
st.caption(
    "Model: Random Forest trained on PhishTank (phishing) and UNB CIC (legitimate) URL datasets. "
    "Predictions are based on lexical/structural features of the URL string only — "
    "no live page content is fetched or analyzed."
)
