import streamlit as st
import joblib
import os

# Load artifacts
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "vectorizer.pkl"))

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Title
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>
        📰 Fake News Detection Dashboard
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Enter a news headline or article below and check if it's <b style='color:red;'>FAKE</b> or <b style='color:green;'>REAL</b>.
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Input box
user_input = st.text_area("✍️ Paste or type news content here:", height=150)

# Predict button
if st.button("🚀 Analyze"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]

        if prediction.lower() == "fake":
            st.error("❌ This looks like **FAKE NEWS**.")
        else:
            st.success("✅ This seems to be **REAL NEWS**.")
    else:
        st.warning("⚠️ Please enter some text before predicting.")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size:14px; color:gray;'>
        Built with ❤️ using Streamlit | Project by Sriyaa
    </p>
    """,
    unsafe_allow_html=True
)
