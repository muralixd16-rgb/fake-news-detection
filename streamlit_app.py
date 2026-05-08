import streamlit as st
import requests

# ================= PAGE CONFIG =================

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ================= CUSTOM CSS =================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 700;
    text-align: center;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

.model-badge-tfidf {
    display: inline-block;
    background: linear-gradient(135deg, #10b981, #059669);
    color: white;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.model-badge-bert {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

.result-fake {
    background: linear-gradient(135deg, #fee2e2, #fecaca);
    border-left: 5px solid #ef4444;
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    font-size: 1.3rem;
    font-weight: 700;
    color: #b91c1c;
}

.result-real {
    background: linear-gradient(135deg, #d1fae5, #a7f3d0);
    border-left: 5px solid #10b981;
    padding: 1.2rem 1.5rem;
    border-radius: 10px;
    font-size: 1.3rem;
    font-weight: 700;
    color: #065f46;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================

st.markdown('<div class="main-title">📰 Fake News Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect whether a news article is REAL or FAKE using ML & Deep Learning</div>', unsafe_allow_html=True)

st.divider()

# ================= MODEL SELECTOR =================

col_l, col_r = st.columns(2)

with col_l:
    model_choice = st.radio(
        "🤖 Choose Model",
        options=["TF-IDF + Passive Aggressive", "BERT (Deep Learning)"],
        index=0,
        help="TF-IDF is faster. BERT is more accurate but slower."
    )

with col_r:
    if model_choice == "TF-IDF + Passive Aggressive":
        st.markdown("""
        **TF-IDF Model**
        - ⚡ Very fast predictions
        - 🪶 Lightweight & low memory
        - 📊 Moderate accuracy
        """)
    else:
        st.markdown("""
        **BERT Model**
        - 🧠 Contextual understanding
        - 🎯 High accuracy
        - 🐢 Slower inference
        """)

st.divider()

# ================= TEXT INPUT =================

text = st.text_area(
    "📄 Paste your news article here:",
    height=200,
    placeholder="e.g. Government introduces new tax policy affecting millions of citizens..."
)

# ================= DETECT BUTTON =================

if st.button("🔍 Detect", use_container_width=True, type="primary"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text first!")

    else:
        # Pick endpoint based on model choice
        if model_choice == "TF-IDF + Passive Aggressive":
            endpoint = "http://127.0.0.1:8001/predict"
            model_label = "TF-IDF + Passive Aggressive Classifier"
            badge_html = '<span class="model-badge-tfidf">TF-IDF</span>'
        else:
            endpoint = "http://127.0.0.1:8001/predict_bert"
            model_label = "BERT (bert-base-uncased fine-tuned)"
            badge_html = '<span class="model-badge-bert">BERT</span>'

        with st.spinner(f"Analyzing with {model_label}..."):
            try:
                response = requests.post(endpoint, json={"text": text}, timeout=60)
                response.raise_for_status()
                result = response.json()

                label      = result["label"]
                confidence = result["confidence"]

                st.divider()

                # ================= RESULT CARD =================

                if label == "FAKE":
                    st.markdown(
                        '<div class="result-fake">🚨 This news is predicted as <u>FAKE</u></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="result-real">✅ This news is predicted as <u>REAL</u></div>',
                        unsafe_allow_html=True
                    )

                st.markdown("")

                # ================= CONFIDENCE =================

                st.markdown(f"#### Confidence Score: `{confidence * 100:.1f}%`")
                st.progress(float(confidence))

                # ================= METRICS =================

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Prediction", label)

                with col2:
                    st.metric("Confidence", f"{confidence * 100:.1f}%")

                with col3:
                    st.metric("Model", "BERT" if "BERT" in model_choice else "TF-IDF")

                # ================= MODEL INFO =================

                st.markdown(f"Model Used: {badge_html} {model_label}", unsafe_allow_html=True)

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ **Cannot connect to FastAPI backend.**\n\n"
                    "Make sure the FastAPI server is running:\n"
                    "```\ncd app\npython -m uvicorn main:app --port 8001\n```"
                )

            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ **Request timed out.** BERT inference can be slow. "
                    "Please wait and try again."
                )

            except Exception as e:
                st.error(f"⚠️ Unexpected error:\n\n`{e}`")

# ================= FOOTER =================

st.divider()
st.caption("Fake News Detection System  |  TF-IDF vs BERT Comparative Analysis  |  Murali Manohar")