# Streamlit UI for Korean Sentiment Analysis
# Loads the model directly — no separate API server needed

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Page Config ---
st.set_page_config(
    page_title="Korean Sentiment Analyzer",
    page_icon="🇰🇷",
    layout="centered"
)

# --- Constants ---
# Read model name from config so it's not hardcoded
import yaml
from pathlib import Path

_config_path = Path(__file__).parent / "configs" / "model_config.yaml"
with open(_config_path) as _f:
    _config = yaml.safe_load(_f)
MODEL_NAME = _config["model_name"]

# --- Emotion Emoji Map ---
EMOTION_EMOJIS = {
    "기쁨(행복한)": "😊",
    "슬픔": "😢",
    "분노": "😡",
    "불안": "😰",
    "상처(배신당한)": "💔",
    "당황": "😳",
    "기쁨": "😊",
    "놀람": "😲",
    "혐오": "🤮",
    "공포": "😱",
    "중립": "😐",
}


def get_emoji(label: str) -> str:
    """Get emoji for a given emotion label."""
    return EMOTION_EMOJIS.get(label, "🔮")


@st.cache_resource
def load_model():
    """Load model and tokenizer once, cached across reruns."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def analyze_sentiment(text: str) -> dict | None:
    """Run sentiment prediction directly on the model."""
    try:
        tokenizer, model = load_model()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        probabilities = F.softmax(logits, dim=-1)
        predicted_class_id = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class_id].item()

        if hasattr(model.config, "id2label"):
            label = model.config.id2label[predicted_class_id]
        else:
            label = str(predicted_class_id)

        return {"label": label, "confidence": round(confidence, 4)}
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None


# --- UI ---
st.title("🇰🇷 Korean Sentiment Analyzer")
st.markdown("Analyze the emotion in Korean text using AI (KcELECTRA model)")

# Load model on startup (shows spinner first time)
with st.spinner("Loading AI model... (first time only)"):
    load_model()
st.sidebar.success("✅ Model loaded")

st.divider()

# --- Input Section ---
text_input = st.text_area(
    "Enter Korean text to analyze:",
    placeholder="예: 이 영화 정말 재미있어요!",
    height=120
)

# Example buttons
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("😊 Happy", use_container_width=True):
        text_input = "오늘 하루 너무 행복해요! 좋은 일이 가득했어요."
        st.session_state["text"] = text_input

with col2:
    if st.button("😢 Sad", use_container_width=True):
        text_input = "너무 슬퍼서 눈물이 나요. 왜 이렇게 힘들까."
        st.session_state["text"] = text_input

with col3:
    if st.button("😡 Angry", use_container_width=True):
        text_input = "정말 화가 나요! 이건 너무 불공평해요."
        st.session_state["text"] = text_input

# Use session state for example buttons
if "text" in st.session_state and not text_input:
    text_input = st.session_state["text"]

st.divider()

# --- Analyze Button ---
if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
    if not text_input or not text_input.strip():
        st.warning("Please enter some Korean text first!")
    else:
        with st.spinner("Analyzing..."):
            result = analyze_sentiment(text_input.strip())

        if result:
            label = result["label"]
            confidence = result["confidence"]
            emoji = get_emoji(label)

            # Results display
            st.markdown("### Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    label="Detected Emotion",
                    value=f"{emoji} {label}"
                )

            with result_col2:
                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.1f}%"
                )

            # Confidence bar
            st.progress(confidence)

            # Interpretation
            if confidence >= 0.8:
                st.success(f"High confidence — the model is quite sure this is **{label}**")
            elif confidence >= 0.5:
                st.info(f"Moderate confidence — likely **{label}**, but could be mixed emotions")
            else:
                st.warning(f"Low confidence — the model is unsure. Best guess: **{label}**")

# --- Footer ---
st.divider()
st.caption("Powered by KcELECTRA · Built with Streamlit · Hosted on Streamlit Community Cloud")
