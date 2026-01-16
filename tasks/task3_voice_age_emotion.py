import streamlit as st
import tempfile
import os

from core_models.voice_gender import predict_voice_gender
from core_models.voice_age import predict_voice_age

st.set_page_config(page_title="Voice-based Age Detection", layout="centered")

st.title("🎙️ Voice-based Age Detection (Male-only Logic)")
st.write("⚠️ This system processes **male voices only** as per internship task logic.")

uploaded_audio = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(uploaded_audio.read())
        audio_path = temp.name

    st.audio(audio_path)

    with st.spinner("Analyzing voice..."):
        gender = predict_voice_gender(audio_path)

    st.subheader(f"Detected Gender: {gender}")

    # 🚫 Female logic
    if gender.lower() == "female":
        st.error("🚫 This task is restricted to male voices only.")
        os.remove(audio_path)
        st.stop()

    # ✅ Age prediction
    age_group = predict_voice_age(audio_path)
    st.subheader(f"Predicted Age Group: {age_group}")

    # 🧠 LOGIC-BASED emotion handling
    if age_group == "60+":
        st.info("Senior citizen detected.")
        st.write("🧠 Emotion Analysis: **Not enabled (Logic-based task)**")
        st.write("📌 As per task requirement, emotion detection is conditionally triggered for seniors.")
    else:
        st.info("Emotion analysis not required for non-senior age groups.")

    os.remove(audio_path)
