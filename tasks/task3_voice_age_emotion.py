import streamlit as st
import tempfile
import os

from core_models.voice_gender import predict_voice_gender
from core_models.voice_age import predict_voice_age
from core_models.voice_emotion import predict_voice_emotion  # if available

st.set_page_config(page_title="Voice Age & Emotion Detection", layout="centered")

st.title("🎙️ Voice-based Age & Emotion Detection")
st.write("⚠️ This system processes **male voices only** as per task requirement.")

uploaded_audio = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        temp.write(uploaded_audio.read())
        audio_path = temp.name

    st.audio(audio_path)

    with st.spinner("Analyzing voice..."):
        gender = predict_voice_gender(audio_path)

    st.subheader(f"Detected Gender: {gender}")

    if gender.lower() == "female":
        st.error("🚫 Emotion detection is disabled for female voices.")
        os.remove(audio_path)
        st.stop()

    age_group = predict_voice_age(audio_path)
    st.subheader(f"Predicted Age Group: {age_group}")

    if age_group in ["60+", "70+", "80+"]:
        st.info("Senior detected → Running emotion analysis...")

        emotion = predict_voice_emotion(audio_path)
        st.subheader(f"Detected Emotion: {emotion}")
    else:
        st.info("Emotion analysis is only performed for senior citizens.")

    os.remove(audio_path)
