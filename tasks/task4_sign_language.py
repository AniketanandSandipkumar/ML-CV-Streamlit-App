import streamlit as st
import numpy as np
from datetime import datetime
import cv2
import random

st.set_page_config(page_title="Sign Language Detection", layout="centered")

st.title("🤟 Sign Language Detection System")
st.write("⏰ Predictions allowed only between **6 PM and 10 PM**")

current_hour = datetime.now().hour

if not (18 <= current_hour <= 22):
    st.error("🚫 Sign language detection is only active between 6 PM and 10 PM.")
    st.stop()

st.success("✅ System active. You may upload an image.")

uploaded_image = st.file_uploader("Upload hand sign image", type=["jpg", "png", "jpeg"])

SIGNS = ["A", "B", "C", "D", "E"]

def predict_sign(image):
    return random.choice(SIGNS)

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))

    st.image(img, caption="Uploaded Image", channels="BGR")

    sign = predict_sign(img)
    st.subheader(f"✋ Detected Sign: **{sign}**")
