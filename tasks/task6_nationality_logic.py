import streamlit as st
import cv2
import numpy as np

from core_models.face_emotion import predict_face_emotion

st.set_page_config(page_title="Nationality Detection Logic", layout="centered")
st.title("🌍 Nationality Detection (Logic Showcase)")

uploaded_image = st.file_uploader("Upload Face Image", type=["jpg", "png", "jpeg"])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def estimate_skin_tone(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(face_img)
    avg_v = np.mean(v)

    if avg_v < 80:
        return "African"
    elif avg_v < 140:
        return "Indian"
    elif avg_v >= 140:
        return "USA"
    else:
        return "Others"

def detect_dress_color(image, face_box):
    x, y, w, h = face_box
    dress_region = image[y+h:y+2*h, x:x+w]
    if dress_region.size == 0:
        return "Unknown"

    avg_color = np.mean(dress_region.reshape(-1, 3), axis=0)
    b, g, r = avg_color

    if r > g and r > b:
        return "Red"
    elif b > r and b > g:
        return "Blue"
    elif g > r and g > b:
        return "Green"
    else:
        return "Mixed"

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("No face detected")
    else:
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]

        nationality = estimate_skin_tone(face)
        emotion = predict_emotion(face)
        dress_color = detect_dress_color(img, (x, y, w, h))

        st.image(img, channels="BGR", caption="Input Image")

        st.subheader("🧾 Output Panel")

        if nationality == "Indian":
            st.write(f"🇮🇳 Nationality: Indian")
            st.write(f"🎭 Emotion: {emotion}")
            st.write(f"👕 Dress Color: {dress_color}")

        elif nationality == "USA":
            st.write(f"🇺🇸 Nationality: USA")
            st.write(f"🎭 Emotion: {emotion}")

        elif nationality == "African":
            st.write(f"🌍 Nationality: African")
            st.write(f"🎭 Emotion: {emotion}")
            st.write(f"👕 Dress Color: {dress_color}")

        else:
            st.write(f"🌐 Nationality: Others")
            st.write(f"🎭 Emotion: {emotion}")
