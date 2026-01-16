import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load emotion model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "face_emotion.h5")

emotion_model = load_model(MODEL_PATH, compile=False)

# Emotion class mapping
EMOTION_MAP = {
    0: "Angry",
    1: "Happy",
    2: "Sad",
    3: "Neutral"
}

# Preprocess face image
def preprocess_face(face_img):
    """
    Input:
        face_img -> BGR or RGB image
    Output:
        (1, 48, 48, 1)
    """
    if face_img is None:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(gray, (48, 48))
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)

    return gray

# Predict emotion
def predict_face_emotion(face_img):
    """
    Returns:
        emotion (str)
    """
    processed = preprocess_face(face_img)
    if processed is None:
        return "Unknown"

    preds = emotion_model.predict(processed, verbose=0)
    emotion_class = np.argmax(preds)

    return EMOTION_MAP.get(emotion_class, "Unknown")
