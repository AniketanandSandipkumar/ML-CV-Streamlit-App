import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "face_age.h5")

age_model = load_model(MODEL_PATH)


# Age class mapping 
AGE_CLASS_MAP = {
    0: "0-20",
    1: "21-30",
    2: "31-60",
    3: "60+"
}

# Preprocess face image
def preprocess_face(face_img):
    """
    Input:
        face_img -> BGR or RGB image (numpy array)
    Output:
        preprocessed image of shape (1, 64, 64, 3)
    """
    if face_img is None:
        return None

    # Ensure RGB
    if face_img.shape[-1] == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

    face_img = cv2.resize(face_img, (64, 64))
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    return face_img


# Predict age group
def predict_face_age(face_img):
    """
    Returns:
        age_group (str)
    """
    processed = preprocess_face(face_img)
    if processed is None:
        return "Unknown"

    preds = age_model.predict(processed, verbose=0)
    age_class = np.argmax(preds)

    return AGE_CLASS_MAP.get(age_class, "Unknown")
