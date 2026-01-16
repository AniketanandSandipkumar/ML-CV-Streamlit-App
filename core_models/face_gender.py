import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load model (once)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "face_gender.h5")

gender_model = load_model(MODEL_PATH, compile=False)
# Gender mapping
GENDER_MAP = {
    0: "Male",
    1: "Female"
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

# Predict gender
def predict_face_gender(face_img):
    """
    Returns:
        gender (str): 'Male' or 'Female'
    """
    processed = preprocess_face(face_img)
    if processed is None:
        return "Unknown"

    pred = gender_model.predict(processed, verbose=0)

    # Binary classifier → sigmoid output
    gender_class = int(pred[0][0] > 0.5)

    return GENDER_MAP.get(gender_class, "Unknown")
