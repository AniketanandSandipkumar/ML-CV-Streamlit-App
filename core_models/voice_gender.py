import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load voice gender model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "voice_gender.h5")

voice_gender_model = load_model(MODEL_PATH)
# MFCC extractor 
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

        if mfcc.shape[1] < max_len:
            pad_width = max_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            mfcc = mfcc[:, :max_len]

        mfcc = mfcc[..., np.newaxis]  
        mfcc = np.expand_dims(mfcc, axis=0) 
        return mfcc

    except Exception:
        return None

# Predict voice gender
def predict_voice_gender(audio_path):
    """
    Returns:
        "Male" or "Female" or "Unknown"
    """
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        return "Unknown"

    pred = voice_gender_model.predict(mfcc, verbose=0)[0][0]
    return "Male" if pred >= 0.5 else "Female"
