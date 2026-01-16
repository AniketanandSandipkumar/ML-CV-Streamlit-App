import librosa
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load voice age model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "voice_age.h5")

voice_age_model = load_model(MODEL_PATH, compile=False)
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

# Convert numeric age to range
def age_to_range(age_value):
    if age_value < 20:
        return "Below 20"
    elif age_value < 30:
        return "20–30"
    elif age_value < 40:
        return "30–40"
    elif age_value < 60:
        return "40–60"
    else:
        return "60+ (Senior)"

# Predict voice age
def predict_voice_age(audio_path):
    """
    Returns:
        age_range (str), raw_age_value (float)
    """
    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        return "Unknown", None

    age_value = voice_age_model.predict(mfcc, verbose=0)[0][0]
    return age_to_range(age_value), round(float(age_value), 2)
