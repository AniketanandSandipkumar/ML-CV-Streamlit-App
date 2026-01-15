import cv2
import numpy as np

from core_models.face_age import predict_face_age
from core_models.face_gender import predict_face_gender

# Simple hair length heuristic
def detect_hair_length(face_img):
    """
    Heuristic:
    More dark pixels below face center → long hair
    """
    h, w, _ = face_img.shape
    lower_half = face_img[h//2:, :]

    gray = cv2.cvtColor(lower_half, cv2.COLOR_BGR2GRAY)
    dark_pixels = np.sum(gray < 80)

    return "Long" if dark_pixels > 0.25 * gray.size else "Short"

# Main logic function
def classify_person(face_img):
    """
    Returns:
        age_range, final_gender, hair_length
    """

    age_range, _ = predict_face_age(face_img)
    predicted_gender = predict_face_gender(face_img)
    hair_length = detect_hair_length(face_img)

    # Logic override
    if age_range == "20–30":
        final_gender = "Female" if hair_length == "Long" else "Male"
    else:
        final_gender = predicted_gender

    return age_range, final_gender, hair_length


# DEMO (Image-based)
if __name__ == "__main__":
    img = cv2.imread("assets/test_face.jpg")

    age, gender, hair = classify_person(img)

    print("Age Range:", age)
    print("Hair Length:", hair)
    print("Final Gender:", gender)
