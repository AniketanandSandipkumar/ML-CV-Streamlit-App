import cv2
import csv
from datetime import datetime

from core_models.face_age import predict_face_age
from core_models.face_gender import predict_face_gender

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# CSV Logger Setup
csv_file = "logs/senior_citizen_log.csv"

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Age Range", "Gender", "Senior Citizen"])

# Age group → numeric decision
def is_senior(age_range):
    return age_range in ["60+", "70+", "80+"]

# Main video loop
cap = cv2.VideoCapture(0)

print("Press 'q' to exit...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        age_range, _ = predict_face_age(face_img)
        gender = predict_face_gender(face_img)

        senior_flag = "YES" if is_senior(age_range) else "NO"

        label = f"{gender}, {age_range}, Senior: {senior_flag}"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

        # Log to CSV
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                age_range,
                gender,
                senior_flag
            ])

    cv2.imshow("Senior Citizen Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
