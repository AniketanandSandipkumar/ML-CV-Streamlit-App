import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Car Color & People Count", layout="centered")
st.title("🚗 Car Color Detection & 👥 People Count")

st.write("Detects cars and people using YOLO and applies color-based logic.")

# Load YOLO model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

def detect_car_color(car_roi):
    avg_color = car_roi.mean(axis=(0, 1))
    # BGR format
    if avg_color[0] > avg_color[2]:
        return "Blue"
    return "Other"

if uploaded_image:
    image = Image.open(uploaded_image)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(frame)

    car_count = 0
    person_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            roi = frame[y1:y2, x1:x2]

            if cls == 2:  # Car
                car_count += 1
                car_color = detect_car_color(roi)

                if car_color == "Blue":
                    box_color = (0, 0, 255)  # Red box
                else:
                    box_color = (255, 0, 0)  # Blue box

                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(
                    frame,
                    f"Car ({car_color})",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    box_color,
                    2,
                )

            elif cls == 0:  # Person
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    st.image(frame, caption="Detection Result", use_container_width=True)

    st.success(f"🚗 Cars detected: {car_count}")
    st.success(f"👥 People detected: {person_count}")
