import streamlit as st
import cv2
import numpy as np

st.set_page_config(page_title="Car Color & People Count", layout="centered")
st.title("🚗 Car Color Detection & People Counting")

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Load detectors
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

car_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_car.xml"
)

def detect_dominant_color(image):
    image = image.reshape((-1, 3))
    avg_color = np.mean(image, axis=0)
    b, g, r = avg_color
    return "blue" if b > r and b > g else "other"

if uploaded_image:
    file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (640, 480))

    # People detection
    people, _ = hog.detectMultiScale(img_resized)
    people_count = len(people)

    # Car detection
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw people boxes
    for (x, y, w, h) in people:
        cv2.rectangle(img_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw car boxes with color logic
    for (x, y, w, h) in cars:
        car_roi = img_resized[y:y+h, x:x+w]
        color = detect_dominant_color(car_roi)

        box_color = (0, 0, 255) if color == "blue" else (255, 0, 0)
        cv2.rectangle(img_resized, (x, y), (x+w, y+h), box_color, 3)

    st.image(img_resized, channels="BGR", caption="Detection Result")

    st.subheader(f"👥 People Count: {people_count}")
    st.subheader(f"🚗 Cars Detected: {len(cars)}")
