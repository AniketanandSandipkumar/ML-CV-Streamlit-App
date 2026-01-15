import streamlit as st

st.set_page_config(page_title="AI Vision & Voice System", layout="wide")

st.sidebar.title("📌 Internship Tasks")

task = st.sidebar.selectbox(
    "Select Task",
    [
        "Task 1 – Long Hair Detection",
        "Task 2 – Senior Citizen Detection",
        "Task 3 – Voice Age & Emotion",
        "Task 4 – Sign Language Detection",
        "Task 5 – Car Color & People Count",
        "Task 6 – Nationality Detection"
    ]
)

if task == "Task 1 – Long Hair Detection":
    import tasks.task1_long_hair

elif task == "Task 2 – Senior Citizen Detection":
    import tasks.task2_senior_citizen

elif task == "Task 3 – Voice Age & Emotion":
    import tasks.task3_voice_age_emotion

elif task == "Task 4 – Sign Language Detection":
    import tasks.task4_sign_language

elif task == "Task 5 – Car Color & People Count":
    import tasks.task5_car_color_people

elif task == "Task 6 – Nationality Detection":
    import tasks.task6_nationality_logic
