# FitMode App — White Theme Update
from dashboard_module import show_dashboard
from dashboard_module import add_to_session_data
import streamlit as st


import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import threading
import tempfile
import pandas as pd
import os
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# ========== TEXT TO SPEECH ==========
engine = pyttsx3.init()
def speak(text):
    def run_tts():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_tts, daemon=True).start()

# ========== MEDIAPIPE POSE SETUP ==========
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

calories_per_rep = {
    "squat": 0.32,
    "pushup": 0.29,
    "curl": 0.15,
    "lunge": 0.28,
    "press": 0.22
}

class PoseEstimator(VideoTransformerBase):
    def __init__(self, exercise):
        self.exercise = exercise
        self.counter = 0
        self.stage = None
        self.calories = 0
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.logged_reps = set()  # To avoid logging same rep multiple times
        speak(f"Get ready for {self.exercise}s. Let's do this!")

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        return self.process_frame(image)

    def process_frame(self, image):
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            feedback_given = False

            if self.exercise == "squat":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                angle = calculate_angle(hip, knee, ankle) 

                if angle > 160:
                    self.stage = "up"
                if angle < 70 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                    self.calories += calories_per_rep[self.exercise]
                    feedback_given = True

            elif self.exercise == "pushup":
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    self.stage = "up"
                if angle < 90 and self.stage == "up":
                    self.stage = "down"
                    self.counter += 1
                    self.calories += calories_per_rep[self.exercise]
                    feedback_given = True

            elif self.exercise == "curl":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle > 160:
                        self.stage = "down"
                    if angle < 40 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                        self.calories += calories_per_rep[self.exercise]
                        feedback_given = True

            elif self.exercise == "lunge":
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)

                    if angle > 160:
                        self.stage = "up"
                    if angle < 90 and self.stage == "up":
                        self.stage = "down"
                        self.counter += 1
                        self.calories += calories_per_rep[self.exercise]
                        feedback_given = True
            
            elif self.exercise == "press":
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)

                    if angle < 70:
                        self.stage = "down"
                    if angle > 160 and self.stage == "down":
                        self.stage = "up"
                        self.counter += 1
                        self.calories += calories_per_rep[self.exercise]
                        # Log only once per new rep
                        if self.counter not in self.logged_reps:
                            self.logged_reps.add(self.counter)

                            if "session_data" not in st.session_state:
                                st.session_state["session_data"] = [] 

                            st.session_state["session_data"].append({
                                "exercise": self.exercise,
                                "reps": self.counter,
                                "calories": round(self.calories, 2)
                            })

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(image, f"{self.exercise.capitalize()} | Reps: {self.counter} | Calories: {self.calories:.2f} kcal",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        return image

# ========== DIET RECOMMENDATION ==========
df = pd.read_csv("Indian_Nutrient_Databank.csv")
numeric_columns = ['energy_kcal', 'carb_g', 'protein_g', 'fat_g', 'freesugar_g', 'fibre_g', 'sodium_mg', 'potassium_mg', 'cholesterol_mg']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

non_veg_keywords = ["chicken", "egg", "fish", "mutton", "meat", "pork", "beef", "prawn", "shrimp"]
easy_to_digest_non_veg = ["chicken", "egg", "fish"]

def get_diet_cards(age, health, preference):
    filtered = df.copy()
    if health == 'high_protein': filtered = filtered[filtered['protein_g'] > 15]
    elif health == 'low_sugar': filtered = filtered[filtered['freesugar_g'] < 5]
    elif health == 'hypertension': filtered = filtered[filtered['sodium_mg'] < 200]
    elif health == 'kidney_disease': filtered = filtered[filtered['potassium_mg'] < 300]
    elif health == 'diabetes': filtered = filtered[(filtered['carb_g'] < 50) & (filtered['fibre_g'] > 5)]
    elif health == 'obesity': filtered = filtered[(filtered['fat_g'] < 10) & (filtered['energy_kcal'] < 500)]
    elif health == 'heart_disease': filtered = filtered[(filtered['cholesterol_mg'] < 50)]

    if age >= 50:
        if preference == 'non-vegetarian':
            filtered = filtered[filtered['food_name'].str.contains('|'.join(easy_to_digest_non_veg), case=False)]
        else:
            filtered = filtered[~filtered['food_name'].str.contains('|'.join(non_veg_keywords), case=False)]

    if preference == 'vegetarian':
        filtered = filtered[~filtered['food_name'].str.contains('|'.join(non_veg_keywords), case=False)]
    elif preference == 'non-vegetarian':
        filtered = filtered[filtered['food_name'].str.contains('|'.join(non_veg_keywords), case=False)]

    return filtered.sample(n=6 if len(filtered) >= 6 else len(filtered))

# ========== UI STYLES ==========
st.set_page_config(layout="wide")
st.markdown("""
    <style>
    .stApp {
        background: 000000;
        color: #111111;
        font-family: 'Open Sans', sans-serif;
    }
    .food-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        height: 100%;
        color: #111111;
        transition: all 0.3s ease;
    }
    .food-card h4 {
        font-size: 20px;
        font-weight: bold;
        color: #ff9800;
        margin-bottom: 10px;
    }
    ul.food-info {
        list-style: none;
        padding-left: 0;
        line-height: 1.6;
    }
    ul.food-info li::before {
        content: "🥗 ";
    }
    h1, h2, h3, .stMarkdown h2, .stMarkdown h3 {
        text-align: center;
    }
    .bordered-header {
        text-align: center;
        margin-top: -10px;
        margin-bottom: 15px;
        font-size: 26px;
        font-weight: bold;
        color: white;
    }
    .main-title {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: white;
        margin-top: -25px;
        margin-bottom: 30px;
        font-family: 'Open Sans', sans-serif;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(1),
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 20px;
        background-color: #fefefe;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏋️ FitMode: AI Coach + 🥗 Diet Planner</div>", unsafe_allow_html=True)

# Remaining code (left/right columns, webrtc, and diet generation) continues unchanged...

left, right = st.columns([1, 1])

with left:
    workout_container = st.container()

    with workout_container:
        st.markdown("<div class='bordered-header'>💪 Workout Coach</div>", unsafe_allow_html=True)
        exercise = st.selectbox("Choose Exercise", ["squat", "pushup", "curl", "lunge", "press"])
        mode = st.radio("Input Mode", ["Webcam", "Upload Video"], horizontal=True)

        if mode == "Webcam":
            st.info("Start moving. Voice guidance enabled.")
            webrtc_streamer(key="pose", video_transformer_factory=lambda: PoseEstimator(exercise))
        else:
            video_file = st.file_uploader("Upload Workout Video", type=["mp4", "mov", "avi"])
            if video_file:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_file.read())
                cap = cv2.VideoCapture(tfile.name)
                estimator = PoseEstimator(exercise)
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (640, 480))
                    result = estimator.process_frame(frame)
                    stframe.image(result, channels="BGR")
                cap.release()
                st.success(f"Reps: {estimator.counter} | Calories: {estimator.calories:.2f} kcal")
                from dashboard_module import add_to_session_data
                add_to_session_data(exercise, estimator.counter, estimator.calories)

    # This should go RIGHT AFTER the content inside the left column
st.markdown(
    """
    <style>
    [data-testid="stHorizontalBlock"] > div:nth-child(1) {
        border: 2px solid white;
        border-radius: 30px;
        padding: 20px;
        background-color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True
)



with right:
    with st.container():
        # st.markdown("<div style='border: 2px solid white; border-radius: 10px; padding: 20px; background-color: #111;'>", unsafe_allow_html=True)
        st.markdown("<div class='bordered-header'>🥗 Diet Plan</div>", unsafe_allow_html=True)
        age = st.slider("Age", 10, 100, 25)
        condition = st.selectbox("Health Condition", ["none", "high_protein", "low_sugar", "hypertension", "kidney_disease", "diabetes", "obesity", "heart_disease"])
        preference = st.radio("Diet Preference", ["vegetarian", "non-vegetarian"])

        if st.button("Generate Diet Plan"):
            meals_df = get_diet_cards(age, condition, preference)
            for i in range(0, len(meals_df), 2):
                diet_cols = st.columns(2)
                for j in range(2):
                    if i + j < len(meals_df):
                        row = meals_df.iloc[i + j]
                        with diet_cols[j]:
                            st.markdown(f"""
                                <div class='food-card'>
                                    <h4>🍛 {row['food_name']}</h4>
                                    <ul class='food-info'>
                                        <li><b>Energy:</b> {row['energy_kcal']} kcal</li>
                                        <li><b>Protein:</b> {row['protein_g']} g</li>
                                        <li><b>Carbs:</b> {row['carb_g']} g</li>
                                        <li><b>Fat:</b> {row['fat_g']} g</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    # Apply border to the right column
st.markdown(
    """
    <style>
    [data-testid="stHorizontalBlock"] > div:nth-child(2) {
        border: 2px solid white;
        border-radius: 30px;
        padding: 20px;
        background-color: #111;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Show the workout summary dashboard
show_dashboard()



