import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Global buffer
pose_data_buffer = {"exercise": None, "reps": 0, "calories": 0}

# Calorie map
calories_per_rep = {"press": 0.22}

# Angle calculation helper
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

# Pose estimator class
mp_pose = mp.solutions.pose
class PoseEstimator(VideoTransformerBase):
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.calories = 0
        self.logged_reps = set()
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

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
                self.calories += calories_per_rep["press"]

                if self.counter not in self.logged_reps:
                    self.logged_reps.add(self.counter)
                    pose_data_buffer["exercise"] = "press"
                    pose_data_buffer["reps"] = self.counter
                    pose_data_buffer["calories"] = round(self.calories, 2)
                    print(f"✅ BUFFER UPDATED: {pose_data_buffer}")

        return image

# Streamlit UI
st.title("💪 Press Exercise Tracker (Test)")
webrtc_streamer(key="pose-test", video_transformer_factory=PoseEstimator)

# Sync buffer to session state
if pose_data_buffer["reps"] > 0:
    if "session_data" not in st.session_state:
        st.session_state["session_data"] = []
    st.session_state["session_data"].append(pose_data_buffer.copy())
    st.success(f"Saved: {pose_data_buffer['reps']} reps ({pose_data_buffer['calories']} kcal)")
    pose_data_buffer["reps"] = 0

# Show dashboard
st.markdown("## 📊 Workout Summary")
if "session_data" in st.session_state and st.session_state["session_data"]:
    df = st.session_state["session_data"]
    st.write(df)
else:
    st.info("No data yet — do some reps!")
