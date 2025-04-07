import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import numpy as np
import av
import time
from datetime import datetime

st.title("🧠 Head Position & Attention Tracker")

# Initialize session state
if 'eye_contact_list' not in st.session_state:
    st.session_state.eye_contact_list = []
if 'attention_level_list' not in st.session_state:
    st.session_state.attention_level_list = []
if 'head_position_logs' not in st.session_state:
    st.session_state.head_position_logs = []
if 'last_direction' not in st.session_state:
    st.session_state.last_direction = None
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = time.time()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Landmarks
NOSE_TIP = 1
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386
HORIZONTAL_THRESHOLD = 40

def get_landmark(landmarks, index, w, h):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        direction = "Unknown"
        eye_contact = 0
        attention = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                nose = get_landmark(landmarks, NOSE_TIP, w, h)
                left_eye = get_landmark(landmarks, LEFT_EYE_CENTER, w, h)
                right_eye = get_landmark(landmarks, RIGHT_EYE_CENTER, w, h)

                # Compute eye center
                eye_center_x = (left_eye[0] + right_eye[0]) // 2
                dx = nose[0] - eye_center_x

                if abs(dx) < HORIZONTAL_THRESHOLD:
                    direction = "Straight"
                elif dx > HORIZONTAL_THRESHOLD:
                    direction = "Right"
                else:
                    direction = "Left"

                current_time = time.time()
                if (direction != st.session_state.last_direction) and (current_time - st.session_state.last_log_time > 1):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    st.session_state.head_position_logs.append(f"{timestamp} - Looking {direction}")
                    st.session_state.last_log_time = current_time
                    st.session_state.last_direction = direction

                for point in [nose, left_eye, right_eye]:
                    cv2.circle(image, point, 3, (0, 255, 0), -1)

                # Simulate scores
                eye_contact = np.random.randint(70, 100)
                attention = np.random.randint(60, 95)

                st.session_state.eye_contact_list.append(eye_contact)
                st.session_state.attention_level_list.append(attention)

                cv2.putText(image, f"Looking: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(key="head-tracker", video_processor_factory=VideoProcessor)

# Show final results
if st.button("📊 Show Final Analysis"):
    avg_eye_contact = np.mean(st.session_state.eye_contact_list) if st.session_state.eye_contact_list else 0
    avg_attention = np.mean(st.session_state.attention_level_list) if st.session_state.attention_level_list else 0

    st.write(f"**👀 Average Eye Contact:** {avg_eye_contact:.2f}%")
    st.write(f"**🧠 Average Attention Level:** {avg_attention:.2f}%")

    st.write("## 📜 Head Position Log")
    for log in st.session_state.head_position_logs:
        st.write(log)
