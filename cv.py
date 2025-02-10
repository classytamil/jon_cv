import cv2
import mediapipe as mp
import numpy as np
import time
import streamlit as st
from datetime import datetime

# Initialize Streamlit
st.title("Head Position & Attention Tracker")
st.write("This app tracks head position, eye contact, and attention level in real-time.")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Landmark indices for key points
NOSE_TIP = 1
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
HORIZONTAL_THRESHOLD = 40

# Store values for averaging
if "eye_contact_list" not in st.session_state:
    st.session_state.eye_contact_list = []
if "attention_level_list" not in st.session_state:
    st.session_state.attention_level_list = []
if "head_position_logs" not in st.session_state:
    st.session_state.head_position_logs = []
if "last_direction" not in st.session_state:
    st.session_state.last_direction = None
if "last_log_time" not in st.session_state:
    st.session_state.last_log_time = time.time()

# Initialize session state for tracking
if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False

# Start video capture
cap = None
stframe = st.empty()
st_metrics = st.empty()

def get_landmark(landmarks, index, w, h):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

def calculate_eye_aspect_ratio(landmarks):
    def aspect_ratio(eye_points):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
        vertical1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
        vertical2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
        horizontal = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
        return (vertical1 + vertical2) / (2 * horizontal)

    left_ear = aspect_ratio(LEFT_EYE)
    right_ear = aspect_ratio(RIGHT_EYE)
    return (left_ear + right_ear) / 2

def calculate_eye_contact(landmarks):
    ear = calculate_eye_aspect_ratio(landmarks)
    eye_contact_score = max(0, min(100, (ear - 0.15) * 400))  # Scale between 0-100
    return eye_contact_score

def calculate_attention_level(landmarks):
    nose = landmarks[NOSE_TIP]
    chin = landmarks[CHIN]
    left_cheek = landmarks[LEFT_CHEEK]
    right_cheek = landmarks[RIGHT_CHEEK]

    head_tilt = abs(left_cheek.y - right_cheek.y) * 100  
    vertical_movement = abs(nose.y - chin.y) * 100  

    attention_score = max(0, min(100, 100 - (head_tilt * 2) - (vertical_movement * 1.5)))
    return attention_score

def process_frame(frame):
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # Default direction if no face is detected
    direction = "No Face Detected"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            nose = get_landmark(landmarks, NOSE_TIP, w, h)
            left_eye = get_landmark(landmarks, LEFT_EYE_CENTER, w, h)
            right_eye = get_landmark(landmarks, RIGHT_EYE_CENTER, w, h)

            # Compute eye center
            eye_center_x = (left_eye[0] + right_eye[0]) // 2
            dx = nose[0] - eye_center_x  # Horizontal deviation

            # Detect head direction
            if abs(dx) < HORIZONTAL_THRESHOLD:
                direction = "Straight"
            elif dx > HORIZONTAL_THRESHOLD:
                direction = "Right"
            else:
                direction = "Left"

            # Calculate eye contact and attention level
            eye_contact = calculate_eye_contact(landmarks)
            attention_level = calculate_attention_level(landmarks)

            # Store values for averaging
            st.session_state.eye_contact_list.append(eye_contact)
            st.session_state.attention_level_list.append(attention_level)

            # Log position change
            current_time = time.time()
            if direction != st.session_state.last_direction and (current_time - st.session_state.last_log_time) > 1:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.head_position_logs.append(f"{timestamp} - Looking {direction}")
                st.session_state.last_log_time = current_time
                st.session_state.last_direction = direction

            # Draw indicators
            for point in [nose, left_eye, right_eye]:
                cv2.circle(frame, point, 3, (80, 200, 120), -1)

            # Display info
            cv2.putText(frame, f"Looking: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Eye Contact: {eye_contact:.2f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Attention Level: {attention_level:.2f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return frame, direction

# Streamlit UI
start_button = st.button("Start Tracking")
stop_button = st.button("Stop Tracking")

if start_button:
    st.session_state.tracking_active = True

if stop_button:
    st.session_state.tracking_active = False
    if cap:
        cap.release()

if st.session_state.tracking_active:
    cap = cv2.VideoCapture(0)
    while st.session_state.tracking_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break

        frame, direction = process_frame(frame)
        stframe.image(frame, channels="BGR")
        st_metrics.markdown(f"### Looking: {direction}")

    if cap:
        cap.release()

    # Display final outputs
    if st.session_state.eye_contact_list and st.session_state.attention_level_list:
        avg_eye_contact = np.mean(st.session_state.eye_contact_list)
        avg_attention = np.mean(st.session_state.attention_level_list)

        st.write("## Final Metrics")
        st.write(f"Average Eye Contact: {avg_eye_contact:.2f}%")
        st.write(f"Average Attention Level: {avg_attention:.2f}%")

    st.write("## Head Position Log")
    for log in st.session_state.head_position_logs:
        st.write(log)