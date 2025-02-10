import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from datetime import datetime

# Streamlit UI Setup
st.title("Head Position & Attention Tracker")
st.write("Click 'Start' to begin real-time analysis. Click 'Stop' to view the results.")

# Initialize session state
if 'eye_contact_list' not in st.session_state:
    st.session_state.eye_contact_list = []
if 'attention_level_list' not in st.session_state:
    st.session_state.attention_level_list = []
if 'head_position_logs' not in st.session_state:
    st.session_state.head_position_logs = []

# Buttons for controlling tracking
start_button = st.button("Start Tracking")
stop_button = st.button("Stop Tracking")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Landmark indices for key points
NOSE_TIP = 1
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386
HORIZONTAL_THRESHOLD = 40

last_direction = None
last_log_time = time.time()

def get_landmark(landmarks, index, w, h):
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

def process_frame(frame):
    global last_direction, last_log_time
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
            dx = nose[0] - eye_center_x  # Horizontal deviation
            
            # Detect head direction
            if abs(dx) < HORIZONTAL_THRESHOLD:
                direction = "Straight"
            elif dx > HORIZONTAL_THRESHOLD:
                direction = "Right"
            else:
                direction = "Left"
            
            # Log position change
            current_time = time.time()
            if direction != last_direction and (current_time - last_log_time) > 1:
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.head_position_logs.append(f"{timestamp} - Looking {direction}")
                last_log_time = current_time
                last_direction = direction
            
            # Draw indicators
            for point in [nose, left_eye, right_eye]:
                cv2.circle(frame, point, 3, (80, 200, 120), -1)
            
            # Compute Eye Contact and Attention Level (Dummy Values for Now)
            eye_contact = np.random.randint(70, 100)  # Simulating attention %
            attention = np.random.randint(60, 95)  # Simulating attention %
            st.session_state.eye_contact_list.append(eye_contact)
            st.session_state.attention_level_list.append(attention)
            
            # Display info
            cv2.putText(frame, f"Looking: {direction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame, direction, eye_contact, attention

# Start video capture if 'Start' is clicked
if start_button:
    cap = cv2.VideoCapture(0)
    stframe = st.image([])
    st_metrics = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video")
            break
        
        frame, direction, eye_contact, attention = process_frame(frame)
        stframe.image(frame, channels="BGR")
        st_metrics.markdown(f"### Looking: {direction} | Eye Contact: {eye_contact}% | Attention Level: {attention}%")
        
        if stop_button:
            cap.release()
            break

# Show final analysis when tracking stops
if stop_button:
    st.write("## Final Analysis")
    avg_eye_contact = np.mean(st.session_state.eye_contact_list) if st.session_state.eye_contact_list else 0
    avg_attention = np.mean(st.session_state.attention_level_list) if st.session_state.attention_level_list else 0
    
    st.write(f"**Average Eye Contact:** {avg_eye_contact:.2f}%")
    st.write(f"**Average Attention Level:** {avg_attention:.2f}%")
    
    st.write("## Head Position Log")
    for log in st.session_state.head_position_logs:
        st.write(log)
