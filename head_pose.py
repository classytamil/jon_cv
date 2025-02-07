import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices for key points
NOSE_TIP = 1
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386

# Start video capture
cap = cv2.VideoCapture(0)

# Define thresholds
HORIZONTAL_THRESHOLD = 40  # Threshold for left-right movement

# Store head position logs
head_position_logs = []
last_direction = None
last_log_time = time.time()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get frame dimensions
                h, w, _ = frame.shape
                landmarks = face_landmarks.landmark

                # Function to convert normalized coordinates to pixel values
                def get_landmark(landmark_idx):
                    return int(landmarks[landmark_idx].x * w), int(landmarks[landmark_idx].y * h)

                # Get key points
                nose = get_landmark(NOSE_TIP)
                left_eye = get_landmark(LEFT_EYE_CENTER)
                right_eye = get_landmark(RIGHT_EYE_CENTER)

                # Calculate midpoint of eyes for better alignment
                eye_center_x = (left_eye[0] + right_eye[0]) // 2
                dx = nose[0] - eye_center_x  # Horizontal deviation

                # Detect head direction
                if abs(dx) < HORIZONTAL_THRESHOLD:
                    direction = "Straight"
                elif dx > HORIZONTAL_THRESHOLD:
                    direction = "Right"
                else:
                    direction = "Left"

                # Display detected direction
                cv2.putText(frame, f"Looking: {direction}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw key landmarks
                for point in [nose, left_eye, right_eye]:
                    cv2.circle(frame, point, 3, (80, 200, 120), -1)

                # Log position change
                current_time = time.time()
                if direction != last_direction and (current_time - last_log_time) > 1:  # Log only when changed
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    head_position_logs.append(f"{timestamp} - Looking {direction}")
                    last_log_time = current_time
                    last_direction = direction

                # Print logs every 5 seconds
                if current_time - last_log_time >= 5:
                    print("\nHead Position Log:")
                    for log in head_position_logs[-5:]:  # Show last 5 movements
                        print(log)
                    last_log_time = current_time

        # Show frame
        cv2.imshow("Head Position Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Print final head position data
    print("\nFinal Head Position Data:")
    for log in head_position_logs:
        print(log)
