import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime

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

# Threshold for detecting head position
HORIZONTAL_THRESHOLD = 40  

# Store values for averaging
eye_contact_list = []
attention_level_list = []
head_position_logs = []
last_direction = None
last_log_time = time.time()

# Start video capture
cap = cv2.VideoCapture(0)

def get_landmark(landmarks, index, w, h):
    """ Convert normalized coordinates to pixel values """
    return int(landmarks[index].x * w), int(landmarks[index].y * h)

def calculate_eye_aspect_ratio(landmarks):
    """ Compute the Eye Aspect Ratio (EAR) for detecting eye openness and gaze. """
    def aspect_ratio(eye_points):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
        vertical1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
        vertical2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
        horizontal = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])
        return (vertical1 + vertical2) / (2 * horizontal)

    left_ear = aspect_ratio(LEFT_EYE)
    right_ear = aspect_ratio(RIGHT_EYE)
    return (left_ear + right_ear) / 2

def calculate_eye_contact(face_landmarks):
    """ Compute eye contact score based on Eye Aspect Ratio (EAR). """
    ear = calculate_eye_aspect_ratio(face_landmarks)
    eye_contact_score = max(0, min(100, (ear - 0.15) * 400))  # Scale between 0-100
    return eye_contact_score

def calculate_attention_level(face_landmarks):
    """ Compute attention level based on head tilt and movement. """
    nose = face_landmarks[NOSE_TIP]
    chin = face_landmarks[CHIN]
    left_cheek = face_landmarks[LEFT_CHEEK]
    right_cheek = face_landmarks[RIGHT_CHEEK]

    head_tilt = abs(left_cheek.y - right_cheek.y) * 100  
    vertical_movement = abs(nose.y - chin.y) * 100  

    attention_score = max(0, min(100, 100 - (head_tilt * 2) - (vertical_movement * 1.5)))
    return attention_score

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape  # Frame dimensions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Get key points for head position
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
                eye_contact_list.append(eye_contact)
                attention_level_list.append(attention_level)

                # Draw key landmarks
                for point in [nose, left_eye, right_eye]:
                    cv2.circle(frame, point, 3, (80, 200, 120), -1)

                # Display metrics on screen
                cv2.putText(frame, f"Looking: {direction}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Eye Contact: {eye_contact:.2f}%", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Attention Level: {attention_level:.2f}%", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Log position change
                current_time = time.time()
                if direction != last_direction and (current_time - last_log_time) > 1:  
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    head_position_logs.append(f"{timestamp} - Looking {direction}")
                    last_log_time = current_time
                    last_direction = direction



        # Show frame
        cv2.imshow("Head Position & Attention Tracker", frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Compute and print final stats
    if eye_contact_list and attention_level_list:
        avg_eye_contact = np.mean(eye_contact_list)
        avg_attention = np.mean(attention_level_list)

    
    
    print("\nFinal Head Position Log:")
    for log in head_position_logs:
        print(log)

    print(f"\nAverage Eye Contact: {avg_eye_contact:.2f}%")
    print(f"Average Attention Level: {avg_attention:.2f}%")