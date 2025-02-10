import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define facial landmark indices for eyes and face orientation
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Store values for averaging
eye_contact_list = []
attention_level_list = []

# Start webcam capture
cap = cv2.VideoCapture(0)

def calculate_eye_aspect_ratio(landmarks, eye_points):
    """ Compute the Eye Aspect Ratio (EAR) to check blinking and gaze. """
    p1, p2, p3, p4, p5, p6 = [landmarks.landmark[i] for i in eye_points]

    vertical1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
    vertical2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
    horizontal = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])

    ear = (vertical1 + vertical2) / (2 * horizontal)
    return ear

def calculate_eye_contact(face_landmarks):
    """ Compute an improved Eye Contact Score based on EAR. """
    left_ear = calculate_eye_aspect_ratio(face_landmarks, LEFT_EYE)
    right_ear = calculate_eye_aspect_ratio(face_landmarks, RIGHT_EYE)

    avg_ear = (left_ear + right_ear) / 2
    eye_contact_score = max(0, min(100, (avg_ear - 0.15) * 400))  # Rescale EAR to a score

    return eye_contact_score

def calculate_attention_level(face_landmarks):
    """ Improved attention calculation using head pose and nose deviation """
    nose = face_landmarks.landmark[NOSE_TIP]
    chin = face_landmarks.landmark[CHIN]
    left_cheek = face_landmarks.landmark[LEFT_CHEEK]
    right_cheek = face_landmarks.landmark[RIGHT_CHEEK]

    head_tilt = abs(left_cheek.y - right_cheek.y) * 100  # Head tilt difference
    vertical_head_movement = abs(nose.y - chin.y) * 100  # Forward/backward movement
    
    attention_score = max(0, min(100, 100 - (head_tilt * 2) - (vertical_head_movement * 1.5)))
    
    return attention_score

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_contact = calculate_eye_contact(face_landmarks)
                attention_level = calculate_attention_level(face_landmarks)

                # Store values for averaging
                eye_contact_list.append(eye_contact)
                attention_level_list.append(attention_level)

                # Display values on screen
                cv2.putText(frame, f"Eye Contact: {eye_contact:.2f}%", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Attention Level: {attention_level:.2f}%", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Enhanced Attention Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nProgram stopped manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    # Calculate mean values
    if eye_contact_list and attention_level_list:
        mean_eye_contact = np.mean(eye_contact_list)
        mean_attention_level = np.mean(attention_level_list)
        
        print(f"\nAverage Eye Contact Score: {mean_eye_contact:.2f}%")
        print(f"Average Attention Level Score: {mean_attention_level:.2f}%")
    else:
        print("\nNo valid data recorded.")
