import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Define facial landmark indices
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

def calculate_eye_contact(face_landmarks, frame_width):
    """ Estimate gaze deviation using multiple eye landmarks """
    left_eye_x = np.mean([face_landmarks.landmark[i].x for i in LEFT_EYE])
    right_eye_x = np.mean([face_landmarks.landmark[i].x for i in RIGHT_EYE])
    
    gaze_deviation = abs(left_eye_x - right_eye_x) * frame_width
    eye_contact_score = max(0, 100 - (gaze_deviation * 2))  # Adjust scaling for better sensitivity

    return eye_contact_score

def calculate_attention_level(face_landmarks, frame_height):
    """ Estimate head tilt and deviation from frontal position """
    nose = face_landmarks.landmark[NOSE_TIP]
    chin = face_landmarks.landmark[CHIN]
    left_cheek = face_landmarks.landmark[LEFT_CHEEK]
    right_cheek = face_landmarks.landmark[RIGHT_CHEEK]

    head_tilt = abs(left_cheek.y - right_cheek.y) * frame_height
    vertical_head_movement = abs(nose.y - chin.y) * frame_height
    
    attention_score = max(0, 100 - (head_tilt * 1.5) - (vertical_head_movement * 1.5))  

    return attention_score

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        frame_height, frame_width, _ = frame.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                eye_contact = calculate_eye_contact(face_landmarks, frame_width)
                attention_level = calculate_attention_level(face_landmarks, frame_height)

                # Store values in the lists
                eye_contact_list.append(eye_contact)
                attention_level_list.append(attention_level)

                # Display on screen
                cv2.putText(frame, f"Eye Contact: {eye_contact:.2f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Attention Level: {attention_level:.2f}", (10, 60), 
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
        
        print(f"\nAverage Eye Contact Score: {mean_eye_contact:.2f}")
        print(f"Average Attention Level Score: {mean_attention_level:.2f}")
    else:
        print("\nNo valid data recorded.")
