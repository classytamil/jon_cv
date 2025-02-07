import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

def calculate_confidence(landmarks):
    """Calculate confidence level based on facial features"""
    if not landmarks:
        return 0  # No face detected, confidence is 0

    # Extract key landmarks (e.g., eyes, mouth, head tilt)
    left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
    left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])
    right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
    right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])
    
    # Eye openness ratio (higher means eyes are more open)
    left_eye_ratio = np.linalg.norm(left_eye_top - left_eye_bottom)
    right_eye_ratio = np.linalg.norm(right_eye_top - right_eye_bottom)
    
    # Blink detection (if eyes are too closed)
    eye_threshold = 0.015
    blink_detected = left_eye_ratio < eye_threshold and right_eye_ratio < eye_threshold

    # Mouth openness (stress indicator)
    mouth_top = np.array([landmarks[13].x, landmarks[13].y])
    mouth_bottom = np.array([landmarks[14].x, landmarks[14].y])
    mouth_ratio = np.linalg.norm(mouth_top - mouth_bottom)

    # Head tilt detection
    left_face = np.array([landmarks[234].x, landmarks[234].y])
    right_face = np.array([landmarks[454].x, landmarks[454].y])
    head_tilt = np.linalg.norm(left_face - right_face)

    # Compute confidence score
    confidence = 100  # Start with full confidence

    # Reduce confidence if excessive blinking detected
    if blink_detected:
        confidence -= 30

    # Reduce confidence if mouth is too open
    if mouth_ratio > 0.05:
        confidence -= 20

    # Reduce confidence if head tilt is too high
    if head_tilt > 0.25:
        confidence -= 15

    # Ensure confidence is within range 1-100
    return max(1, min(100, confidence))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    confidence_score = 0  # Default score

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            confidence_score = calculate_confidence(face_landmarks.landmark)

            # Draw confidence score on screen
            cv2.putText(frame, f"Confidence Level: {confidence_score}%", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Confidence Level Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
