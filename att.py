import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Indices for key facial landmarks
LEFT_EYE = [33, 160, 158, 159, 157, 173]  # Left eye
RIGHT_EYE = [263, 387, 385, 386, 384, 373]  # Right eye
NOSE = 1
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454

# Camera calibration (assuming standard webcam settings)
FOCAL_LENGTH = 500  # Approximate focal length
CAMERA_CENTER = (320, 240)  # Assumed camera center

# Start video capture
cap = cv2.VideoCapture(1)

def eye_aspect_ratio(landmarks, img_w, img_h):
    """Calculate the Eye Aspect Ratio (EAR) to detect blinking."""
    left_eye_pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in LEFT_EYE]
    right_eye_pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in RIGHT_EYE]
    
    # Calculate vertical and horizontal distances
    def get_ear(eye_pts):
        vertical1 = distance.euclidean(eye_pts[1], eye_pts[5])
        vertical2 = distance.euclidean(eye_pts[2], eye_pts[4])
        horizontal = distance.euclidean(eye_pts[0], eye_pts[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)

    left_ear = get_ear(left_eye_pts)
    right_ear = get_ear(right_eye_pts)
    
    return (left_ear + right_ear) / 2.0  # Average EAR

def get_attention_score(landmarks, img_w, img_h):
    """Calculate attention level on a scale of 1 to 100 using multiple factors."""
    # Convert normalized coordinates to pixel values
    nose_x, nose_y = int(landmarks[NOSE].x * img_w), int(landmarks[NOSE].y * img_h)
    chin_x, chin_y = int(landmarks[CHIN].x * img_w), int(landmarks[CHIN].y * img_h)
    left_cheek_x = int(landmarks[LEFT_CHEEK].x * img_w)
    right_cheek_x = int(landmarks[RIGHT_CHEEK].x * img_w)

    # Head tilt (Yaw)
    head_tilt = abs(nose_x - ((left_cheek_x + right_cheek_x) // 2))

    # Head pose estimation
    head_roll = abs(nose_y - chin_y)
    head_score = max(0, 100 - (head_tilt / img_w) * 200 - (head_roll / img_h) * 150)

    # Gaze direction (based on nose alignment)
    gaze_center_x = (left_cheek_x + right_cheek_x) // 2
    gaze_deviation = abs(nose_x - gaze_center_x)
    gaze_score = max(0, 100 - (gaze_deviation / img_w) * 200)

    # Eye openness (Blink detection)
    ear = eye_aspect_ratio(landmarks, img_w, img_h)
    if ear < 0.2:  # Blink detected
        blink_score = 40
    else:
        blink_score = 100

    # Combine scores with weights
    attention_score = int((0.5 * gaze_score) + (0.3 * head_score) + (0.2 * blink_score))
    return max(1, min(100, attention_score))  # Keep within 1-100 range

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert color
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape

    # Process frame
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw face mesh
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # Get attention score
            attention_score = get_attention_score(face_landmarks.landmark, img_w, img_h)

            # Display score
            cv2.putText(frame, f"Attention Score: {attention_score}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Attention Level Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
