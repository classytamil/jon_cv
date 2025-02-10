import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 160, 158, 159, 157, 133]
RIGHT_EYE = [362, 385, 387, 386, 384, 263]
LEFT_IRIS, RIGHT_IRIS = 468, 473
NOSE, CHIN = 1, 152
LEFT_CHEEK, RIGHT_CHEEK = 234, 454

# Function to compute gaze deviation
def iris_gaze_tracking(landmarks, img_w, img_h):
    left_pupil = np.array([landmarks[LEFT_IRIS].x * img_w, landmarks[LEFT_IRIS].y * img_h])
    right_pupil = np.array([landmarks[RIGHT_IRIS].x * img_w, landmarks[RIGHT_IRIS].y * img_h])
    left_eye_center = np.mean([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in LEFT_EYE], axis=0)
    right_eye_center = np.mean([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in RIGHT_EYE], axis=0)
    left_gaze_offset = np.linalg.norm(left_pupil - left_eye_center) / img_w
    right_gaze_offset = np.linalg.norm(right_pupil - right_eye_center) / img_w
    return max(0, min(100, 100 - (left_gaze_offset + right_gaze_offset) * 500))

# Function to compute Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(landmarks, img_w, img_h):
    def get_ear(eye_pts):
        vertical1 = distance.euclidean(eye_pts[1], eye_pts[5])
        vertical2 = distance.euclidean(eye_pts[2], eye_pts[4])
        horizontal = distance.euclidean(eye_pts[0], eye_pts[3])
        return (vertical1 + vertical2) / (2.0 * horizontal)
    left_eye_pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in LEFT_EYE]
    right_eye_pts = [(int(landmarks[i].x * img_w), int(landmarks[i].y * img_h)) for i in RIGHT_EYE]
    return (get_ear(left_eye_pts) + get_ear(right_eye_pts)) / 2.0

# Function to calculate attention score
def get_attention_score(landmarks, img_w, img_h):
    nose_x, nose_y = int(landmarks[NOSE].x * img_w), int(landmarks[NOSE].y * img_h)
    chin_x, chin_y = int(landmarks[CHIN].x * img_w), int(landmarks[CHIN].y * img_h)
    left_cheek_x, right_cheek_x = int(landmarks[LEFT_CHEEK].x * img_w), int(landmarks[RIGHT_CHEEK].x * img_w)
    head_tilt = abs(nose_x - ((left_cheek_x + right_cheek_x) // 2))
    head_roll = abs(nose_y - chin_y)
    head_score = max(0, 100 - (head_tilt / img_w) * 200 - (head_roll / img_h) * 150)
    gaze_score = iris_gaze_tracking(landmarks, img_w, img_h)
    ear = eye_aspect_ratio(landmarks, img_w, img_h)
    blink_score = 40 if ear < 0.2 else 100
    return max(1, min(100, int((0.5 * gaze_score) + (0.3 * head_score) + (0.2 * blink_score))))

# Video capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w, _ = frame.shape
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
            eye_contact_level = iris_gaze_tracking(face_landmarks.landmark, img_w, img_h)
            attention_score = get_attention_score(face_landmarks.landmark, img_w, img_h)
            cv2.putText(frame, f"Eye Contact: {eye_contact_level}%", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Attention: {attention_score}%", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Eye Contact & Attention Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
