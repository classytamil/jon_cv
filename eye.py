import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Indices for facial landmarks (eyes)
LEFT_EYE = [33, 160, 158, 159, 157, 173]
RIGHT_EYE = [263, 387, 385, 386, 384, 373]

# Camera center
CAMERA_CENTER = (320, 240)

# Function to calculate Eye Aspect Ratio (EAR) for eye openness
def eye_aspect_ratio(eye_landmarks, img_w, img_h):
    # Calculate the distance between vertical and horizontal eye landmarks
    A = np.linalg.norm(np.array([eye_landmarks[1].x * img_w, eye_landmarks[1].y * img_h]) - np.array([eye_landmarks[5].x * img_w, eye_landmarks[5].y * img_h]))
    B = np.linalg.norm(np.array([eye_landmarks[2].x * img_w, eye_landmarks[2].y * img_h]) - np.array([eye_landmarks[4].x * img_w, eye_landmarks[4].y * img_h]))
    C = np.linalg.norm(np.array([eye_landmarks[0].x * img_w, eye_landmarks[0].y * img_h]) - np.array([eye_landmarks[3].x * img_w, eye_landmarks[3].y * img_h]))

    # Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate gaze deviation from the center of the camera
def gaze_deviation(landmarks, img_w, img_h):
    left_eye_center = np.mean([[landmarks[33].x, landmarks[33].y], [landmarks[159].x, landmarks[159].y]], axis=0)
    right_eye_center = np.mean([[landmarks[263].x, landmarks[263].y], [landmarks[386].x, landmarks[386].y]], axis=0)

    eye_center_x = (left_eye_center[0] + right_eye_center[0]) / 2 * img_w
    eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2 * img_h

    # Calculate the deviation from the center of the camera (CAMERA_CENTER)
    deviation_x = abs(CAMERA_CENTER[0] - eye_center_x)
    deviation_y = abs(CAMERA_CENTER[1] - eye_center_y)

    # Combine deviations (scaled to max of 100)
    gaze_score = 100 - (min(deviation_x, img_w - deviation_x) + min(deviation_y, img_h - deviation_y)) / (max(img_w, img_h) / 2) * 100
    return max(0, min(100, gaze_score))

# Function to calculate the eye contact level (1-100)
def get_eye_contact_level(landmarks, img_w, img_h):
    # Calculate EAR (Eye Aspect Ratio) for both eyes
    left_eye_ear = eye_aspect_ratio([landmarks[i] for i in LEFT_EYE], img_w, img_h)
    right_eye_ear = eye_aspect_ratio([landmarks[i] for i in RIGHT_EYE], img_w, img_h)

    # Average EAR for both eyes
    avg_ear = (left_eye_ear + right_eye_ear) / 2.0

    # Gaze deviation from the center of the camera
    gaze_score = gaze_deviation(landmarks, img_w, img_h)

    # Calculate eye contact level as a weighted score (combine both)
    # Higher EAR and lower gaze deviation implies better eye contact
    eye_contact_level = int((avg_ear * 50) + (gaze_score * 50))

    return max(1, min(100, eye_contact_level))

# Start video capture
cap = cv2.VideoCapture()

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

            # Get eye contact level
            eye_contact_level = get_eye_contact_level(face_landmarks.landmark, img_w, img_h)

            # Display eye contact level
            cv2.putText(frame, f"Eye Contact Level: {eye_contact_level}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Eye Contact Level Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
