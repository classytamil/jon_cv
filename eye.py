import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_EYE = [33, 160, 158, 159, 157, 133]  # Left eye outline
RIGHT_EYE = [362, 385, 387, 386, 384, 263]  # Right eye outline
LEFT_IRIS = 468  # Left iris center
RIGHT_IRIS = 473  # Right iris center

# Function to compute gaze deviation using iris position
def iris_gaze_tracking(landmarks, img_w, img_h):
    left_pupil = np.array([landmarks[LEFT_IRIS].x * img_w, landmarks[LEFT_IRIS].y * img_h])
    right_pupil = np.array([landmarks[RIGHT_IRIS].x * img_w, landmarks[RIGHT_IRIS].y * img_h])
    
    left_eye_center = np.mean([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in LEFT_EYE], axis=0)
    right_eye_center = np.mean([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in RIGHT_EYE], axis=0)

    left_gaze_offset = np.linalg.norm(left_pupil - left_eye_center) / img_w
    right_gaze_offset = np.linalg.norm(right_pupil - right_eye_center) / img_w

    # Scale the gaze score (lower deviation = better eye contact)
    gaze_score = max(0, min(100, 100 - (left_gaze_offset + right_gaze_offset) * 500))
    return gaze_score

# Function to compute overall eye contact level
def get_eye_contact_level(landmarks, img_w, img_h):
    gaze_score = iris_gaze_tracking(landmarks, img_w, img_h)
    return int(gaze_score)

# Start video capture
cap = cv2.VideoCapture(0)
eye_contact_levels = []  # Store predictions

start_time = time.time()  # Track time

try:
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
                
                eye_contact_level = get_eye_contact_level(face_landmarks.landmark, img_w, img_h)
                
                # Display eye contact level
                cv2.putText(frame, f"Eye Contact Level: {eye_contact_level}%", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check if 5 seconds have passed
                if time.time() - start_time >= 5:
                    eye_contact_levels.append(eye_contact_level)
                    print(f"Eye Contact Level Recorded: {eye_contact_level}%")
                    start_time = time.time()  # Reset timer

        cv2.imshow("Eye Contact Detection with Iris Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    # Calculate and print the mean eye contact level
    if eye_contact_levels:
        mean_eye_contact = sum(eye_contact_levels) / len(eye_contact_levels)
        print(f"Mean Eye Contact Level: {mean_eye_contact:.2f}%")
    else:
        print("No eye contact data recorded.")
