import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices for key points
NOSE_TIP = 1
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386
CHIN = 152
FOREHEAD = 10

# Start video capture
cap = cv2.VideoCapture(0)

# Define the thresholds for head direction detection
HORIZONTAL_THRESHOLD = 40  # Threshold for horizontal deviation (left-right)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the dimensions of the frame
            h, w, _ = frame.shape
            landmarks = face_landmarks.landmark
            
            # Function to convert normalized coordinates to pixel values
            def get_landmark(landmark_idx):
                return int(landmarks[landmark_idx].x * w), int(landmarks[landmark_idx].y * h)
            
            # Get positions of key landmarks
            nose = get_landmark(NOSE_TIP)
            left_eye = get_landmark(LEFT_EYE_CENTER)
            right_eye = get_landmark(RIGHT_EYE_CENTER)
            chin = get_landmark(CHIN)
            forehead = get_landmark(FOREHEAD)

            # Calculate the midpoint of the eyes for better horizontal alignment
            eye_center_x = (left_eye[0] + right_eye[0]) // 2
            dx = nose[0] - eye_center_x  # Horizontal deviation (left-right)

            # Logic for detecting head position
            if abs(dx) < HORIZONTAL_THRESHOLD:
                direction = "Straight"
            elif dx > HORIZONTAL_THRESHOLD:
                direction = "Right"
            elif dx < -HORIZONTAL_THRESHOLD:
                direction = "Left"
            else:
                direction = "Straight"  # Default to straight if no other direction is detected

            # Display the detected direction on the frame
            cv2.putText(frame, f"Looking: {direction}", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the landmarks on the face for visualization
            for point in [nose, left_eye, right_eye, chin, forehead]:
                cv2.circle(frame, point, 3, (80, 200, 120), -1)

    # Show the frame with the detected head position
    cv2.imshow("Head Position Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
