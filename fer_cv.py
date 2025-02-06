import cv2
import mediapipe as mp
import numpy as np
from fer import FER

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize FER for facial expression recognition
emotion_detector = FER()

# Open webcam
cap = cv2.VideoCapture(0)

def estimate_head_pose(landmarks, frame_width, frame_height):
    """Estimates head pose (pitch, yaw, roll) using facial landmarks."""
    
    # Define 2D model points (based on face landmarks)
    image_points = np.array([
        landmarks[1],  # Nose tip
        landmarks[152], # Chin
        landmarks[33],  # Left eye left corner
        landmarks[263], # Right eye right corner
        landmarks[61],  # Left mouth corner
        landmarks[291]  # Right mouth corner
    ], dtype="double")

    # Define 3D model points (approximated based on a standard face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0),# Left eye left corner
        (225.0, 170.0, -135.0), # Right eye right corner
        (-150.0, -150.0, -125.0),# Left mouth corner
        (150.0, -150.0, -125.0) # Right mouth corner
    ])

    # Camera matrix approximation
    focal_length = frame_width
    center = (frame_width / 2, frame_height / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assume no lens distortion

    # Solve PnP to estimate head pose
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    if success:
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Extract Euler angles (Pitch, Yaw, Roll)
        euler_angles = cv2.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))[6]

        pitch, yaw, roll = euler_angles.flatten()
        return pitch, yaw, roll
    else:
        return None, None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB (MediaPipe requires RGB input)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract facial landmarks
            landmarks = {i: (int(face_landmarks.landmark[i].x * frame_width), 
                             int(face_landmarks.landmark[i].y * frame_height)) 
                         for i in range(len(face_landmarks.landmark))}

            # Estimate head pose
            pitch, yaw, roll = estimate_head_pose(landmarks, frame_width, frame_height)
            
            # Display head pose values
            if pitch is not None and yaw is not None and roll is not None:
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw face mesh
            for _, (x, y) in landmarks.items():
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    # Detect facial expression
    emotions = emotion_detector.detect_emotions(frame)
    if emotions:
        emotion_data = emotions[0]["emotions"]
        dominant_emotion = max(emotion_data, key=emotion_data.get)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Head Pose & Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
