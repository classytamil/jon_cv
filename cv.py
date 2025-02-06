import cv2
import numpy as np
import mediapipe as mp

# 3D model points of key facial features (nose, eyes, etc.)
model_points = np.array([
    (0.0, 0.0, 0.0),       # Nose tip
    (0.0, -330.0, -65.0),  # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype="double")



def estimate_head_pose(image, face_landmarks):
    # Camera parameters (assume basic intrinsics for now)
    camera_matrix = np.array([
        [image.shape[1], 0, image.shape[1] / 2],
        [0, image.shape[0], image.shape[0] / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # No lens distortion

    # Extract specific 2D landmarks for SolvePnP
    image_points = np.array([
        face_landmarks[1],  # Nose tip
        face_landmarks[152],  # Chin
        face_landmarks[263],  # Right eye corner
        face_landmarks[33],  # Left eye corner
        face_landmarks[287],  # Right mouth corner
        face_landmarks[57],  # Left mouth corner
    ], dtype="double")

    # SolvePnP for rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    if success:
        # Convert rotation vector to angles
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        euler_angles = cv2.decomposeProjectionMatrix(rotation_matrix)[6]
        yaw, pitch, roll = euler_angles[1], euler_angles[0], euler_angles[2]
        return pitch, yaw, roll
    return None, None, None


def classify_head_position(pitch, yaw):
    if yaw > 10:
        return "Looking Right"
    elif yaw < -10:
        return "Looking Left"
    elif pitch > 10:
        return "Looking Up"
    elif pitch < -10:
        return "Looking Down"
    else:
        return "Looking Straight"

import time

cap = cv2.VideoCapture(0)
last_checked = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Get 2D facial landmarks
            face_landmarks = [(lm.x, lm.y) for lm in landmarks.landmark]

            # Estimate head pose
            pitch, yaw, roll = estimate_head_pose(frame, face_landmarks)

            if time.time() - last_checked > 10:  # Check every 10 seconds
                head_position = classify_head_position(pitch, yaw)
                print(f"Head Position: {head_position}")
                last_checked = time.time()

    cv2.imshow("Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
