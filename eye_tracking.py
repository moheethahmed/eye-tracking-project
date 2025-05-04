import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indices for the eye landmarks
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]
LEFT_IRIS = 468
RIGHT_IRIS = 473

# Helper function to extract eye position
def lm(idx, landmarks, w, h):
    pt = landmarks[idx]
    return np.array([int(pt.x * w), int(pt.y * h)])

# Function to calculate the ratio for gaze direction (horizontal or vertical)
def get_ratio(p1, p2, center, axis=0):
    eye_range = p2[axis] - p1[axis]
    offset = center[axis] - p1[axis]
    return offset / eye_range if eye_range != 0 else 0.5

# Calculate gaze direction
def get_gaze_direction(h_ratio, v_ratio):
    # Horizontal direction
    if h_ratio < 0.36:
        horizontal = "Left"
    elif h_ratio > 0.60:
        horizontal = "Right"
    else:
        horizontal = "Center"

    # Vertical direction
    if v_ratio < 0.35:
        vertical = "closed/Down"
    elif v_ratio > 0.42:
        vertical = "up"
    else:
        vertical = "Center"

    return f"Looking {vertical} {horizontal}"

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Get landmarks in pixel coordinates
        le = [lm(i, landmarks, w, h) for i in LEFT_EYE]
        re = [lm(i, landmarks, w, h) for i in RIGHT_EYE]
        le_tb = [lm(i, landmarks, w, h) for i in LEFT_EYE_LIDS]
        re_tb = [lm(i, landmarks, w, h) for i in RIGHT_EYE_LIDS]
        le_iris = lm(LEFT_IRIS, landmarks, w, h)
        re_iris = lm(RIGHT_IRIS, landmarks, w, h)

        # Calculate gaze ratios (Horizontal and Vertical)
        h_ratio = (get_ratio(le[0], le[1], le_iris) + get_ratio(re[0], re[1], re_iris)) / 2
        v_ratio = (get_ratio(le_tb[0], le_tb[1], le_iris, axis=1) +
                   get_ratio(re_tb[0], re_tb[1], re_iris, axis=1)) / 2

        # Get gaze direction
        gaze_direction = get_gaze_direction(h_ratio, v_ratio)

        # Display gaze direction on screen
        cv2.putText(frame, gaze_direction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Draw green dot on the eyes
        cv2.circle(frame, tuple(le_iris), 5, (0, 255, 0), -1)  # Left iris
        cv2.circle(frame, tuple(re_iris), 5, (0, 255, 0), -1)  # Right iris
        cv2.circle(frame, tuple(le[0]), 5, (0, 255, 0), -1)  # Left eye
        cv2.circle(frame, tuple(re[0]), 5, (0, 255, 0), -1)  # Right eye

    # Show the frame with gaze direction and green dots
    cv2.imshow("Eye Gaze Tracker", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
