Eye Gaze Tracking

This project utilizes **MediaPipe** and **OpenCV** to track the movement of a person's eyes in real-time using a webcam. The system calculates the gaze direction based on eye and iris positions, and displays the gaze direction (left, right, up, down, or center) on the screen.

Features

* Real-time eye tracking using webcam feed.
* Calculates the horizontal and vertical gaze direction.
* Displays gaze direction text on the video feed.
* Highlights eye landmarks (iris and eye corners) with green dots.
* Uses **MediaPipe Face Mesh** for facial landmark detection.

Requirements

* Python 3.x
* `opencv-python`
* `mediapipe`
* `numpy`

You can install the required packages using the following commands:


pip install opencv-python mediapipe numpy


How to Run
1. Clone or download the repository to your local machine.
2. Install the required libraries listed above.
3. Run the `eye_tracking.py` file.


python eye_tracking.py

4. The webcam feed will start, and you will see the gaze direction text displayed on the screen.
5. Press 'q' to exit the application.

 How It Works
* The **Face Mesh** model from MediaPipe detects facial landmarks, including the eyes and iris.
* Eye movement is tracked by calculating the horizontal and vertical positions of the iris relative to the eye’s boundaries.
* The system calculates the gaze direction by comparing the position of the iris to the outer eye landmarks.
* A green dot is drawn on the iris and eye corners, and the gaze direction is displayed on the screen.

#CODE

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

# Indices for eye landmarks
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]
LEFT_IRIS = 468
RIGHT_IRIS = 473

# Helper function to convert normalized landmarks to pixel coordinates
def lm(idx, landmarks, w, h):
    pt = landmarks[idx]
    return np.array([int(pt.x * w), int(pt.y * h)])

# Function to calculate the ratio for gaze direction
def get_ratio(p1, p2, center, axis=0):
    eye_range = p2[axis] - p1[axis]
    offset = center[axis] - p1[axis]
    return offset / eye_range if eye_range != 0 else 0.5

# Calculate gaze direction based on ratios
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
        vertical = "Down/Closed"
    elif v_ratio > 0.42:
        vertical = "Up"
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

 # Calculate gaze ratios
   h_ratio = (get_ratio(le[0], le[1], le_iris) + get_ratio(re[0], re[1], re_iris)) / 2
        v_ratio = (get_ratio(le_tb[0], le_tb[1], le_iris, axis=1) +
                   get_ratio(re_tb[0], re_tb[1], re_iris, axis=1)) / 2

# Get and display gaze direction
   gaze_direction = get_gaze_direction(h_ratio, v_ratio)
        cv2.putText(frame, gaze_direction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
# Draw markers on iris and corners
for pt in [le_iris, re_iris, le[0], re[0]]:
            cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)

   cv2.imshow("Eye Gaze Tracker", frame)

   if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


Code Overview

* **`lm()`**: Extracts the coordinates of the landmarks and converts them to pixel values.
* **`get_ratio()`**: Calculates the ratio of iris position relative to the eye’s width or height.
* **`get_gaze_direction()`**: Determines the gaze direction (left, right, up, down, or center) based on horizontal and vertical ratios.
* **Real-time webcam feed**: Captures frames, processes them for face and eye detection, and displays the gaze direction.

Contributions

Feel free to fork this project and make improvements or add new features. If you have any suggestions or encounter issues, feel free to open an issue.

![Screenshot 2025-04-22 103551](https://github.com/user-attachments/assets/bc2ec889-82f5-4ab7-88b1-771f9cfaa6d8)
![Screenshot 2025-04-22 103603](https://github.com/user-attachments/assets/05512c8e-9e69-4277-b1cd-813b7607f259)
![Screenshot 2025-04-22 103610](https://github.com/user-attachments/assets/112317a1-6a74-4c27-a14a-5656461002e2)
![Screenshot 2025-04-22 103619](https://github.com/user-attachments/assets/48bcd1de-3488-4790-a1b0-6267e483d2fb)
![Screenshot 2025-04-22 103705](https://github.com/user-attachments/assets/edd5e9f0-c933-4f8a-a31d-3af6c35dfd05)



