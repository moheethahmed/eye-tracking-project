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

Code Overview

* **`lm()`**: Extracts the coordinates of the landmarks and converts them to pixel values.
* **`get_ratio()`**: Calculates the ratio of iris position relative to the eye’s width or height.
* **`get_gaze_direction()`**: Determines the gaze direction (left, right, up, down, or center) based on horizontal and vertical ratios.
* **Real-time webcam feed**: Captures frames, processes them for face and eye detection, and displays the gaze direction.

Contributions

Feel free to fork this project and make improvements or add new features. If you have any suggestions or encounter issues, feel free to open an issue.
