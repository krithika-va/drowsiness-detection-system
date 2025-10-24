import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from collections import deque

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2 * euclidean_distance(p1, p4))

def calculate_mar(landmarks):
    top = np.mean([landmarks[13], landmarks[14]], axis=0)
    bottom = np.mean([landmarks[312], landmarks[317]], axis=0)
    left, right = landmarks[78], landmarks[308]
    return euclidean_distance(top, bottom) / np.linalg.norm(np.array(left)-np.array(right))

def calculate_head_pitch(landmarks):
    nose_tip = np.array(landmarks[1])
    chin = np.array(landmarks[152])
    return chin[1] - nose_tip[1]

# Setup video and CSV
cap = cv2.VideoCapture(0)
csv_file = open("dataset.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "EAR", "MAR", "PERCLOS", "HeadPitch", "Drowsy_Label"])

# Parameters
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
PITCH_THRESHOLD = 30
CONSEC_FRAMES = 15
WINDOW_SIZE = 60
ear_window = deque(maxlen=WINDOW_SIZE)
drowsy_frames = 0

last_written_time = 0  # in seconds

print("Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    avg_ear, mar, pitch, perclos = 0, 0, 0, 0
    label = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks)
            pitch = calculate_head_pitch(landmarks)

            ear_window.append(avg_ear)
            perclos = np.mean(np.array(ear_window) < EAR_THRESHOLD)

            condition_count = sum([
                avg_ear < EAR_THRESHOLD,
                mar > MAR_THRESHOLD,
                perclos > 0.4,
                pitch > PITCH_THRESHOLD
            ])

            if condition_count >= 2:
                drowsy_frames += 1
            else:
                drowsy_frames = 0

            if drowsy_frames >= CONSEC_FRAMES:
                label = 1
                cv2.putText(frame, "DROWSY ALERT!", (70, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Display live metrics
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"HeadPitch: {pitch:.1f}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Record data 5 times per second (~every 0.2 sec)
            current_time = time.time()
            if current_time - last_written_time >= 0.2:
                timestamp = time.strftime("%H:%M:%S")
                csv_writer.writerow([timestamp, avg_ear, mar, perclos, pitch, label])
                last_written_time = current_time

    cv2.imshow("Drowsiness Detection Live Feed (Press 'q' to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("Data saved to dataset.csv")
