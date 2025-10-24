import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import matlab.engine
from collections import deque
import winsound

# === Start MATLAB Engine ===
eng = matlab.engine.start_matlab()
fis = eng.readfis("drowsy_fis_model_improved.fis")
norm_constants = eng.load('feature_norm_constants.mat')
minF = np.array(norm_constants['minF']).flatten()
maxF = np.array(norm_constants['maxF']).flatten()

# === MediaPipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

# === Parameters ===
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.7
CONSEC_FRAMES = 5
WINDOW_SIZE = 60
FPS = 5  # Process ~5 frames per second

# === Rolling windows ===
metrics_window = {
    "ear": deque(maxlen=WINDOW_SIZE),
    "mar": deque(maxlen=WINDOW_SIZE),
    "pitch": deque(maxlen=WINDOW_SIZE),
    "perclos": deque(maxlen=WINDOW_SIZE)
}

drowsy_frames = 0
alarm_playing = False
prev_time = 0

# === CSV Setup ===
csv_file = open("anfis_predictions.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Timestamp", "EAR", "MAR", "PERCLOS", "HeadPitch", "Drowsy_Label"])

# === Utility Functions ===
def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_points]
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2 * euclidean_distance(p1, p4))

def calculate_mar(landmarks):
    top = np.mean([landmarks[13], landmarks[14]], axis=0)
    bottom = np.mean([landmarks[312], landmarks[317]], axis=0)
    left, right = landmarks[78], landmarks[308]
    return euclidean_distance(top, bottom) / euclidean_distance(left, right)

def calculate_head_pitch(landmarks):
    nose_tip = np.array(landmarks[1])
    chin = np.array(landmarks[152])
    return chin[1] - nose_tip[1]

def normalize_features(features, minF, maxF):
    return [(f - mn) / (mx - mn + 1e-6) for f, mn, mx in zip(features, minF, maxF)]

def predict_drowsiness(features_norm):
    fis_input = matlab.double([features_norm])
    output = eng.evalfis(fis_input, fis)
    return float(np.clip(output, 0.0, 1.0))

def alert_logic(label):
    global drowsy_frames, alarm_playing
    if label == 1:
        drowsy_frames += 1
    else:
        drowsy_frames = 0
        alarm_playing = False

    if drowsy_frames >= CONSEC_FRAMES:
        if not alarm_playing:
            winsound.PlaySound("alarm.wav", winsound.SND_ASYNC)
            alarm_playing = True
        return True
    return False

# === Video Capture ===
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- FPS Control ---
    if time.time() - prev_time < 1/FPS:
        continue
    prev_time = time.time()

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    avg_ear, mar, pitch, perclos = 0, 0, 0, 0
    label = 0

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            # --- Calculate Metrics ---
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = calculate_mar(landmarks)
            pitch = calculate_head_pitch(landmarks)
            metrics_window["ear"].append(avg_ear)
            metrics_window["mar"].append(mar)
            metrics_window["pitch"].append(pitch)
            metrics_window["perclos"].append(avg_ear < EAR_THRESHOLD)
            perclos = np.mean(metrics_window["perclos"])

            # --- Normalize & Predict ---
            features = [np.mean(metrics_window["ear"]), np.mean(metrics_window["mar"]),
                        perclos, np.mean(metrics_window["pitch"])]
            features_norm = normalize_features(features, minF, maxF)
            fis_output = predict_drowsiness(features_norm)
            label = 1 if fis_output >= 0.5 else 0

            # --- Alert ---
            if alert_logic(label):
                cv2.putText(frame, "DROWSY ALERT!", (70, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # --- Display Metrics ---
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"HeadPitch: {pitch:.1f}", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            timestamp = time.strftime("%H:%M:%S")
            csv_writer.writerow([timestamp, avg_ear, mar, perclos, pitch, label])

    cv2.imshow("Drowsiness Detection Live Feed (Press 'q' to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
csv_file.close()
print("Data saved to anfis_predictions.csv")
