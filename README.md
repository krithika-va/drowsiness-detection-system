# 😴 Real-Time Drowsiness Detection System

A multi-modal driver safety system that combines **Computer Vision** with **Neuro-Fuzzy (ANFIS)** modeling to detect fatigue levels in real-time.
This project goes beyond simple eye-tracking. It monitors multiple physiological and behavioral cues via a live webcam feed, processing them through a hybrid AI architecture to achieve superior classification accuracy compared to traditional SVM or KNN models.

### 🧠 The ANFIS Advantage
While standard machine learning models offer binary classification, the **Adaptive Neuro-Fuzzy Inference System (ANFIS)** handles the uncertainty and "fuzziness" of human fatigue, leading to a more robust and reliable warning system.

---

## ⚙️ Technical Workflow

1. **Live Acquisition:** High-speed video capture via OpenCV.
2. **Feature Extraction:** Leveraging **MediaPipe** to calculate:
   * **EAR (Eye Aspect Ratio):** To detect blinking patterns and closure.
   * **MAR (Mouth Aspect Ratio):** To monitor yawning frequency.
   * **PERCLOS:** Percentage of eye closure over time.
   * **Head Pitch:** Tracking head nods or "micro-sleep" posture.
3. **Classification:** Features are fed into a **MATLAB-trained ANFIS model**, providing real-time drowsiness scoring.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **Language** | Python 3.x, MATLAB |
| **Vision** | OpenCV, MediaPipe |
| **Intelligence** | ANFIS (Neuro-Fuzzy), Scikit-Learn (for Benchmarking) |
| **Libraries** | NumPy, SciPy, Matplotlib |

---

## 📊 Performance Comparison

Extensive testing showed that the **ANFIS model** significantly outperformed other supervised learning approaches:

| Model | Accuracy | Robustness |
| :--- | :--- | :--- |
| **ANFIS** | **Highest** | **High (Handles uncertainty)** |
| SVM | Moderate | Sensitive to noise |
| KNN | Low-Moderate | Computationally expensive |

---
y
