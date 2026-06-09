# 🎓 Heart Rate Estimation using Eye Blink Detection

A Flask-based web application that estimates heart rate using eye blink detection from webcam feed. It uses Computer Vision (OpenCV + MediaPipe) and simple ML logic to calculate heart rate based on blink frequency.

---

## 🚀 Features

- 👁️ Live webcam face and eye tracking
- 👀 Eye-only detection using MediaPipe
- ⚡ Blink detection using Eye Aspect Ratio (EAR) method
- ❤️ Heart rate estimation from blink frequency
- 📊 Real-time dashboard with graphs
- 🚫 Heart rate becomes 0 BPM when eyes are hidden

---

## 🧠 Example Output
<img width="1502" height="915" alt="image" src="https://github.com/user-attachments/assets/a8e8553b-ef73-486f-9577-64b499852875" />

| Blink Frequency   | Heart Rate (BPM) |
|------------------|------------------|
| Low blinking     | 60–75 BPM        |
| Moderate blinking| 70–85 BPM        |
| High blinking    | 85–100 BPM       |
| Eyes hidden      | 0 BPM            |

---

## 🛠️ Tech Stack

- Flask
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- HTML, CSS, JavaScript

---

## ▶️ Installation & Run

```bash
pip install flask opencv-python mediapipe numpy
python app.py
