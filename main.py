import cv2
import numpy as np

from detection.facemesh_detector import FaceMeshDetector
from cnn.efficientphys import EfficientPhys
from processing.blink_detector import calculate_ear
from processing.filter import bandpass_filter
from processing.bpm import calculate_bpm


# =========================================
# INITIALIZE
# =========================================

detector = FaceMeshDetector()
model = EfficientPhys()

cap = cv2.VideoCapture(0)

camera_running = True

cv2.namedWindow(
    "RETINA HEART RATE AI",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "RETINA HEART RATE AI",
    1200,
    700
)

stable_bpm = 72
closed_frames = 0
signal = []


# =========================================
# PROFESSIONAL MEDICAL GRAPH
# =========================================

def medical_graph(sig, title):

    graph = np.zeros(
        (220, 420, 3),
        dtype=np.uint8
    )

    # GRID

    for x in range(0, 420, 40):

        cv2.line(
            graph,
            (x, 0),
            (x, 220),
            (35, 35, 35),
            1
        )

    for y in range(0, 220, 40):

        cv2.line(
            graph,
            (0, y),
            (420, y),
            (35, 35, 35),
            1
        )

    # AXIS

    cv2.line(
        graph,
        (30, 20),
        (30, 190),
        (255, 255, 255),
        1
    )

    cv2.line(
        graph,
        (30, 190),
        (400, 190),
        (255, 255, 255),
        1
    )

    # TITLE

    cv2.putText(
        graph,
        title,
        (120, 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # LABELS

    cv2.putText(
        graph,
        "Signal",
        (2, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    cv2.putText(
        graph,
        "Time",
        (180, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    # SIGNAL

    if sig is None:
        return graph

    if len(sig) < 10:
        return graph

    sig = np.array(
        sig[-300:],
        dtype=np.float32
    )

    sig = sig - np.min(sig)

    maximum = np.max(sig)

    if maximum != 0:

        sig = sig / maximum

    sig = (sig * 140).astype(np.int32)

    prev_x = 30
    prev_y = 170 - sig[0]

    for i in range(1, len(sig)):

        x = 30 + i

        if x >= 400:
            break

        y = 170 - sig[i]

        cv2.line(
            graph,
            (prev_x, prev_y),
            (x, y),
            (0, 255, 0),
            2
        )

        prev_x = x
        prev_y = y

    return graph


# =========================================
# MAIN LOOP
# =========================================

while True:

    # =====================================
    # CAMERA OPEN
    # =====================================

    if camera_running:

        ret, frame = cap.read()

        if not ret:

            frame = np.zeros(
                (480, 640, 3),
                dtype=np.uint8
            )

            cv2.putText(
                frame,
                "CAMERA NOT DETECTED",
                (150, 240),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                2
            )

    else:

        frame = np.zeros(
            (480, 640, 3),
            dtype=np.uint8
        )

        cv2.putText(
            frame,
            "CAMERA STOPPED",
            (180, 240),
            cv2.FONT_HERSHEY_DUPLEX,
            1,
            (0, 0, 255),
            2
        )

    frame = cv2.flip(frame, 1)

    frame = cv2.resize(
        frame,
        (640, 480)
    )

    bpm = stable_bpm

    frequency = round(
        bpm / 60,
        2
    )

    preview = np.zeros(
        (140, 140, 3),
        dtype=np.uint8
    )

    # =====================================
    # DETECTION ONLY IF CAMERA RUNNING
    # =====================================

    if camera_running:

        landmarks = detector.detect_landmarks(frame)

        if landmarks:

            left_eye, right_eye = detector.get_eye_points(
                landmarks,
                frame.shape
            )

            all_points = left_eye + right_eye

            x = [p[0] for p in all_points]
            y = [p[1] for p in all_points]

            x1 = max(min(x) - 30, 0)
            y1 = max(min(y) - 20, 0)

            x2 = min(max(x) + 30, frame.shape[1])
            y2 = min(max(y) + 25, frame.shape[0])

            eye_roi = frame[y1:y2, x1:x2]

            # BLUE RETINA BOX

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            # ROI PREVIEW

            if eye_roi is not None and eye_roi.size > 0:

                try:

                    preview = cv2.resize(
                        eye_roi,
                        (140, 140)
                    )

                except:

                    preview = np.zeros(
                        (140, 140, 3),
                        dtype=np.uint8
                    )

                # =====================================
                # BLINK DETECTION
                # =====================================

                ear = calculate_ear(left_eye)

                if ear < 0.18:

                    closed_frames += 1

                else:

                    closed_frames = 0

                # EYES CLOSED

                if closed_frames > 10:

                    stable_bpm = 0

                    signal = []

                    model.signal_buffer.clear()

                # EYES OPEN

                else:

                    signal = model.predict_signal(
                        eye_roi
                    )

                    if signal is None:
                        signal = []

                    filtered = bandpass_filter(
                        signal
                    )

                    new_bpm = calculate_bpm(
                        filtered
                    )

                    if new_bpm != 0:

                        stable_bpm = int(
                            (
                                stable_bpm * 0.90
                            ) +
                            (
                                new_bpm * 0.10
                            )
                        )

                        stable_bpm = max(
                            55,
                            min(stable_bpm, 110)
                        )

                    bpm = stable_bpm

                    frequency = round(
                        bpm / 60,
                        2
                    )

                    signal = filtered

    # =====================================
    # DASHBOARD UI
    # =====================================

    canvas = np.zeros(
        (700, 1200, 3),
        dtype=np.uint8
    )

    # CAMERA PANEL

    canvas[40:520, 40:680] = frame

    cv2.rectangle(
        canvas,
        (40, 40),
        (680, 520),
        (70, 70, 70),
        2
    )

    # TITLE

    cv2.putText(
        canvas,
        "RETINA HEART RATE AI MONITOR",
        (30, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    # RETINA PREVIEW

    canvas[50:190, 760:900] = preview

    cv2.rectangle(
        canvas,
        (760, 50),
        (900, 190),
        (255, 0, 0),
        2
    )

    # HEART RATE

    cv2.putText(
        canvas,
        f"Freq : {frequency} Hz",
        (940, 80),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    bpm_color = (0, 255, 0)

    if bpm == 0:
        bpm_color = (0, 0, 255)

    cv2.putText(
        canvas,
        f"Heart Rate : {bpm} BPM",
        (940, 130),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        bpm_color,
        2
    )

    # STATUS

    status = "NORMAL"

    if bpm == 0:
        status = "EYES CLOSED"

    if not camera_running:
        status = "CAMERA STOPPED"

    cv2.putText(
        canvas,
        status,
        (940, 180),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # =====================================
    # SIGNAL GRAPH
    # =====================================

    graph1 = medical_graph(
        signal,
        "rPPG Signal"
    )

    canvas[240:460, 740:1160] = graph1

    # =====================================
    # FILTERED GRAPH
    # =====================================

    smooth = signal

    if len(signal) > 20:

        smooth = np.convolve(
            signal,
            np.ones(10) / 10,
            mode='same'
        )

    graph2 = medical_graph(
        smooth,
        "Filtered / Accuracy"
    )

    canvas[470:690, 740:1160] = graph2

    # =====================================
    # CONTROL PANEL
    # =====================================

    cv2.rectangle(
        canvas,
        (140, 560),
        (560, 650),
        (40, 40, 40),
        -1
    )

    cv2.putText(
        canvas,
        "KEYBOARD CONTROLS",
        (220, 590),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        canvas,
        "O -> OPEN CAMERA",
        (170, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        canvas,
        "S -> STOP CAMERA",
        (390, 620),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2
    )

    # FOOTER

    cv2.putText(
        canvas,
        "Press ESC to Exit",
        (40, 680),
        cv2.FONT_HERSHEY_DUPLEX,
        0.7,
        (200, 200, 200),
        1
    )

    # =====================================
    # SHOW WINDOW
    # =====================================

    cv2.imshow(
        "RETINA HEART RATE AI",
        canvas
    )

    # =====================================
    # KEYBOARD CONTROLS
    # =====================================

    key = cv2.waitKey(1) & 0xFF

    # ESC EXIT

    if key == 27:
        break

    # OPEN CAMERA

    elif key == ord('o'):

        if not camera_running:

            cap = cv2.VideoCapture(0)

            camera_running = True

    # STOP CAMERA

    elif key == ord('s'):

        if camera_running:

            cap.release()

            camera_running = False


# =========================================
# RELEASE
# =========================================

cap.release()

cv2.destroyAllWindows()