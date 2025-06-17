import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
import os
from collections import defaultdict
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker import CentroidTracker

from tracker import CentroidTracker  

VIDEO_PATH = "../test1.avi"
OUTPUT_PATH = "output/final_defect_output.mp4"
MODEL_PATH = "../03_autoencoder_module/models/autoencoder_320_best_v2.h5"
IMG_SIZE = 320
SSIM_THRESHOLD = 0.85
ANOMALY_HISTORY_FRAMES = 6
ANOMALY_CONFIRM_COUNT = 3

model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 360
fourcc = cv2.VideoWriter_fourcc(*"XVID")
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
tracker = CentroidTracker()
anomaly_history = defaultdict(list)

frame_idx = 0
processed_frames = 0

def detect_wires(image, min_length=10, max_width=12, aspect_ratio_thresh=1.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV)

    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wires = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = max(w, h) / (min(w, h) + 1e-5)

        if max(w, h) >= min_length and min(w, h) <= max_width and aspect_ratio > aspect_ratio_thresh:
            wires.append((x, y, w, h))

    return wires

while True:
    ret, frame = cap.read()
 
    frame_resized = cv2.resize(frame, (width, height))
    fgmask = fgbg.apply(frame_resized)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [(x, y, w, h) for c in contours if cv2.contourArea(c) > 7000
             for (x, y, w, h) in [cv2.boundingRect(c)]]

    objects = tracker.update(rects)

    for ((x, y, w, h), objectID) in zip(rects, objects):
        crop = frame_resized[y:y+h, x:x+w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=(0, -1))

        reconstructed = model.predict(gray.reshape(1, 320, 320, 1), verbose=0)
        score = ssim(gray.squeeze(), reconstructed.squeeze(), data_range=1.0)

        anomaly_history[objectID].append(score < SSIM_THRESHOLD)
        if len(anomaly_history[objectID]) > ANOMALY_HISTORY_FRAMES:
            anomaly_history[objectID].pop(0)

        is_anomaly = sum(anomaly_history[objectID]) >= ANOMALY_CONFIRM_COUNT
        label = f"ID:{objectID} Normal ({score:.2f})"
        color = (0, 255, 0)

        if is_anomaly:
            defects = detect_wires(crop)
            if defects:
                label = f"ID:{objectID} Defect ({score:.2f})"
                color = (0, 0, 255)
                for (dx, dy, dw, dh) in defects:
                    cv2.rectangle(crop, (dx, dy), (dx+dw, dy+dh), (255, 0, 255), 2)

        cv2.rectangle(frame_resized, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame_resized, label, (x+10, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    writer.write(frame_resized)
    cv2.imshow("Tracking (Stable)", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()