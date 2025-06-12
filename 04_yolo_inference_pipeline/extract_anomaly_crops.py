import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker import CentroidTracker
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim


model = load_model("/Users/aligokkaya/Desktop/CV_task/3/models/autoencoder_320_best_v2.h5")
IMG_SIZE = 320
VIDEO_PATH = "/Users/aligokkaya/Desktop/CV_task/test1.avi"
OUTPUT_PATH = "output/final_output.avi"
SSIM_THRESHOLD = 0.85
ANOMALY_HISTORY_FRAMES = 6
ANOMALY_CONFIRM_COUNT = 3


os.makedirs("output", exist_ok=True)
os.makedirs("anomaly_crops", exist_ok=True)
crop_id_counter = 0


cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 360
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
tracker = CentroidTracker()
anomaly_history = defaultdict(list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

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
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gray = gray.astype("float32") / 255.0
        gray = np.expand_dims(gray, axis=(0, -1))

        reconstructed = model.predict(gray, verbose=0)
        original = gray.squeeze()
        reconstructed = reconstructed.squeeze()

        score_ssim, _ = ssim(original, reconstructed, data_range=1.0, full=True)

        anomaly_history[objectID].append(score_ssim < SSIM_THRESHOLD)
        if len(anomaly_history[objectID]) > ANOMALY_HISTORY_FRAMES:
            anomaly_history[objectID].pop(0)

        is_anomaly = sum(anomaly_history[objectID]) >= ANOMALY_CONFIRM_COUNT
        label = f"ID:{objectID} {'Anomaly' if is_anomaly else 'Normal'} ({score_ssim:.2f})"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_resized, label, (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if is_anomaly:
            crop_path = os.path.join("anomaly_crops", f"anomaly_{crop_id_counter:04d}.png")
            cv2.imwrite(crop_path, crop)
            crop_id_counter += 1

    writer.write(frame_resized)
    cv2.imshow("Anomaly + Tracking", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()