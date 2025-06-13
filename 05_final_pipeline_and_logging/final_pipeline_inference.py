import cv2
import numpy as np
import os
import csv
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.models import load_model
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker import CentroidTracker

from ultralytics import YOLO
from datetime import datetime


model = load_model("../03_autoencoder_module/models/autoencoder_320_best_v2.h5")
yolo_model = YOLO("content/runs/detect/train/weights/best.pt")  # YOLOv8 modelini buraya koy

IMG_SIZE = 320
VIDEO_PATH = "../test1.avi"
OUTPUT_PATH = "output/final_output.mp4"
LOG_PATH = "output/anomaly_log.csv"
SSIM_THRESHOLD = 0.85
ANOMALY_HISTORY_FRAMES = 6
ANOMALY_CONFIRM_COUNT = 3

# Yüklemeler
tracker = CentroidTracker()
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
anomaly_history = defaultdict(list)

# Video ayarları
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 360
fourcc = cv2.VideoWriter_fourcc(*"XVID")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Log CSV başlığı
with open(LOG_PATH, mode="w", newline="") as file:
    writer_csv = csv.writer(file)
    writer_csv.writerow(["Frame", "ObjectID", "SSIM", "YOLO_Class", "X", "Y", "W", "H", "Timestamp"])

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_resized = cv2.resize(frame, (width, height))
    fgmask = fgbg.apply(frame_resized)

    # Gürültü temizleme
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

        # Autoencoder çıkışı
        reconstructed = model.predict(gray, verbose=0)
        score, _ = ssim(gray.squeeze(), reconstructed.squeeze(), data_range=1.0, full=True)

        anomaly_history[objectID].append(score < SSIM_THRESHOLD)
        if len(anomaly_history[objectID]) > ANOMALY_HISTORY_FRAMES:
            anomaly_history[objectID].pop(0)

        is_anomaly = sum(anomaly_history[objectID]) >= ANOMALY_CONFIRM_COUNT
        label = f"ID:{objectID} {'Anomaly' if is_anomaly else 'Normal'} ({score:.2f})"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        # Anomaliler için YOLO sınıflandırması
        yolo_class = "-"
        if is_anomaly:
            temp_crop = cv2.resize(crop, (640, 640))  # YOLOv8 input
            results = yolo_model.predict(temp_crop, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    yolo_class = r.names[cls_id]
                    bbox = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(crop, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(crop, yolo_class, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Log CSV
        with open(LOG_PATH, mode="a", newline="") as file:
            writer_csv = csv.writer(file)
            writer_csv.writerow([frame_count, objectID, round(score, 4), yolo_class, x, y, w, h,
                                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_resized, f"{label} / {yolo_class}", (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    writer.write(frame_resized)
    cv2.imshow("Anomaly + YOLO", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()