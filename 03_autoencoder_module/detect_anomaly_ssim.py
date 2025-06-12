import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tracker import CentroidTracker
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim
import os

# Ayarlar
model = load_model("models/autoencoder_320_best_v2.h5")
IMG_SIZE = 320
VIDEO_PATH = "test1.avi"
OUTPUT_PATH = "output/final_output.avi"
SSIM_THRESHOLD = 0.85
ANOMALY_HISTORY_FRAMES = 6
ANOMALY_CONFIRM_COUNT = 3

# Video işlemleri
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*"XVID")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 640, 360
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Arka plan çıkarma ve takip
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
tracker = CentroidTracker()
anomaly_history = defaultdict(list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (width, height))
    fgmask = fgbg.apply(frame_resized)

    # Gürültü temizleme
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    # Kontur bul
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

        # Rekonstrüksiyon
        reconstructed = model.predict(gray, verbose=0)
        original = gray.squeeze()
        reconstructed = reconstructed.squeeze()

        # SSIM hesapla
        score, _ = ssim(original, reconstructed, data_range=1.0, full=True)

        # Geçmişe kaydet
        anomaly_history[objectID].append(score < SSIM_THRESHOLD)
        if len(anomaly_history[objectID]) > ANOMALY_HISTORY_FRAMES:
            anomaly_history[objectID].pop(0)

        # Son 6 frame'in 3'ünden fazlası anomaly ise...
        is_anomaly = sum(anomaly_history[objectID]) >= ANOMALY_CONFIRM_COUNT
        label = f"ID:{objectID} {'Anomaly' if is_anomaly else 'Normal'} ({score:.2f})"
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)

        cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame_resized, label, (x+10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    writer.write(frame_resized)
    cv2.imshow("Anomaly + Tracking", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()