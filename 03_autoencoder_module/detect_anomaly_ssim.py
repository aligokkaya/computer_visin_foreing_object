import cv2
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model
from collections import defaultdict
from skimage.metrics import structural_similarity as ssim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker import CentroidTracker

# Ayarlar
model = load_model("models/autoencoder_320_best_v2.h5")
IMG_SIZE = 320
VIDEO_PATH = "../test1.avi"
OUTPUT_PATH = "output/final_output_task3.mp4"
SSIM_THRESHOLD = 0.85
ANOMALY_HISTORY_FRAMES = 6
ANOMALY_CONFIRM_COUNT = 3

# Video ve takip ayarları
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"XVID"), fps, (640, 360))
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
tracker = CentroidTracker(max_disappeared=5)
anomaly_history = defaultdict(list)

# ID eşleme için çizgi takibi
crossed_ids = set()
object_id_map = {}
current_id = 1
line_x = 320  # Ortadaki çizgi

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    roi = frame_resized[:-50, :]  # Alt 50 piksel dışarıda bırakılır
    fgmask = fgbg.apply(roi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10000 < area < 200000:
            x, y, w, h = cv2.boundingRect(cnt)
            if not any(abs(x - rx) < 30 and abs(y - ry) < 30 for (rx, ry, rw, rh) in rects):
                rects.append((x, y, w, h))

    objects = tracker.update(rects)

    for objectID, centroid in objects.items():
        cx, cy = centroid

        # Çizgiyi geçince ID ata
        if cx > line_x and objectID not in crossed_ids:
            crossed_ids.add(objectID)
            object_id_map[objectID] = current_id
            current_id += 1

        mapped_id = object_id_map.get(objectID, "-")

        for (x, y, w, h) in rects:
            if x < cx < x + w and y < cy < y + h:
                crop = frame_resized[y:y + h, x:x + w]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                gray = gray.astype("float32") / 255.0
                gray = np.expand_dims(gray, axis=(0, -1))

                reconstructed = model.predict(gray, verbose=0)
                original = gray.squeeze()
                reconstructed = reconstructed.squeeze()

                score, _ = ssim(original, reconstructed, data_range=1.0, full=True)
                anomaly_history[objectID].append(score < SSIM_THRESHOLD)
                if len(anomaly_history[objectID]) > ANOMALY_HISTORY_FRAMES:
                    anomaly_history[objectID].pop(0)

                is_anomaly = sum(anomaly_history[objectID]) >= ANOMALY_CONFIRM_COUNT
                label = f"ID:{mapped_id} {'Anomaly' if is_anomaly else 'Normal'} ({score:.2f})"
                color = (0, 0, 255) if is_anomaly else (0, 255, 0)

                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame_resized, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                break

        cv2.putText(frame_resized, f"TrackID {objectID}", (cx - 20, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.circle(frame_resized, (cx, cy), 4, (0, 0, 255), -1)

    writer.write(frame_resized)
    cv2.imshow("Tracking (Stable)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
writer.release()
cv2.destroyAllWindows()