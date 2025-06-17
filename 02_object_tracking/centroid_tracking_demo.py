import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker import CentroidTracker

cap = cv2.VideoCapture("../test1.avi")
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
ct = CentroidTracker(max_disappeared=5)

fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = cap.get(cv2.CAP_PROP_FPS)
writer = cv2.VideoWriter("tracking_output.mp4", fourcc, fps, (640, 360))

# Geçiş yapılan ID'ler ve çizgi pozisyonları
crossed_ids = set()
object_id_map = {}
current_id = 1
line_x = 320  # Dikey çizgi x koordinatı (frame genişliği 640)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    roi = frame_resized[:-50, :] 
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

            # Yakın dikdörtgenleri filtrelemek için:
            if not any(abs(x - rx) < 30 and abs(y - ry) < 30 for (rx, ry, rw, rh) in rects):
                rects.append((x, y, w, h))

    objects = ct.update(rects)

    for objectID, centroid in objects.items():
        cx, cy = centroid
        # Dikey çizgiye göre geçiş kontrolü
        if cx > line_x and objectID not in crossed_ids:
            crossed_ids.add(objectID)
            object_id_map[objectID] = current_id
            current_id += 1

        # Görsel bilgi yaz
        text = f"TrackID {objectID}"
        cv2.putText(roi, text, (cx - 20, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.circle(roi, (cx, cy), 4, (0, 0, 255), -1)

    for (x, y, w, h) in rects:
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Dikey çizgiyi çiz
    #cv2.line(roi, (line_x, 0), (line_x, roi.shape[0]), (255, 0, 0), 2)

    writer.write(frame_resized)
    cv2.imshow("Tracking (Stable)", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()