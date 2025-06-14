import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tracker import CentroidTracker

cap = cv2.VideoCapture("../test1.avi")
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)
ct = CentroidTracker()

# Video Writer tanımı
fourcc = cv2.VideoWriter_fourcc(*"XVID")
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = "tracking_output.mp4"
writer = cv2.VideoWriter(output_path, fourcc, fps, (640, 360))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    mask = fgbg.apply(frame_resized)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, w, h))
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)

    objects = ct.update(rects)

    for objectID, centroid in objects.items():
        text = f"ID {objectID}"
        cv2.putText(frame_resized, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame_resized, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

    # Yaz video dosyasına
    writer.write(frame_resized)

    cv2.imshow("Tracking", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()
