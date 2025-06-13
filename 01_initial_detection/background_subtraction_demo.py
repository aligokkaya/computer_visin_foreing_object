import cv2
import numpy as np

cap = cv2.VideoCapture("../test1.avi")

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

# VideoWriter başlat
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("initial_detection_full.mp4", fourcc, 20.0, (640, 360))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360)) 
    fgmask = fgbg.apply(frame_resized)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

    out.write(frame_resized)
    cv2.imshow("Detected Objects", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()