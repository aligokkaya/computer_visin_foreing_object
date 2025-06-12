import cv2
import numpy as np
import os

cap = cv2.VideoCapture("autoencoding_clean.avi")
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

output_dir = "dataset/train/shishe_crops"
os.makedirs(output_dir, exist_ok=True)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    fgmask = fgbg.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 7000:
            x, y, w, h = cv2.boundingRect(cnt)
            crop = gray[y:y+h, x:x+w]

            # Kare şekline getirmek için yeniden boyutlandır
            crop_square = cv2.resize(crop, (128, 128))

            filename = os.path.join(output_dir, f"crop_{count:06d}.png")
            cv2.imwrite(filename, crop_square)
            count += 1

            cv2.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Detected Objects", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ {count} crop kaydedildi: {output_dir}")