import cv2
import os
import numpy as np
from tqdm import tqdm

VIDEO_PATH = "../autoencoding_clean.avi"
OUTPUT_DIR = "dataset/train/clean_crops_320"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = 320
CROP_RATIO = 0.6
AREA_THRESHOLD = 5000

fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=False)

cap = cv2.VideoCapture(VIDEO_PATH)
crop_count = 0

def center_crop_and_resize(img, ratio=0.6, size=320):
    h, w = img.shape[:2]
    new_h, new_w = int(h * ratio), int(w * ratio)
    y1, x1 = (h - new_h) // 2, (w - new_w) // 2
    crop = img[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(crop, (size, size))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, (640, 360))
    mask = fgbg.apply(frame_resized)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < AREA_THRESHOLD:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = frame_resized[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        processed = center_crop_and_resize(gray, ratio=CROP_RATIO, size=IMG_SIZE)
        out_path = os.path.join(OUTPUT_DIR, f"crop_{crop_count:05}.png")
        cv2.imwrite(out_path, processed)
        crop_count += 1

cap.release()
print(f"{crop_count} adet 320x320 crop kaydedildi.")