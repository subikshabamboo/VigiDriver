"""Capture real webcam frames and log the full detection chain."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")
import time
import numpy as np
import cv2

import models as m
import vision as v

svc = m.get_service()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("WEBCAM NOT AVAILABLE")
    sys.exit(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(5):
    cap.read()  # warm up

print(f"{'f':>3} {'bright':>7} {'haar':>18} {'lm':>5} {'EAR':>6} {'MAR':>6} {'p_open':>7} {'p_closed':>8} cropMean")
print("-" * 90)
saved = 0
for i in range(25):
    ok, fr = cap.read()
    if not ok:
        print(f"{i:3d}  read failed")
        continue
    bright = float(fr.mean())
    box = v.detect_face(fr)
    ear = mar = None
    lm = False
    p_open = p_closed = float("nan")
    cmean = "None"
    if box is not None:
        r = v.extract_eye_mouth_regions(fr, box)
        ear = r.get("ear"); mar = r.get("mar"); lm = r.get("landmarks")
        le, re = r.get("left_eye_crop"), r.get("right_eye_crop")
        crops = [c for c in (le, re) if c is not None]
        if crops:
            ps = [svc.predict_eye(c) for c in crops]
            p_open = float(np.mean([p["p_open"] for p in ps]))
            p_closed = float(np.mean([p["p_closed"] for p in ps]))
            cmean = f"{crops[0].mean():5.1f}"
            if saved < 3:
                cv2.imwrite(f"diag_crop_{saved}.png", crops[0])
                cv2.imwrite(f"diag_frame_{saved}.png", fr)
                saved += 1
    bt = "None" if box is None else f"{box}"
    print(f"{i:3d} {bright:7.1f} {bt:>18} {str(lm):>5} "
          f"{('%.3f'%ear) if ear is not None else '  -  ':>6} "
          f"{('%.3f'%mar) if mar is not None else '  -  ':>6} "
          f"{p_open:7.3f} {p_closed:8.3f} {cmean}")
    time.sleep(0.05)

cap.release()
print("\nsaved diag_frame_*.png and diag_crop_*.png")
