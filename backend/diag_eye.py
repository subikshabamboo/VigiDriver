"""Diagnostic: run the REAL eye pipeline on simulated open->closed frames.

Prints, per frame: simulated openness, landmarker success, EAR, CNN p_open for
each eye crop, and the combined closure decision. This isolates whether the
failure is (a) the eye CNN, (b) the crop, or (c) the EAR/landmark path.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")

import numpy as np
import cv2

import models as m
import vision as v
import app as a

svc = m.get_service()

BOX = (180, 80, 250, 310)

def run(frame, tag):
    regions = v.extract_eye_mouth_regions(frame, BOX)
    le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
    pl = svc.predict_eye(le) if le is not None else None
    pr = svc.predict_eye(re) if re is not None else None
    p_open = np.mean([r["p_open"] for r in (pl, pr) if r]) if (pl or pr) else float("nan")
    p_closed = np.mean([r["p_closed"] for r in (pl, pr) if r]) if (pl or pr) else float("nan")
    ear = regions.get("ear")
    lm = regions.get("landmarks")
    # Crop brightness stats (a closed eye is mostly skin = brighter/low variance)
    def cstat(c):
        if c is None: return "None"
        return f"mean={c.mean():6.1f} std={c.std():5.1f}"
    print(f"{tag:26s} open_sim={OPEN:4.2f} lm={str(lm):5s} EAR={('%.3f'%ear) if ear is not None else ' None'} "
          f"p_open={p_open:5.3f} p_closed={p_closed:5.3f} | L[{cstat(le)}] R[{cstat(re)}]")
    return p_open, p_closed

print("="*118)
print("SIMULATED DRIVER: openness ramps 1.0 -> 0.06 after frame 20")
print("="*118)
for fc in [2, 5, 8, 12, 18, 25, 35, 45, 60, 70, 90, 120]:
    global OPEN
    OPEN = a._eyelid_openness(fc)
    frame = a._simulate_frame(fc)
    # refresh global for print
    OPEN = a._eyelid_openness(fc)
    run(frame, f"sim frame {fc}")

print()
print("="*118)
print("SYNTHETIC EYE CROPS (direct CNN sensitivity, no landmarker)")
print("="*118)
def crop_open():
    img = np.full((64,64), 170, np.uint8)
    img[24:40, 8:56] = 225
    cv2.circle(img, (32,32), 9, 55, -1)
    cv2.circle(img, (32,32), 4, 25, -1)
    return img
def crop_closed():
    img = np.full((64,64), 170, np.uint8)
    img[30:34, 8:56] = 60
    return img
def crop_half():
    img = np.full((64,64), 170, np.uint8)
    img[24:34, 8:56] = 225
    img[22:27, 8:56] = 140
    cv2.circle(img, (32,30), 7, 55, -1)
    return img

for name, c in [("open", crop_open()), ("half", crop_half()), ("closed", crop_closed())]:
    r = svc.predict_eye(c)
    print(f"  {name:8s} -> p_open={r['p_open']:.4f}  p_closed={r['p_closed']:.4f}  label={r['label']}")

print()
print("="*118)
print("WEB CAMERA (if available)")
print("="*118)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if cap.isOpened():
    ok, fr = cap.read()
    if ok:
        b = v.detect_face(fr)
        print("  webcam frame grabbed; haar face box:", b)
        if b:
            r2 = v.extract_eye_mouth_regions(fr, b)
            le, re = r2.get("left_eye_crop"), r2.get("right_eye_crop")
            print("  landmarks:", r2.get("landmarks"), "EAR:", r2.get("ear"), "MAR:", r2.get("mar"))
            for tag, c in (("L", le), ("R", re)):
                if c is not None:
                    rr = svc.predict_eye(c)
                    print(f"  {tag} crop mean={c.mean():6.1f} -> p_open={rr['p_open']:.4f} p_closed={rr['p_closed']:.4f}")
        else:
            print("  no Haar face detected in webcam frame")
    else:
        print("  webcam opened but frame read failed")
    cap.release()
else:
    print("  webcam not available / busy")
