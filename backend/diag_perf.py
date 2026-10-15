"""Stop session, grab one real frame, profile each pipeline stage."""
import os, sys, time, json, urllib.request
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, ".")
import numpy as np
import cv2

# stop any live session so the camera is free
try:
    req = urllib.request.Request("http://127.0.0.1:8000/api/monitor/stop",
                                 data=b"{}", headers={"Content-Type": "application/json"},
                                 method="POST")
    urllib.request.urlopen(req, timeout=5).read()
    print("session stopped")
except Exception as e:
    print("stop:", e)
time.sleep(2.0)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
for _ in range(15):
    cap.read(); time.sleep(0.03)
ok, frame = cap.read()
cap.release()
if not ok:
    print("no frame from camera"); raise SystemExit
cv2.imwrite("diag_face.png", frame)
print("frame mean:", round(float(frame.mean()), 1))

import vision as v
import models as m
import xai as x

svc = m.get_service()
print("face detect box:", v.detect_face(frame))

# pre-extract regions so we can time the CNN separately
t = time.perf_counter(); regions = v.extract_eye_mouth_regions(frame, (180, 80, 250, 310)); t_regions = time.perf_counter()-t
le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
print(f"landmarks={regions.get('landmarks')} ear={regions.get('ear')} mar={regions.get('mar')}")

def timeit(fn, n=5):
    fn()
    s = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter()-s)/n

print()
print(f"{'stage':28s} {'ms':>8}")
print("-"*40)
print(f"{'detect_face':28s} {timeit(lambda: v.detect_face(frame))*1e3:8.1f}")
print(f"{'extract_eye_mouth_regions':28s} {t_regions*1e3:8.1f}")
if le is not None:
    print(f"{'predict_eye x2':28s} {timeit(lambda: (svc.predict_eye(le), svc.predict_eye(re)))*1e3:8.1f}")
    if re is None: re = le
    print(f"{'eye_gradcam_png':28s} {timeit(lambda: x.eye_gradcam_png(svc, le), 3)*1e3:8.1f}")
print(f"{'enhance_low_light':28s} {timeit(lambda: v.enhance_low_light(frame))*1e3:8.1f}")

# full analyze_frame (one-shot to avoid state mutation)
import app as a
print(f"{'analyze_frame(one-shot)':28s} {timeit(lambda: a.analyze_frame(frame, temporal=False), 3)*1e3:8.1f}")
