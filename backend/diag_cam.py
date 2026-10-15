"""Probe webcam indices / backends / formats to find one that returns real frames."""
import cv2
import numpy as np
import time

BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]

def probe(idx, name, backend):
    cap = cv2.VideoCapture(idx, backend)
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    # try MJPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    for _ in range(15):
        cap.read()
        time.sleep(0.03)
    frames = []
    for _ in range(5):
        ok, fr = cap.read()
        if ok and fr is not None:
            frames.append(fr)
        time.sleep(0.05)
    cap.release()
    if not frames:
        return ("no frames",)
    arr = np.stack([f.astype(np.float32) for f in frames])
    return (float(arr.mean()), float(arr.min()), float(arr.max()), float(arr.std()))

print(f"{'idx':>3} {'backend':8s} {'mean':>7} {'min':>6} {'max':>6} {'std':>7}")
print("-" * 45)
for idx in (0, 1, 2):
    for name, be in BACKENDS:
        try:
            r = probe(idx, name, be)
        except Exception as e:
            r = (f"ERR {type(e).__name__}",)
        if r and len(r) == 4:
            print(f"{idx:3d} {name:8s} {r[0]:7.1f} {r[1]:6.0f} {r[2]:6.0f} {r[3]:7.2f}")
        else:
            print(f"{idx:3d} {name:8s} {r}")
