"""Capture a dark frame; test if enhancement makes the face detectable."""
import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
for _ in range(20):
    cap.read(); time.sleep(0.03)
ok, fr = cap.read()
cap.release()
if not ok:
    print("no frame"); raise SystemExit

cv2.imwrite("diag_dark.png", fr)
g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
print(f"raw brightness mean={g.mean():.1f} min={g.min()} max={g.max()}")

import vision as v

variants = {}
variants["raw"] = fr
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
g_clahe = clahe.apply(g)
variants["clahe2"] = cv2.cvtColor(g_clahe, cv2.COLOR_GRAY2BGR)
clahe4 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
variants["clahe4"] = cv2.cvtColor(clahe4.apply(g), cv2.COLOR_GRAY2BGR)
variants["equalize"] = cv2.cvtColor(cv2.equalizeHist(g), cv2.COLOR_GRAY2BGR)

# Strong gamma + CLAHE
gamma = 0.4
lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
g_gam = cv2.LUT(g, lut)
variants["gamma0.4"] = cv2.cvtColor(g_gam, cv2.COLOR_GRAY2BGR)
variants["gamma+clahe"] = cv2.cvtColor(clahe.apply(g_gam), cv2.COLOR_GRAY2BGR)

# Gain normalize
p2, p98 = np.percentile(g, (2, 98))
gn = np.clip((g.astype(np.float32) - p2) / max(1, p98 - p2) * 255, 0, 255).astype(np.uint8)
variants["p2norm"] = cv2.cvtColor(gn, cv2.COLOR_GRAY2BGR)

print(f"\n{'variant':14s} {'mean':>6} {'face_box(vision.detect_face)':>30}")
print("-" * 60)
for name, img in variants.items():
    gg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = v.detect_face(img)
    print(f"{name:14s} {gg.mean():6.1f} {str(box):>30}")
    cv2.imwrite(f"diag_ll_{name}.png", img)
print("\nsaved diag_dark.png + diag_ll_*.png")
