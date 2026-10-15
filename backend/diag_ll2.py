"""Haar vs MediaPipe on dark frame, raw and enhanced."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np
import mediapipe as mp
import vision as v

frame = cv2.imread("diag_dark.png")
if frame is None:
    print("diag_dark.png missing"); raise SystemExit

def enhance(bgr, target=115.0, clip=3.0):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    m = max(float(gray.mean()), 1e-3)
    if m >= 70.0:
        return bgr
    g = float(np.clip(np.log(target / 255.0) / np.log(m / 255.0), 0.35, 3.0))
    lut = np.array([((i / 255.0) ** g) * 255 for i in range(256)]).astype(np.uint8)
    out = cv2.LUT(bgr, lut)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def landmarks_ok(bgr):
    lm = v.get_landmarker()
    if lm is None:
        return False, None
    img = mp.Image(image_format=mp.ImageFormat.SRGB,
                   data=np.ascontiguousarray(bgr[:, :, ::-1]))
    res = lm.detect(img)
    if not res.face_landmarks:
        return False, None
    return True, len(res.face_landmarks[0])

print(f"raw mean={cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean():.1f}")
for tag, img in [("raw", frame), ("enhanced", enhance(frame))]:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    haar = v.detect_face(img)
    lm_ok, n = landmarks_ok(img)
    print(f"{tag:9s} mean={g.mean():6.1f}  haar={haar}  mediapipe_face={lm_ok} ({n} pts)")
    cv2.imwrite(f"diag_ll2_{tag}.png", img)
