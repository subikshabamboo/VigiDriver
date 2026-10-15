"""Live webcam EAR/CNN baseline: brightness, landmarks, EAR components, CNN.

Runs the same low-light enhancement the production loop applies, so face
detection behaves as it does in the real monitor.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")

import numpy as np
import cv2

import vision as v
import models as m
from vision import enhance_low_light

svc = m.get_service()

print("camera backends: trying DSHOW then default", flush=True)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("webcam unavailable")
    sys.exit(0)

ear_l, ear_r, ear_all, cnn_open, app_ear = [], [], [], [], []
n = 0
for i in range(40):
    ok, fr = cap.read()
    if not ok:
        print(f"frame {i}: read failed")
        break
    raw_bright = float(fr.mean())
    raw_std = float(fr.std())
    fr2 = enhance_low_light(fr)
    box = v.detect_face(fr2)
    if i < 6:
        print(f"frame {i:2d}: raw_mean={raw_bright:5.1f} raw_std={raw_std:5.1f} "
              f"enh_mean={fr2.mean():5.1f} haar={box}", flush=True)
    if not box:
        continue
    regions = v.extract_eye_mouth_regions(fr2, box)
    n += 1
    ear = regions.get("ear")
    le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
    pl = svc.predict_eye(le) if le is not None else None
    pr = svc.predict_eye(re) if re is not None else None
    p_open = np.mean([r["p_open"] for r in (pl, pr) if r]) if (pl or pr) else float("nan")

    geo = ""
    if regions.get("landmarks"):
        try:
            landmarker = v.get_landmarker()
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(fr2[:, :, ::-1]))
            res = landmarker.detect(img)
            lm = res.face_landmarks[0]
            H, W = fr2.shape[:2]

            def d(i, j):
                return float(np.hypot((lm[i].x - lm[j].x) * W,
                                      (lm[i].y - lm[j].y) * H))
            l_v1, l_v2, l_h = d(160, 153), d(158, 144), d(33, 133)
            l_ap = d(159, 145)
            geo = (f"diagEAR={(l_v1 + l_v2) / (2 * l_h):.3f} "
                   f"ap/w={l_ap / l_h:.3f} v1={l_v1:.1f} v2={l_v2:.1f} w={l_h:.1f}")
            ear_l.append(d(160, 153)); ear_r.append(d(158, 144))
        except Exception as exc:
            geo = f"geo-fail {exc}"
    if ear is not None:
        ear_all.append(ear)
    cnn_open.append(float(p_open))
    if i < 8 or i % 10 == 0:
        print(f"frame {i:2d}: bright={raw_bright:5.1f} box={str(box):24s} "
              f"lm={str(regions.get('landmarks')):5s} EAR={('%7.4f' % ear) if ear is not None else '  None'} "
              f"CNN_p_open={p_open:.3f} | {geo}", flush=True)

cap.release()
print()
if ear_all:
    a = np.array(ear_all)
    print(f"EAR  (code): n={len(a)} min={a.min():.4f} p10={np.percentile(a,10):.4f} "
          f"mean={a.mean():.4f} p90={np.percentile(a,90):.4f} max={a.max():.4f}")
    print(f"  -> EarClosure thresh = max(0.60*base, 0.12); with base=p90: "
          f"{max(0.60 * np.percentile(a, 90), 0.12):.4f}")
if cnn_open:
    b = np.array(cnn_open)
    print(f"CNN p_open: n={len(b)} min={b.min():.4f} mean={b.mean():.4f} max={b.max():.4f}")
