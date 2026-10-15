"""Compare the eye pipeline on a REAL closed-eye image vs live open-eye frames.

Prints, for each source: MediaPipe landmarks success, raw EAR components
(vertical pairs, horizontal width), resulting EAR, EAR-normalised eyelid
aperture, and the CNN's raw (unclamped) p_open/p_closed.

This isolates whether the EAR authority is inflating closed-eye readings
(diagonal landmark pairings) and thereby vetoing the CNN's closure verdict.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")

import numpy as np
import cv2

import vision as v
import models as m

svc = m.get_service()

SHOT_CANDIDATES = [
    r"C:\Users\subim\OneDrive\Pictures\Screenshots\Screenshot 2026-10-15 034324.png",
    r"C:/Users/subim/OneDrive/Pictures/Screenshots/Screenshot 2026-10-15 034324.png",
]


def pair_report(frame, box, tag):
    regions = v.extract_eye_mouth_regions(frame, box)
    lm_ok = regions.get("landmarks")
    ear = regions.get("ear")
    le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")

    pl = svc.predict_eye(le) if le is not None else None
    pr = svc.predict_eye(re) if re is not None else None

    # Raw landmark geometry (why EAR reads what it reads)
    geo = ""
    if lm_ok:
        try:
            landmarker = v.get_landmarker()
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(frame[:, :, ::-1]))
            res = landmarker.detect(img)
            lm = res.face_landmarks[0]
            H, W = frame.shape[:2]

            def d(i, j):
                return float(np.hypot((lm[i].x - lm[j].x) * W,
                                      (lm[i].y - lm[j].y) * H))

            # code's EAR pairings
            l_v1, l_v2, l_h = d(160, 153), d(158, 144), d(33, 133)
            r_v1, r_v2, r_h = d(385, 373), d(387, 380), d(362, 263)
            # true vertical lid pairs (same set _eyelid_from_landmarks uses)
            l_ap, r_ap = d(159, 145), d(386, 374)
            geo = (f"| Lpair {l_v1:.1f}/{l_v2:.1f} w={l_h:.1f} -> "
                   f"diagEAR={(l_v1 + l_v2) / (2 * l_h):.3f} "
                   f"aperture/w={l_ap / l_h:.3f} "
                   f"|| Rpair {r_v1:.1f}/{r_v2:.1f} w={r_h:.1f} -> "
                   f"diagEAR={(r_v1 + r_v2) / (2 * r_h):.3f} "
                   f"aperture/w={r_ap / r_h:.3f}")
        except Exception as exc:
            geo = f"| geo failed: {exc}"

    def fmt(r):
        return (f"p_open={r['p_open']:.3f} p_closed={r['p_closed']:.3f}"
                if r else "n/a")
    print(f"{tag:26s} lm={str(lm_ok):5s} EAR={('%7.4f' % ear) if ear is not None else '   None'} "
          f"CNN[{fmt(pl)}] {geo}", flush=True)
    return regions


print("=" * 40, "CLOSED-EYE SCREENSHOT", "=" * 40)
shot = None
for p in SHOT_CANDIDATES:
    if os.path.exists(p):
        shot = p
        break
if shot is None:
    print("  screenshot not found")
else:
    img = cv2.imread(shot)
    print("  loaded", shot, "shape", img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = v.detect_face(img)
    print("  haar box:", box)
    # The dashboard face is drawn at 202x202 in the video panel; try haar box
    # first, then a generous crop around the cabin-feed panel.
    if box:
        pair_report(img, box, "screenshot haar-box")
    H, W = img.shape[:2]
    # cabin feed panel occupies roughly the left 48% of the screenshot
    crop = img[0:H, 0:int(W * 0.48)]
    box2 = v.detect_face(crop)
    print("  haar box (left panel):", box2)
    if box2:
        pair_report(crop, box2, "screenshot panel-crop")
    else:
        # fall back to the known dashboard bbox scaled into screenshot coords
        pair_report(crop, (int(W * 0.48 * 0.20), 60, int(W * 0.48 * 0.55),
                           int(W * 0.48 * 0.55)), "screenshot assumed-box")

print()
print("=" * 40, "LIVE WEBCAM (eyes open expected)", "=" * 40)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("  webcam unavailable")
    sys.exit(0)
ear_hist, cnn_hist = [], []
for i in range(12):
    ok, fr = cap.read()
    if not ok:
        print("  frame read failed")
        break
    fr = v.enhance_low_light(fr) if hasattr(v, "enhance_low_light") else fr
    box = v.detect_face(fr)
    if not box:
        print(f"  frame {i:2d}: no face")
        continue
    regions = pair_report(fr, box, f"webcam frame {i:2d}")
    if regions.get("ear") is not None:
        ear_hist.append(regions["ear"])
    le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
    pl = svc.predict_eye(le) if le is not None else None
    if pl:
        cnn_hist.append(pl["p_open"])
cap.release()

if ear_hist:
    a = np.array(ear_hist)
    print(f"\nwebcam EAR: n={len(a)} mean={a.mean():.3f} min={a.min():.3f} "
          f"max={a.max():.3f}")
if cnn_hist:
    b = np.array(cnn_hist)
    print(f"webcam CNN p_open: mean={b.mean():.3f} min={b.min():.3f} max={b.max():.3f}")
