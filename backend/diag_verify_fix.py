"""Verify the eye-closure fix using trustworthy labels and measured live values.

Ground truth for the labelled frames comes from the sclera fraction recorded
at capture time (in the filename), which was measured on the eye ROI and
agrees with the capture protocol. The CNN and EAR readouts here come from the
production extraction path.

Checks:
  1. CNN reports closure on every pixel-labelled shut eye.
  2. The corrected EAR separates open (0.505) from shut (0.30-0.36).
  3. EarClosure fires on shut values and stays quiet on open values.
  4. CNN-or-EAR classifies every labelled frame correctly.
  5. The sustained tracker escalates to micro-sleep at 2.0 s of wall clock,
     at both a fast and a slow loop rate, while blinks stay protected.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")

import glob
import re
import numpy as np
import cv2

import vision as v
import models as m

svc = m.get_service()
HERE = os.path.dirname(os.path.abspath(__file__))
SHUT_SCLERA = 0.12      # capture-time label: sclera below this = eyes shut
OPEN_SCLERA = 0.20      # capture-time label: sclera above this = eyes open


def box_from_landmarks(img):
    landmarker = v.get_landmarker()
    import mediapipe as mp
    im = mp.Image(image_format=mp.ImageFormat.SRGB,
                  data=np.ascontiguousarray(img[:, :, ::-1]))
    r = landmarker.detect(im)
    if not r.face_landmarks:
        return None
    lm = r.face_landmarks[0]
    H, W = img.shape[:2]
    xs = [p.x * W for p in lm]
    ys = [p.y * H for p in lm]
    return (int(min(xs)), int(min(ys)),
            max(1, int(max(xs) - min(xs))), max(1, int(max(ys) - min(ys))))


def measure(path):
    """Return (sclera_label, ear, cnn_open) via the production path."""
    img = cv2.imread(path)
    if img is None:
        return None
    img = v.enhance_low_light(img)
    box = box_from_landmarks(img)
    if box is None:
        return None
    regions = v.extract_eye_mouth_regions(img, box)
    if not regions.get("landmarks"):
        return None
    l_crop = regions.get("left_eye_crop")
    r_crop = regions.get("right_eye_crop")
    pl = svc.predict_eye(l_crop) if l_crop is not None else None
    pr = svc.predict_eye(r_crop) if r_crop is not None else None
    vals = [x["p_open"] for x in (pl, pr) if x]
    cnn = float(np.mean(vals)) if vals else float("nan")
    m = re.search(r"_sf([0-9.]+)\.png$", os.path.basename(path))
    sf = float(m.group(1)) if m else float("nan")
    return sf, regions.get("ear"), cnn


print("=" * 84)
files = sorted(glob.glob(os.path.join(HERE, "cap_closed", "*.png")))
rows = []
for p in files:
    r = measure(p)
    if r:
        rows.append((os.path.basename(p),) + r)

print("LABELLED FRAMES  (capture-time sclera label vs production readouts)")
print(f"  {'file':20s} {'sclera':>7s} {'label':>7s} {'EAR':>7s} {'CNN':>7s}")
for name, sf, ear, cnn in rows:
    lab = "SHUT" if sf < SHUT_SCLERA else ("OPEN" if sf > OPEN_SCLERA else "mid")
    print(f"  {name:20s} {sf:7.3f} {lab:>7s} "
          f"{(ear if ear is not None else float('nan')):7.4f} {cnn:7.3f}")

shut = [(e, c) for _, s, e, c in rows if s < SHUT_SCLERA and e is not None]
open_ = [(e, c) for _, s, e, c in rows if s > OPEN_SCLERA and e is not None]
print()
ok = True

# 1. CNN on pixel-labelled shut eyes (models.py convention: p_open <= 0.5 is closed)
if shut:
    cn = np.array([c for _, c in shut])
    hit = int((cn <= 0.5).sum())
    print(f"[1] CNN on {len(shut)} pixel-labelled SHUT eyes: "
          f"p_open mean={cn.mean():.3f} max={cn.max():.3f} -> closure on {hit}/{len(shut)}")
    ok &= hit == len(shut)
else:
    print("[1] no shut frames"); ok = False

# 2. EAR separation
if shut and open_:
    eo = np.array([e for e, _ in open_])
    ec = np.array([e for e, _ in shut])
    print(f"[2] EAR: open mean={eo.mean():.4f}  shut mean={ec.mean():.4f}  "
          f"ratio={ec.mean() / eo.mean():.2f}  (old pairing ratio was 0.92 -> undetectable)")
    ok &= ec.mean() < eo.mean() * 0.85

# 3. EarClosure behaviour on measured values
closure = v.EarClosure()
base_open = [e for e, _ in open_] or [0.505, 0.505]
for _ in range(30):
    closure.update(float(np.mean(base_open)))
thr = max(closure.rel_factor * np.percentile(closure._hist, 90), closure.abs_floor)
fired = [closure.update(e) for e, _ in shut]
quiet = [closure.update(e) for e, _ in open_]
n_fire = sum(1 for x in fired if x is True)
n_quiet = sum(1 for x in quiet if x is True)
print(f"[3] EarClosure baseline={np.percentile(closure._hist, 90):.4f} "
      f"threshold={thr:.4f}: shut -> closed on {n_fire}/{len(fired)}, "
      f"open -> closed on {n_quiet}/{len(quiet)} (want 0)")
ok &= n_fire == len(fired) and n_quiet == 0

# 4. CNN-or-EAR on every labelled frame
comb_bad = 0
for _, s, e, c in rows:
    if e is None or np.isnan(c) or s < 0 or (s >= SHUT_SCLERA and s <= OPEN_SCLERA):
        continue
    truth_shut = s < SHUT_SCLERA
    closure.update(e)
    guess = (c <= 0.5) or (closure._hist and
                          e < max(closure.rel_factor * np.percentile(closure._hist, 90),
                                  closure.abs_floor))
    if guess != truth_shut:
        comb_bad += 1
n_labelled = sum(1 for _, s, e, c in rows
                 if e is not None and not np.isnan(c)
                 and (s < SHUT_SCLERA or s > OPEN_SCLERA))
print(f"[4] CNN-or-EAR combined decision: {n_labelled - comb_bad}/{n_labelled} "
      f"labelled frames correct")
ok &= comb_bad == 0 and n_labelled > 0

# 5. Sustained timing
print()
tr = v.EyeBehaviorTracker(fps=15.0)
t1 = None
for i in range(60):
    b = tr.update(0.05, dt=1.0 / 15.0)
    if b["sustained_start"] and t1 is None:
        t1 = (i + 1) / 15.0
print(f"[5a] 4 s closed @15 fps -> sustained_start t={t1:.2f}s (want 2.00)")
ok &= abs(t1 - 2.0) < 0.15

tr = v.EyeBehaviorTracker(fps=15.0)
t2 = None
for i in range(20):
    b = tr.update(0.05, dt=0.2)
    if b["sustained_start"] and t2 is None:
        t2 = (i + 1) * 0.2
print(f"[5b] 4 s closed @5 fps  -> sustained_start t={t2:.2f}s (want 2.00) "
      f"— previously this needed 30 frames = 6 s")
ok &= abs(t2 - 2.0) < 0.25

tr = v.EyeBehaviorTracker(fps=15.0)
for _ in range(4):
    tr.update(0.05, dt=1 / 15)
for _ in range(10):
    tr.update(0.95, dt=1 / 15)
print(f"[5c] 0.27 s blink -> blink_count={tr.blink_count} "
      f"sustained={tr.sustained_now} (want 1 / False)")
ok &= tr.blink_count == 1 and not tr.sustained_now

print("=" * 84)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — inspect above")
print("=" * 84)
sys.exit(0 if ok else 1)
