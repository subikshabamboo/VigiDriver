"""Per-frame dump: pixel label (sclera) vs EAR vs CNN, sorted by label quality.

Used to judge how trustworthy the pixel ground truth is on each captured
frame and whether the CNN disagrees because of a bad crop.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")

import glob
import numpy as np
import cv2

import vision as v
import models as m

svc = m.get_service()
HERE = os.path.dirname(os.path.abspath(__file__))


def sclera_frac(gray_eye):
    if gray_eye is None or gray_eye.size == 0:
        return float("nan")
    border = np.concatenate([gray_eye[0, :], gray_eye[-1, :],
                             gray_eye[:, 0], gray_eye[:, -1]])
    skin = float(np.median(border))
    inner = gray_eye[1:-1, 1:-1] if min(gray_eye.shape) > 4 else gray_eye
    if inner.size == 0:
        return float("nan")
    return float((inner > skin + 22).mean())


def one(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = v.enhance_low_light(img)
    try:
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
        box = (int(min(xs)), int(min(ys)),
               max(1, int(max(xs) - min(xs))), max(1, int(max(ys) - min(ys))))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        exs = [lm[i].x * W for i in (33, 133, 159, 145)]
        eys = [lm[i].y * H for i in (159, 145, 153, 144)]
        x1, x2 = int(max(0, min(exs) - 3)), int(min(W, max(exs) + 3))
        y1, y2 = int(max(0, min(eys) - 3)), int(min(H, max(eys) + 3))
        eb = gray[y1:y2, x1:x2] if x2 - x1 > 6 and y2 - y1 > 4 else None
        sf = sclera_frac(eb)
    except Exception:
        return None

    regions = v.extract_eye_mouth_regions(img, box)
    lm_ok = bool(regions.get("landmarks"))
    ear = regions.get("ear")
    le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
    pl = svc.predict_eye(le) if le is not None else None
    pr = svc.predict_eye(re) if re is not None else None
    vals = [x["p_open"] for x in (pl, pr) if x]
    cnn = float(np.mean(vals)) if vals else float("nan")
    return {"path": path, "sf": sf, "ear": ear, "cnn": cnn, "lm": lm_ok,
            "le_mean": float(le.mean()) if le is not None else float("nan"),
            "re_mean": float(re.mean()) if re is not None else float("nan")}


rows = []
for d in ("cap_pairs", "cap_closed", "cap_open"):
    for p in sorted(glob.glob(os.path.join(HERE, d, "*.png"))):
        r = one(p)
        if r:
            rows.append(r)

rows.sort(key=lambda r: (np.isnan(r["sf"]), -r["sf"]))
print(f"{'source':11s} {'sclera':>7s} {'EAR':>7s} {'CNN':>7s} {'lm':>3s} "
      f"{'Lmean':>6s} {'Rmean':>6s}  file")
for r in rows:
    src = os.path.basename(os.path.dirname(r["path"]))
    print(f"{src:11s} {r['sf']:7.3f} "
          f"{(r['ear'] if r['ear'] is not None else float('nan')):7.4f} "
          f"{r['cnn']:7.3f} {str(r['lm']):>3s} "
          f"{r['le_mean']:6.1f} {r['re_mean']:6.1f}  "
          f"{os.path.basename(r['path'])[:28]}")

print()
strong_open = [r for r in rows if r["sf"] > 0.25]
strong_shut = [r for r in rows if r["sf"] < 0.08]
for nm, grp in (("STRONG OPEN (sclera>0.25)", strong_open),
                ("STRONG CLOSED (sclera<0.08)", strong_shut)):
    if not grp:
        print(f"{nm}: none")
        continue
    ears = np.array([r["ear"] for r in grp if r["ear"] is not None])
    cnns = np.array([r["cnn"] for r in grp])
    print(f"{nm}: n={len(grp)} EAR mean={ears.mean():.4f} "
          f"[{ears.min():.4f},{ears.max():.4f}] | CNN mean={cnns.mean():.3f} "
          f"[{cnns.min():.3f},{cnns.max():.3f}] | CNN<0.5: "
          f"{(cnns < 0.5).sum()}/{len(cnns)}")
