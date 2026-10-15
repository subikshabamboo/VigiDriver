"""Find MediaPipe landmark pairs that actually collapse when the eye shuts.

Captures fresh webcam frames, labels each one from PIXELS (sclera fraction,
not from any model), then decomposes candidate upper/lower lid pairs into
their horizontal (dx) and vertical (dy) components normalized by eye width.

A usable EAR pair has:
  * dx/w ~ 0          (truly vertical -> distance collapses when shut)
  * dy/w large when open, ~0 when closed  (discriminates)

Also re-analyses the previously saved cap_closed/*.png frames.
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
CAP_DIR = os.path.join(HERE, "cap_pairs")
os.makedirs(CAP_DIR, exist_ok=True)

LEFT_UP = [157, 158, 159, 160, 161, 163]
LEFT_DN = [144, 145, 153, 154, 155, 7]
RIGHT_UP = [384, 385, 386, 387, 388, 466]
RIGHT_DN = [373, 374, 375, 376, 377, 380]

CURRENT_LEFT = [(160, 153), (158, 144)]
CURRENT_RIGHT = [(385, 373), (387, 380)]


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


def analyse(img, tag):
    """Return dict of label + per-pair normalized dx/dy for both eyes."""
    lmres = None
    try:
        landmarker = v.get_landmarker()
        import mediapipe as mp
        im = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=np.ascontiguousarray(img[:, :, ::-1]))
        r = landmarker.detect(im)
        if r.face_landmarks:
            lmres = r.face_landmarks[0]
    except Exception:
        return None
    if lmres is None:
        return None

    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def px(i):
        return lmres[i].x * W, lmres[i].y * H

    def pair(u, d):
        ux, uy = px(u)
        dx_, dy_ = px(d)
        return abs(ux - dx_), abs(uy - dy_)

    # ground-truth label from pixels around the left eye
    xs = [px(i)[0] for i in [33, 133, 159, 160]]
    ys = [px(i)[1] for i in [159, 145, 153, 144]]
    x1, x2 = int(max(0, min(xs) - 4)), int(min(W, max(xs) + 4))
    y1, y2 = int(max(0, min(ys) - 4)), int(min(H, max(ys) + 4))
    eye_box = gray[y1:y2, x1:x2] if x2 - x1 > 6 and y2 - y1 > 4 else None
    sf = sclera_frac(eye_box)

    lw = float(np.hypot(px(33)[0] - px(133)[0], px(33)[1] - px(133)[1]))
    rw = float(np.hypot(px(362)[0] - px(263)[0], px(362)[1] - px(263)[1]))
    if lw <= 0 or rw <= 0:
        return None

    out = {"tag": tag, "sf": sf, "lw": lw, "rw": rw, "pairs": {}}
    for side, ups, dns, width in (("L", LEFT_UP, LEFT_DN, lw),
                                  ("R", RIGHT_UP, RIGHT_DN, rw)):
        for u in ups:
            for d in dns:
                dx_, dy_ = pair(u, d)
                out["pairs"][(side, u, d)] = (dx_ / width, dy_ / width)

    # CNN readout for cross-reference
    le = None
    try:
        regions = v.extract_eye_mouth_regions(img, _box_from_lm(lmres, W, H))
        le = regions.get("left_eye_crop")
    except Exception:
        pass
    if le is not None:
        r = svc.predict_eye(le)
        out["cnn_open"] = r["p_open"]
    else:
        out["cnn_open"] = float("nan")
    return out


def _box_from_lm(lm, W, H):
    xs = [p.x * W for p in lm]
    ys = [p.y * H for p in lm]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def capture(n=45):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("CAMERA UNAVAILABLE")
        return []
    print(f"capturing {n} frames (no timing needed; labels come from pixels)...",
          flush=True)
    saved = []
    for i in range(n):
        ok, fr = cap.read()
        if not ok:
            continue
        fr = v.enhance_low_light(fr)
        p = os.path.join(CAP_DIR, f"c{i:03d}.png")
        cv2.imwrite(p, fr)
        saved.append(p)
    cap.release()
    print(f"  saved {len(saved)} frames")
    return saved


def main():
    frames = []
    for p in sorted(glob.glob(os.path.join(HERE, "cap_closed", "*.png"))):
        frames.append((p, "saved-closed"))
    frames += [(p, "fresh") for p in capture()]

    results = []
    for p, tag in frames:
        img = cv2.imread(p)
        if img is None:
            continue
        r = analyse(img, tag)
        if r is not None:
            results.append(r)

    open_r = [r for r in results if not np.isnan(r["sf"]) and r["sf"] > 0.18]
    shut_r = [r for r in results if not np.isnan(r["sf"]) and r["sf"] < 0.12]
    print(f"\nlabelled frames: OPEN={len(open_r)}  CLOSED={len(shut_r)}  "
          f"unlabelled={len(results) - len(open_r) - len(shut_r)}")

    print("\ncurrent code's pairs (LEFT):")
    for pr in CURRENT_LEFT:
        for nm, grp in (("OPEN", open_r), ("CLOSED", shut_r)):
            if not grp:
                continue
            vals = [r["pairs"][("L", pr[0], pr[1])] for r in grp if ("L", *pr) in r["pairs"]]
            if vals:
                a = np.array(vals)
                print(f"  {str(pr):14s} {nm:6s} dx/w={a[:, 0].mean():.3f} "
                      f"dy/w={a[:, 1].mean():.3f} "
                      f"dist/w={np.hypot(a[:, 0], a[:, 1]).mean():.3f}")

    print("\ncandidate pair screen (need dx/w small + dy/w separates):")
    print(f"  {'pair':16s} {'dx/w':>7s} {'dy OPEN':>9s} {'dy CLOSED':>10s} {'sep':>7s}")
    allkeys = set()
    for r in results:
        allkeys |= set(r["pairs"].keys())
    rows = []
    for k in sorted(allkeys):
        o = np.array([r["pairs"][k] for r in open_r if k in r["pairs"]])
        c = np.array([r["pairs"][k] for r in shut_r if k in r["pairs"]])
        if len(o) < 3 or len(c) < 3:
            continue
        dx = float(o[:, 0].mean())
        dyo, dyc = float(o[:, 1].mean()), float(c[:, 1].mean())
        sep = dyo - dyc
        rows.append((k, dx, dyo, dyc, sep))
    rows.sort(key=lambda t: (abs(t[1]), -t[4]))
    for k, dx, dyo, dyc, sep in rows[:14]:
        flag = " <== GOOD" if abs(dx) < 0.06 and sep > 0.10 else ""
        print(f"  {str(k):16s} {dx:7.3f} {dyo:9.3f} {dyc:10.3f} {sep:7.3f}{flag}")

    print("\nCNN cross-check:")
    for nm, grp in (("OPEN", open_r), ("CLOSED", shut_r)):
        if grp:
            c = np.array([r["cnn_open"] for r in grp])
            print(f"  {nm:6s} n={len(c)} CNN p_open mean={c.mean():.3f} "
                  f"min={c.min():.3f} max={c.max():.3f}")


if __name__ == "__main__":
    main()
