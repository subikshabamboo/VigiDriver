"""Run the eye pipeline on saved still frames and print ASCII of the eye crop.

ASCII lets an agent (or a human reading a terminal) see whether the crop
actually contains an open or closed eye, independent of what any model says.
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

IMAGES = [
    os.path.join(os.path.dirname(__file__), "diag_face.png"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "reports", "smoke_cam_frame.jpg"),
]


def ascii_art(img, cols=56, rows=22):
    """Render a grayscale crop as ASCII (dark = dense glyph)."""
    g = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    rng = (hi - lo) or 1.0
    ramp = " .:-=+*#%@"
    lines = []
    for r in range(rows):
        line = "".join(ramp[min(9, int((1.0 - (g[r, c] - lo) / rng) * 9.99))]
                       for c in range(cols))
        lines.append(line)
    return lines, (lo, hi)


def report(path):
    img = cv2.imread(path)
    if img is None:
        print(f"MISSING {path}")
        return
    print("=" * 100)
    print(f"{path}  shape={img.shape} mean={img.mean():.1f}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = v.detect_face(img)
    print(f"  haar face box: {box}")
    if box is None:
        return
    regions = v.extract_eye_mouth_regions(img, box)
    lm = regions.get("landmarks")
    ear = regions.get("ear")
    print(f"  landmarks={lm} EAR={ear} MAR={regions.get('mar')}")

    if lm:
        try:
            landmarker = v.get_landmarker()
            import mediapipe as mp
            im = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(img[:, :, ::-1]))
            plm = landmarker.detect(im).face_landmarks[0]
            H, W = img.shape[:2]

            def d(i, j):
                dx = (plm[i].x - plm[j].x) * W
                dy = (plm[i].y - plm[j].y) * H
                return abs(dx), abs(dy)
            dx1, dy1 = d(160, 153)
            dx2, dy2 = d(158, 144)
            dwl = d(33, 133)
            wl = float(np.hypot(dwl[0], dwl[1]))
            ap = d(159, 145)
            apw = float(np.hypot(ap[0], ap[1])) / wl
            print(f"  left pair dx=({dx1:.1f},{dx2:.1f}) dy=({dy1:.1f},{dy2:.1f}) "
                  f"width={wl:.1f} -> diagEAR="
                  f"{(np.hypot(dx1, dy1) + np.hypot(dx2, dy2)) / (2 * wl):.4f} "
                  f"floor(dy=0)={(dx1 + dx2) / (2 * wl):.4f} aperture/w={apw:.4f}")
        except Exception as exc:
            print("  geometry failed:", exc)

    for tag, key in (("LEFT", "left_eye_crop"), ("RIGHT", "right_eye_crop")):
        crop = regions.get(key)
        if crop is None:
            print(f"  {tag}: no crop")
            continue
        r = svc.predict_eye(crop)
        lines, (lo, hi) = ascii_art(crop)
        print(f"\n  {tag} eye crop  64x64 mean={crop.mean():.1f} std={crop.std():.1f} "
              f"range=[{lo:.0f},{hi:.0f}] CNN p_open={r['p_open']:.4f}")
        for ln in lines:
            print("    |" + ln + "|")
        print()


for p in IMAGES:
    report(p)
