"""Capture labelled OPEN and CLOSED eye frames + per-frame sensor readout.

Phases (wall clock):
    0 -  6 s  EYES OPEN (calibration)
    6 - 18 s  CLOSE YOUR EYES AND KEEP THEM SHUT
   18 - 24 s  OPEN AGAIN

Full-resolution frames are saved to backend/cap_open/ and backend/cap_closed/
so later fix iterations can be tested offline against real ground truth.

Ground truth comes from pixels, not from a model: inside the eye box we count
"sclera" pixels (clearly brighter than the surrounding skin). An open eye has
them; a shut eye does not. That gives an independent label per frame.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import sys
sys.path.insert(0, ".")

import time
import numpy as np
import cv2

import vision as v
import models as m

svc = m.get_service()
closure = v.EarClosure()

HERE = os.path.dirname(os.path.abspath(__file__))
DIR_OPEN = os.path.join(HERE, "cap_open")
DIR_CLOSED = os.path.join(HERE, "cap_closed")
os.makedirs(DIR_OPEN, exist_ok=True)
os.makedirs(DIR_CLOSED, exist_ok=True)

PHASES = [(0.0, 6.0, "OPEN"), (6.0, 18.0, "CLOSED"), (18.0, 24.0, "OPEN")]
SAVED = {"OPEN": 0, "CLOSED": 0}
MAX_SAVE = 18


def phase_of(t):
    for a, b, n in PHASES:
        if a <= t < b:
            return n
    return "DONE"


def open_camera():
    for backend, tag in ((cv2.CAP_DSHOW, "DSHOW"), (None, "default")):
        cap = cv2.VideoCapture(0) if backend is None else cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"camera opened via {tag}", flush=True)
                return cap
            cap.release()
    return None


def eye_box_pixels(frame, lm, H, W, idxs):
    xs = [lm[i].x * W for i in idxs]
    ys = [lm[i].y * H for i in idxs]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 3:
        return None
    return frame[y1:y2, x1:x2]


def sclera_frac(gray_eye):
    """Fraction of clearly-brighter-than-skin pixels inside the eye box.

    Open eye -> sclera is brighter than surrounding skin (non-zero).
    Shut eye -> only skin and a dark lash line (near zero).
    """
    if gray_eye is None or gray_eye.size == 0:
        return float("nan")
    border = np.concatenate([gray_eye[0, :], gray_eye[-1, :],
                             gray_eye[:, 0], gray_eye[:, -1]])
    skin = float(np.median(border))
    inner = gray_eye[1:-1, 1:-1] if min(gray_eye.shape) > 4 else gray_eye
    if inner.size == 0:
        return float("nan")
    return float((inner > skin + 22).mean())


def ascii_art(img, cols=46, rows=13):
    g = cv2.resize(img, (cols, rows), interpolation=cv2.INTER_AREA).astype(np.float32)
    lo, hi = float(g.min()), float(g.max())
    rng = (hi - lo) or 1.0
    ramp = " .:-=+*#%@"
    return ["".join(ramp[min(9, int((1.0 - (g[r, c] - lo) / rng) * 9.99))]
                    for c in range(cols)) for r in range(rows)]


def main():
    cap = open_camera()
    if cap is None:
        print("CAMERA UNAVAILABLE")
        sys.exit(1)
    print("=" * 96, flush=True)
    print("0-6s EYES OPEN | 6-18s CLOSE YOUR EYES AND KEEP THEM SHUT | 18-24s OPEN", flush=True)
    print("=" * 96, flush=True)

    t0 = time.time()
    last_phase = None
    last_ascii_t = -9
    rows = []
    while True:
        t = time.time() - t0
        ph = phase_of(t)
        if ph != last_phase:
            print(f"\n--- t={t:5.1f}s  >>> {ph} <<<", flush=True)
            last_phase = ph
        if ph == "DONE":
            break
        ok, fr = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        bright = float(fr.mean())
        fr2 = v.enhance_low_light(fr)
        box = v.detect_face(fr2)
        if box is None:
            print(f"t={t:5.1f} {ph:6s} bright={bright:5.1f} NO FACE", flush=True)
            time.sleep(0.03)
            continue

        regions = v.extract_eye_mouth_regions(fr2, box)
        lm_ok = bool(regions.get("landmarks"))
        ear = regions.get("ear")
        state = closure.update(ear)

        le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
        pl = svc.predict_eye(le) if le is not None else None
        pr = svc.predict_eye(re) if re is not None else None
        vals = [r["p_open"] for r in (pl, pr) if r]
        cnn = float(np.mean(vals)) if vals else float("nan")

        # Independent pixel ground truth
        sf = float("nan")
        apw = float("nan")
        if lm_ok:
            try:
                landmarker = v.get_landmarker()
                import mediapipe as mp
                im = mp.Image(image_format=mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(fr2[:, :, ::-1]))
                plm = landmarker.detect(im).face_landmarks[0]
                H, W = fr2.shape[:2]
                gray = cv2.cvtColor(fr2, cv2.COLOR_BGR2GRAY)
                eb = eye_box_pixels(gray, plm, H, W, [33, 160, 158, 133, 153, 144])
                if eb is not None:
                    sf = sclera_frac(eb)
                ap = float(np.hypot((plm[159].x - plm[145].x) * W,
                                     (plm[159].y - plm[145].y) * H))
                wl = float(np.hypot((plm[33].x - plm[133].x) * W,
                                    (plm[33].y - plm[133].y) * H))
                apw = ap / wl if wl else float("nan")
            except Exception:
                pass

        rows.append({"t": t, "phase": ph, "ear": ear, "state": state,
                     "cnn": cnn, "sf": sf, "apw": apw})

        # Save labelled ground-truth frames
        if SAVED[ph] < MAX_SAVE and not np.isnan(sf):
            dst = DIR_OPEN if ph == "OPEN" else DIR_CLOSED
            idx = SAVED[ph]
            cv2.imwrite(os.path.join(dst, f"f{idx:02d}_sf{sf:.3f}.png"), fr2)
            SAVED[ph] += 1

        print(f"t={t:5.1f} {ph:6s} ear={('None' if ear is None else round(ear,4))} "
              f"state={state} cnn_open={cnn:.3f} sclera={sf:.3f} ap/w={apw:.3f} "
              f"box={box}", flush=True)

        if t - last_ascii_t >= 2.0 and le is not None:
            last_ascii_t = t
            print(f"   LEFT EYE CROP @t={t:.1f}s ({ph})", flush=True)
            for ln in ascii_art(le):
                print("    |" + ln + "|", flush=True)
        time.sleep(0.03)
    cap.release()

    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)
    for pname in ("OPEN", "CLOSED"):
        sel = [r for r in rows if r["phase"] == pname]
        if not sel:
            print(f"  {pname}: no samples")
            continue
        ears = np.array([r["ear"] for r in sel if r["ear"] is not None])
        cnns = np.array([r["cnn"] for r in sel if not np.isnan(r["cnn"])])
        sfs = np.array([r["sf"] for r in sel if not np.isnan(r["sf"])])
        aps = np.array([r["apw"] for r in sel if not np.isnan(r["apw"])])
        states = sorted({str(r["state"]) for r in sel})
        print(f"  {pname:6s} n={len(sel):3d} | EAR mean={ears.mean():.4f} "
              f"min={ears.min():.4f} | CNN mean={cnns.mean():.3f} min={cnns.min():.3f} | "
              f"sclera mean={sfs.mean():.3f} | ap/w mean={aps.mean():.3f} | states={states}")
    all_ears = [r["ear"] for r in rows if r["ear"] is not None]
    if all_ears:
        base = float(np.percentile(all_ears, 90))
        print(f"\n  baseline(90pct)={base:.4f}  threshold=max(0.6*base,0.12)="
              f"{max(0.6 * base, 0.12):.4f}")
        cl = [r['ear'] for r in rows if r['phase'] == 'CLOSED' and r['ear'] is not None]
        if cl:
            print(f"  CLOSED min EAR={min(cl):.4f} -> "
                  f"{'WOULD trigger' if min(cl) < max(0.6 * base, 0.12) else 'WOULD NOT trigger'}")
    print(f"  saved: {SAVED['OPEN']} open / {SAVED['CLOSED']} closed frames")
    print("=" * 96)


if __name__ == "__main__":
    main()
