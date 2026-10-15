"""Controlled open/closed eye protocol against the REAL production pipeline.

Phases (wall clock from start):
    0.0 -  5.0 s  -> KEEP EYES OPEN   (calibrates the EAR open-eye baseline)
    5.0 - 13.0 s  -> CLOSE YOUR EYES  (8 s, well past the 2 s alarm rule)
   13.0 - 17.0 s  -> OPEN AGAIN

For every frame it records the code's EAR, the landmark-pair decomposition
(diagonal vs vertical components), the true eyelid aperture ratio, the raw
(unclamped) CNN probabilities, and what EarClosure concludes.

Run this while the monitor is STOPPED so the camera is free.
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

PHASES = [(0.0, 5.0, "OPEN"), (5.0, 13.0, "CLOSED"), (13.0, 17.0, "OPEN")]


def phase_of(t):
    for a, b, name in PHASES:
        if a <= t < b:
            return name
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


def measure(frame, box, tag):
    """Full geometry + CNN readout for one frame."""
    regions = v.extract_eye_mouth_regions(frame, box)
    out = {"lm": bool(regions.get("landmarks")), "ear": regions.get("ear"),
           "cnn_open": float("nan"), "apw": float("nan"), "dx_floor": float("nan")}
    le, re = regions.get("left_eye_crop"), regions.get("right_eye_crop")
    pl = svc.predict_eye(le) if le is not None else None
    pr = svc.predict_eye(re) if re is not None else None
    vals = [r["p_open"] for r in (pl, pr) if r]
    if vals:
        out["cnn_open"] = float(np.mean(vals))
        out["cnn_closed"] = float(np.mean([r["p_closed"] for r in (pl, pr) if r]))

    if out["lm"]:
        try:
            landmarker = v.get_landmarker()
            import mediapipe as mp
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=np.ascontiguousarray(frame[:, :, ::-1]))
            lm = landmarker.detect(img).face_landmarks[0]
            H, W = frame.shape[:2]

            def raw(i, j):
                dx = (lm[i].x - lm[j].x) * W
                dy = (lm[i].y - lm[j].y) * H
                return abs(dx), abs(dy)

            dx1, dy1 = raw(160, 153)   # code's left pair 1
            dx2, dy2 = raw(158, 144)   # code's left pair 2
            wl = float(np.hypot((lm[33].x - lm[133].x) * W,
                                (lm[33].y - lm[133].y) * H))
            # true vertical lid separation (same points _eyelid uses)
            ap_l = float(np.hypot((lm[159].x - lm[145].x) * W,
                                  (lm[159].y - lm[145].y) * H))
            rdx1, rdy1 = raw(385, 373)
            rdx2, rdy2 = raw(387, 380)
            wr = float(np.hypot((lm[362].x - lm[263].x) * W,
                                (lm[362].y - lm[263].y) * H))
            ap_r = float(np.hypot((lm[386].x - lm[374].x) * W,
                                  (lm[386].y - lm[374].y) * H))

            diag = (np.hypot(dx1, dy1) + np.hypot(dx2, dy2)) / (2 * wl)
            # EAR if the eyelids were fully closed (dy -> 0, dx unchanged)
            floor = (dx1 + dx2) / (2 * wl)
            out["diag_l"] = diag
            out["floor_l"] = floor
            out["apw"] = ((ap_l / wl) + (ap_r / wr)) / 2.0
            out["dx"] = (dx1 + dx2) / 2.0
            out["dy"] = (dy1 + dy2) / 2.0
            out["w"] = wl
            out["ap"] = ap_l
        except Exception as exc:
            out["geo_err"] = str(exc)
    return out


def main():
    cap = open_camera()
    if cap is None:
        print("CAMERA UNAVAILABLE")
        sys.exit(1)

    print("=" * 100, flush=True)
    print("PHASES:  0-5s EYES OPEN | 5-13s CLOSE YOUR EYES | 13-17s OPEN AGAIN", flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()
    last_phase = None
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
            print("frame read failed")
            time.sleep(0.05)
            continue
        bright = float(fr.mean())
        fr2 = v.enhance_low_light(fr)
        box = v.detect_face(fr2)
        if box is None:
            print(f"t={t:5.1f} {ph:6s} bright={bright:5.1f} NO FACE", flush=True)
            time.sleep(0.03)
            continue
        r = measure(fr2, box, ph)
        state = closure.update(r["ear"])
        r["ear_state"] = state
        r["t"] = t
        r["phase"] = ph
        rows.append(r)
        print(f"t={t:5.1f} {ph:6s} ear={r['ear'] if r['ear'] is None else round(r['ear'],4)} "
              f"state={state} floorL={r.get('floor_l', float('nan')):.3f} "
              f"ap/w={r.get('apw', float('nan')):.3f} "
              f"cnn_open={r['cnn_open']:.3f} "
              f"box={box}", flush=True)
        time.sleep(0.03)
    cap.release()

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for pname in ("OPEN", "CLOSED"):
        sel = [r for r in rows if r["phase"] == pname and r.get("ear") is not None]
        if not sel:
            print(f"  {pname}: no samples")
            continue
        ears = np.array([r["ear"] for r in sel])
        cnn = np.array([r["cnn_open"] for r in sel if not np.isnan(r["cnn_open"])])
        floors = np.array([r.get("floor_l", np.nan) for r in sel])
        aps = np.array([r.get("apw", np.nan) for r in sel])
        states = [str(r["ear_state"]) for r in sel]
        print(f"  {pname:6s} n={len(sel):3d} EAR mean={ears.mean():.4f} "
              f"min={ears.min():.4f} max={ears.max():.4f} | "
              f"floorL mean={np.nanmean(floors):.4f} | ap/w mean={np.nanmean(aps):.4f} | "
              f"CNN p_open mean={cnn.mean():.3f} min={cnn.min():.3f} | "
              f"ear_states={sorted(set(states))}")
    base = np.percentile([r["ear"] for r in rows if r.get("ear") is not None], 90) \
        if rows else 0.0
    print(f"\n  open-eye baseline (90th pct over run) = {base:.4f}")
    print(f"  EarClosure threshold = max(0.60*base, 0.12) = {max(0.60 * base, 0.12):.4f}")
    closed = [r for r in rows if r["phase"] == "CLOSED" and r.get("ear") is not None]
    if closed:
        ce = np.array([r["ear"] for r in closed])
        fl = np.array([r.get("floor_l", np.nan) for r in closed])
        print(f"  CLOSED-phase EAR: mean={ce.mean():.4f} min={ce.min():.4f}")
        print(f"  CLOSED-phase geometric floor (dy=0) mean={np.nanmean(fl):.4f}")
        print(f"  => fraction of closed frames below threshold: "
              f"{(ce < max(0.60 * base, 0.12)).mean():.2%}")
    print("=" * 100)


if __name__ == "__main__":
    main()
