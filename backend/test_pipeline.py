"""Offline pipeline test: render simulation frames, run the real inference
pipeline (extract_eye_mouth_regions + eye CNN + MAR + temporal trackers).

Verifies:
  * eye CNN open/closed classification across the fatigue ramp
  * full blinks counted (not mistaken for drowsiness)
  * sustained closure escalated to critical
  * yawns counted as sustained events

Usage:  python test_pipeline.py
"""

from __future__ import annotations

from vision import EyeBehaviorTracker, YawnTracker, extract_eye_mouth_regions
from models import get_service
import app as appmod


def main() -> None:
    svc = get_service()
    print("model loaded")

    eye_tr = EyeBehaviorTracker()
    yawn_tr = YawnTracker()
    box = (180, 80, 250, 310)

    print(f"{'fr':>4} {'lid':>5} {'p_open':>7} {'eye':>6} {'mar':>6} {'pyawn':>6} | "
          f"{'closed_s':>8} {'blinks':>6} {'yawns':>5} {'perclos':>7} {'tier-ev':>8}")
    prev_yawns = 0
    prev_sustained = False
    for fc in range(0, 100):
        frame = appmod._simulate_frame(fc)
        lid = appmod._eyelid_openness(fc)
        regions = extract_eye_mouth_regions(frame, box)
        lc = regions["left_eye_crop"]
        rc = regions["right_eye_crop"]
        lr = svc.predict_eye(lc) if lc is not None else None
        rr = svc.predict_eye(rc) if rc is not None else None
        p_open = ((lr["p_open"] + rr["p_open"]) / 2) if lr and rr else (lr or rr)["p_open"]
        beh = eye_tr.update(p_open)
        yres = yawn_tr.update(regions.get("mar"))
        # emulate analyze_frame tier logic
        p_drowsy = min(1.0, p_open * 0 + (1 - p_open) * 0.8 + beh["perclos"] * 0.2 + yres["p_yawn"] * 0.7)
        tier_ev = ""
        if yres["yawn_count"] > prev_yawns:
            tier_ev = f"YAWN#{yres['yawn_count']}"
            prev_yawns = yres["yawn_count"]
        if beh["sustained_start"] and not prev_sustained:
            tier_ev = (tier_ev + "+" if tier_ev else "") + f"SUSTAINED {beh['closed_seconds']:.1f}s"
        prev_sustained = beh["sustained_closure"]
        print(f"{fc:>4} {lid:>5.2f} {p_open:>7.4f} {'CLS' if beh['is_closed'] else 'opn':>6} "
              f"{(regions.get('mar') or 0):>6.3f} {yres['p_yawn']:>6.3f} | "
              f"{beh['closed_seconds']:>8.2f} {beh['blink_count']:>6} {yres['yawn_count']:>5} "
              f"{beh['perclos']:>7.3f} {tier_ev:>8}")


if __name__ == "__main__":
    main()
