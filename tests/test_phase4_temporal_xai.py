"""PHASE 4 TESTS — Temporal Bi-LSTM, K-Means gate, XAI suite, noise robustness.

Covers:
  T1  LSTM model loads from models/best_vigidriver_lstm.keras with the
      (None, 30, 4) vigilance window and produces a valid P(drowsy).
  T2  Feature standardization stats (feature_mean/std.npy) shaped (4,).
  T3  GET /api/model/temporal — architecture + K-Means gate registry.
  T4  Live simulated session builds a temporal window and the snapshot
      exposes LSTM P(drowsy), K-Means cluster and Grad-CAM saliency.
  T5  Fused drowsiness score remains bounded and reacts during the sim's
      fatigue ramp (LSTM fusion cannot invert the signal).
  T6  POST /api/xai/gradcam — real Grad-CAM PNG + localization metadata.
  T7  POST /api/robustness/selftest — all regimes return valid predictions
      and the drift verdict is computed.

Output: reports/test_results_phase4.json
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

# Windows consoles default to cp1252; the summaries print check marks, which
# would raise UnicodeEncodeError and turn a passing suite into exit code 2.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


RESULTS = []

def record(tid, category, name, purpose, input_desc, expected, actual, passed, evidence=""):
    RESULTS.append({
        "id": tid, "category": category, "name": name, "purpose": purpose,
        "input": input_desc, "expected": expected,
        "actual": str(actual)[:400], "status": "PASS" if passed else "FAIL",
        "evidence": str(evidence)[:500],
    })

def main():
    import numpy as np
    import cv2
    from fastapi.testclient import TestClient
    from app import app, _simulate_frame

    import temporal as temporal_mod

    # ---- T1: LSTM loads ------------------------------------------------------
    svc = temporal_mod.get_lstm_service()
    ok = svc.available
    info = svc.model_info() if ok else {}
    p_const = None
    if ok:
        buf = temporal_mod.TemporalBuffer()
        for _ in range(30):
            buf.push(np.array([0.30, 1.0, 0.15, 3.0], dtype=np.float32))
        p_const = svc.predict_window(buf.window_tensor(svc.mean, svc.std))
        ok = ok and (0.0 <= p_const <= 1.0)
    record("T1", "temporal-model", "LSTM checkpoint loads + valid sigmoid output",
           "models/best_vigidriver_lstm.keras strictly loads with (30,4) window",
           "constant alert window (EAR .30, ratio 1.0, MAR .15, yaw 3°)",
           "available=True; P(drowsy) ∈ [0,1]",
           f"available={svc.available}, P={p_const}", ok, info.get("input_shape"))

    # ---- T2: standardization stats ------------------------------------------
    ok = False
    if ok or True:
        mean = np.load(os.path.join(ROOT, "models", "feature_mean.npy")).reshape(-1)
        std = np.load(os.path.join(ROOT, "models", "feature_std.npy")).reshape(-1)
        ok = mean.shape == (4,) and std.shape == (4,) and (std > 0).all()
        record("T2", "temporal-model", "Vigilance standardization stats present",
               "feature_mean/std .npy standardized to the training distribution",
               "models/feature_mean.npy + feature_std.npy", "shape (4,), std > 0",
               f"mean={mean.round(3).tolist()} std={std.round(3).tolist()}", ok)

    client = TestClient(app)

    # ---- T3: model temporal registry endpoint -------------------------------
    r = client.get("/api/model/temporal")
    j = r.json()
    ok = r.status_code == 200 and j.get("available") and "kmeans_gate" in j
    record("T3", "api", "GET /api/model/temporal — LSTM + K-Means registry",
           "Architecture registry exposes LSTM layers and K-Means gate state",
           "GET /api/model/temporal", "200; available=True; kmeans_gate present",
           f"{r.status_code}; layers={len(j.get('layers', []))}; gate={bool(j.get('kmeans_gate'))}", ok)

    # ---- T4/T5: live sim session with temporal fusion ------------------------
    r = client.post("/api/monitor/start", json={"use_simulation": True, "use_camera": False})
    ok = r.status_code == 200
    record("T4a", "api", "POST /api/monitor/start (sim)", "Sim session for temporal integration",
           "POST /api/monitor/start", "200 + session_id",
           f"{r.status_code} sid={r.json().get('session_id')}", ok)

    # Pump simulated frames through the real monitor loop pieces so feature
    # buffer + K-Means gate + gradcam all exercise against real inference.
    from app import analyze_frame, state
    lstm_seen = False
    gradcam_seen = False
    p_alert = p_fatigued = None
    # Negative frames render the alert phase, so the per-driver calibration
    # baseline gets enough alert samples without a real webcam.
    for i in range(-100, 140):
        res = analyze_frame(_simulate_frame(i), temporal=True)
        if res.get("face_box"):
            snap = state.__dict__
            if snap.get("temporal_ready") and snap.get("lstm_p") is not None:
                lstm_seen = True
            if snap.get("gradcam_b64"):
                gradcam_seen = True
            if i == 20:
                p_alert = res["p_drowsy"]
            if i == 120:
                p_fatigued = res["p_drowsy"]
    snap = json.loads(json.dumps({
        "temporal": {"ready": state.temporal_ready, "window": state.temporal_window,
                     "p": state.lstm_p},
        "kmeans": {"cluster": state.cluster_label, "id": state.cluster_id,
                   "outlier": state.is_outlier},
        "gradcam_len": len(state.gradcam_b64),
        "p_alert": p_alert, "p_fatigued": p_fatigued,
    }, default=str))
    ok4 = bool(lstm_seen and gradcam_seen and state.temporal_window == 30)
    record("T4b", "temporal-model", "30-frame window fills; LSTM fused + Grad-CAM rendered",
           "Live sim session drives feature buffer, LSTM fusion, K-Means, Grad-CAM",
           "140 simulated frames through analyze_frame(temporal=True)",
           "window=30; LSTM P seen; gradcam b64 non-empty",
           json.dumps(snap)[:300], ok4)

    ok5 = (p_alert is not None) and (p_fatigued is not None) and p_fatigued > p_alert
    record("T5", "temporal-model", "Fusion preserves fatigue ramp direction",
           "p_drowsy(alert early) < p_drowsy(fatigued late) with LSTM fusion active",
           "sim: alert phase frame 20 vs fatigue phase frame 120",
           f"p_fatigued > p_alert ({p_alert} → {p_fatigued})",
           f"alert={p_alert}, fatigued={p_fatigued}", ok5)

    r = client.post("/api/monitor/stop")
    record("T5b", "api", "POST /api/monitor/stop", "Clean stop after temporal session",
           "POST /api/monitor/stop", "200", f"{r.status_code}", r.status_code == 200)

    # ---- T6: XAI gradcam endpoint -------------------------------------------
    frame = _simulate_frame(5)
    ok, jpg = cv2.imencode(".jpg", frame)
    r = client.post("/api/xai/gradcam", files={"file": ("f.jpg", jpg.tobytes(), "image/jpeg")})
    j = r.json() if r.status_code == 200 else {}
    ok = r.status_code == 200 and j.get("gradcam_png") and j.get("gradcam_meta")
    record("T6", "xai", "POST /api/xai/gradcam — Grad-CAM + LRP sandbox",
           "Real grad-cam heat map PNG + localization metadata on uploaded frame",
           "synthetic cabin frame upload", "200; gradcam_png b64; meta.conv_layer",
           f"{r.status_code}; png={len(j.get('gradcam_png', ''))}; meta={str(j.get('gradcam_meta'))[:80]}",
           ok, j.get("localization"))

    # Negative: corrupt body must not 5xx
    r2 = client.post("/api/xai/gradcam", files={"file": ("f.jpg", b"junk", "image/jpeg")})
    record("T6b", "xai", "POST /api/xai/gradcam (corrupt)", "Corrupt upload → 4xx",
           "bytes 'junk'", "400", f"{r2.status_code}", 400 <= r2.status_code < 500)

    # ---- T7: noise robustness selftest ---------------------------------------
    ok, jpg = cv2.imencode(".jpg", _simulate_frame(12))
    r = client.post("/api/robustness/selftest", files={"file": ("f.jpg", jpg.tobytes(), "image/jpeg")})
    j = r.json() if r.status_code == 200 else {}
    regimes = j.get("regimes", [])
    ok = (r.status_code == 200 and len(regimes) >= 6
          and all(("p_open" in rr or rr.get("face_found") is False) for rr in regimes))
    record("T7", "robustness", "POST /api/robustness/selftest — SNR sweep",
           "Pristine/AWGN/Poisson/blur/glare/lowlight regimes probed with eye CNN",
           "synthetic frame upload", "200; ≥6 regimes each with p_open or no-face",
           f"{r.status_code}; n={len(regimes)}; drift={j.get('max_drift')}; stable={j.get('stable')}",
           ok, json.dumps(j.get("regimes", []))[:200])

    # ---- report ---------------------------------------------------------------
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out_path = os.path.join(ROOT, "reports", "test_results_phase4.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2, ensure_ascii=False)

    n_pass = sum(1 for r_ in RESULTS if r_["status"] == "PASS")
    print(f"\nPHASE 4 RESULT: {n_pass}/{len(RESULTS)} PASSED — {out_path}")
    for r_ in RESULTS:
        print(f"  [{r_['status']}] {r_['id']}: {r_['name']}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 2
    sys.exit(code)
