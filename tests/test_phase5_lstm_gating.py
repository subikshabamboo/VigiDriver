"""PHASE 5 TESTS — LSTM self-gating (calibration x support x CNN agreement).

Verifies the reliability gate that scales the Bi-LSTM's contribution to the
fused drowsiness score. The defect it guards against is a real observed one:
the LSTM returning P(drowsy) ~= 0.9996 on a frame whose geometric eye signal
reported "wide open", which at a fixed 30 % fusion weight produced a false
fatigue alert.

Output: reports/test_results_phase5.json
"""
from __future__ import annotations

import json
import os
import sys
import traceback

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

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


def record(tid, name, purpose, expected, actual, passed, evidence=""):
    RESULTS.append({
        "id": tid, "category": "lstm-gating", "name": name, "purpose": purpose,
        "expected": expected, "actual": str(actual)[:400],
        "status": "PASS" if passed else "FAIL", "evidence": str(evidence)[:500],
    })


class _StubSvc:
    """Minimal LSTMSvc stand-in that returns a fixed raw P(drowsy)."""

    def __init__(self, raw: float):
        self._raw = float(raw)
        self.available = True
        self.mean = np.zeros((1, 1, 4), dtype=np.float32)
        self.std = np.ones((1, 1, 4), dtype=np.float32)
        self._model = None

    def predict_window(self, z) -> float:
        return self._raw


ALERT_FEATURES = [
    np.array([0.30, 1.0, 0.10, 3.70], dtype=np.float32) for _ in range(30)
]


def _buffer(rows=None):
    import temporal as T
    b = T.TemporalBuffer()
    for r in (rows or ALERT_FEATURES):
        b.push(r)
    return b


def _extractor(calibrated: bool):
    import temporal as T
    fe = T.VigilanceFeatureExtractor()
    fe.calibrated = calibrated
    return fe


def main():
    import temporal as T

    # ---- G1: CNN agreement curve -------------------------------------------
    a_free = T._cnn_agreement(0.90, 0.95)          # gap 0.05, inside tolerance
    a_mid = T._cnn_agreement(0.70, 0.30)           # gap 0.40, just past tol
    a_none = T._cnn_agreement(0.99, None)          # no CNN signal -> no penalty
    a_full = T._cnn_agreement(1.0, 0.0)            # full-scale disagreement
    ok1 = (a_free == 1.0 and a_none == 1.0
           and 0.0 < a_mid < 1.0 and a_full == 0.0)
    record("G1", "CNN agreement term is monotone and reaches 0 at full disagreement",
           "Agreement must be 1 inside tolerance, decay past it, hit 0 at gap=1, "
           "and stay 1 when no CNN signal is available",
           "a(0.90,0.95)=1; a(0.99,None)=1; a(1.0,0.0)=0",
           f"free={a_free} mid={a_mid} none={a_none} full={a_full}", ok1)

    # ---- G2: manifold support ---------------------------------------------
    z_in = np.zeros((1, 30, 4), dtype=np.float32)
    z_out = np.full((1, 30, 4), 9.0, dtype=np.float32)
    s_in, s_out = T._manifold_support(z_in), T._manifold_support(z_out)
    z_half = np.full((1, 30, 4), 9.0, dtype=np.float32)
    z_half[:, :15, :] = 0.0
    s_half = T._manifold_support(z_half)
    ok2 = (s_in == 1.0 and s_out == 0.0 and abs(s_half - 0.5) < 1e-6)
    record("G2", "Manifold support measures in-distribution fraction",
           "Fraction of |z| <= 3 sigma; OOD windows must score 0",
           "all-in=1.0; all-out=0.0; half=0.5",
           f"in={s_in} out={s_out} half={s_half}", ok2)

    # ---- G3: calibration is a hard gate ------------------------------------
    rel = T.gate_for_fusion(
        svc=_StubSvc(0.99), feature_extractor=_extractor(False),
        temporal_buffer=_buffer(), p_cnn=0.0,
    )
    ok3 = rel.alpha_lstm == 0.0 and rel.separable is False and rel.gated_p_drowsy == 0.0
    record("G3", "Uncalibrated driver neutralises the temporal stream",
           "Before the per-driver baseline exists the LSTM must not enter fusion",
           "alpha=0; separable=False; gated=0.0",
           f"alpha={rel.alpha_lstm} sep={rel.separable} gated={rel.gated_p_drowsy}", ok3)

    # ---- G4: the observed false-drowsy defect is suppressed ----------------
    rel_bad = T.gate_for_fusion(
        svc=_StubSvc(0.99), feature_extractor=_extractor(True),
        temporal_buffer=_buffer(), p_cnn=0.0,     # CNN: eyes wide OPEN
    )
    geo = 0.0
    fused_ungated = 0.70 * geo + 0.30 * 0.99
    fused_gated = 0.70 * geo + 0.30 * rel_bad.gated_p_drowsy
    ok4 = (rel_bad.raw_p_drowsy == 0.99 and rel_bad.alpha_lstm < 0.05
           and fused_gated < 0.05 and fused_gated < fused_ungated / 10.0)
    record("G4", "LSTM 0.99 vs CNN 'eyes open' no longer injects a false alert",
           "Reproduces the live defect (raw P(drowsy)=0.99 while p_closed=0.0) and "
           "confirms the gated fusion collapses back onto the geometric score",
           "alpha<0.05; fused_gated < 0.05 and < fused_ungated/10",
           f"raw={rel_bad.raw_p_drowsy} alpha={rel_bad.alpha_lstm:.4f} "
           f"fused {fused_ungated:.3f} -> {fused_gated:.4f} "
           f"agreement={rel_bad.agreement:.4f} support={rel_bad.support:.3f}", ok4)

    # ---- G5: a genuinely corroborated drowsy verdict still fuses ----------
    rel_good = T.gate_for_fusion(
        svc=_StubSvc(0.90), feature_extractor=_extractor(True),
        temporal_buffer=_buffer(), p_cnn=0.95,    # CNN: eyes CLOSED
    )
    ok5 = (rel_good.alpha_lstm > 0.99 and rel_good.separable is True
           and abs(rel_good.gated_p_drowsy - 0.90) < 1e-6)
    record("G5", "Corroborated drowsy verdict keeps full fusion weight",
           "When the CNN agrees (p_closed=0.95) and the window is in-distribution, "
           "alpha must saturate so the original 70/30 fusion is reproduced",
           "alpha>0.99; separable=True; gated==0.90",
           f"alpha={rel_good.alpha_lstm:.4f} sep={rel_good.separable} "
           f"gated={rel_good.gated_p_drowsy}", ok5)

    # ---- G6: real model + real extractor, end to end ----------------------
    svc = T.get_lstm_service()
    fe = T.VigilanceFeatureExtractor()
    buf = T.TemporalBuffer()
    ctx = {"ear": 0.30, "p_open_eff": 1.0, "mar": 0.10,
           "face_box": (100, 100, 200, 200), "eye_mid": (200.0, 174.0),
           "eye_mid_y": 174.0}
    for _ in range(40):                       # auto-calibrates at 30 samples
        f = fe.compute_features(ctx)
        buf.push(f)
    rel_real = T.gate_for_fusion(
        svc=svc, feature_extractor=fe, temporal_buffer=buf, p_cnn=0.0,
    )
    alpha_ok = 0.0 <= rel_real.alpha_lstm <= 1.0
    blend_ok = (min(0.0, rel_real.raw_p_drowsy) <= rel_real.gated_p_drowsy
                <= max(0.0, rel_real.raw_p_drowsy))
    ok6 = (svc.available and fe.calibrated and alpha_ok and blend_ok)
    record("G6", "Real LSTM + real vigilance extractor produce a bounded gate",
           "End-to-end path: 40 alert frames auto-calibrate the baseline, then the "
           "gate blends the real LSTM output between the geometric score and raw",
           "available; calibrated; alpha in [0,1]; gated between geo(=0) and raw",
           f"raw={rel_real.raw_p_drowsy:.4f} alpha={rel_real.alpha_lstm:.4f} "
           f"gated={rel_real.gated_p_drowsy:.4f} support={rel_real.support:.3f} "
           f"agreement={rel_real.agreement:.3f} "
           f"margin={rel_real.manifold_margin:.3f} grad={rel_real.gradient_norm:.4f}",
           ok6)

    # ---- G7: live fused score keeps the fatigue ramp direction -------------
    #     (mirrors the production fusion formula with alpha applied)
    def fused(raw, p_cnn, calibrated=True):
        r = T.gate_for_fusion(
            svc=_StubSvc(raw), feature_extractor=_extractor(calibrated),
            temporal_buffer=_buffer(), p_cnn=p_cnn)
        return 0.70 * p_cnn + 0.30 * r.gated_p_drowsy, r.alpha_lstm

    alert, a_alert = fused(0.99, 0.02)     # over-confident LSTM, eyes open
    drowsy, a_drowsy = fused(0.90, 0.90)   # corroborated closure
    ok7 = drowsy > 0.5 > alert and a_drowsy > a_alert
    record("G7", "Gated fusion preserves the alert -> drowsy ramp",
           "An over-confident LSTM on alert frames must not out-score a "
           "corroborated drowsy frame",
           "alert<0.5, drowsy>0.5, alpha_drowsy>alpha_alert",
           f"alert={alert:.4f} (a={a_alert:.3f}) drowsy={drowsy:.4f} "
           f"(a={a_drowsy:.3f})", ok7)

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", "test_results_phase5.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(RESULTS, fh, indent=2, ensure_ascii=False)

    n_pass = sum(1 for r in RESULTS if r["status"] == "PASS")
    print(f"\nPHASE 5 RESULT: {n_pass}/{len(RESULTS)} PASSED - {out}")
    for r in RESULTS:
        print(f"  [{r['status']}] {r['id']}: {r['name']}")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        traceback.print_exc()
        code = 2
    sys.exit(code)
