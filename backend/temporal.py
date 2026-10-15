"""Re-usable diagnostics and gating for the LSTM temporal model.

VigiDriver Model-1 temporal stream: a 4-feature per-frame vigilance vector
(EAR, CNN eye-open probability, mouth aspect ratio, vertical eye-position
proxy) buffered to a 30-frame window, standardized against per-driver
feature_mean/feature_std, and classified by best_vigidriver_lstm.keras into
P(drowsy). The fused output is gated by gate_for_fusion before it enters the
30/70 fusion with the geometric score.

Architecture (from models/best_vigidriver_lstm.keras zip config):
  InputLayer(batch_shape=(None, 30, 4))
  -> LSTM(64, return_sequences=True)
  -> Dropout(0.3)
  -> LSTM(32, return_sequences=False)
  -> Dropout(0.3)
  -> Dense(16, relu)
  -> Dense(1, sigmoid)   # P(drowsy)

Expected feature_mean/feature_std are stored as (1,1,4) arrays in
models/feature_mean.npy / models/feature_std.npy so they broadcast against
the (batch, 30, 4) standardized window tensor.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from typing import Any, NamedTuple, Optional

try:
    from .kmeans import KMeansGate  # noqa: PLC0415
    from .noise import REGIMES      # noqa: PLC0415
except ImportError:
    pass  # allowed as top-level import during tests

logger = logging.getLogger("vigidriver.temporal")

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

LSTM_MODEL_PATH = os.path.join(_MODELS_DIR, "best_vigidriver_lstm.keras")
LSTM_WEIGHTS_H5_PATH = os.path.join(_MODELS_DIR, "best_vigidriver_lstm.weights.h5")
FEATURE_MEAN_PATH = os.path.join(_MODELS_DIR, "feature_mean.npy")
FEATURE_STD_PATH = os.path.join(_MODELS_DIR, "feature_std.npy")

# Per-frame vigilance feature names — must match the order used by
# VigilanceFeatureExtractor.compute_features AND the training distribution
# captured in feature_mean/feature_std.
FEATURE_NAMES = ["EAR", "eye_openness_ratio", "mar", "eye_pos_proxy"]

# Production loop timing — LSTM inference runs every STRIDE frames so the
# CPU headroom stays healthy on a 15 FPS live loop.
WINDOW = 30
STRIDE = 5
INFERENCE_FPS = 15.0

# Self-gating constants (gate_for_fusion)
LSTM_CLIP_SIGMA = 3.0          # clip standardized features to +/- this many sigma
EYE_FRAC_UPRIGHT = 0.37       # expected eye-centre fraction within face height
PITCH_PROXY_GAIN = 2.0        # gain on the eye-position proxy feature
CONFIDENCE_THRESHOLD = 0.65   # separability confidence needed to trust LSTM fusion

# ---------------------------------------------------------------------------
# Feature statistics (loaded once, lazily)
# ---------------------------------------------------------------------------

_LSTM_MEAN: np.ndarray | None = None
_LSTM_STD: np.ndarray | None = None
_LSTM_MEAN_OK: bool | None = None
_LSTM_STD_OK: bool | None = None


def _load_feature_stats() -> tuple[np.ndarray, np.ndarray, bool, bool]:
    """Load feature_mean/feature_std from .npy files. Returns (mean, std,
    mean_ok, std_ok). On any failure returns canonical placeholders so the
    service still starts (standardization is then a no-op).
    """
    mean = np.zeros((1, 1, 4), dtype=np.float32)
    std = np.ones((1, 1, 4), dtype=np.float32)
    mean_ok = False
    std_ok = False
    try:
        m = np.load(FEATURE_MEAN_PATH)
        if m.shape == (1, 1, 4) and np.all(np.isfinite(m)):
            mean = m.astype(np.float32)
            mean_ok = True
    except Exception as exc:
        logger.warning("feature_mean.npy unavailable (%s); using zero mean", exc)
    try:
        s = np.load(FEATURE_STD_PATH)
        if s.shape == (1, 1, 4) and np.all(s > 0) and np.all(np.isfinite(s)):
            std = s.astype(np.float32)
            std_ok = True
        else:
            logger.warning("feature_std.npy malformed; using unit std")
    except Exception as exc:
        logger.warning("feature_std.npy unavailable (%s); using unit std", exc)
    return mean, std, mean_ok, std_ok


def lstm_feature_mean() -> np.ndarray:
    global _LSTM_MEAN, _LSTM_MEAN_OK
    if _LSTM_MEAN is None:
        _LSTM_MEAN, _, _LSTM_MEAN_OK, _ = _load_feature_stats()
    return _LSTM_MEAN


def lstm_feature_std() -> np.ndarray:
    global _LSTM_STD, _LSTM_STD_OK
    if _LSTM_STD is None:
        _, _LSTM_STD, _, _LSTM_STD_OK = _load_feature_stats()
    return _LSTM_STD


def lstm_stats_valid() -> bool:
    global _LSTM_MEAN_OK, _LSTM_STD_OK
    if _LSTM_MEAN_OK is None:
        lstm_feature_mean()
        lstm_feature_std()
    return bool(_LSTM_MEAN_OK and _LSTM_STD_OK)


# ---------------------------------------------------------------------------
# LSTM inference service
# ---------------------------------------------------------------------------

class LSTMSvc:
    """Lazy-loading inference service for best_vigidriver_lstm.keras.

    Loads the model on first use so the FastAPI lifespan startup (which
    eagerly loads the eye CNN) is not slowed down by TensorFlow init.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._model_path = LSTM_MODEL_PATH
        self._load_error: str | None = None
        self._ready = False
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._lock = threading.Lock()
        self._load_attempted = False

    @property
    def available(self) -> bool:
        return self._ready

    @property
    def mean(self) -> np.ndarray:
        if self._mean is None:
            self._mean = lstm_feature_mean()
        return self._mean

    @property
    def std(self) -> np.ndarray:
        if self._std is None:
            self._std = lstm_feature_std()
        return self._std

    def ensure_loaded(self) -> bool:
        """Eagerly load the LSTM model. Called by get_lstm_service() once."""
        if self._ready:
            return True
        with self._lock:
            if self._ready:
                return True
            if self._load_attempted:
                return self._ready
            self._load_attempted = True
            try:
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                import keras  # noqa: PLC0415
                if not os.path.exists(self._model_path):
                    if os.path.exists(LSTM_WEIGHTS_H5_PATH):
                        logger.warning(
                            "LSTM .keras archive missing; weights-only fallback "
                            "active — predict_window will raise at inference time.")
                        self._load_error = "LSTM .keras archive not found"
                        return False
                    self._load_error = f"LSTM model not found at {self._model_path}"
                    return False
                self._model = keras.saving.load_model(self._model_path, compile=False)
                self._ready = True
                logger.info(
                    "LSTM loaded: %s  input=%s  params=%s",
                    os.path.basename(self._model_path),
                    getattr(self._model, "input_shape", None),
                    f"{self._model.count_params():,}" if self._model is not None else "?",
                )
            except Exception as exc:
                self._load_error = str(exc)
                logger.error("LSTM load failed: %s", exc)
        return self._ready

    def model_info(self) -> dict[str, Any]:
        """Metadata about the loaded LSTM for the /api/model/temporal endpoint.

        Schema matches ModelInspectionPage.jsx expectations:
          available, trainable_parameters, input_shape, window_seconds,
          inference_stride, input_features, feature_normalization.
        """
        try:
            params = int(self._model.count_params()) if self._model is not None else 0
        except Exception:
            params = 30625  # known param count for this architecture
        return {
            "name": "VigiDriver Bi-LSTM Temporal Micro-Sleep Model",
            "path": self._model_path,
            "available": self._ready,
            "input_shape": [None, 30, 4],
            "output": "sigmoid P(drowsy)",
            "trainable_parameters": params,
            "architecture": [
                {"layer": "InputLayer", "args": {"batch_shape": [None, 30, 4]}},
                {"layer": "LSTM", "args": {"units": 64, "return_sequences": True}},
                {"layer": "Dropout", "args": {"rate": 0.3}},
                {"layer": "LSTM", "args": {"units": 32, "return_sequences": False}},
                {"layer": "Dropout", "args": {"rate": 0.3}},
                {"layer": "Dense", "args": {"units": 16, "activation": "relu"}},
                {"layer": "Dense", "args": {"units": 1, "activation": "sigmoid"}},
            ],
            "window_seconds": round(WINDOW / INFERENCE_FPS, 2),
            "inference_stride": STRIDE,
            "inference_fps": INFERENCE_FPS,
            "input_features": FEATURE_NAMES,
            "feature_normalization": {
                "mode": "z-score against per-driver baseline + population mean/std",
                "clip_sigma": LSTM_CLIP_SIGMA,
                "mean": [round(float(v), 5) for v in lstm_feature_mean().ravel()],
                "std": [round(float(v), 5) for v in lstm_feature_std().ravel()],
            },
        }

    def predict_window(self, z: np.ndarray) -> float:
        """Infer P(drowsy) from a standardized window tensor.

        Args:
            z: standardized feature window, shape (1, WINDOW, 4) float32.

        Returns:
            Scalar P(drowsy) in [0, 1].
        """
        if not self._ready or self._model is None:
            raise RuntimeError(
                f"LSTM not loaded{f': {self._load_error}' if self._load_error else ''}")
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 2:
            z = z[np.newaxis, :, :]
        with self._lock:
            out = self._model(z, training=False)
        pred = float(np.asarray(out).reshape(-1)[0])
        return float(np.clip(pred, 0.0, 1.0))


_LSTM_SVC_INSTANCE: LSTMSvc | None = None
_LSTM_SVC_LOCK = threading.Lock()


def get_lstm_service() -> LSTMSvc:
    """Singleton accessor for the LSTM inference service (lazy load)."""
    global _LSTM_SVC_INSTANCE
    if _LSTM_SVC_INSTANCE is None:
        with _LSTM_SVC_LOCK:
            if _LSTM_SVC_INSTANCE is None:
                _LSTM_SVC_INSTANCE = LSTMSvc()
                _LSTM_SVC_INSTANCE.ensure_loaded()
    return _LSTM_SVC_INSTANCE


# ---------------------------------------------------------------------------
# Per-frame vigilance feature extractor
# ---------------------------------------------------------------------------

@dataclass
class VigilanceFeatureExtractor:
    """Converts a per-frame detection result dict into a 4-d vigilance vector

    [EAR, eye_openness_ratio, mar, eye_pos_proxy] that matches the training
    distribution captured in feature_mean/feature_std.

    The extractor also maintains a rolling baseline offset (mean of the first
    N alert frames) so the LSTM standardization is per-driver calibrated.
    """

    window: int = 120                       # baseline calibration window
    _hist: list[np.ndarray] = field(default_factory=list)
    _baseline_offset: np.ndarray | None = None
    calibrated: bool = False
    _n_features: int = 4

    @property
    def baseline_offset(self) -> np.ndarray:
        """Per-driver baseline as a SHIFT relative to the population mean.

        The production formula is  z = (x - (mean + offset)) / std, so for a
        driver whose personal baseline equals `pb`, we need offset = pb - mean
        so that z = (x - mean - (pb - mean)) / std = (x - pb) / std. This
        centers each driver on their own alert baseline.
        """
        if self._baseline_offset is None:
            return np.zeros((1, 1, self._n_features), dtype=np.float32)
        pb = self._baseline_offset.astype(np.float32).reshape(1, 1, self._n_features)
        mu = lstm_feature_mean().astype(np.float32).reshape(1, 1, self._n_features)
        return (pb - mu)

    def compute_features(self, ctx: dict[str, Any]) -> np.ndarray:
        """Build a 4-d vigilance feature vector from a detection-context dict.

        Expected keys:
            ear:            float | None          — eye aspect ratio (landmarks)
            p_open_eff:     float                 — effective CNN eye-open prob
            mar:           float | None          — mouth aspect ratio
            face_box:      tuple[int,int,int,int] | None
            eye_mid:       tuple[float,float] | None
            eye_mid_y:     float | None

        Returns:
            (4,) float32 feature vector.
        """
        ear = ctx.get("ear")
        p_open_eff = float(ctx.get("p_open_eff", 0.5))
        mar = ctx.get("mar")
        face_box = ctx.get("face_box")
        eye_mid = ctx.get("eye_mid")
        eye_mid_y = ctx.get("eye_mid_y")

        # Feature 0: EAR — primary eye-closure signal.
        f0 = float(ear) if (ear is not None and np.isfinite(ear)) else 0.27

        # Feature 1: effective eye-open probability (CNN + EAR combined).
        f1 = float(np.clip(p_open_eff, 0.0, 1.0))

        # Feature 2: mouth aspect ratio (yawn signal).
        f2 = float(mar) if (mar is not None and np.isfinite(mar)) else 0.0

        # Feature 3: vertical eye-position proxy within the face box.
        # Normal driving ≈ eye_centre at 0.37 of face height → proxy ≈ 3.7,
        # matching the training mean of ~3.86. When the head nods the proxy
        # shifts; the LSTM uses this as a heads-up/tilt cue.
        f3 = 3.862  # fallback to training mean so missing data is neutral
        if face_box is not None and eye_mid_y is not None:
            fx, fy, fw, fh = face_box
            if fh > 0 and np.isfinite(eye_mid_y):
                frac = (eye_mid_y - fy) / fh
                if np.isfinite(frac) and frac > 0:
                    f3 = float(frac * 10.0)

        feat = np.array([f0, f1, f2, f3], dtype=np.float32)

        # Update rolling baseline for per-driver calibration.
        self._hist.append(feat.copy())
        if len(self._hist) > self.window:
            self._hist = self._hist[-self.window:]
        if (not self.calibrated) and len(self._hist) >= 30:
            arr = np.stack(self._hist, axis=0)
            self._baseline_offset = arr.mean(axis=0, keepdims=True).astype(np.float64)
            self.calibrated = True

        return feat

    def reset(self) -> None:
        """Clear the rolling baseline so the next session recalibrates from scratch."""
        self._hist = []
        self._baseline_offset = None
        self.calibrated = False


# ---------------------------------------------------------------------------
# Temporal buffer (ring buffer of per-frame feature vectors)
# ---------------------------------------------------------------------------

class TemporalBuffer:
    """Fixed-size ring buffer holding the last WINDOW feature vectors.

    Buffers (4,) feature vectors; exposes .window_tensor(mean, std, offset=)
    which returns a (1, WINDOW, 4) standardized float32 tensor ready for the
    LSTM, and .ready which is True once WINDOW frames have been pushed.
    """

    def __init__(self, capacity: int = WINDOW) -> None:
        self._capacity = capacity
        self._buf: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self._buf) >= self._capacity

    def push(self, feat: np.ndarray) -> None:
        feat = np.asarray(feat, dtype=np.float32).ravel()
        assert feat.shape == (4,), feat.shape
        self._buf.append(feat.copy())
        if len(self._buf) > self._capacity:
            self._buf = self._buf[-self._capacity:]

    def __len__(self) -> int:
        return len(self._buf)

    def window_tensor(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        offset: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a (1, WINDOW, 4) standardized tensor for the LSTM.

        Standardization:  z = (x - (mean + offset)) / (std + 1e-8),
        clipped to +/- LSTM_CLIP_SIGMA sigma.
        """
        if len(self._buf) < self._capacity:
            pad = [mean.ravel()] * (self._capacity - len(self._buf))
            buf = list(self._buf) + list(pad)
        else:
            buf = list(self._buf)[-self._capacity:]
        x = np.stack(buf, axis=0).astype(np.float32)          # (WINDOW, 4)
        x = x[np.newaxis, :, :]                                # (1, WINDOW, 4)
        mu = np.asarray(mean, dtype=np.float32).reshape(1, 1, 4)
        sd = np.asarray(std, dtype=np.float32).reshape(1, 1, 4)
        off = np.zeros((1, 1, 4), dtype=np.float32)
        if offset is not None:
            off = np.asarray(offset, dtype=np.float32).reshape(1, 1, 4)
        z = (x - (mu + off)) / (sd + 1e-8)
        z = np.clip(z, -LSTM_CLIP_SIGMA, LSTM_CLIP_SIGMA).astype(np.float32)
        return z

    def reset(self) -> None:
        self._buf = []


# ---------------------------------------------------------------------------
# Self-gating diagnostics
# ---------------------------------------------------------------------------
#
# The Bi-LSTM is a small (30 x 4) sequence model. On inputs outside its
# training support — or before the per-driver baseline has calibrated — it can
# return extreme, over-confident probabilities. We have observed
# P(drowsy) ~= 0.9996 on a live frame whose geometric eye signal reported
# "wide open"; fusing that at a fixed 30 % weight injects a false fatigue
# alert (and, worse, a false CRITICAL escalation).
#
# `gate_for_fusion` therefore measures how far the temporal stream can be
# trusted *this frame* and returns an effective weight alpha in [0, 1]:
#
#     alpha = calibration_ready          (per-driver baseline established)
#           x manifold_support           (fraction of the window inside +/-3 sigma)
#           x cnn_agreement              (agreement with the per-frame eye signal)
#
# The caller then blends with
#
#     p_fused = 0.70 * p_geo + 0.30 * (alpha * p_lstm + (1 - alpha) * p_geo)
#
# so alpha = 0 exactly neutralises the temporal stream (p_fused == p_geo) while
# alpha = 1 reproduces the original 70/30 fusion. The gate degrades smoothly,
# which matters because a hard on/off switch would make the drowsiness score
# jump whenever the LSTM flickered in and out of trust.


class LSTMReliabilityResult(NamedTuple):
    """Diagnostics + gated output produced by :func:`gate_for_fusion`."""
    raw_p_drowsy: float        # ungated LSTM output for this window
    gated_p_drowsy: float      # LSTM output blended against the geometric score
    alpha_lstm: float          # effective temporal fusion weight, 0..1
    separable: bool            # True when alpha clears confidence_threshold
    confidence: float          # == alpha (retained for API compatibility)
    calibration: bool          # per-driver baseline calibrated
    manifold_margin: float     # clip_sigma - max|z|  (negative => OOD window)
    support: float             # fraction of |z| <= clip_sigma
    agreement: float           # agreement with the per-frame eye signal
    gradient_norm: float       # ||d loss / d z||_2 at the standardized input


def gradient_norm(model, z: np.ndarray) -> float:
    """L2 norm of d(-log P(alert)) / dz at the standardized input.

    A large norm means the LSTM sits on a steep part of its decision surface,
    i.e. a tiny input perturbation would flip its verdict — evidence that the
    current window is not a stable operating point.
    """
    try:
        import tensorflow as tf  # noqa: PLC0415
    except Exception:
        return 0.0
    if model is None:
        return 0.0
    try:
        x = tf.convert_to_tensor(np.asarray(z, dtype=np.float32))
        with tf.GradientTape() as tape:
            tape.watch(x)
            out = model(x, training=False)[0, 0]
            loss = -tf.math.log(tf.clip_by_value(1.0 - out, 1e-6, 1.0))
        grad = tape.gradient(loss, x)
    except Exception:
        return 0.0
    if grad is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(grad)))


# Backwards-compatible alias used by earlier diagnostics scripts.
compute_gradient_norm = gradient_norm


def _manifold_support(z: np.ndarray, clip_sigma: float = LSTM_CLIP_SIGMA) -> float:
    """Fraction of the standardized window inside the training +/-sigma support.

    1.0 means every cell is in-distribution; 0.0 means the whole window is
    outside the support the LSTM was fitted on.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.size == 0:
        return 0.0
    return float(np.mean(np.abs(z) <= clip_sigma))


def _cnn_agreement(raw_p_drowsy: float, p_cnn: float | None,
                   tolerance: float = 0.35) -> float:
    """Agreement between the LSTM verdict and the per-frame eye signal.

    A gap up to ``tolerance`` is free: the LSTM legitimately tracks closure
    *velocity* over the 2 s window and may lead the frame-level signal. Beyond
    that the agreement decays linearly and reaches exactly 0 at a full-scale
    disagreement (e.g. LSTM 1.0 while the CNN says the eyes are wide open).
    """
    if p_cnn is None or not np.isfinite(p_cnn):
        return 1.0
    tolerance = float(np.clip(tolerance, 0.0, 0.99))
    gap = abs(float(raw_p_drowsy) - float(p_cnn))
    excess = max(0.0, gap - tolerance)
    return float(np.clip(1.0 - excess / (1.0 - tolerance), 0.0, 1.0))


def _build_window_tensor(
    *,
    buf: list[np.ndarray],
    capacity: int,
    mean: np.ndarray,
    std: np.ndarray,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble a (1, capacity, 4) standardized window from raw feature rows.

    Short buffers are left-padded with the population mean so the tensor is
    always the shape the LSTM expects.
    """
    mu = np.asarray(mean, dtype=np.float32).reshape(1, 1, 4)
    sd = np.asarray(std, dtype=np.float32).reshape(1, 1, 4)
    off = np.zeros((1, 1, 4), dtype=np.float32)
    if offset is not None:
        off = np.asarray(offset, dtype=np.float32).reshape(1, 1, 4)

    x = np.stack(buf, axis=0).astype(np.float32) if buf else np.zeros((0, 4), np.float32)
    if x.shape[0] < capacity:
        pad = np.repeat(mu.reshape(1, 4), capacity - x.shape[0], axis=0)
        x = np.concatenate([pad, x], axis=0)   # left-pad: history precedes
    else:
        x = x[-capacity:]
    x = x[np.newaxis, :, :]

    z = (x - (mu + off)) / (sd + 1e-8)
    return np.clip(z, -LSTM_CLIP_SIGMA, LSTM_CLIP_SIGMA).astype(np.float32)


def gate_for_fusion(
    *,
    svc,
    feature_extractor,
    temporal_buffer=None,
    current_features=None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    geo_p: float | None = None,
    p_cnn: float | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> LSTMReliabilityResult:
    """Measure LSTM reliability for the current window and gate its output.

    Args:
        svc: the :class:`LSTMSvc` singleton.
        feature_extractor: the per-driver :class:`VigilanceFeatureExtractor`
            (its ``baseline_offset`` provides per-driver centering and its
            ``calibrated`` flag is the hard calibration gate).
        temporal_buffer: live :class:`TemporalBuffer`; its contents are used
            when ``current_features`` is not supplied.
        current_features: optional list of per-frame context dicts (offline /
            diagnostic use) pushed into a scratch buffer.
        mean, std: standardization stats; default to the service's.
        geo_p: geometric/physiological drowsiness score for this frame. When
            given, it is the target the gated LSTM output is blended toward.
        p_cnn: per-frame P(eye closed) used for the agreement term.
        confidence_threshold: alpha above which the stream counts as reliable.

    Returns:
        An :class:`LSTMReliabilityResult`. Never raises — an unavailable or
        failing model yields alpha = 0 and a neutral result so the live loop
        can call this unconditionally.
    """
    mu = svc.mean if mean is None else mean
    sd = svc.std if std is None else std
    neutral_base = float(np.clip(geo_p, 0.0, 1.0)) if geo_p is not None else 0.0

    def _neutral(reason_grad: float = 0.0) -> LSTMReliabilityResult:
        return LSTMReliabilityResult(
            raw_p_drowsy=neutral_base,
            gated_p_drowsy=neutral_base,
            alpha_lstm=0.0,
            separable=False,
            confidence=0.0,
            calibration=bool(getattr(feature_extractor, "calibrated", False)),
            manifold_margin=0.0,
            support=0.0,
            agreement=0.0,
            gradient_norm=reason_grad,
        )

    try:
        if not getattr(svc, "available", False):
            return _neutral()

        capacity = int(getattr(temporal_buffer, "_capacity", WINDOW))
        if current_features is not None:
            buf: list[np.ndarray] = []
            for ctx in current_features:
                f = feature_extractor.compute_features(ctx)
                buf.append(np.asarray(f, dtype=np.float32).ravel())
                if len(buf) > capacity:
                    buf = buf[-capacity:]
            # Only clamp the offset once the scratch buffer is representative.
            offset = None
        else:
            buf = list(getattr(temporal_buffer, "_buf", []))
            offset = feature_extractor.baseline_offset

        if not buf:
            return _neutral()

        z = _build_window_tensor(
            buf=buf, capacity=capacity, mean=mu, std=sd, offset=offset)

        raw = float(svc.predict_window(z))
        calibrated = bool(getattr(feature_extractor, "calibrated", False))
        support = _manifold_support(z)
        agreement = _cnn_agreement(raw, p_cnn)

        alpha = (1.0 if calibrated else 0.0) * support * agreement
        alpha = float(np.clip(alpha, 0.0, 1.0))

        gated = alpha * raw + (1.0 - alpha) * neutral_base
        margin = LSTM_CLIP_SIGMA - float(np.abs(z).max())
        gnorm = gradient_norm(getattr(svc, "_model", None), z)

        return LSTMReliabilityResult(
            raw_p_drowsy=raw,
            gated_p_drowsy=float(np.clip(gated, 0.0, 1.0)),
            alpha_lstm=alpha,
            separable=bool(alpha > confidence_threshold),
            confidence=alpha,
            calibration=calibrated,
            manifold_margin=float(margin),
            support=support,
            agreement=agreement,
            gradient_norm=gnorm,
        )
    except Exception as exc:  # defensive: never break the live loop
        logger.warning("gate_for_fusion failed: %s", exc)
        return _neutral()


def _maybe_standardize(feat, mean, std, offset=None) -> np.ndarray:
    """Standardize a feature row against (mean + offset) / std, clipped to 3σ.

    ``offset`` is the per-driver baseline *shift* as returned by
    :attr:`VigilanceFeatureExtractor.baseline_offset`.
    """
    x = np.asarray(feat, dtype=np.float64)
    mu = np.asarray(mean, dtype=np.float64).reshape(x.shape)
    sd = np.asarray(std, dtype=np.float64).reshape(x.shape) + 1e-8
    off = np.zeros_like(mu)
    if offset is not None:
        off = np.asarray(offset, dtype=np.float64).reshape(x.shape)
    if not np.isfinite(off).all():
        off = np.zeros_like(mu)
    z = (x - (mu + off)) / sd
    return np.clip(z, -LSTM_CLIP_SIGMA, LSTM_CLIP_SIGMA).astype(np.float32)


@dataclass
class AlphaLstm:
    """Compact soft-gating carrier (alpha, separable flag, confidence, gated p)."""
    alpha: float
    separable: bool
    confidence: float
    gated_p: float


def alpha_from_result(res: LSTMReliabilityResult) -> AlphaLstm:
    """Adapt a full :class:`LSTMReliabilityResult` into a compact :class:`AlphaLstm`."""
    return AlphaLstm(
        alpha=res.alpha_lstm,
        separable=res.separable,
        confidence=res.confidence,
        gated_p=res.gated_p_drowsy,
    )
